#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import random
import cv2
import numpy as np
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_tex_3dgs
import sys
import torch.nn.functional as F
from scene import Scene, GaussianModel, SplatFieldsModel
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from extract_geo import query_nn, morans_loss
import wandb
DISABLE_WANDB_IMAGES = True
import torchvision
import datetime
import os  

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False




def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    # logs
    date_str = datetime.datetime.now().strftime("%m%d-%H%M%S")
    filedir = 'log_train'
    filedir += date_str
    log_dir = os.path.join(args.model_path, filedir)
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(log_dir, False, fname="log.out")
    sys.stderr = Logger(log_dir, False, fname="log.err")
    
    return tb_writer
 

def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic= True
        torch.backends.cudnn.benchmark = False


def save_torch_img(file, path):
    try:
        torchvision.utils.save_image(file, path)
    except Exception as e:
        print(f"[WARN] save image failed: {e}")


def get_gaussian_dict(_viewpoint_cam, gaussians, deform, iteration=None, warm_up=None, static=False, n_splats=-1):
    fid = _viewpoint_cam.fid.cuda()
    if static:
        fid = 0*fid
    # animate
    if static or (iteration is not None and warm_up is not None and iteration < warm_up):
        gaussian_dict = {
            'means3D': gaussians.get_xyz,
            'active_sh_degree': gaussians.active_sh_degree, # 0
            'gaussian_opacity': gaussians.get_opacity, # N,1 in range [0,1]
            'gaussian_features': gaussians.get_features, # N, 16, 3
            'gaussian_scales': gaussians.get_scaling, # N,3
            'gaussian_rotations': gaussians.get_rotation, # N,4
        }
        overwrite_attributes = None
    else:
        active_sh_degree = gaussians.active_sh_degree
        xyz = gaussians.get_xyz.detach()
        scaling = gaussians.get_scaling.detach()
        gaussian_features = gaussians.get_features
        if n_splats > 0 and n_splats < xyz.shape[0]: # sample splats if needed
            idx = torch.randperm(xyz.shape[0])[:n_splats]
            xyz = xyz[idx]
            scaling = scaling[idx]
            gaussian_features = gaussian_features[idx]
        else:
            idx = None

        N = xyz.shape[0]
        time_input = fid.detach().unsqueeze(0).expand(N, -1)
        ret = deform.step(xyz, time_input)
        gaussian_dict = {
            'idx': idx,
            'means3D': ret['means3D'],
            'active_sh_degree': active_sh_degree, # 0
            'gaussian_opacity': ret['opacity'], # N,1 in range [0,1]
            'gaussian_scales': torch.clamp(ret['scales'] + scaling, 1e-6, 0.3),
            'gaussian_rotations': ret['rotations'], # N,4
            'gradient_error': ret.get('gradient_error', None),
            'color_tex': ret.get('color_tex', None),
            'opacity_tex': ret.get('opacity_tex', None)
        }
        if 'gaussian_features' in ret:
            gaussian_dict['gaussian_features'] = ret['gaussian_features'].view(gaussian_features.shape)*0.1 + gaussian_features
        elif 'rgb' in ret:
            gaussian_dict['gaussian_rgb'] = ret['rgb']
        elif 'rgb_fnc' in ret:
            gaussian_dict['gaussian_rgb_fnc'] = ret['rgb_fnc']
        else:
            gaussian_dict['gaussian_features'] = gaussian_features


        overwrite_attributes = {}
        overwrite_attributes["xyz"] = gaussian_dict['means3D']
        # gaussian_features = gaussian_dict.get('gaussian_features', None)
        # if gaussian_features is None:
        #     gaussian_features = gaussians.get_features 
        f_dc, f_rest = gaussians.get_features.split([
            gaussians._features_dc.shape[1], 
            gaussians._features_rest.shape[1]
        ], dim=1)
        overwrite_attributes["f_dc"] = f_dc
        overwrite_attributes["f_rest"] = f_rest
        overwrite_attributes["opacity"] = gaussian_dict['gaussian_opacity']
        overwrite_attributes["scaling"] = ret['scales']
        overwrite_attributes["rotation"] = gaussian_dict['gaussian_rotations']

    return gaussian_dict, overwrite_attributes

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, resume_ckpt, resume_iteration):
    _n_frames = dataset.load_time_step
    setattr(hyper, 'n_frames', _n_frames if _n_frames > 1 else 0)
    is_static = dataset.is_static
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.use_isotropic = hyper.use_isotropic
    ENABLE_G_OPT = not opt.disable_gaussian_opt #True
    scene = Scene(dataset, gaussians)
    deform = SplatFieldsModel(hyper, radius=scene.cameras_extent)
    deform.train_setting(opt)

    if resume_ckpt:
        deform.load_weights(resume_ckpt, resume_iteration)
        gaussians.load_ply(os.path.join(resume_ckpt, 'point_cloud',f'iteration_{resume_iteration}', 'point_cloud.ply'))

    gaussians.training_setup(opt)
    gaussians.iteration = 0
    lambda_mask = opt.lambda_mask
    lambda_norm = opt.lambda_norm
    lambda_corr = opt.lambda_corr
    lambda_corr_color = opt.lambda_corr_color
    lambda_norm_mean = opt.lambda_norm_mean
    lambda_depth = opt.lambda_depth
    lambda_opacity = opt.lambda_opacity
    lambda_depthl1 = opt.lambda_depthl1
    lambda_gradient = opt.lambda_gradient
    overwrite_loc = opt.overwrite_loc

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    # smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    N_SPLATS = opt.n_splats
    for iteration in range(1, opt.iterations + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if ENABLE_G_OPT and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        _viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack) - 1)]
        gaussian_dict, overwrite_attributes = get_gaussian_dict(_viewpoint_cam, gaussians, deform, iteration, opt.warm_up, static=is_static, n_splats=N_SPLATS)
        if iteration > 1500 and overwrite_loc:
            # gaussians.overwrite_loc(mean3D)
            gaussians._xyz = gaussians._xyz*0 + gaussian_dict['means3D'].clone().detach()

        # pick training views
        if opt.all_training:
            viewpoint_cam_list = [_vp for _vp in viewpoint_stack if _vp.fid == _viewpoint_cam.fid]
        else:
            viewpoint_cam_list = [_viewpoint_cam]
        random.shuffle(viewpoint_cam_list)
        viewpoint_cam_list = viewpoint_cam_list[:opt.num_views]
        
        loss_list, Ll1_list, loss_mask_list, loss_depth_list, loss_depthl1_list = [], [], [], [], []
        loss_corr = []
        loss_corr_color = []
        # _ind = 0
        for viewpoint_cam in viewpoint_cam_list:
            # print(f'Processing {_ind}/{len(viewpoint_cam_list)}')
            # _ind += 1
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()

            # Render
            render_pkg_re = render_tex_3dgs(viewpoint_cam, gaussian_dict, pipe, background, return_opacity=lambda_mask > 0.0)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

            gt_image = viewpoint_cam.original_image.cuda()
            _Ll1 = l1_loss(image, gt_image)

            _loss = (1.0 - opt.lambda_dssim) * _Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            # mask loss
            _loss_mask, _loss_depth, _loss_depthl1 = torch.tensor(0).cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda()
            _loss_corr = torch.tensor(0).cuda()
            _loss_corr_color = torch.tensor(0).cuda()
            if lambda_mask > 0.0:
                opacity_image = torch.clamp(render_pkg_re["opacity"], 0.0, 1.0)
                gt_mask = viewpoint_cam.mask.cuda()
                _loss_mask = F.l1_loss(opacity_image.view(-1), gt_mask.view(-1))
                _loss += lambda_mask*_loss_mask

            if lambda_norm > 0.0:
                _loss_norm = gaussian_dict['means3D'].norm(dim=1).mean()
                _loss += lambda_norm*_loss_norm
            if lambda_norm_mean > 0.0:
                mean_val = gaussian_dict['means3D'].detach().mean(dim=0, keepdim=True)
                _loss_norm = (gaussian_dict['means3D']-mean_val).norm(dim=1).mean()
                _loss += lambda_norm_mean*_loss_norm

            if lambda_corr > 0.0:
                weight_mat, nn_ix = query_nn(gaussian_dict['means3D'])
                moran_scale = morans_loss(weight_mat, gaussian_dict['gaussian_scales'][nn_ix])
                moran_rotation = morans_loss(weight_mat, gaussian_dict['gaussian_rotations'][nn_ix])
                moran_opacity = morans_loss(weight_mat, gaussian_dict['gaussian_opacity'][nn_ix])
                moran_rgb = morans_loss(weight_mat, gaussian_dict['gaussian_features'].view(gaussian_dict['gaussian_features'].shape[0], -1)[nn_ix])
                _loss_corr = moran_scale + moran_rotation + moran_opacity + moran_rgb
                _loss += lambda_corr*_loss_corr

            if lambda_corr_color > 0.0:
                weight_mat, nn_ix = query_nn(gaussian_dict['means3D'])
                _loss_corr_color = morans_loss(weight_mat, gaussian_dict['gaussian_features'].view(gaussian_dict['gaussian_features'].shape[0], -1)[nn_ix])
                _loss += lambda_corr*_loss_corr_color

            if lambda_depth > 0.0:
                gt_depth = viewpoint_cam.depth.cuda().squeeze()
                _dmask = gt_depth > 0
                rnd_depth = render_pkg_re["depth"].squeeze()
                _loss_depth = ssim((rnd_depth*_dmask).unsqueeze(-1), (gt_depth*_dmask).unsqueeze(-1))
                _loss += lambda_depth*_loss_depth

            if lambda_depthl1 > 0.0:
                gt_depth = viewpoint_cam.depth.cuda().squeeze()
                _dmask = gt_depth > 0
                rnd_depth = render_pkg_re["depth"].squeeze()
                _loss_depthl1 = F.l1_loss((rnd_depth*_dmask).unsqueeze(-1), (gt_depth*_dmask).unsqueeze(-1))
                _loss += lambda_depthl1*_loss_depthl1
            if 'opacity_tex' in render_pkg_re and render_pkg_re['opacity_tex'] is not None:
                opacity_tex_norm_loss = render_pkg_re["opacity_tex"].abs().mean()
                color_tex_norm_loss = render_pkg_re["color_tex"].abs().mean()
                _loss += opacity_tex_norm_loss * 0.1 + color_tex_norm_loss * 0.01
                # if iteration % 100 == 0:
                #     # print(f'[DEBUG] opacity_tex_norm mean:{render_pkg_re["opacity_tex"].abs().mean()}, color_tex_norm mean:{render_pkg_re["color_tex"].abs().mean()}')
                #     print(f'[DEBUG] opacity_tex_norm mean max:{render_pkg_re["opacity_tex"].abs().mean()}, {render_pkg_re["opacity_tex"].abs().max()}, color_tex_norm mean max:{render_pkg_re["color_tex"].abs().mean()}, {render_pkg_re["color_tex"].abs().max()}')


            loss_list.append(_loss)
            Ll1_list.append(_Ll1)
            loss_mask_list.append(_loss_mask)
            loss_depth_list.append(_loss_depth)
            loss_depthl1_list.append(_loss_depthl1)
            loss_corr.append(_loss_corr)
            loss_corr_color.append(_loss_corr_color)

            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device('cpu')
                
        if iteration % 100 == 0:
            rgb = render_pkg_re["render"]
            gt_rgb = viewpoint_cam.original_image
            os.makedirs(f"{dataset.model_path}/log_images", exist_ok = True)
            final_vis = np.concatenate([gt_rgb.permute(1,2,0).detach().cpu().numpy(),
                                        rgb.permute(1,2,0).detach().cpu().clamp(0.0, 1.0).numpy(),
                                        ], axis=1)[..., ::-1] * 255
            cv2.imwrite(f"{dataset.model_path}/log_images/rgb_{iteration}.jpg", final_vis)

        loss = sum(loss_list)/len(loss_list)
        loss_opacity = torch.tensor(0).cuda()
        if lambda_opacity > 0.0:
            loss_opacity = ((gaussian_dict['gaussian_opacity'] - 1.0)**2).mean()
            loss += lambda_opacity*loss_opacity
        loss_gradient = torch.tensor(0).cuda()
        if lambda_gradient > 0.0 and gaussian_dict.get('gradient_error', None) is not None:
            loss_gradient = gaussian_dict['gradient_error']
            loss += lambda_gradient*loss_gradient        

        loss.backward()
        def _sanitize_grads(optimizer):
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None: 
                        continue
                    if not torch.isfinite(p.grad).all():
                        p.grad = torch.nan_to_num(p.grad, 0.0, 1e3, -1e3)  # 或者置 None 跳过该参数更新
                        # p.grad = None

        if ENABLE_G_OPT:
            _sanitize_grads(gaussians.optimizer)
        _sanitize_grads(deform.optimizer)

        torch.nn.utils.clip_grad_norm_(deform.deform.parameters(), 1)
        _idx = gaussian_dict.get('idx', None)
        torch.cuda.empty_cache()
        loss_dict = {
            'mask': sum(loss_mask_list).detach()/len(loss_mask_list),
            'depth': sum(loss_depth_list).detach()/len(loss_depth_list),
            'depthl1': sum(loss_depthl1_list).detach()/len(loss_depthl1_list),
            'corr': sum(loss_corr).detach()/len(loss_corr),
            'corr_color': sum(loss_corr_color).detach()/len(loss_corr_color),
            'opacity': loss_opacity,
            'loss_gradient': loss_gradient, 
        }
        loss_dict.update(deform.log_variables())
        Ll1 = sum(Ll1_list).detach()/len(Ll1_list)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if ENABLE_G_OPT:
                if _idx is None:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                else:
                    _max_radii2d = torch.empty_like(gaussians.max_radii2D[_idx], device="cuda")
                    _max_radii2d[visibility_filter] = torch.max(gaussians.max_radii2D[_idx][visibility_filter], radii[visibility_filter])
                    gaussians.max_radii2D[_idx] = _max_radii2d

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss.detach(), l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render_tex_3dgs, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, loss_dict=loss_dict, is_static=is_static)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                vis_geometric = args.vis_geometric
                if not vis_geometric:
                    overwrite_attributes = None
                scene.save(iteration, overwrite_attributes=overwrite_attributes, vis_geometric=vis_geometric)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if ENABLE_G_OPT and iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, _idx=_idx)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    opacity = gaussian_dict['gaussian_opacity']
                    scale = gaussian_dict['gaussian_scales']
                    
                    # dead_mask = (opacity <= 0.001).squeeze(-1)
                    # print('[opacity statistic]:', opacity.min().item(), opacity.max().item(), opacity.mean().item())
                    # print('[scale statistic]:', scale.min().item(), scale.max().item(), scale.mean().item())

                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.001, scene.cameras_extent, size_threshold, opacity, scale)


            # Optimizer step
            if iteration < opt.iterations:
                if ENABLE_G_OPT:
                    gaussians.optimizer.step()
                    gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                if ENABLE_G_OPT:
                    gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
        
        del gaussian_dict

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))

class Logger(object):
    def __init__(self, summary_dir, silent=False, fname="logfile.txt"):
        self.terminal = sys.stdout
        self.silent = silent
        self.log = open(os.path.join(summary_dir, fname), "a") 
        cmdline = " ".join(sys.argv)+"\n"
        self.log.write(cmdline) 
    def write(self, message):
        if not self.silent: 
            self.terminal.write(message)
        self.log.write(message) 
    def flush(self):
        self.terminal.flush()
        self.log.flush()
   
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, loss_dict=dict(), is_static=False):
    import torchvision
    import numpy as np
    import cv2
    import torch

    def compute_psnr_torch(pred, gt, mask=None, eps=1e-12):
        """
        pred, gt: [3,H,W], float[0,1]
        mask: [H,W] in {0,1}, optional
        """
        pred = pred.float()
        gt = gt.float()
        if mask is not None:
            m = (mask > 0).float()  # [H,W]
            m3 = m.unsqueeze(0)
            diff2 = ((pred - gt) ** 2) * m3
            denom = m.sum() * pred.shape[0]  # Valid pixels * number of channels
            mse = diff2.sum() / (denom.clamp_min(1.0))
        else:
            mse = torch.mean((pred - gt) ** 2)
        if mse <= eps:
            return torch.tensor(float('inf'), device=pred.device)
        return 20.0 * torch.log10(1.0 / torch.sqrt(mse + 1e-12))

    def make_diff_rgb(pred, gt, mask=None):
        """
        Directly returns RGB difference map (abs(pred-gt)), uint8 RGB [H,W,3]
        pred/gt: [3,H,W] float[0,1]; mask: [H,W] in {0,1}
        """
        with torch.no_grad():
            a = pred.clamp(0,1)
            b = gt.clamp(0,1)
            if mask is not None:
                m = (mask > 0).float()  # [H,W]
                a = a * m.unsqueeze(0)
                b = b * m.unsqueeze(0)
            diff = (a - b).abs().detach().cpu()  # [3,H,W]
            diff_np = (diff.numpy().transpose(1,2,0) * 255.0).astype(np.uint8)  # RGB
            return diff_np

    def draw_text_with_bg_np(rgb_img_np, text, x, y, font_scale=0.7, thickness=1):
        """
        Draws text on a numpy RGB image with a semi-transparent black background
        """
        bgr = cv2.cvtColor(rgb_img_np, cv2.COLOR_RGB2BGR)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x0 = max(0, x - 6); y0 = max(0, y - th - 8)
        x1 = min(bgr.shape[1], x0 + tw + 12); y1 = min(bgr.shape[0], y0 + th + 12)
        overlay = bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        alpha = 0.5
        bgr[y0:y1, x0:x1] = (overlay[y0:y1, x0:x1] * alpha + bgr[y0:y1, x0:x1] * (1 - alpha)).astype(np.uint8)
        cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        for key, val in loss_dict.items():
            tb_writer.add_scalar('train_loss_patches/{}'.format(key), val.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    wandb_log_dict = {
    "train_loss_patches/l1_loss": Ll1.item(),
    "train_loss_patches/total_loss": loss.item(),
    "iter_time": elapsed,
    }
    for key, val in loss_dict.items():
        wandb_log_dict[f"train_loss_patches/{key}"] = val.item()

    wandb.log(wandb_log_dict, step=iteration)

    test_psnr = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test_list, psnr_test_list = [], []
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()

                    gaussian_dict, _ = get_gaussian_dict(viewpoint, scene.gaussians, deform, static=is_static)

                    has_mask = viewpoint.mask is not None
                    render_out = renderFunc(viewpoint, gaussian_dict, *renderArgs, return_opacity=has_mask)
                    has_mask = ("opacity" in render_out) and has_mask and (render_out['opacity'] is not None)

                    image = torch.clamp(render_out["render"], 0.0, 1.0)          # [3,H,W]
                    gt_image = torch.clamp(viewpoint.original_image.to(image.device), 0.0, 1.0)

                    gt_mask = None
                    if has_mask:
                        gt_mask = torch.clamp(viewpoint.mask.to(image.device), 0.0, 1.0).squeeze()  # [H,W]
                        opacity = torch.clamp(render_out["opacity"], 0.0, 1.0).squeeze()            # [H,W]
                        mask_vis = torch.cat((gt_mask, opacity), dim=1).unsqueeze(0).repeat_interleave(3, dim=0)

                    # metrics
                    if has_mask:
                        m3 = gt_mask.unsqueeze(0)
                        l1_test_list.append(torch.mean((image - gt_image).abs() * m3))
                    else:
                        l1_test_list.append(l1_loss(image, gt_image))
                    # psnr_val = compute_psnr_torch(image, gt_image, mask=gt_mask)
                    psnr_val = compute_psnr_torch(image, gt_image)
                    psnr_test_list.append(psnr_val)

                    # ===== Visualization: Direct diff_rgb + PSNR annotation =====
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                        if has_mask:
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render_mask", mask_vis[None], global_step=iteration)

                        # depth (original logic)
                        depth = render_out["depth"].squeeze()
                        if has_mask:
                            depth = depth * gt_mask
                        if viewpoint.depth is not None:
                            depth_vis = torch.cat((depth, viewpoint.depth.squeeze()), dim=1).unsqueeze(0).repeat_interleave(3, dim=0)
                        else:
                            depth_vis = depth.unsqueeze(0).repeat_interleave(3, dim=0)
                        depth_vis = torch.clip(depth_vis, DEPTH_MIN, depth_vis.max())
                        depth_vis = (depth_vis - DEPTH_MIN) / (depth_vis.max() - DEPTH_MIN + 1e-6)
                        depth_vis = torch.clip(depth_vis, 0.0, 1.0)
                        _depth_vis = depth_vis.cpu().numpy().transpose(1,2,0)
                        _depth_vis = cv2.applyColorMap((_depth_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                        depth_vis = torch.from_numpy(_depth_vis.transpose(2,0,1)).to(image.device).float() / 255.0
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render_depth", depth_vis[None], global_step=iteration)

                        # diff_rgb panel: GT | Pred (with PSNR) | Diff (with PSNR)
                        diff_rgb = make_diff_rgb(image, gt_image, mask=gt_mask)                   # np RGB
                        gt_np   = (gt_image.clamp(0,1).detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
                        pred_np = (image.clamp(0,1).detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

                        psnr_text = f"PSNR: {float(psnr_val):.2f}"
                        pred_np = draw_text_with_bg_np(pred_np, psnr_text, 10, 26, font_scale=0.7, thickness=1)
                        diff_rgb = draw_text_with_bg_np(diff_rgb, psnr_text, 10, 26, font_scale=0.7, thickness=1)

                        panel_np = np.concatenate([gt_np, pred_np, diff_rgb], axis=1)  # [H,3W,3]
                        panel_t  = torch.from_numpy(panel_np.transpose(2,0,1)).to(image.device).float() / 255.0
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/panel", panel_t[None], global_step=iteration)

                    # ===== Saving to disk & wandb =====
                    save_it = (int(viewpoint.image_name) % 10 == 0 and config['name'] == 'test') or (config['name'] == 'train' and (idx < 5))
                    if save_it:
                        save_dir = os.path.join(args.model_path, 'test_img')
                        os.makedirs(save_dir, exist_ok=True)

                        # render
                        if not DISABLE_WANDB_IMAGES:
                            wandb.log({f"{config['name']}_view_{viewpoint.image_name}/render":
                                    [wandb.Image(image.detach().cpu())]}, step=iteration)
                        # try:
                        #     torchvision.utils.save_image(
                        #         image.detach().cpu(),
                        #         os.path.join(save_dir, f"iter{iteration:06d}_{config['name']}_view_{viewpoint.image_name}_render.png")
                        #     )
                        # except Exception as e:
                        #     print(f"[WARN] save render image failed: {e}")
                        save_torch_img(image.detach().cpu(),
                                       os.path.join(save_dir, f"iter{iteration:06d}_{config['name']}_view_{viewpoint.image_name}_render.png"))

                        # gt
                        if not DISABLE_WANDB_IMAGES:
                            wandb.log({f"{config['name']}_view_{viewpoint.image_name}/ground_truth":
                                    [wandb.Image(gt_image.detach().cpu())]}, step=iteration)
                                    
                        save_torch_img(
                            gt_image.detach().cpu(),
                            os.path.join(save_dir, f"iter{iteration:06d}_{config['name']}_view_{viewpoint.image_name}_gt.png")
                        )

                        # mask
                        if has_mask:
                            if not DISABLE_WANDB_IMAGES:
                                wandb.log({f"{config['name']}_view_{viewpoint.image_name}/render_mask":
                                        [wandb.Image(mask_vis.detach().cpu())]}, step=iteration)
                            save_torch_img(
                                mask_vis.detach().cpu(),
                                os.path.join(save_dir, f"iter{iteration:06d}_{config['name']}_view_{viewpoint.image_name}_mask.png")
                            )

                        # diff
                        diff_rgb = make_diff_rgb(image, gt_image, mask=None)
                        psnr_text = f"PSNR: {float(psnr_val):.2f}"
                        diff_rgb_anno = draw_text_with_bg_np(diff_rgb, psnr_text, 10, 26, font_scale=0.7, thickness=1)
                        diff_t = torch.from_numpy(diff_rgb_anno.transpose(2,0,1)).to(image.device).float() / 255.0

                        if not DISABLE_WANDB_IMAGES:
                            wandb.log({f"{config['name']}_view_{viewpoint.image_name}/diff_rgb":
                                    [wandb.Image(diff_t.detach().cpu())]}, step=iteration)
                        save_torch_img(
                            diff_t.detach().cpu(),
                            os.path.join(save_dir, f"iter{iteration:06d}_{config['name']}_view_{viewpoint.image_name}_diff.png")
                        )

                        # panel
                        gt_np   = (gt_image.clamp(0,1).detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
                        pred_np = (image.clamp(0,1).detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
                        pred_np = draw_text_with_bg_np(pred_np, psnr_text, 10, 26, font_scale=0.7, thickness=1)
                        panel_np = np.concatenate([gt_np, pred_np, diff_rgb_anno], axis=1)
                        panel_t  = torch.from_numpy(panel_np.transpose(2,0,1)).to(image.device).float() / 255.0

                        if not DISABLE_WANDB_IMAGES:
                            wandb.log({f"{config['name']}_view_{viewpoint.image_name}/panel":
                                    [wandb.Image(panel_t.detach().cpu())]}, step=iteration)
                        save_torch_img(
                            panel_t.detach().cpu(),
                            os.path.join(save_dir, f"iter{iteration:06d}_{config['name']}_view_{viewpoint.image_name}_panel.png")
                        )

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')

                # === Summary Metrics ===
                l1_test = sum(l1_test_list)/len(l1_test_list)
                psnr_test = sum(psnr_test_list)/len(psnr_test_list)
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if config['name'] == 'train':
                    wandb.log({"psnr_train": psnr_test.item()}, step=iteration)
                elif config['name'] == 'test':
                    wandb.log({"psnr_test": psnr_test.item()}, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr

if __name__ == "__main__":
    set_rand_seed(42)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--capmax_ratio', type=float, default=1.0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*5000 for i in range(0,20000)] + [1, 200, 500, 100_000, 200_000])
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--resume_iteration', type=int, default=-1)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100,500,1000] + [7_000, 10_000, 20_000, 30_000, 40_000, 50_000, 100_000, 200_000])
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

        args.cap_max = int(args.capmax_ratio * args.args.cap_max)


    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    model_path_parts = [p for p in args.model_path.split('/') if p]
    wandb_name = model_path_parts[-2] + '_' + model_path_parts[-1] if len(model_path_parts) >= 2 else model_path_parts[-1]
    wandb.init(project="NTS_splatfields", name=wandb_name)



    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    DEPTH_MIN = 9.0
    torch.backends.cudnn.benchmark=False
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.resume_ckpt, args.resume_iteration)

    # All done
    print("\nTraining complete.")
