import torch
import torch.nn.functional as F
from utils.time_utils import SplatFields
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, get_warmup_hold_expon_lr_func


class SplatFieldsModel:
    def __init__(self, hyper_args, radius=None):
        self.deform = SplatFields(radius=radius, **hyper_args.__dict__).cuda()
        pytorch_total_params = sum(p.numel() for p in self.deform.parameters())
        print('DEFORM #params:', pytorch_total_params/1_000_000, 'M')
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.iteration = -1

    def log_variables(self):
        return self.deform.log_variables()

    def step(self, xyz, time_emb): # -> tuple of: d_xyz, d_rotation, d_scaling

        use_tex = self.iteration > 10000 or self.iteration == -1
        return self.deform(xyz, time_emb, use_tex=use_tex)


    def train_setting(self, training_args):
        param_groups = []

        # xyz modules (lr=0, frozen)
        base_xyz_modules = [
            self.deform.mlp_deform,
            getattr(self.deform, "mlp_flow", None),
            getattr(self.deform, "mlp_flow_head", None),  
        ]
        base_xyz_params = []
        for m in base_xyz_modules:
            if m is not None:
                base_xyz_params += list(m.parameters())
        if len(base_xyz_params) > 0:
            param_groups.append({
                "params": base_xyz_params,
                "lr": 0.0,
                "name": "flow_xyz"
            })

        # other base modules (trainable)
        base_modules = [
            self.deform.encoder,
            self.deform.mlp_refine_feat,
            self.deform.mlp_rgb,
            getattr(self.deform, "mlp_rgb_viewdep", None),
            self.deform.mlp_scale,
            self.deform.mlp_opacity,
            self.deform.mlp_rotation,
        ]
        base_params = []
        for m in base_modules:
            if m is not None:
                base_params += list(m.parameters())
        if len(base_params) > 0:
            param_groups.append({
                "params": base_params,
                "lr": 0.0,   # will be scheduled
                "name": "base"
            })

        # tex modules
        tex_modules = [
            getattr(self.deform, "mlp_refine_feat_tex", None),
            getattr(self.deform, "tex_encoder", None),
            getattr(self.deform, "mlp_color_residue", None),
            getattr(self.deform, "mlp_opacity_residue", None),
        ]
        tex_params = []
        for m in tex_modules:
            if m is not None:
                tex_params += list(m.parameters())
        if len(tex_params) > 0:
            param_groups.append({
                "params": tex_params,
                "lr": 0.0,   # will be scheduled
                "name": "tex"
            })

        # construct optimizer
        # self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-8)

        # schedulers
        self.schedulers = {}
        self.schedulers["base"] =  get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)
        self.schedulers["flow_xyz"] =  get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)
        if len(tex_params) > 0:
            self.schedulers["tex"] =  get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                        lr_delay_steps = 30000,
                                                        # lr_delay_steps = 0,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)


    def update_learning_rate(self, iteration):
        self.iteration = iteration
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name in self.schedulers:
                lr = self.schedulers[name](iteration)
                param_group["lr"] = lr


    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))



    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        print('load weights_path', weights_path)
        # self.deform.load_state_dict(torch.load(weights_path), strict=True)
        # self.deform.load_state_dict(torch.load(weights_path), strict=False)


        ckpt = torch.load(weights_path, map_location="cpu")


        exclude_prefixes = []
        
        new_ckpt = {}
        for k, v in ckpt.items():
            if not any(k.startswith(p) for p in exclude_prefixes):
                new_ckpt[k] = v
        print(f"Skipped keys: {[k for k in ckpt.keys() if k not in new_ckpt]}")

        self.deform.load_state_dict(new_ckpt, strict=False)

