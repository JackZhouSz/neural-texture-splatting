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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import collections
import trimesh
from packaging.version import parse as parse_version


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if parse_version(torch.__version__) < parse_version('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    import mcubes
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()
    if len(vertices) > 0:
        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    mesh = trimesh.Trimesh(vertices, triangles)
    return mesh

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    device = bound_max.device
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device=device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device=device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device=device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def ArrayToTorch(array, resolution):
    # resized_image = np.resize(array, resolution)
    resized_image_torch = torch.from_numpy(array)

    if len(resized_image_torch.shape) == 3:
        return resized_image_torch.permute(2, 0, 1)
    else:
        return resized_image_torch.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def get_warmup_hold_expon_lr_func(
    lr_start: float,          # warmup 起点学习率（通常很小，甚至 0）
    lr_max: float,            # plateau 期间的最大学习率
    lr_final: float,          # 训练结束时的最终学习率
    warmup_steps: int = 1000, # warmup 步数（例如 1000）
    hold_steps: int = 10000,  # plateau 保持步数（例如 10000）
    max_steps: int = 100000,  # 总步数（含 warmup 与 hold），剩余步数做衰减
    eps: float = 1e-12        # 数值稳定
):
    """
    三阶段学习率调度：线性 warmup -> 常数保持 -> 对数线性（指数）衰减
    返回的是 '绝对学习率'，方便直接设置到 optimizer.param_groups[i]['lr']。

    约定：
    - step < warmup_steps：从 lr_start 线性升到 lr_max
    - warmup_steps <= step < warmup_steps + hold_steps：保持 lr_max
    - 其后直到 max_steps：从 lr_max 指数式衰减到 lr_final
    - step >= max_steps：保持 lr_final
    """
    warmup_steps = max(int(warmup_steps), 0)
    hold_steps   = max(int(hold_steps),   0)
    max_steps    = max(int(max_steps),    warmup_steps + hold_steps + 1)  # 至少留 1 步给衰减阶段

    decay_steps = max(max_steps - warmup_steps - hold_steps, 1)

    lr_start = float(lr_start)
    lr_max   = float(lr_max)
    lr_final = float(lr_final)

    def helper(step: int) -> float:
        if step < 0:
            return 0.0

        # Phase 1: linear warmup
        if step < warmup_steps and warmup_steps > 0:
            t = step / warmup_steps
            return lr_start + (lr_max - lr_start) * t

        # Phase 2: hold at lr_max
        if step < warmup_steps + hold_steps:
            return lr_max

        # Phase 3: exponential decay from lr_max -> lr_final
        if step < max_steps:
            t = (step - warmup_steps - hold_steps) / decay_steps  # in [0,1]
            # 对数线性插值（等价于指数衰减）
            log_lerp = np.exp(
                np.log(max(lr_max, eps)) * (1.0 - t) + np.log(max(lr_final, eps)) * t
            )
            return float(log_lerp)

        # After max_steps: stay at lr_final
        return lr_final

    return helper
