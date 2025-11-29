import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scene.triplane_net import Triplane
from scene.embedder import get_embedder
from scene.siren_basic import SirenNet



@torch.no_grad()
def contract_mipnerf360(xyz, roi_min, roi_max):
    xyz_unit = (xyz - roi_min) / (roi_max - roi_min) # roi_to_unit roi -> [0, 1]^3
    xyz_unit = xyz_unit * 2.0 - 1.0 # roi -> [-1, 1]^3
    xyz_norm = torch.norm(xyz_unit, dim=-1, keepdim=True)
    _ind = xyz_norm.squeeze(-1) > 1.0
    xyz_inv_norm = 1./xyz_norm[_ind]
    xyz_unit[_ind] = (2.0 - 1.0 * xyz_inv_norm) * (xyz_unit[_ind] * xyz_inv_norm)
    # xyz_unit = xyz_unit * 0.25 + 0.5 # [-1, 1]^3 -> [0.25, 0.75]^3
    return xyz_unit

@torch.no_grad()
def contract_unit(xyz, roi_min, roi_max):
    xyz_unit = (xyz - roi_min) / (roi_max - roi_min) # roi_to_unit roi -> [0, 1]^3
    xyz_unit = xyz_unit * 2.0 - 1.0 # roi -> [-1, 1]^3
    xyz_unit = xyz_unit.clamp(-1.0, 1.0)
    return xyz_unit



class TexNet(nn.Module):
    def __init__(self, uv_res, hidden_dim = 128, layer_num=2, decompose_type='CP', cnn_generator=False, view_dependent_color = True):
        super(TexNet, self).__init__()
        self.uv_res = uv_res

        self.view_encoder = get_embedder(2, 3)


        self.use_tex_sh = False
        self.max_tex_sh_degree = 1


        self.opacity_dim = 3
        self.tex_color_dim = 3 * (self.max_tex_sh_degree + 1) ** 2 * 3 if self.use_tex_sh else 3 * 3

        self.triplane_opacity = Triplane(triplane_feat_dim=16, cnn_generator=cnn_generator)
        self.triplane_color = Triplane(triplane_feat_dim=16, cnn_generator=cnn_generator)
   


        self.decompose_type = decompose_type
        if decompose_type == 'CP':
            self.texture_model_dim = self.uv_res + self.uv_res
        elif decompose_type == 'Tucker':
            self.texture_model_dim = self.uv_res * 3
        else:
            self.texture_model_dim = self.uv_res * self.uv_res
   

        self.opacity_net = SirenNet(in_channels=self.triplane_opacity.output_dim + 3,
                            out_channels=self.opacity_dim * self.texture_model_dim, init_siren_frequency=15,
                            D = layer_num,
                            W = hidden_dim)

        color_input_dim = self.triplane_color.output_dim + 3
        if view_dependent_color:
            color_input_dim += self.view_encoder[1]
        self.color_net =  nn.Sequential(
                nn.Linear(color_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.tex_color_dim * self.texture_model_dim),
            )


    def forward_opacity(self, query_xyz, additional_feat):

        query_xyz = query_xyz.clamp(-1.0, 1.0)
        opacity_base = self.triplane_opacity(query_xyz)
        opacity_base = self.opacity_net(torch.cat([opacity_base, additional_feat], dim=-1))

        if self.decompose_type == 'CP':
            opacity_base = opacity_base.reshape(-1, self.uv_res + self.uv_res, self.opacity_dim)

            opacity_vector_x = opacity_base[:, :self.uv_res, :][..., None, :]  # [B, uv_res, 1, opacity_dim]
            opacity_vector_y = opacity_base[:, self.uv_res:, :][:, None, :, :]  # [B, 1, uv_res, opacity_dim]

            opacity = opacity_vector_x * opacity_vector_y  # [B, uv_res, uv_res, opacity_dim]

        elif self.decompose_type == 'Tucker':
            opacity_base = opacity_base.reshape(-1, self.uv_res * 3, self.opacity_dim)

            opacity_vector_x = opacity_base[:, :self.uv_res, :][..., None, :]  # [B, uv_res, 1, opacity_dim]
            opacity_vector_y = opacity_base[:, self.uv_res:self.uv_res*2, :][:, None, :, :]  # [B, 1, uv_res, opacity_dim]

            opacity_vector_s = opacity_base[:, -self.uv_res:, :]  # [B, uv_res, opacity_dim]
            G = torch.zeros(opacity_vector_s.shape[0], self.uv_res, self.uv_res, self.opacity_dim, device=opacity_vector_s.device)
            diag_indices = torch.arange(self.uv_res)
            G[:, diag_indices, diag_indices, :] = opacity_vector_s

            opacity = opacity_vector_x * opacity_vector_y  # [B, uv_res, uv_res, opacity_dim]
            opacity = opacity * G  # [B, uv_res, uv_res, opacity_dim]

        else:
            opacity = opacity_base  # assumed to be [B, uv_res, uv_res, opacity_dim]

        opacity = opacity.reshape(-1, self.uv_res, self.uv_res, self.opacity_dim)

        return opacity


    def forward_color(self, query_xyz, additional_feat):
        query_xyz = query_xyz.clamp(-1.0, 1.0)
        color_base = self.triplane_color(query_xyz)
        color_base = self.color_net(torch.cat([color_base, additional_feat], dim=-1))

        if self.decompose_type == 'CP':
            color_base = color_base.reshape(-1, self.uv_res + self.uv_res, self.tex_color_dim)

            color_vector_x = color_base[:, :self.uv_res, :][..., None, :] # [B, uv_res, 1, self.tex_color_dim]
            color_vector_y = color_base[:, self.uv_res:, :][:, None, :, :] # [B, 1, uv_res, self.tex_color_dim]

            color = color_vector_x * color_vector_y   # [B, uv_res, uv_res, self.tex_color_dim]

        elif self.decompose_type == 'Tucker':
            color_base = color_base.reshape(-1, self.uv_res + self.uv_res + self.uv_res, self.tex_color_dim)

            color_vector_x = color_base[:, :self.uv_res, :][..., None, :] # [B, uv_res, 1, self.tex_color_dim]
            color_vector_y = color_base[:, self.uv_res:self.uv_res*2, :][:, None, :, :] # [B, 1, uv_res, self.tex_color_dim]

            color_vector_s = color_base[:, -self.uv_res:, :] # [B, uv_res, self.tex_color_dim]
            G = torch.zeros(color_vector_s.shape[0], self.uv_res, self.uv_res, self.tex_color_dim, device=color_vector_s.device)
            diag_indices = torch.arange(self.uv_res)
            G[:, diag_indices, diag_indices, :] = color_vector_s  # assign diagonal values

            color = color_vector_x * color_vector_y # [B, uv_res, uv_res, self.tex_color_dim]
            color = color * G # [B, uv_res, uv_res, self.tex_color_dim]

        else:
            color = color_base
        color = color.reshape(-1, self.uv_res, self.uv_res, self.tex_color_dim)
        return color

