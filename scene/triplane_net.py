import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .siren_basic import SirenNet


class Triplane(nn.Module):
    def __init__(self, input_dim=3, triplane_feat_dim=16, resolution=192, cnn_generator=False):
        super().__init__()
        
        self.triplane_feat_dim = triplane_feat_dim
        self.output_dim = self.triplane_feat_dim * 3


        self.resolution = resolution # 

        self.cnn_generator = cnn_generator
        if self.cnn_generator:
            self.plane_coef = torch.nn.ModuleList([
                Tensorial2D(out_ch = self.triplane_feat_dim)  # splatFields
                for i in range(3)
            ])
        else:
            self.plane_coef = nn.ParameterList([nn.Parameter(torch.randn(self.triplane_feat_dim, self.resolution, self.resolution)*0.001) for _ in range(3)]) # todo: set to different resolution for ablation


    def get_plane_coef(self):
        if self.cnn_generator:
            ret = []
            for i in range(3):
                ret.append(self.plane_coef[i](frame_id=0))
            ret = torch.cat(ret, dim=0)
            return ret
        else:
            return self.plane_coef

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates, additional_feat=None):
        batch_size, n_coords, n_dims = coordinates.shape
        plane_coef = self.get_plane_coef()
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], plane_coef[0].unsqueeze(0)).squeeze(0)
        yz_embed = self.sample_plane(coordinates[..., 1:3], plane_coef[1].unsqueeze(0)).squeeze(0)
        xz_embed = self.sample_plane(coordinates[..., :3:2], plane_coef[2].unsqueeze(0)).squeeze(0)

        features = torch.cat([xy_embed, yz_embed, xz_embed], dim = -1) # [N, 64]
        features = features.reshape(batch_size, n_coords, features.shape[-1])
        return features

