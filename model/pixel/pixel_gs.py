import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import imageio
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import warnings
from einops import rearrange, einsum
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor
from ..utils.ops import get_ray_directions, get_rays
from torch.nn.init import normal_


@MODELS.register_module()
class PixelGaussian(BaseModule):

    def __init__(self,
                 in_embed_dim=128,
                 out_embed_dims=[128, 256, 512, 512],
                 near=0.1,
                 far=1000.0,
                 use_checkpoint=False,
                 **kwargs,
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint     
        self.plucker_to_embed = nn.Linear(6, out_embed_dims[0])        
        
        # output & post-process
        self.near = near
        self.far = far
        self.num_surfaces = 1

        self.upsampler = nn.Sequential(
            nn.Conv2d(in_embed_dim, out_embed_dims[0], 3, 1, 1),
            nn.Upsample(
                scale_factor=4,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )

        gs_channels = 1 + 1 + 3 + 4 + 3 # offset, opacity, scale, rotation, rgb
        self.gs_channels = gs_channels
        self.to_gaussians = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(out_embed_dims[0], gs_channels, 1),
        )
        self.opt_act = torch.sigmoid
        self.scale_act = lambda x: torch.exp(x) * 0.01
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid
        
        self.delta_clamp = lambda x: x.clamp(-10.0, 6.0)
        self.delta_act = torch.exp
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def plucker_embedder(
        self, 
        rays_o,
        rays_d
    ):
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)
        plucker = torch.cat([torch.cross(rays_o, rays_d, dim=2), rays_d], dim=2)
        return plucker
    
    def forward(self, img_feats, depths_in, confs_in, pluckers, origins, directions, status="train"):
        """Forward training function."""
        # upsample 4x downsampled img features to original size
        bs, v, _, _, _ = origins.shape

        gaussians = self.to_gaussians(img_feats)
        gaussians = rearrange(gaussians, "(b v) (n c) h w -> b (v h w n) c",
                              b=bs, v=v, n=1, c=self.gs_channels)
        offsets = gaussians[..., :1]
        opacities = self.opt_act(gaussians[..., 1:2])
        scales = self.scale_act(gaussians[..., 2:5])
        rotations = self.rot_act(gaussians[..., 5:9])
        rgbs = self.rgb_act(gaussians[..., 9:12])

        depths_in = rearrange(depths_in, "b v c h w-> b (v h w) c", b=bs, v=v)

        origins = rearrange(origins, "b v h w c -> b (v h w) c")
        origins = origins.unsqueeze(-2)
        directions = rearrange(directions, "b v h w c -> b (v h w) c")
        directions = directions.unsqueeze(-2)
        depth_pred = (depths_in + offsets).clamp(min=0.0)
        means = origins + directions * depth_pred[..., None]
        means = rearrange(means, "b r n c -> b (r n) c")
        # means = means + offsets

        gaussians = torch.cat([means, rgbs, opacities, rotations, scales], dim=-1)
        features = rearrange(img_feats, "(b v) c h w -> b (v h w) c", b=bs, v=v)
        
        return gaussians, features, depth_pred