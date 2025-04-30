import os
import numpy as np
import torch
import imageio
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import warnings
from einops import rearrange
from vis_feat import single_features_to_RGB

@MODELS.register_module()
class VolumeGaussianCos(BaseModule):

    def __init__(self,
                 encoder=None,
                 gs_decoder=None,
                 use_checkpoint=False,
                 **kwargs,
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint

        if encoder:
            self.encoder = MODELS.build(encoder)
        if gs_decoder:
            self.gs_decoder = MODELS.build(gs_decoder)

        self.tpv_h = self.encoder.tpv_h  #y
        self.tpv_w = self.encoder.tpv_w  #x
        self.tpv_z = self.encoder.tpv_z  #z
        self.pc_range = self.encoder.pc_range
        self.pc_xrange = self.pc_range[3] - self.pc_range[0]
        self.pc_yrange = self.pc_range[4] - self.pc_range[1]
        self.pc_zrange = self.pc_range[5] - self.pc_range[2]
        # k = torch.arange(self.tpv_h) # 0 to M-1
        # h_k = -1 + (k + 0.5) * (2.0 / self.tpv_h) # M 个 cos(phi) 的值，范围在 (-1, 1) 内
        # # 限制在 [-1, 1] 防止数值误差
        # h_k = torch.clip(h_k, -1.0, 1.0)

        h_k = torch.linspace(-0.98, 0.98, steps=self.tpv_h)
        # 计算对应的 phi 值
        self.phis = torch.flip(torch.arccos(h_k), dims=[0]) # M 个 phi 的值，范围在 (0, pi) 内

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img_feats, candidate_gaussians, candidate_feats, candidate_uv_map, candidate_depth_pred, img_metas=None, status="train"):
        """Forward training function.
        """
        if candidate_gaussians is not None and candidate_feats is not None:
            bs = len(candidate_feats)
            _, c = candidate_feats[0].shape
            
            project_feats_hw = candidate_feats[0].new_zeros((bs, self.tpv_h, self.tpv_w, c))
            project_feats_zh = candidate_feats[0].new_zeros((bs, self.tpv_z, self.tpv_h, c))
            project_feats_wz = candidate_feats[0].new_zeros((bs, self.tpv_w, self.tpv_z, c))
            
            for i in range(bs):
                candidate_xyzs_i = candidate_gaussians[i][..., :3]
                
                # Decare
                # candidate_hs_i = (self.tpv_h * (candidate_xyzs_i[..., 1] - self.pc_range[1]) / self.pc_yrange - 0.5).int()
                # candidate_ws_i = (self.tpv_w * (candidate_xyzs_i[..., 0] - self.pc_range[0]) / self.pc_xrange - 0.5).int()
                # candidate_zs_i = (self.tpv_z * (candidate_xyzs_i[..., 2] - self.pc_range[2]) / self.pc_zrange - 0.5).int()
                
                # Spherical
                eps = 1e-5
                x = candidate_xyzs_i[..., 0]
                y = candidate_xyzs_i[..., 1]
                z = candidate_xyzs_i[..., 2]

                phi = (torch.atan2(y, torch.sqrt(x**2 + z**2 + eps))  + torch.pi/2)
                abs_diff = torch.abs(phi.unsqueeze(1) - self.phis.unsqueeze(0).to(phi.device))

                candidate_hs_i = torch.argmin(abs_diff, dim=1)
                candidate_ws_i = (self.tpv_w * (torch.atan2(x, z) + torch.pi)/(2 * torch.pi) - eps).int()
                candidate_zs_i = (self.tpv_z * candidate_depth_pred[i][:, 0] / self.pc_zrange - eps).int()
                
                # original
                # candidate_hs_i = candidate_uv_map[i][:, 1]
                # candidate_ws_i = candidate_uv_map[i][:, 0]
                # candidate_zs_i = (self.tpv_z * candidate_depth_pred[i][:, 0] / self.pc_zrange - 0.5).int()
                # n, c
                #candidate_feats_i = candidate_feats[[i, valid_mask]]
                candidate_feats_i = candidate_feats[i]
                # hw: n, 2
                candidate_coords_hw_i = torch.stack([candidate_hs_i, candidate_ws_i], dim=-1)
                linear_inds_hw_i = (candidate_coords_hw_i[..., 0] * self.tpv_w + candidate_coords_hw_i[..., 1]).to(dtype=torch.int64)
                project_feats_hw_i = project_feats_hw[i].view(-1, c)
                project_feats_hw_i.scatter_add_(0, linear_inds_hw_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_hw_i = project_feats_hw_i.new_zeros((self.tpv_h * self.tpv_w, c), dtype=torch.float32)
                ones_hw_i = torch.ones_like(candidate_feats_i)
                count_hw_i.scatter_add_(0, linear_inds_hw_i.unsqueeze(-1).expand(-1, c), ones_hw_i)
                count_hw_i = torch.where(count_hw_i == 0, torch.ones_like(count_hw_i), count_hw_i)
                project_feats_hw_i = (project_feats_hw_i / count_hw_i).view(self.tpv_h, self.tpv_w, c)
                project_feats_hw[i] = project_feats_hw_i

                # zh: n, 2
                candidate_coords_zh_i = torch.stack([candidate_zs_i, candidate_hs_i], dim=-1)
                linear_inds_zh_i = (candidate_coords_zh_i[..., 0] * self.tpv_h + candidate_coords_zh_i[..., 1]).to(dtype=torch.int64)
                project_feats_zh_i = project_feats_zh[i].view(-1, c)
                project_feats_zh_i.scatter_add_(0, linear_inds_zh_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_zh_i = project_feats_zh_i.new_zeros((self.tpv_z * self.tpv_h, c), dtype=torch.float32)
                ones_zh_i = torch.ones_like(candidate_feats_i)
                count_zh_i.scatter_add_(0, linear_inds_zh_i.unsqueeze(-1).expand(-1, c), ones_zh_i)
                count_zh_i = torch.where(count_zh_i == 0, torch.ones_like(count_zh_i), count_zh_i)
                project_feats_zh_i = (project_feats_zh_i / count_zh_i).view(self.tpv_z, self.tpv_h, c)
                project_feats_zh[i] = project_feats_zh_i

                # wz: n, 2
                candidate_coords_wz_i = torch.stack([candidate_ws_i, candidate_zs_i], dim=-1)
                linear_inds_wz_i = (candidate_coords_wz_i[..., 0] * self.tpv_z + candidate_coords_wz_i[..., 1]).to(dtype=torch.int64)
                project_feats_wz_i = project_feats_wz[i].view(-1, c)
                project_feats_wz_i.scatter_add_(0, linear_inds_wz_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_wz_i = project_feats_wz_i.new_zeros((self.tpv_w * self.tpv_z, c), dtype=torch.float32)
                ones_wz_i = torch.ones_like(candidate_feats_i)
                count_wz_i.scatter_add_(0, linear_inds_wz_i.unsqueeze(-1).expand(-1, c), ones_wz_i)
                count_wz_i = torch.where(count_wz_i == 0, torch.ones_like(count_wz_i), count_wz_i)
                project_feats_wz_i = (project_feats_wz_i / count_wz_i).view(self.tpv_w, self.tpv_z, c)
                project_feats_wz[i] = project_feats_wz_i
            
            project_feats_hw = rearrange(project_feats_hw, "b h w c -> b c h w")
            project_feats_zh = rearrange(project_feats_zh, "b h w c -> b c h w")
            project_feats_wz = rearrange(project_feats_wz, "b h w c -> b c h w")
            project_feats = [project_feats_hw, project_feats_zh, project_feats_wz]
        else:
            project_feats = [None, None, None]

        # single_features_to_RGB(project_feats_hw[-1:], img_name='feat_hw.png')
        # single_features_to_RGB(project_feats_zh[-1:], img_name='feat_zh.png')
        # single_features_to_RGB(project_feats_wz[-1:], img_name='feat_wz.png')

        if self.use_checkpoint and status != "test":
            input_vars_enc = (img_feats, project_feats, img_metas)
            outs = torch.utils.checkpoint.checkpoint(
                self.encoder, *input_vars_enc, use_reentrant=False
            )
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, outs, use_reentrant=False)
        else:
            outs = self.encoder(img_feats, project_feats, img_metas)
            gaussians = self.gs_decoder(outs)
        bs = gaussians.shape[0]
        n_feature = gaussians.shape[-1]
        gaussians = gaussians.reshape(bs, -1, n_feature)

        # K = 5000                 # 要保留的 top K 个高斯球
        # feature_index = 6         # 第 7 个特征的索引 (0-based)
        # ranking_feature = gaussians[:, :, feature_index]
        # _, top_indices = torch.topk(ranking_feature, k=K, dim=1) # dim=1 是高斯球数量的维度
        # indices_for_gather = top_indices.unsqueeze(-1).expand(-1, -1, 14) # Shape: [6, 10000, 14]
        # filtered_gaussians = torch.gather(gaussians, dim=1, index=indices_for_gather) # Shape: [6, 10000, 14]

        return gaussians
