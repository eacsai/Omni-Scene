
import torch, torch.nn as nn, torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from sample_anchors import sample_concentrating_sphere, project_onto_planes
import math
from vis_feat import single_features_to_RGB
from .depth_predictor import DepthPredictor
import numpy as np

def sigmoid_scaling(scaling:torch.Tensor, lower_bound=0.005, upper_bound=0.02):
    sig = torch.sigmoid(scaling)
    return lower_bound * (1 - sig) + upper_bound * sig


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. 首先，定义上采样层，将H和W放大2倍
        # 使用'bilinear'即为线性插值
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 2. 然后，定义一个卷积层来处理和细化放大后的特征
        # kernel_size=3, padding=1 是保持分辨率不变的经典组合
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # 3. (可选) 添加一个激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 按顺序执行：先上采样，再卷积，再激活
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv(x)
        return x


@MODELS.register_module()
class VolumeGaussianDecoder(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, pc_range, gs_dim=14,
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, gpv=4, offset_max=None, scale_max=None,
        use_checkpoint=False, task='spherical'
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.gpv = gpv
        self.pc_depth = math.sqrt(pc_range[0]**2 + pc_range[1]**2 + pc_range[2]**2)
        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.gs_decoder = nn.Linear(out_dims, gs_dim*gpv)
        self.use_checkpoint = use_checkpoint

        # set activations
        # TODO check if optimal
        self.pos_act = lambda x: torch.tanh(x)
        # if offset_max is None:
        #     self.offset_max = [1.0] * 3 # meters
        # else:
        #     self.offset_max = offset_max
        self.offset_max = [0.02] * 3
        #self.scale_act = lambda x: sigmoid_scaling(x, lower_bound=0.005, upper_bound=0.02)
        # if scale_max is None:
        #     self.scale_max = [1.0] * 3 # meters
        # else:
        #     self.scale_max = scale_max
        self.scale_max = [0.02] * 3 
        self.scale_act = lambda x: torch.sigmoid(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x)

        self.compressNet = nn.Conv3d(in_channels=128 * 3, out_channels=128 * 4, kernel_size=(16, 1, 1))
        self.upsample_x4_layer = nn.Sequential(
            UpsampleBlock(in_channels=128 * 4, out_channels=128 *2),
            UpsampleBlock(in_channels=128 * 2, out_channels=128)
        )
        self.depth_predictor = DepthPredictor(
            128,
            32,
            1,
            False,
        )

        scale_cylinder = self.get_scale_spherical(
            theta_res=tpv_w, 
            phi_res=tpv_h,
            r_res=tpv_z,
            r_min=pc_range[2],
            r_max=pc_range[5]
        )
        self.register_buffer('scale_spherical', scale_cylinder)

    @staticmethod
    def get_reference_points(H, W, Z, pc_range, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            H, W: spatial shape of tpv plane.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, -1).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, -1, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(2, 1, 0, 3).contiguous() # w, h, z, 3
        ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref_3d = ref_3d.view(-1,3) # num_points, 3
        return ref_3d[:,None,:] # num_points, 1, 3
    
    @staticmethod
    def get_square_coordinates(ref_3d, pc_depth):
        eps = 1e-8
        theta = (torch.atan2(ref_3d[..., 0], ref_3d[..., 2]) + torch.pi)/(2 * torch.pi) * 2 - 1
        phi = (torch.atan2(ref_3d[..., 1], torch.sqrt(ref_3d[..., 0]**2 + ref_3d[..., 2]**2 + eps)) + torch.pi/2)/torch.pi * 2 - 1         
        r = torch.sqrt(ref_3d[..., 0]**2 + ref_3d[..., 1]**2 + ref_3d[..., 2]**2) / pc_depth * 2 - 1
        return torch.cat((theta,phi,r), dim=-1) # num_points, 3

    @staticmethod
    def get_panorama_reference_points(H, W, Z, pc_range, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            H, W: spatial shape of tpv plane.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space
        rs = (pc_range[5] - pc_range[2]) * torch.linspace(0, Z, Z+1, dtype=dtype,
                            device=device)[:-1].view(-1, 1, 1).expand(Z, H, W) / Z  + pc_range[2]
        thetas = 2 * torch.pi * torch.linspace(0, W, W+1, dtype=dtype,
                            device=device)[:-1].view(1, 1, -1).expand(Z, H, W) / W
        phis = torch.pi * torch.linspace(0, H, H+1, dtype=dtype,
                            device=device)[:-1].view(1, -1, 1).expand(Z, H, W) / H
        
        xs = -torch.sin(phis) * torch.sin(thetas) * rs
        ys = -torch.cos(phis) * rs
        zs = -torch.sin(phis) * torch.cos(thetas) * rs

        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(2, 1, 0, 3) # w, h, z, 3
        # ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        # ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        # ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1, 1) # b, w, h, z, 3
        return ref_3d
    
    @staticmethod
    def get_scale_spherical(
        theta_res: int, 
        phi_res: int,
        r_res: int,
        r_min: float, 
        r_max: float
    ) -> torch.Tensor:
        """
        为球坐标系下均匀分布的点云计算初始的3DGS尺度。
        (注意：这里返回的是直接的scale，而不是log_scale，以匹配您提供的代码)

        返回:
            一个形状为 [theta_res, phi_res, r_res, 3] 的张量，存储了每个点的 (scale_x, scale_y, scale_z)。
        """
        # 1. 计算步长 (弧度)
        delta_r = (r_max - r_min) / r_res
        delta_phi = np.pi / phi_res          # 改回: 使用 delta_phi
        delta_theta = 2 * np.pi / theta_res
        # delta_z 不再需要

        # 2. 创建球坐标网格
        # 我们取每个格子的中心点作为高斯球的中心
        theta_vals = torch.linspace(0, 2 * np.pi, theta_res + 1)[:-1] + delta_theta / 2
        phi_vals = torch.linspace(0, np.pi, phi_res + 1)[:-1] + delta_phi / 2 # 改回: 创建 phi_vals
        r_vals = torch.linspace(r_min, r_max, r_res + 1)[:-1] + delta_r / 2
        
        # 改回: 网格现在是 theta, phi, r
        grid_theta, grid_phi, grid_r = torch.meshgrid(theta_vals, phi_vals, r_vals, indexing='ij')

        # 3. 为每个点计算其格子的笛卡尔尺寸 (Δx, Δy, Δz)
        # 改回: 需要计算 phi 的 sin 和 cos
        cos_phi, sin_phi = torch.cos(grid_phi), torch.sin(grid_phi)
        cos_theta, sin_theta = torch.cos(grid_theta), torch.sin(grid_theta)

        # 改回: 使用球坐标系的尺寸计算公式
        # Z方向的尺寸现在依赖于 r 和 phi
        delta_y = torch.abs(cos_phi) * delta_r + torch.abs(grid_r * sin_phi) * delta_phi
        
        # X和Y方向的尺寸计算也使用球坐标的完整公式
        delta_z = torch.abs(sin_phi * cos_theta) * delta_r + \
                torch.abs(grid_r * cos_phi * cos_theta) * delta_phi + \
                torch.abs(grid_r * sin_phi * sin_theta) * delta_theta
                    
        delta_x = torch.abs(sin_phi * sin_theta) * delta_r + \
                torch.abs(grid_r * cos_phi * sin_theta) * delta_phi + \
                torch.abs(grid_r * sin_phi * cos_theta) * delta_theta

        # 4. 计算初始Scale (格子尺寸的一半)，并乘以一个小的放大系数（可选）
        scales = torch.stack([delta_x / 2, delta_y / 2, delta_z / 2], dim=-1) * 1.1
        
        return scales
    
    
    def forward(self, tpv_list, deterministic=False, debug=False):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape

        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w) # [theta, phi]
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h) # [phi, r]
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z) # [r, theta]

        # sample method
        # anchors_coordinates = self.anchors_coordinates[None, :, :].repeat(bs, 1, 1)
        # anchors_plane_coordinates = project_onto_planes(anchors_coordinates) # [bs, n_planes, M, 2]

        # tpv_reshape_list = [tpv_hw, tpv_zh, tpv_wz]
        # anchors_feats = []
        # for l in range(3):
        #     tpv_feat = tpv_reshape_list[l]
        #     anchors_feat = F.grid_sample(tpv_feat, anchors_plane_coordinates[:,l:l+1,:,:], mode='bilinear', padding_mode='zeros', align_corners=False)
        #     anchors_feats.append(anchors_feat.squeeze(2))
        # anchors_feats = anchors_feats[0] + anchors_feats[1] + anchors_feats[2] #[bs, M, c]
        # single_features_to_RGB(anchors_feats[0], img_name='feat_hw.png')
        # single_features_to_RGB(anchors_feats[1], img_name='feat_zh.png')
        # single_features_to_RGB(anchors_feats[2], img_name='feat_wz.png')
        # _, _, num_points = anchors_feats.shape

        # gaussians = anchors_feats.permute(0,2,1)

        # if self.scale_h != 1 or self.scale_w != 1:
        #     tpv_hw = F.interpolate(
        #         tpv_hw, 
        #         size=(self.tpv_h*self.scale_h, self.tpv_w*self.scale_w),
        #         mode='bilinear'
        #     )
        # if self.scale_z != 1 or self.scale_h != 1:
        #     tpv_zh = F.interpolate(
        #         tpv_zh, 
        #         size=(self.tpv_z*self.scale_z, self.tpv_h*self.scale_h),
        #         mode='bilinear'
        #     )
        # if self.scale_w != 1 or self.scale_z != 1:
        #     tpv_wz = F.interpolate(
        #         tpv_wz, 
        #         size=(self.tpv_w*self.scale_w, self.tpv_z*self.scale_z),
        #         mode='bilinear'
        #     )

        # #print("before voxelize:{}".format(torch.cuda.memory_allocated(0)))
        tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
        tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1)
        tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1)

        gaussians_tpv = torch.cat([tpv_hw, tpv_zh, tpv_wz], dim=1).permute(0, 1, 4, 2, 3)  # bs c w h z
        gaussians_theta_z = self.compressNet(gaussians_tpv).squeeze(2) # 3dCNN

        features = self.upsample_x4_layer(gaussians_theta_z)
        _, _, theta, phi = features.shape
        features = features.permute(0, 2, 3, 1).view(bs, -1, c) # bs, theta, phi, c
        _, n, _ = features.shape
        depths, opacity = self.depth_predictor.forward(
            features,
            torch.tensor([0.2], device=features.device).repeat(bs),
            torch.tensor([20.0], device=features.device).repeat(bs),
            deterministic,
            1 if deterministic else self.gpv,
        )

        thetas = 2 * torch.pi * torch.linspace(0, theta, theta+1, dtype=torch.float,
                            device=features.device)[:-1].view(1, -1, 1, 1).expand(bs, theta, phi, self.gpv).contiguous().view(bs, -1, self.gpv) / theta

        phis = torch.pi * torch.linspace(0, phi, phi+1, dtype=torch.float,
                            device=features.device)[:-1].view(1, 1, -1, 1).expand(bs, theta, phi, self.gpv).contiguous().view(bs, -1, self.gpv) / phi
        
        xs = -torch.sin(phis) * torch.sin(thetas) * depths
        ys = -torch.cos(phis) * depths
        zs = -torch.sin(phis) * torch.cos(thetas) * depths
        ref_3d = torch.stack((xs, ys, zs), -1)

        if self.use_checkpoint:
            gaussians = torch.utils.checkpoint.checkpoint(self.decoder, features, use_reentrant=False)
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, gaussians, use_reentrant=False)
            # gaussians = gaussians.view(bs, num_points, self.gpv, -1)
            gaussians = gaussians.view(bs, n, self.gpv, -1)
        else:
            gaussians = self.decoder(features)
            gaussians = self.gs_decoder(gaussians)
            # gaussians = gaussians.view(bs, num_points, self.gpv, -1)
            gaussians = gaussians.view(bs, n, self.gpv, -1)
        #print("after decode:{}".format(torch.cuda.memory_allocated(0)))
        gs_offsets_x = self.pos_act(gaussians[..., :1]) * self.scale_spherical[None, ..., None, :1].view() # bs, w, h, z, 3
        gs_offsets_y = self.pos_act(gaussians[..., 1:2]) * self.scale_spherical[None, ..., None, 1:2] # bs, w, h, z, 3
        gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.scale_spherical[None, ..., None, 2:3]# bs, w, h, z, 3
        #gs_offsets = gaussians[..., :3]
        gs_positions = torch.cat([gs_offsets_x, gs_offsets_y, gs_offsets_z], dim=-1) + ref_3d
        x = torch.cat([gs_positions, gaussians[..., 3:]], dim=-1)
        rgbs = self.rgb_act(x[..., 3:6])
        rotation = self.rot_act(x[..., 6:10])
        scale_x = self.scale_act(x[..., 10:11]) * self.scale_spherical[None, ..., None, :1]
        scale_y = self.scale_act(x[..., 11:12]) * self.scale_spherical[None, ..., None, 1:2]
        scale_z = self.scale_act(x[..., 12:13]) * self.scale_spherical[None, ..., None, 2:3]

        if debug:
            scale_x[:] = 0.5
            scale_y[:] = 0.5
            scale_z[:] = 0.5
        # rgbs[..., 0] = 0.0
        # rgbs[..., 1] = 0.0
        # rgbs[..., 2] = 0.0

        gaussians = torch.cat([gs_positions, rgbs, opacity.unsqueeze(-1) / self.gpv, rotation, scale_x, scale_y, scale_z], dim=-1) # bs, w, h, z, gpv, 14
        predict_depth = depths * opacity
        return gaussians
