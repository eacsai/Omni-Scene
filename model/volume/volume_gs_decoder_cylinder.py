
import torch, torch.nn as nn, torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from sample_anchors import sample_concentrating_sphere, project_onto_planes
import math
from vis_feat import single_features_to_RGB
import numpy as np

def sigmoid_scaling(scaling:torch.Tensor, lower_bound=0.005, upper_bound=0.02):
    sig = torch.sigmoid(scaling)
    return lower_bound * (1 - sig) + upper_bound * sig

@MODELS.register_module()
class VolumeGaussianDecoderCylinder(BaseModule):
    def __init__(
        self, tpv_theta, tpv_r, tpv_z, pc_range, gs_dim=14,
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_theta=2, scale_r=2, scale_z=2, gpv=4, offset_max=None, scale_max=None,
        use_checkpoint=False
    ):
        super().__init__()
        self.tpv_theta = tpv_theta
        self.tpv_r = tpv_r
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.scale_theta = scale_theta
        self.scale_r = scale_r
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
        self.offset_max = [pc_range[3] - pc_range[0], 2*torch.pi/tpv_theta, (pc_range[5] - pc_range[2])/tpv_z] # r, theta, z
        #self.scale_act = lambda x: sigmoid_scaling(x, lower_bound=0.005, upper_bound=0.02)
        # if scale_max is None:
        #     self.scale_max = [1.0] * 3 # meters
        # else:
        #     self.scale_max = scale_max
        self.scale_max = [1.0] * 3 
        self.scale_act = lambda x: torch.sigmoid(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x)


        # obtain anchor points for gaussians        
        # r = torch.linspace(0.5, self.tpv_z-0.5, self.tpv_z, device='cuda')
        # anchors_coordinates = sample_concentrating_sphere(r, 2000, threshold=3.0, device='cuda') # [N_radii * n_samples, 3]
        gs_anchors = self.get_panorama_reference_points(tpv_theta * scale_theta, tpv_r * scale_r, tpv_z * scale_z, pc_range) # 1, w, h, z, 3
        scale_cylinder = self.get_scale_cylinder(tpv_theta * scale_theta, tpv_r * scale_r, tpv_z * scale_z, pc_range[0], pc_range[3], pc_range[2], pc_range[5])
        # gs_anchors = self.get_sample_reference_points(anchors_coordinates, pc_range[5] - pc_range[2]) # [num_points, 1, 3]
        # mask_lower = gs_anchors[:, 0, 1] >= -2.5
        # mask_upper = gs_anchors[:, 0, 1] <= 2.5
        # combined_mask = torch.logical_and(mask_lower, mask_upper)
        self.register_buffer('gs_anchors', gs_anchors)
        self.register_buffer('scale_cylinder', scale_cylinder)
        # self.register_buffer('anchors_coordinates', anchors_coordinates[combined_mask])

    
    @staticmethod
    def get_scale_cylinder(
        theta_res: int, 
        r_res: int,     # 修改: phi_res -> r_res
        z_res: int,     # 修改: 新增 z_res
        r_min: float, 
        r_max: float,
        z_min: float,   # 修改: 新增 z_min
        z_max: float    # 修改: 新增 z_max
    ) -> torch.Tensor:
        """
        为柱坐标系下均匀分布的点云计算初始的3DGS对数尺度。

        返回:
            一个形状为 [theta_res, r_res, z_res, 3] 的张量，存储了每个点的 (scale_x, scale_y, scale_z)。
        """
        # 1. 计算步长 (弧度)
        delta_r = (r_max - r_min) / r_res
        delta_z = (z_max - z_min) / z_res  # 修改: 计算 delta_z
        delta_theta = 2 * np.pi / theta_res
        # delta_phi 不再需要

        # 2. 创建柱坐标网格
        # 我们取每个格子的中心点作为高斯球的中心
        theta_vals = torch.linspace(0, 2 * np.pi, theta_res + 1)[:-1] + delta_theta / 2
        r_vals = torch.linspace(r_min, r_max, r_res + 1)[:-1] + delta_r / 2
        z_vals = torch.linspace(z_min, z_max, z_res + 1)[:-1] + delta_z / 2 # 修改: 创建 z_vals
        
        # 修改: 网格现在是 theta, r, z
        grid_r, grid_theta, grid_z = torch.meshgrid(r_vals, theta_vals, z_vals, indexing='ij')

        # 3. 为每个点计算其格子的笛卡尔尺寸 (Δx, Δy, Δz)
        cos_theta, sin_theta = torch.cos(grid_theta), torch.sin(grid_theta)
        # cos_phi, sin_phi 不再需要

        # 修改: Z方向的尺寸现在是一个常数，与位置无关
        # 因为z轴是独立的，所以格子的“高度”就是delta_z
        delta_y_values = torch.full_like(grid_z, delta_z)
        
        # 修改: X和Y方向的尺寸计算变得更简单，只和r, theta有关
        # 这描述了一个2D极坐标网格单元在x,y方向上的尺寸
        delta_z = torch.abs(cos_theta) * delta_r + torch.abs(grid_r * sin_theta) * delta_theta
                    
        delta_x = torch.abs(sin_theta) * delta_r + torch.abs(grid_r * cos_theta) * delta_theta

        # 4. 计算初始Scale (格子尺寸的一半)
        scales = torch.stack([delta_x / 2, delta_y_values / 2, delta_z / 2], dim=-1) * 1.01
        
        return scales

    @staticmethod
    def get_panorama_reference_points(THETA, R, Z, pc_range, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            THETA, R: spatial shape of tpv plane.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space
        rs = (pc_range[3] - pc_range[0]) * torch.linspace(0, R, R+1, dtype=dtype,
                            device=device)[:-1].view(1, 1, -1).expand(Z, THETA, R) / R 
        thetas = 2 * torch.pi * torch.linspace(0, THETA, THETA+1, dtype=dtype,
                            device=device)[:-1].view(1, -1, 1).expand(Z, THETA, R) / THETA
        zs = (pc_range[5] - pc_range[2]) * torch.linspace(0, Z, Z+1, dtype=dtype,
                            device=device)[:-1].view(-1, 1, 1).expand(Z, THETA, R) / Z
        
        xs = -torch.sin(thetas) * rs
        ys = zs + pc_range[2] 
        zs = -torch.cos(thetas) * rs

        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(2, 1, 0, 3) # w, h, z, 3
        # ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        # ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        # ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1, 1) # b, w, h, z, 3
        return ref_3d
    
    def get_offsets_reference_points(self, 
                                     THETA, 
                                     R, 
                                     Z, 
                                     offset_T, 
                                     offset_R, 
                                     offset_Z,
                                     delta_T,
                                     delta_R,
                                     delta_Z, 
                                     pc_range, 
                                     dim='3d', 
                                     bs=1, 
                                     device='cuda', 
                                     dtype=torch.float,
                                     ):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            THETA, R: spatial shape of tpv plane.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # 1. 计算步长 (弧度)
        delta_r = (pc_range[3] - pc_range[0]) * delta_R / R
        delta_z = (pc_range[5] - pc_range[2]) * delta_Z / Z  # 修改: 计算 delta_z
        delta_theta = 2 * np.pi * delta_T / THETA

        # reference points in 3D space
        rs = (pc_range[3] - pc_range[0]) * torch.linspace(0, R, R+1, dtype=dtype,
                            device=device)[:-1].view(-1, 1, 1).expand(R, THETA, Z)[None,:,:,:,None,None] / R + offset_R 
        thetas = 2 * torch.pi * torch.linspace(0, THETA, THETA+1, dtype=dtype,
                            device=device)[:-1].view(1, -1, 1).expand(R, THETA, Z)[None,:,:,:,None,None] / THETA + offset_T
        zs = (pc_range[5] - pc_range[2]) * torch.linspace(0, Z, Z+1, dtype=dtype,
                            device=device)[:-1].view(1, 1, -1).expand(R, THETA, Z)[None,:,:,:,None,None] / Z + offset_Z
        # rs = torch.clamp(rs, min=0)
        xs = -torch.sin(thetas) * rs
        ys = zs + pc_range[2] 
        zs = -torch.cos(thetas) * rs

        # 2. 计算每个点的笛卡尔尺寸 (Δx, Δy, Δz)
        cos_theta, sin_theta = torch.cos(thetas), torch.sin(thetas)
        delta_y = delta_z
        delta_z_values = torch.abs(cos_theta) * delta_r + torch.abs(rs * sin_theta) * delta_theta                    
        delta_x_values = torch.abs(sin_theta) * delta_r + torch.abs(rs * cos_theta) * delta_theta

        ref_3d = torch.cat((xs, ys, zs), -1)
        # scale_3d = torch.cat((delta_x_values / 2, delta_y / 2, delta_z_values / 2), -1) * 1.01
        scale_3d = torch.cat((delta_x_values, delta_y, delta_z_values), -1) * 1.01
        # ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        # ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        # ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        return ref_3d, scale_3d

    def forward(self, tpv_list, debug=False):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_thetar, tpv_ztheta, tpv_rz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_thetar.shape

        tpv_thetar = tpv_thetar.permute(0, 2, 1).reshape(bs, c, self.tpv_theta, self.tpv_r) # [theta, phi]
        tpv_ztheta = tpv_ztheta.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_theta) # [phi, r]
        tpv_rz = tpv_rz.permute(0, 2, 1).reshape(bs, c, self.tpv_r, self.tpv_z) # [r, theta]

        # #print("before voxelize:{}".format(torch.cuda.memory_allocated(0)))
        tpv_thetar = tpv_thetar.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
        tpv_ztheta = tpv_ztheta.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_r*self.tpv_r, -1, -1)
        tpv_rz = tpv_rz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_theta*self.tpv_theta, -1)

        gaussians = tpv_thetar + tpv_ztheta + tpv_rz
        #print("after voxelize:{}".format(torch.cuda.memory_allocated(0)))
        gaussians = gaussians.permute(0, 2, 3, 4, 1) # bs, w, h, z, c
        bs, w, h, z, _ = gaussians.shape

        if self.use_checkpoint:
            gaussians = torch.utils.checkpoint.checkpoint(self.decoder, gaussians, use_reentrant=False)
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, gaussians, use_reentrant=False)
            # gaussians = gaussians.view(bs, num_points, self.gpv, -1)
            gaussians = gaussians.view(bs, w, h, z, self.gpv, -1)
        else:
            gaussians = self.decoder(gaussians)
            gaussians = self.gs_decoder(gaussians)
            # gaussians = gaussians.view(bs, num_points, self.gpv, -1)
            gaussians = gaussians.view(bs, w, h, z, self.gpv, -1)
        #print("after decode:{}".format(torch.cuda.memory_allocated(0)))
        # gs_offsets_x = self.pos_act(gaussians[..., :1]) * self.scale_cylinder[None, ..., None, :1] # bs, w, h, z, 3
        # gs_offsets_y = self.pos_act(gaussians[..., 1:2]) * self.scale_cylinder[None, ..., None, 1:2] # bs, w, h, z, 3
        # gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.scale_cylinder[None, ..., None, 2:3] # bs, w, h, z, 3
        
        # gs_offsets_x = self.pos_act(gaussians[..., :1]) # bs, w, h, z, 3
        # gs_offsets_y = self.pos_act(gaussians[..., 1:2]) # bs, w, h, z, 3
        # gs_offsets_z = self.pos_act(gaussians[..., 2:3]) # bs, w, h, z, 3

        gs_offsets_r = self.pos_act(gaussians[..., :1]) * self.offset_max[0] # r
        gs_offsets_theta = self.pos_act(gaussians[..., 1:2]) * self.offset_max[1] # theta
        gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.offset_max[2] # z

        scale_x = self.scale_act(gaussians[..., 3:4])
        scale_y = self.scale_act(gaussians[..., 4:5])
        scale_z = self.scale_act(gaussians[..., 5:6])

        #gs_offsets = gaussians[..., :3]
        gs_positions, scale_3d = self.get_offsets_reference_points(
            self.tpv_theta * self.scale_theta, 
            self.tpv_r * self.scale_r, 
            self.tpv_z * self.scale_z, 
            gs_offsets_theta, gs_offsets_r, gs_offsets_z,
            scale_x, scale_y, scale_z, 
            self.pc_range, bs=bs, device=gaussians.device
        )
        # gs_positions = torch.cat([gs_offsets_x, gs_offsets_y, gs_offsets_z], dim=-1) + self.gs_anchors[:, :, :, :, None, :]
        
        x = torch.cat([gs_positions, scale_3d, gaussians[..., 6:]], dim=-1)
        rgbs = self.rgb_act(x[..., 6:9])
        opacity = self.opacity_act(x[..., 9:10])
        rotation = self.rot_act(x[..., 10:14])

        # scale_x = self.scale_act(x[..., 11:12])
        # scale_y = self.scale_act(x[..., 12:13])
        # scale_z = self.scale_act(x[..., 13:14])

        if debug:
            opacity[:] = 1.0
            scale_x[:] = 0.5
            scale_y[:] = 0.5
            scale_z[:] = 0.5
            rgbs[..., 0] = 1.0
            rgbs[..., 1] = 0.0
            rgbs[..., 2] = 0.0

        gaussians = torch.cat([gs_positions, rgbs, opacity, rotation, scale_3d], dim=-1) # bs, w, h, z, gpv, 14
    
        return gaussians
