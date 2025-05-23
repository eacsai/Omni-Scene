
import torch, torch.nn as nn, torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from sample_anchors import sample_concentrating_sphere, project_onto_planes
import math
from vis_feat import single_features_to_RGB
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
        self.offset_max = [1.0] * 3
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
        # gs_anchors = self.get_sample_reference_points(anchors_coordinates, pc_range[5] - pc_range[2]) # [num_points, 1, 3]
        # mask_lower = gs_anchors[:, 0, 1] >= -2.5
        # mask_upper = gs_anchors[:, 0, 1] <= 2.5
        # combined_mask = torch.logical_and(mask_lower, mask_upper)
        self.register_buffer('gs_anchors', gs_anchors)
        # self.register_buffer('anchors_coordinates', anchors_coordinates[combined_mask])

    
    @staticmethod
    def get_square_coordinates(ref_3d, pc_depth):
        eps = 1e-8
        theta = (torch.atan2(ref_3d[..., 0], ref_3d[..., 2]) + torch.pi)/(2 * torch.pi) * 2 - 1
        phi = (torch.atan2(ref_3d[..., 1], torch.sqrt(ref_3d[..., 0]**2 + ref_3d[..., 2]**2 + eps)) + torch.pi/2)/torch.pi * 2 - 1         
        r = torch.sqrt(ref_3d[..., 0]**2 + ref_3d[..., 1]**2 + ref_3d[..., 2]**2) / pc_depth * 2 - 1
        return torch.cat((theta,phi,r), dim=-1) # num_points, 3

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
        gs_offsets_x = self.pos_act(gaussians[..., :1]) * self.offset_max[0] # bs, w, h, z, 3
        gs_offsets_y = self.pos_act(gaussians[..., 1:2]) * self.offset_max[1] # bs, w, h, z, 3
        gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.offset_max[2] # bs, w, h, z, 3
        #gs_offsets = gaussians[..., :3]
        gs_positions = torch.cat([gs_offsets_x, gs_offsets_y, gs_offsets_z], dim=-1) + self.gs_anchors[:, :, :, :, None, :]
        x = torch.cat([gs_positions, gaussians[..., 3:]], dim=-1)
        rgbs = self.rgb_act(x[..., 3:6])
        opacity = self.opacity_act(x[..., 6:7])
        rotation = self.rot_act(x[..., 7:11])
        scale_x = self.scale_act(x[..., 11:12]) * self.scale_max[0]
        scale_y = self.scale_act(x[..., 12:13]) * self.scale_max[1]
        scale_z = self.scale_act(x[..., 13:14]) * self.scale_max[2]

        if debug:
            opacity[:] = 1.0
            scale_x[:] = 0.5
            scale_y[:] = 0.5
            scale_z[:] = 0.5
            rgbs[..., 0] = 1.0
            rgbs[..., 1] = 0.0
            rgbs[..., 2] = 0.0

        gaussians = torch.cat([gs_positions, rgbs, opacity, rotation, scale_x, scale_y, scale_z], dim=-1) # bs, w, h, z, gpv, 14
    
        return gaussians
