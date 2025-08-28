import functools
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torchvision.models import ResNet
from mmengine.registry import MODELS
from torch.utils.checkpoint import checkpoint

from .backbone import Backbone
from .unimatch.utils import split_feature, merge_splits
from .unimatch.position import PositionEmbeddingSine
from .multiview_transformer import MultiViewFeatureTransformer


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list

@dataclass
class BackboneResnetCfg:
    name: Literal["resnet"]
    model: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"
    ]
    num_layers: int
    use_first_pool: bool
    d_out: int

@MODELS.register_module()
class BackboneResnet(Backbone[BackboneResnetCfg]):
    model: ResNet

    def __init__(self, d_in: int = 3, num_cams: int = 2, attn_splits: int = 2, no_cross_attn: bool = False) -> None:
        super().__init__()

        assert d_in == 3
        self.num_layers = 4
        self.use_epipolar_transformer = True
        self.attn_splits = attn_splits
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50", trust_repo=True)
        self.plucker_to_embed = nn.Linear(6, 512)
        # Set up projections
        self.projections = nn.ModuleDict({})
        for index in range(1, self.num_layers):
            key = f"layer{index}"
            block = getattr(self.model, key)
            conv_index = 1
            try:
                while True:
                    d_layer_out = getattr(block[-1], f"conv{conv_index}").out_channels
                    conv_index += 1
            except AttributeError:
                pass
            self.projections[key] = nn.Conv2d(d_layer_out, 512, 1)

        # Add a projection for the first layer.
        self.projections["layer0"] = nn.Conv2d(
            self.model.conv1.out_channels, 512, 1
        )
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512 + 2, 128, kernel_size=1, bias=True),
        )
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.ReLU(),
        )   

        # downscaler
        self.downscaler = nn.Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))

        self.transformer = MultiViewFeatureTransformer(
            num_layers=4,
            d_model=128,
            nhead=1,
            ffn_dim_expansion=4,
            no_cross_attn=no_cross_attn,
        )

        # upscaler
        self.upscaler = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=(4, 4), stride=(4, 4)
            ),
        )

        self.upscale_refinement = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU(approximate='none'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        )

    def forward(
        self,
        img,
        depths_in, 
        confs_in, 
        pluckers,
    ) -> Float[Tensor, "batch view d_out height width"]:
        # Merge the batch dimensions.
        b, v, c, h, w = img.shape
        x = rearrange(img, "b v c h w -> (b v) c h w")
        
        skip = self.high_resolution_skip(x)
        # Layer 0
        x0 = self.model.conv1(x)
        x0 = self.model.bn1(x0)
        x0 = self.model.relu(x0)

        x1 = self.model.layer1(x0)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)

        p4 = self.projections["layer3"](x3)
        p3 = F.interpolate(p4, size=x2.shape[-2:], mode="bilinear", align_corners=True)
        p3 = p3 + self.projections["layer2"](x2)
        p2 = F.interpolate(p3, size=x1.shape[-2:], mode="bilinear", align_corners=True)
        p2 = p2 + self.projections["layer1"](x1)
        p1 = F.interpolate(p2, size=x0.shape[-2:], mode="bilinear", align_corners=True)
        p1 = p1 + self.projections["layer0"](x0)

        features = F.interpolate(p1, (h, w), mode="bilinear", align_corners=True)
    
        pluckers = rearrange(pluckers, "b v c h w -> (b v) h w c")
        plucker_embeds = self.plucker_to_embed(pluckers)
        plucker_embeds = rearrange(plucker_embeds, "(b v) h w c -> b v c h w", b=b, v=v)
        
        features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
        features = features + plucker_embeds
        features = rearrange(features, "b v c h w -> (b v) c h w")
        
        # rearrange pseudo depths and confs
        depths_in = rearrange(depths_in, "b v c h w -> (b v) c h w")
        confs_in = rearrange(confs_in, "b v c h w -> (b v) c h w")
        depth_scale = 2.0

        features = torch.cat([features, depths_in / depth_scale, confs_in], dim=1)
        features = self.backbone_projection(features)

        # downscale
        features = self.downscaler(features)
        features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
        
        # Apply cross-view attention.
        features_list = list(torch.unbind(features, dim=1))

        # Apply sin position embedding.
        # cur_features_list = feature_add_position_list(features_list, self.attn_splits, 128)
        # cur_features_list = self.transformer(cur_features_list, attn_num_splits=self.attn_splits)

        # Apply RoPE position embedding.
        cur_features_list = self.transformer(features_list, attn_num_splits=self.attn_splits)


        features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        # upscale
        features = rearrange(features, "b v c h w -> (b v) c h w")
        features = self.upscaler(features)
        features = self.upscale_refinement(features) + features

        # # Run the epipolar transformer.
        # if self.use_epipolar_transformer:
        #     features, sampling = self.epipolar_transformer(
        #         features,
        #         context["extrinsics"],
        #     )

        features = features + skip
        # Separate batch dimensions.
        return rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

    @property
    def d_out(self) -> int:
        return 512