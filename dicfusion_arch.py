import torch
import torch.nn as nn
import torch.nn.functional as F
import sys,os
sys.path.append(os.getcwd())

from utils.registry import ARCH_REGISTRY
from archs.arch_utils import FusionDecoder, SharedBackbone, SegmentationHead, FusionUpsampler
from archs.attn_utils import CrossTransformerBlock, MAFusion

@ARCH_REGISTRY.register()
class DICFusionNet(nn.Module):
    def __init__(self, in_channels, base_channel, num_classes, channels_list, num_blocks_list):
        super(DICFusionNet, self).__init__()

        self.conv_ir = nn.Sequential(
            nn.Conv2d(in_channels, channels_list[0], kernel_size=3, padding=1),
            #nn.BatchNorm2d(channels_list[0]),
            #nn.LeakyReLU(0.2, inplace=False),
            #nn.Conv2d(channels_list[0], channels_list[0], kernel_size=3, padding=1),
        )
        self.conv_vis = nn.Sequential(
            nn.Conv2d(in_channels, channels_list[0], kernel_size=3, padding=1),
            #nn.BatchNorm2d(channels_list[0]),
            #nn.LeakyReLU(0.2, inplace=False),
            #nn.Conv2d(channels_list[0], channels_list[0], kernel_size=3, padding=1),
        )

        self.conv_fuse = MAFusion(dim=channels_list[0])

        self.backbone = SharedBackbone(
            in_channels=channels_list[0],
            channels_list=channels_list,
            num_blocks_list=num_blocks_list,
        )

        self.fusion_upsampler = FusionUpsampler(channels_list=channels_list)

        self.segmentation_head = SegmentationHead(
            in_channels_list=channels_list,
            base_channel=base_channel,
            num_classes=num_classes
        )
        
        self.cross_attn = CrossTransformerBlock(
                            dim=channels_list[0],  
                            kv_dim=base_channel,
                            num_heads=4,
                            ffn_expansion_factor=2.66,
                            bias=True,
                            LayerNorm_type='WithBias'
                        )
        self.fusion_decoder = FusionDecoder(
                    in_channels=channels_list[0],
                    out_channels=1,  
                    num_blocks=num_blocks_list[3],
                )

    def forward(self, vis_input, ir_input):
        feat_vis = self.conv_vis(vis_input)
        feat_ir = self.conv_ir(ir_input)


        #unified_feat = self.conv_fuse(feat_vis, feat_ir)
        unified_feat = self.conv_fuse(feat_ir, feat_vis)

        deep_features, skips, segmentation_features = self.backbone(unified_feat)

        fusion_feature = self.fusion_upsampler(deep_features, skips)
        boundary_out, binary_out, seg_feature, segmentation_output = self.segmentation_head(segmentation_features)
        
        fusion_feature = self.cross_attn(fusion_feature, seg_feature)
        #fusion_output = self.fusion_decoder(fusion_feature, torch.max(vis_input, ir_input))
        fusion_output = self.fusion_decoder(fusion_feature)
        #fusion_output = self.fusion_decoder(fusion_feature, vis_input)

        return fusion_output, boundary_out, binary_out, segmentation_output