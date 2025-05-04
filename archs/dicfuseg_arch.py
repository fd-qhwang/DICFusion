import torch
import torch.nn as nn
# 加上这句话可以解决绝对引用的问题，但是同时导致了相对引用的问题
import sys,os
sys.path.append(os.getcwd())

from utils.registry import ARCH_REGISTRY
from archs.arch_utils import FusionDecoder, SharedBackbone, SegmentationHead, FusionUpsampler
from archs.attn_utils import CrossTransformerBlock, MAFusion
import sys,os
from torchinfo import summary

import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format

@ARCH_REGISTRY.register()
class DICFusegNet(nn.Module):
    def __init__(self, in_channels, base_channel, num_classes, channels_list, num_blocks_list):
        super(DICFusegNet, self).__init__()

        # 浅层卷积特征提取层
        self.conv_ir = nn.Sequential(
            nn.Conv2d(in_channels, channels_list[0], kernel_size=3, padding=1),
        )
        self.conv_vis = nn.Sequential(
            nn.Conv2d(in_channels, channels_list[0], kernel_size=3, padding=1),
        )

        # 特征融合后的维度调整层
        self.conv_fuse = MAFusion(dim=channels_list[0])

        # 主干网络
        self.backbone = SharedBackbone(
            in_channels=channels_list[0],
            channels_list=channels_list,
            num_blocks_list=num_blocks_list,
        )

        # 融合解码器
        self.fusion_upsampler = FusionUpsampler(channels_list=channels_list)

        # 分割头
        self.segmentation_head = SegmentationHead(
            in_channels_list=channels_list,
            base_channel=base_channel,
            num_classes=num_classes
        )
        
        self.cross_attn = CrossTransformerBlock(
                            dim=channels_list[0],  # 特征的通道数
                            kv_dim=base_channel,
                            num_heads=4,
                            ffn_expansion_factor=2.66,
                            bias=True,
                            LayerNorm_type='WithBias'
                        )
        self.fusion_decoder = FusionDecoder(
                    in_channels=channels_list[0],
                    out_channels=1,  # 假设输出通道数为 1
                    num_blocks=num_blocks_list[3],
                )

    def forward(self, vis_input, ir_input):
        # 每个模态的浅层特征提取
        feat_vis = self.conv_vis(vis_input)
        feat_ir = self.conv_ir(ir_input)


        # 维度调整，确保通道数一致
        unified_feat = self.conv_fuse(feat_ir, feat_vis)

        # 通过主干网络
        deep_features, skips, segmentation_features = self.backbone(unified_feat)

        # 融合解码和分割输出
        fusion_feature = self.fusion_upsampler(deep_features, skips)
        boundary_out, binary_out, seg_feature, segmentation_output = self.segmentation_head(segmentation_features)
        
        fusion_feature = self.cross_attn(fusion_feature, seg_feature) # 可以用于测试不加上分割特征的结果

        # 结合不同模态输入得到最终输出结果
        #fusion_output = self.fusion_decoder(fusion_feature, torch.max(vis_input, ir_input))
        fusion_output = self.fusion_decoder(fusion_feature)
        #fusion_output = self.fusion_decoder(fusion_feature, ir_input)

        return fusion_output, boundary_out, binary_out, segmentation_output

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())  # 确保可正确 import DICFusegNet

    # 1. 如果需要 GPU 且可用，则用 cuda:3，否则用 CPU
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # ========== 2) 创建模型并切换到 eval 模式 ==========
    in_channels = 1       # 输入通道=1
    base_channel = 128
    num_classes = 9
    channels_list = [64, 128, 256]
    num_blocks_list = [4, 6, 4, 2]

    model = DICFusegNet(
        in_channels=in_channels,
        base_channel=base_channel,
        num_classes=num_classes,
        channels_list=channels_list,
        num_blocks_list=num_blocks_list
    ).to(device).eval()

    # ========== 3) 准备测试输入 (B=1, C=1, H=480, W=640) ==========
    ir_input = torch.randn(1, 1, 256, 256, device=device)
    vis_input = torch.randn(1, 1, 256, 256, device=device)

    # ========== 4) 前向推理：模型返回 4 个张量 (fusion_out, boundary_out, binary_out, seg_out) ==========
    with torch.no_grad():
        out_tuple = model(ir_input, vis_input)
    # out_tuple 是一个元组(tuple)，包含 4 个张量
    fusion_out, boundary_out, binary_out, segmentation_out = out_tuple

    # 打印每个输出张量的维度，避免 "tuple has no attribute shape" 错误
    print("[Forward] fusion_out shape:       ", fusion_out.shape)
    print("[Forward] boundary_out shape:     ", boundary_out.shape)
    print("[Forward] binary_out shape:       ", binary_out.shape)
    print("[Forward] segmentation_out shape: ", segmentation_out.shape)

    # ========== 5) 使用 torchinfo 查看网络结构 (需 pip install torchinfo) ==========
    print("\n================== [torchinfo] Summary ==================")
    try:
        from torchinfo import summary
        summary(
            model,
            input_size=[(1, 1, 256, 256), (1, 1, 256, 256)],
            col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
            depth=4  # 可根据需要调整
        )
    except ImportError:
        print("未安装 torchinfo 或导入失败，请先 pip install torchinfo.")

    # ========== 6) 使用 THOP 统计 FLOPs & Params (需 pip install thop) ==========
    print("\n================== [THOP] 统计 FLOPs & Params ==================")
    try:
        from thop import profile, clever_format
        flops_thop, params_thop = profile(model, inputs=(ir_input, vis_input))
        # 转换单位，FLOPs => G，Params => M
        flops_g = flops_thop / 1e9
        params_m = params_thop / 1e6
        print(f"[THOP] FLOPs: {flops_g:.3f} G, Params: {params_m:.3f} M")

        # 如果需要更紧凑的格式，可用 clever_format
        flops_fmt, params_fmt = clever_format([flops_thop, params_thop], "%.3f")
        print(f"[THOP => formatted] FLOPs: {flops_fmt}, Params: {params_fmt}")
    except ImportError:
        print("未安装 thop，请先 pip install thop.")
    except Exception as e:
        print("THOP 统计失败，报错信息：", e)

    # ========== 7) 使用 fvcore 统计 FLOPs & Params (需 pip install fvcore) ==========
    print("\n================== [fvcore] 统计 FLOPs & Params ==================")
    try:
        from fvcore.nn import FlopCountAnalysis
        flop_analyser = FlopCountAnalysis(model, (ir_input, vis_input))
        flops_fvcore = flop_analyser.total()
        flops_fvcore_g = flops_fvcore / 1e9

        # 计算模型参数量
        params_fvcore = sum(p.numel() for p in model.parameters())
        params_fvcore_m = params_fvcore / 1e6
        print(f"[fvcore] FLOPs: {flops_fvcore_g:.3f} G, Params: {params_fvcore_m:.3f} M")
    except ImportError:
        print("未安装 fvcore，请先 pip install fvcore.")
    except Exception as e:
        print("fvcore 统计失败，报错信息：", e)
