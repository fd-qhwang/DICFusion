import torch
import torch.nn as nn
import torch.nn.functional as F
# 加上这句话可以解决绝对引用的问题，但是同时导致了相对引用的问题
import sys,os
sys.path.append(os.getcwd())
from utils.registry import ARCH_REGISTRY
from archs.mambair_arch import VSSBlock
import sys,os

    
##########################################################################
class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

##########################################################################
# conv building blocks from efficient net
class MBConv(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, expansion_factor=4):
        super(MBConv, self).__init__()
        
        self.expanded_channels = channels * expansion_factor
        self.use_residual = stride == 1
        
        self.expand_conv = nn.Sequential(
            nn.Conv2d(channels, self.expanded_channels, kernel_size=1),
            nn.BatchNorm2d(self.expanded_channels),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.expanded_channels, self.expanded_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.expanded_channels),
            nn.BatchNorm2d(self.expanded_channels),
            nn.LeakyReLU(0.2, inplace=False),
            SEModule(self.expanded_channels),
            nn.Conv2d(self.expanded_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=False)
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(self.expand_conv(x))
        else:
            return self.conv(self.expand_conv(x))


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        # 使用全局平均池化和两个全连接层实现
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y
    
##########################################################################
class FusionBlock(nn.Module):
    def __init__(self, channels, d_state=16, mlp_ratio=2.):
        super(FusionBlock, self).__init__()
        # 由于通道数被拆分，每个分支的输入通道数为 channels // 2
        self.channels = channels
        half_channels = channels // 2

        # CNN块
        self.cnn_block = MBConv(half_channels)

        # Mamba块
        self.mamba_block = VSSBlock(
            hidden_dim=half_channels,
            drop_path=0,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            d_state=d_state,
            expand=mlp_ratio,
            is_light_sr=False
        )

        # 通道注意力机制
        self.channel_attention = ChannelAttention(channels)

        # 调整通道数
        self.conv_adjust = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 将通道一分为二
        split_channels = torch.chunk(x, 2, dim=1)  # x: [B, C, H, W], split_channels: list of two tensors each of shape [B, C//2, H, W]
        
        # 分别通过 CNN 和 Mamba 块
        cnn_feat = self.cnn_block(split_channels[0])
        mamba_feat = self.mamba_block(split_channels[1])

        # 特征拼接
        combined_feat = torch.cat([cnn_feat, mamba_feat], dim=1)  # [B, C, H, W]

        # 通道注意力
        attention_feat = self.channel_attention(combined_feat)

        # 调整通道数
        out = self.conv_adjust(attention_feat)
        # 残差连接
        out = out + x
        return out


# 更新后的DICFeatureExtractor
class DICFeatureExtractor(nn.Module):
    def __init__(self, in_channels=36, num_blocks=4, d_state=16, mlp_ratio=2.):
        super(DICFeatureExtractor, self).__init__()
        self.blocks = nn.Sequential(
            *[FusionBlock(in_channels, d_state=d_state, mlp_ratio=mlp_ratio) for _ in range(num_blocks)]
        )

    def forward(self, x):
        out = self.blocks(x)
        return out

# 下采样模块
class Downsample(nn.Module):
    '''
    功能：使用卷积进行下采样，同时扩大通道数
    '''
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True)
        self.shortcut = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)

        self.conv_down = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.lrelu1(self.conv_1(x))
        x = self.lrelu2(self.conv_2(x))
        x += identity

        x_down = self.conv_down(x)
        return x_down, x  # 返回下采样后的特征和用于跳跃连接的特征


class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockUp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lrelu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=False)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.lrelu1(self.conv_1(x))
        x = self.lrelu2(self.conv_2(x))
        x += identity

        return x
    
class Upsample(nn.Module):
    '''
    功能：使用卷积进行上采样，同时减小通道数
    '''
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv_up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True)
        self.block = ResBlockUp(out_channels * 2, out_channels)  # 乘以2是因为会与跳跃连接的特征拼接

    def forward(self, x, skip):
        x = self.conv_up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x
    
class FusionUpsampler(nn.Module):
    def __init__(self, channels_list, out_channels=1):
        super(FusionUpsampler, self).__init__()

        # 上采样模块，使用 channels_list 反向索引
        self.upsample1 = Upsample(channels_list[2], channels_list[1])
        self.upsample2 = Upsample(channels_list[1], channels_list[0])
        self.upsample3 = Upsample(channels_list[0], channels_list[0])  # 最后输出通道数可以自行定义

        # 最终卷积和激活函数
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels_list[0], channels_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_list[0]), #之前训练的大部分都是有bnde
            nn.LeakyReLU(0.2, inplace=False)  # 添加 ReLU 激活函数
        )

    def forward(self, x, skips, inp_img=None):
        # skips 顺序为 [skip1, skip2, skip3]
        x = self.upsample1(x, skips[2])  # 对应 channels_list[2] 和 channels_list[1]
        x = self.upsample2(x, skips[1])  # 对应 channels_list[1] 和 channels_list[0]
        x = self.upsample3(x, skips[0])  # 对应 channels_list[0] 和 channels_list[0]
        x = self.final_conv(x)

        return x


class FusionDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, d_state=16, mlp_ratio=2.):
        super(FusionDecoder, self).__init__()

        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels), #之前训练的大部分都是有bnde
            nn.LeakyReLU(0.2, inplace=False),
            #nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )
        self.dic_extractor = DICFeatureExtractor(
            in_channels=in_channels,
            num_blocks=num_blocks,
            d_state=d_state,
            mlp_ratio=mlp_ratio
        )
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, inp_img=None):
        x = self.decoder1(x)
        x = self.dic_extractor(x)
        # 通过最终卷积和 tanh 激活函数
        x = self.final_conv(x)

        # 如果提供了输入图像，则加到输出上
        if inp_img is not None:
            x = x + inp_img

        x = (self.tanh(x) + 1) / 2  # 将输出调整到 (0, 1) 区间
        
        return x

    
class SharedBackbone(nn.Module):
    def __init__(self, in_channels, channels_list, num_blocks_list):
        super(SharedBackbone, self).__init__()

        # 通道数列表，例如 channels_list = [64, 128, 256]
        self.channels_list = channels_list

        # 下采样模块和 DICFeatureExtractor
        self.downsample1 = Downsample(in_channels, channels_list[0])
        self.cms_block1 = DICFeatureExtractor(
            channels_list[0],
            num_blocks=num_blocks_list[0],
        )

        self.downsample2 = Downsample(channels_list[0], channels_list[1])
        self.cms_block2 = DICFeatureExtractor(
            channels_list[1],
            num_blocks=num_blocks_list[1],
        )

        self.downsample3 = Downsample(channels_list[1], channels_list[2])
        self.cms_block3 = DICFeatureExtractor(
            channels_list[2],
            num_blocks=num_blocks_list[2],
        )

    def forward(self, x):
        # 输入 x 形状为 (B, in_channels, H, W)

        # 下采样到 H/2 x W/2
        x_down1, skip1 = self.downsample1(x)
        # 特征提取
        x_cms1 = self.cms_block1(x_down1)

        # 下采样到 H/4 x W/4
        x_down2, skip2 = self.downsample2(x_cms1)
        # 特征提取
        x_cms2 = self.cms_block2(x_down2)

        # 下采样到 H/8 x W/8
        x_down3, skip3 = self.downsample3(x_cms2)
        # 特征提取
        x_cms3 = self.cms_block3(x_down3)

        # 深层特征
        deep_features = x_cms3  # (B, channels_list[2], H/8, W/8)

        # 跳跃连接
        skips = [skip1, skip2, skip3]  # 分别对应不同的通道数

        # 用于分割头的特征
        segmentation_features = [x_cms1, x_cms2, x_cms3]

        return deep_features, skips, segmentation_features  

# 辅助模块定义
def ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def BasicConv2d(in_channels, out_channels, **kwargs):
    return nn.Conv2d(in_channels, out_channels, **kwargs)

# S2PM 模块定义
class S2PM(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(S2PM, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        out = self.block3(x2)
        return out

# 修改后的 SegmentationHead 类
class SegmentationHead(nn.Module):
    def __init__(self, in_channels_list, base_channel=128, num_classes=9):
        super(SegmentationHead, self).__init__()

        # 定义 FPN 的横向和自顶向下连接
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, base_channel, kernel_size=1, stride=1, padding=0)
            fpn_conv = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        # 使用 S2PM 处理 seg_feature
        self.seg_feature_module = S2PM(in_channel=base_channel, out_channel=base_channel)

        # 使用 S2P2 中的分类器
        feature = base_channel  # 分类器的输入通道数

        self.binary_conv1 = ConvBNReLU(feature, feature // 4, kernel_size=1)
        self.binary_conv2 = nn.Conv2d(feature // 4, 2, kernel_size=3, padding=1)

        self.semantic_conv1 = ConvBNReLU(feature, feature, kernel_size=1)
        self.semantic_conv2 = nn.Conv2d(feature, num_classes, kernel_size=3, padding=1)

        self.boundary_conv = nn.Sequential(
            nn.Conv2d(feature * 2, feature, kernel_size=1),
            nn.BatchNorm2d(feature),
            nn.ReLU6(inplace=True),
            nn.Conv2d(feature, 2, kernel_size=3, padding=1),
        )

        # 上采样层
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, features):
        """
        Args:
            features: List of feature maps from different stages.
                The expected sizes are:
                features[0]: [B, C1, H/2, W/2]
                features[1]: [B, C2, H/4, W/4]
                features[2]: [B, C3, H/8, W/8]
        Returns:
            boundary_out: [B, 2, H, W]
            binary_out: [B, 2, H, W]
            seg_feature: [B, base_channel, H/2, W/2]
            segmentation_output: [B, num_classes, H, W]
        """

        # 确保输入特征图列表长度为 3
        assert len(features) == 3, "Expected 3 feature maps, got {}".format(len(features))

        # 特征图尺寸
        B, _, H_2, W_2 = features[0].shape  # H/2, W/2
        H, W = H_2 * 2, W_2 * 2             # H, W

        # 1. FPN 横向连接和自顶向下融合
        # 横向连接
        laterals = []
        for lateral_conv, feature in zip(self.lateral_convs, features):
            lateral = lateral_conv(feature)  # [B, base_channel, h_i, w_i]
            laterals.append(lateral)

        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            size = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=size, mode='nearest')
            laterals[i - 1] += upsampled

        # 构建 FPN 特征图
        fpn_features = []
        for i in range(len(laterals)):
            fpn_feature = self.fpn_convs[i](laterals[i])  # [B, base_channel, h_i, w_i]
            fpn_features.append(fpn_feature)

        # 2. 将 FPN 特征图上采样到 [H/2, W/2] 并融合
        for i in range(len(fpn_features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=(H_2, W_2), mode='bilinear', align_corners=False)

        fpn_out = sum(fpn_features) / len(fpn_features)  # [B, base_channel, H/2, W/2]

        # 3. 使用 S2PM 处理 seg_feature
        seg_feature = self.seg_feature_module(fpn_out)  # [B, base_channel, H/2, W/2]

        # 4. 二值分割分支
        binary = self.binary_conv2(self.binary_conv1(seg_feature))  # [B, 2, H/2, W/2]
        binary_out = self.up2x(binary)  # 上采样到 [B, 2, H, W]

        '''
        # 计算权重
        weight = torch.exp(binary)
        weight = weight[:, 1:2, :, :] / torch.sum(weight, dim=1, keepdim=True)  # [B, 1, H/2, W/2]
        '''
        
        # 使用 softmax 计算权重
        weight = torch.softmax(binary, dim=1)
        weight = weight[:, 1:2, :, :]  # [B, 1, H/2, W/2]


        # 5. 语义分割分支
        feat_semantic = seg_feature * weight  # 加权特征图 [B, base_channel, H/2, W/2]
        feat_semantic = self.semantic_conv1(feat_semantic)  # [B, base_channel, H/2, W/2]
        segmentation_output = self.semantic_conv2(feat_semantic)  # [B, num_classes, H/2, W/2]
        segmentation_output = self.up2x(segmentation_output)  # 上采样到 [B, num_classes, H, W]

        # 6. 边界分割分支
        feat_boundary = torch.cat([feat_semantic, seg_feature], dim=1)  # [B, base_channel*2, H/2, W/2]
        boundary_out = self.boundary_conv(feat_boundary)  # [B, 2, H/2, W/2]
        boundary_out = self.up2x(boundary_out)  # 上采样到 [B, 2, H, W]

        return boundary_out, binary_out, seg_feature, segmentation_output