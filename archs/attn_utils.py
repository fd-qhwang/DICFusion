import torch.nn as nn
import torch
import torch.nn.functional as F
import numbers
from einops import rearrange
from einops.layers.torch import Rearrange


##########################################################################

class ConcatFusion(nn.Module):
    def __init__(self, feature_dim):
        super(ConcatFusion, self).__init__()
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
                    
    def forward(self, modality1, modality2):
        concat_features = torch.cat([modality1, modality2], dim=1)
        return self.dim_reduce(concat_features)
    

## Layer Norm
def check_tensor(tensor, name, interval=250):
    """
    检查张量的数值分布，并在指定间隔内输出一次统计信息。
    特殊情况（NaN 或 Inf）立即输出。

    Args:
        tensor (torch.Tensor): 要检查的张量。
        name (str): 张量的名称，用于标识输出。
        interval (int): 每隔多少次输出一次统计信息。
    """
    # 静态变量计数器初始化
    if not hasattr(check_tensor, "counter"):
        check_tensor.counter = 0  # 初始化计数器

    # 递增计数器
    check_tensor.counter += 1

    # 每次检查 NaN 和 Inf 并立即打印
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN")
    if torch.isinf(tensor).any():
        print(f"{name} contains Inf")

    # 每 interval 次打印统计信息
    if check_tensor.counter % interval == 0:
        print(f"{name} - max: {tensor.max().item()}, min: {tensor.min().item()}, mean: {tensor.mean().item()}")

    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, kv_dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 分别对 Query 和 Key/Value 进行线性变换
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_proj = nn.Conv2d(kv_dim, dim * 2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_query, x_key_value):
        b, c, h, w = x_query.shape

        # 计算 Query
        q = self.q_dwconv(self.q_proj(x_query))
        # 计算 Key 和 Value
        kv = self.kv_dwconv(self.kv_proj(x_key_value))
        k, v = kv.chunk(2, dim=1)

        # 重排维度以适应多头注意力
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对 Query 和 Key 进行归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 计算注意力得分
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 形状：(b, head, n, n)
        attn = attn.softmax(dim=-1)

        # 计算注意力输出
        out = attn @ v  # 形状：(b, head, c, n)

        # 恢复形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)

        # 输出投影
        out = self.project_out(out)
        return out

class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, kv_dim, num_heads=8, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(CrossTransformerBlock, self).__init__()

        self.norm1_query = LayerNorm(dim, LayerNorm_type)
        self.norm1_key_value = LayerNorm(kv_dim, LayerNorm_type)
        self.attn = CrossAttention(dim, kv_dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x_query, x_key_value):
        x_key_value = F.interpolate(x_key_value, size=x_query.shape[2:], mode='bilinear', align_corners=False)
        # 交叉注意力
        x = x_query + self.attn(self.norm1_query(x_query), self.norm1_key_value(x_key_value))
        # 前馈网络
        x = x + self.ffn(self.norm2(x))
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

    
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
    
    
class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


class Sobelxy(nn.Module):
    """Sobel 算子，用于边缘检测。"""
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

class Laplacian(nn.Module):
    """拉普拉斯算子，用于边缘检测。"""
    def __init__(self):
        super(Laplacian, self).__init__()
        # 定义拉普拉斯卷积核
        kernel = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 升维以适应卷积操作
        self.weight = nn.Parameter(data=kernel, requires_grad=False)  # 不更新权重

    def forward(self, x):
        # x 的形状应为 [B, C, H, W]，但由于卷积核是单通道的，我们需要对每个通道进行卷积
        # 可以使用 group 参数进行分组卷积
        B, C, H, W = x.size()
        weight = self.weight.repeat(C, 1, 1, 1)  # 扩展卷积核以匹配输入通道数
        laplacian = F.conv2d(x, weight, padding=1, groups=C)
        return laplacian
    
class ChannelAttentionIR(nn.Module):
    """针对红外图像的通道注意力模块，更关注全局信息。"""
    def __init__(self, dim, reduction=8):
        super(ChannelAttentionIR, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)  # 使用最大池化
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        x_mp = self.gmp(x).view(b, c)     # [B, C]
        attn = self.fc(x_mp)              # [B, C]
        attn = self.sigmoid(attn).view(b, c, 1, 1)
        return attn

class SpatialAttentionIR(nn.Module):
    """针对红外图像的空间注意力模块，强调边缘和形状信息。"""
    def __init__(self):
        super(SpatialAttentionIR, self).__init__()
        self.sobel = Sobelxy()
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, padding_mode='reflect', bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_gray = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_edge = self.sobel(x_gray)
        x_edge = (x_edge - x_edge.min()) / (x_edge.max() - x_edge.min() + 1e-6)
        attn = self.conv(x_edge)
        #check_tensor(attn, "edge attn before Sigmoid")
        attn = self.sigmoid(attn)
        return attn

class ChannelAttentionVIS(nn.Module):
    """针对可见光图像的通道注意力模块，更关注细节信息。"""
    def __init__(self, dim, reduction=16):
        super(ChannelAttentionVIS, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 使用平均池化
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        x_ap = self.gap(x).view(b, c)     # [B, C]
        attn = self.fc(x_ap)              # [B, C]
        attn = self.sigmoid(attn).view(b, c, 1, 1)
        return attn

class SpatialAttentionVIS(nn.Module):
    """针对可见光图像的空间注意力模块，强调纹理和细节信息。"""
    def __init__(self):
        super(SpatialAttentionVIS, self).__init__()
        self.laplacian = Laplacian()
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, padding_mode='reflect', bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 将输入转换为灰度图
        x_gray = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        # 使用拉普拉斯算子提取高频信息
        x_laplacian = self.laplacian(x_gray)
        # 归一化处理
        x_laplacian = torch.abs(x_laplacian)  # 取绝对值
        x_laplacian = (x_laplacian - x_laplacian.min()) / (x_laplacian.max() - x_laplacian.min() + 1e-6)
        # 卷积和激活
        attn = self.conv(x_laplacian)
        #check_tensor(attn, "laplacian attn before Sigmoid")
        attn = self.sigmoid(attn)
        return attn

class MAFusion(nn.Module):
    """模态感知融合模块，分别处理红外和可见光特征，再进行融合。"""
    def __init__(self, dim, reduction_ir=8, reduction_vis=16):
        super(MAFusion, self).__init__()
        # 红外注意力模块
        self.channel_attn_ir = ChannelAttentionIR(dim, reduction_ir)
        self.spatial_attn_ir = SpatialAttentionIR()
        # 可见光注意力模块
        self.channel_attn_vis = ChannelAttentionVIS(dim, reduction_vis)
        self.spatial_attn_vis = SpatialAttentionVIS()
        # 像素注意力模块
        self.pa = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        # 最终融合卷积
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x_ir, x_vis):
        # 红外特征的通道注意力
        attn_ir_channel = self.channel_attn_ir(x_ir)           # [B, C, 1, 1]
        x_ir_channel = x_ir * attn_ir_channel                  # 通道加权
        # 红外特征的空间注意力
        attn_ir_spatial = self.spatial_attn_ir(x_ir_channel)   # [B, 1, H, W]
        x_ir_attn = x_ir_channel * attn_ir_spatial             # 空间加权

        # 可见光特征的通道注意力
        attn_vis_channel = self.channel_attn_vis(x_vis)        # [B, C, 1, 1]
        x_vis_channel = x_vis * attn_vis_channel               # 通道加权
        # 可见光特征的空间注意力
        attn_vis_spatial = self.spatial_attn_vis(x_vis_channel)  # [B, 1, H, W]
        x_vis_attn = x_vis_channel * attn_vis_spatial          # 空间加权

        # 像素注意力融合
        x_cat = torch.cat([x_ir_attn, x_vis_attn], dim=1)      # [B, 2C, H, W]
        attn_pixel = self.pa(x_cat)                            # [B, C, H, W]
        result = attn_pixel * x_ir_attn + (1 - attn_pixel) * x_vis_attn  # 融合

        result = self.conv(result)                             # 1x1 卷积融合
        return result


