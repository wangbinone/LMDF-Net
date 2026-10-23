import math
import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_

class LayerNorm2D(nn.Module):
    """LayerNorm for channels of 2D tensor(B C H W)"""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None, bn_weight_init=0)

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x


class Stem(nn.Module):
    def __init__(self, in_dim=3, dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, dim // 8, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 2, dim, kernel_size=3, stride=2, padding=1, act_layer=None))

    def forward(self, x):
        x = self.conv(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=4.0):
        super().__init__()
        hidden_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, hidden_dim, kernel_size=1),
            ConvLayer2D(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim),
            SqueezeExcite(hidden_dim, .25),
            ConvLayer2D(hidden_dim, out_dim, kernel_size=1, act_layer=None)
        )

        self.dwconv1 = ConvLayer2D(in_dim, in_dim, 3, padding=1, groups=in_dim, act_layer=None)
        self.dwconv2 = ConvLayer2D(out_dim, out_dim, 3, padding=1, groups=out_dim, act_layer=None)

    def forward(self, x):
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x


class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3 * state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0)
        self.hz_proj = ConvLayer1D(d_model, 2 * self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x, H, W):
        batch, _, L = x.shape
        assert L == H * W, f"Input feature map length {L} does not match H*W ({H}*{W})"
        # H = int(math.sqrt(L))
        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, H, W)).flatten(2)
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        A = (dt + self.A.view(1, -1, 1)).softmax(-1)

        AB = (A * B)
        h = x @ AB.transpose(-2, -1)

        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ C  # B C N, B C L -> B C L

        y = y.view(batch, -1, H, W).contiguous()  # + x * self.D  # B C H W
        return y, h


class EfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(dim)

        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)

        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))

        # LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        # DWconv1
        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)

        # HSM-SSD
        x_prev = x

        B, C, H, W = x.shape
        x_flat = self.norm(x.flatten(2))
        x, h = self.mixer(x_flat, H, W)

        x = (1 - alpha[1]) * x_prev + alpha[1] * x

        # DWConv2
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)

        # FFN
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)
        return x


class MSM(nn.Module):
    """多分支特征提取模块"""

    def __init__(self, in_channels, filters):
        super(MSM, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=5, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=7, stride=1, padding=2, dilation=1),
            nn.LeakyReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        o1 = self.branch1(x)
        o2 = self.branch2(x)
        o3 = self.branch3(x)
        o4 = self.branch4(x)
        return torch.cat([o1, o2, o3, o4], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class AttentionModule(nn.Module):
    """组合注意力模块，结合多分支特征提取和通道注意力"""

    def __init__(self, in_channels, filters, reduction_ratio=4, dim=64, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super(AttentionModule, self).__init__()
        self.inc = MSM(in_channels=in_channels, filters=filters)
        self.adjust_channel = nn.Conv2d(filters * 4, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # 用 EfficientViMBlock 替换 CAM
        self.eff_vim_block = EfficientViMBlock(dim=dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim)

    def forward(self, x):
        inc_out = self.inc(x)
        adjusted_out = self.adjust_channel(inc_out)

        # 用 EfficientViMBlock 处理 adjusted_out
        out = self.eff_vim_block(adjusted_out)

        return out
def window_partition_cf(x, window_size):
    """
    Channel-First (B, C, H, W) 窗口分割
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    # -> (B*num_windows, C, window_size, window_size)
    return windows

def window_reverse_cf(windows, window_size, B, H, W):
    """
    Channel-First (B*num_windows, C, H_win, W_win) 窗口合并
    H, W 必须是 *padding 后* 的 H, W
    """
    C = windows.shape[1]
    H_groups = H // window_size
    W_groups = W // window_size
    # B can be inferred: B = windows.shape[0] // (H_groups * W_groups)
    x = windows.view(B, H_groups, W_groups, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    # -> (B, C, H, W)
    return x


class WindowedEfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64,
                 window_size=7, shift_size=0):  # <-- 新增参数
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.shift_size = shift_size

        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(dim)  # 假设 LayerNorm1D 可以处理 (B, C, L)

        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)

        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))

        # LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        # 1. DWconv1 (局部操作, 无需窗口化)
        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)

        # 2. Windowed HSM-SSD
        x_prev = x

        # --- (开始窗口化) ---

        # 2.1 Pad: 保证 H, W 是 window_size 的整数倍
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x_pad = F.pad(x, (0, pad_r, 0, pad_b), 'reflect')
        B, C, H_pad, W_pad = x_pad.shape

        # 2.2 Shift: 如果 shift_size > 0，执行循环位移
        if self.shift_size > 0:
            x_pad_shifted = torch.roll(x_pad, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            x_pad_shifted = x_pad

        # 2.3 Partition: (B, C, H_pad, W_pad) -> (B_win, C, H_win, W_win)
        x_windows = window_partition_cf(x_pad_shifted, self.window_size)

        # 2.4 Apply Mixer (在窗口内执行)
        B_win, C_win, H_win, W_win = x_windows.shape
        x_windows_flat = self.norm(x_windows.flatten(2))  # (B_win, C, L_win)

        # 关键: 确保 HSMSSD 接收 H_win, W_win (上一问的修复)
        x_mixed, _ = self.mixer(x_windows_flat, H_win, W_win)
        # HSMSSD 的 forward 应该返回 (B_win, C, H_win, W_win)

        # 2.5 Reverse Partition: (B_win, C, H_win, W_win) -> (B, C, H_pad, W_pad)
        x_reversed = window_reverse_cf(x_mixed, self.window_size, B, H_pad, W_pad)

        # 2.6 Reverse Shift:
        if self.shift_size > 0:
            x_reversed_unshifted = torch.roll(x_reversed, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x_reversed_unshifted = x_reversed

        # 2.7 Unpad: 裁剪回原始 H, W
        x_unpadded = x_reversed_unshifted[:, :, :H, :W]

        # --- (结束窗口化) ---

        x = (1 - alpha[1]) * x_prev + alpha[1] * x_unpadded  # 残差连接

        # 3. DWConv2 (局部操作, 无需窗口化)
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)

        # 4. FFN (局部操作, 无需窗口化)
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)

        return x


class NNBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super(NNBlock, self).__init__()

        # 卷积层 + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.residual_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

        # 用 AttentionModule 替换 EfficientViMBlock
        self.attention_module = AttentionModule(in_channels=ch_out, filters=ch_out // 4, reduction_ratio=4,
                                                dim=dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand,
                                                state_dim=state_dim)

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # 替换为 AttentionModule
        out = self.attention_module(out)

        # 残差连接
        return out + residual


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.conv2(x)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Enhanced_EAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=16):
        super().__init__()
        valid_groups = min(num_groups, F_int // 4)

        self.g_conv = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, groups=valid_groups),
            nn.BatchNorm2d(F_int),
            nn.GELU()  # 更平滑的激活
        )

        self.x_conv = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, groups=valid_groups),
            nn.BatchNorm2d(F_int),
            nn.GELU()
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, F_l, 1),  # 多通道注意力
            nn.BatchNorm2d(F_l),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g_proj = self.g_conv(g)
        x_proj = self.x_conv(x)
        attn = self.psi(g_proj + x_proj)
        return x * attn + x


class SPI_Attention(nn.Module):
    def __init__(self, in_dim, kernel_size=3, is_bottom=False):
        super().__init__()
        self.is_bottom = is_bottom
        if not is_bottom:
            self.eag = Enhanced_EAG(in_dim, in_dim, in_dim // 2)
            self.fusion = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, padding=1),
                CP_Attention_block(default_conv, in_dim, kernel_size)
            )
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, padding=1),
                CP_Attention_block(default_conv, in_dim, kernel_size)
            )

    def forward(self, x, skip=None):
        if not self.is_bottom:
            if skip.size()[-2:] != x.size()[-2:]:
                skip = F.interpolate(skip, x.shape[2:], mode='bilinear', align_corners=True)
            x = self.eag(x, skip)
        return self.fusion(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_value = self.avg_pool(x).view(x.size(0), -1)
        max_value = self.max_pool(x).view(x.size(0), -1)
        channel_attention = self.fc(avg_value + max_value).view(x.size(0), x.size(1), 1, 1)
        return channel_attention


class PixelAttention(nn.Module):
    def __init__(self):
        super(PixelAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_pool, max_pool], dim=1)

        pixel_attention = self.conv1(combined)
        pixel_attention = self.sigmoid(pixel_attention)

        return pixel_attention


class ChannelPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelPixelAttention, self).__init__()

        self.channel_attention = ChannelAttention(in_channels)
        self.pixel_attention = PixelAttention()

    def forward(self, x):
        channel_att = self.channel_attention(x)
        pixel_att = self.pixel_attention(x)
        return channel_att * pixel_att * x


class PDU(nn.Module):
    def __init__(self, channel):
        super(PDU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ka = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid())
        self.td = nn.Sequential(
            default_conv(channel, channel, 3),
            default_conv(channel, channel // 8, 3),
            nn.ReLU(inplace=True),
            default_conv(channel // 8, channel, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.avg_pool(x)
        a = self.ka(a)
        t = self.td(x)
        j = torch.mul((1 - t), a) + torch.mul(t, x)
        return j

class PBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super(PBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.pdu = PDU(ch_out)
        self.CPA = ChannelPixelAttention(ch_out)
        self.residual_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.CPA(out)
        out = self.pdu(out)

        return out + residual


class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

            attn_windows = self.attn(qkv)

            # merge windows
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)  # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out


class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x
        if self.use_attn: x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn: x = x * rescale + rebias
        x = identity + x

        identity = x
        if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm: x = x * rescale + rebias
        x = identity + x
        return x
if __name__ == '__main__':
    block = EfficientViMBlock(dim=96).to('cuda')

    input_tensor = torch.rand(4, 96, 32, 32).to('cuda')

    output = block(input_tensor)

    print("Input size:", input_tensor.size())
    print("Output size:", output.size())
