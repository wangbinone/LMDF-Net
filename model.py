import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vm0 import UpConv  # 仅保留被 U_Net 调用的外部模块

# =====================================================================
# 1. 基础工具与小波变换
# =====================================================================
def haar_dwt2d(x):
    """2D 离散小波变换 (Haar). 分解为 LL, HL, LH, HH"""
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    LL = (x00 + x01 + x10 + x11) / 2.0
    HL = (x00 - x01 + x10 - x11) / 2.0
    LH = (x00 + x01 - x10 - x11) / 2.0
    HH = (x00 - x01 - x10 + x11) / 2.0
    return torch.cat([LL, HL, LH, HH], dim=1)

class ChannelPool(nn.Module):
    def forward(self, x):
        # 将 max_pooling 与 global_average_pooling 结果在通道维度拼接
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# =====================================================================
# 2. 基础注意力层 (Spatial, Channel, Pixel)
# =====================================================================
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=False, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.maxPool = nn.AdaptiveMaxPool2d(1)

        # 防止通道数较少时 reduction 导致中间通道数为 0
        mid_channel = max(4, channel // reduction)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, mid_channel, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avgPool(x))
        max_out = self.fc(self.maxPool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class PixelAttention(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 4, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feat // 4, n_feat, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(x)

# =====================================================================
# 3. 核心特征提纯块与融合模块
# =====================================================================
class RCSSC(nn.Module):
    """
    专门为遥感设计的特征校准模块。
    故事线：分离低频雾气分布(SC分支)与高频地物细节(SA/CA分支)，实现保边缘去雾。
    """
    def __init__(self, n_feat, reduction=16):
        super(RCSSC, self).__init__()
        pooling_r = 4
        self.head = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
        )
        # SC: 提取遥感大视角的低频上下文 (Global Haze Distribution Prior)
        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=True),
            nn.GroupNorm(8, n_feat)
        )

        self.SA = spatial_attn_layer()
        self.CA = CALayer(n_feat, reduction)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=1),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True)
        )
        self.ReLU = nn.LeakyReLU(inplace=True)
        self.tail = nn.Conv2d(n_feat, n_feat, 3, padding=1)

    def forward(self, x):
        res = x
        x = self.head(x)

        # 高频细节提纯
        x1 = self.conv1x1(torch.cat([self.SA(x), self.CA(x)], dim=1))

        # 低频雾气门控权重生成
        sc_feat = F.interpolate(self.SC(x), size=x.size()[-2:], mode='bilinear', align_corners=True)
        gate = torch.sigmoid(x + sc_feat)

        out = self.tail(x1 * gate)
        return self.ReLU(out + res)

class W_RCP_RCSSC(nn.Module):
    """
    小波驱动的三重注意力特征校准模块。
    优势：比原版 RCSSC 更强的边缘保护能力，更稳定的全局雾气估计。
    """
    def __init__(self, n_feat, reduction=16):
        super().__init__()
        self.n_feat = n_feat

        self.head = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # SC 分支升级：Wavelet-LL 引导
        self.SC = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=True),
            nn.GroupNorm(8, n_feat),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.SA = spatial_attn_layer()
        self.CA = CALayer(n_feat, reduction)
        self.PA = PixelAttention(n_feat)

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(n_feat * 3, n_feat, kernel_size=1),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=True)
        )

        # 高频边缘锐化器：利用小波高频子带修正门控边缘
        self.edge_refiner = nn.Sequential(
            nn.Conv2d(n_feat * 3, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.ReLU = nn.LeakyReLU(0.1, inplace=True)
        self.tail = nn.Conv2d(n_feat, n_feat, 3, padding=1)

    def forward(self, x):
        res = x
        x = self.head(x)
        B, C, H, W = x.shape

        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        wt = haar_dwt2d(x_pad)
        LL, HL, LH, HH = wt.chunk(4, dim=1)

        sc_feat = self.SC(LL)
        sc_feat = F.interpolate(sc_feat, size=(H, W), mode='bilinear', align_corners=True)

        x_sa = self.SA(x)
        x_ca = self.CA(x)
        x_pa = self.PA(x)
        x_refined = self.conv_fusion(torch.cat([x_sa, x_ca, x_pa], dim=1))

        edge_energy = torch.cat([HL, LH, HH], dim=1)
        edge_mask = self.edge_refiner(edge_energy)
        edge_mask = F.interpolate(edge_mask, size=(H, W), mode='bilinear', align_corners=True)

        gate = torch.sigmoid(x + sc_feat)
        gate = gate * (1 + edge_mask)

        out = self.tail(x_refined * gate)
        return self.ReLU(out + res)

class SFM(nn.Module):
    def __init__(self, C, mode='fft', mid_ratio=0.25, init_cutoff=0.5, kernel_size=3):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(C * 2, C, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=C),
            nn.GELU()
        )
        self.refiner = RCSSC(n_feat=C, reduction=16)

    def forward(self, dec, skip):
        if skip.size()[-2:] != dec.size()[-2:]:
            skip = F.interpolate(skip, size=dec.shape[-2:], mode='bilinear', align_corners=True)
        out = self.fuse(torch.cat([dec, skip], dim=1))
        return self.refiner(out)

class W_RCP_SFM(nn.Module):
    def __init__(self, C, mode='fft', mid_ratio=0.25, init_cutoff=0.5, kernel_size=3):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(C * 2, C, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=C),
            nn.GELU()
        )
        self.refiner = W_RCP_RCSSC(n_feat=C, reduction=16)

    def forward(self, dec, skip):
        if skip.size()[-2:] != dec.size()[-2:]:
            skip = F.interpolate(skip, size=dec.shape[-2:], mode='bilinear', align_corners=True)
        out = self.fuse(torch.cat([dec, skip], dim=1))
        return self.refiner(out)

# =====================================================================
# 4. CBSA 模块
# =====================================================================
class CBSA_2D(nn.Module):
    """
    优化版 CBSA：引入 LayerScale/ReZero 思想，确保加入注意力机制后训练初期绝对稳定，保底不掉点。
    """
    def __init__(self, dim, heads=4, dim_head=16, pool_size=(8, 8)):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.proj = nn.Conv2d(dim, inner_dim, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, kernel_size=1)

        self.step_x = nn.Parameter(torch.ones(heads, 1, 1) * 1e-4)
        self.step_rep = nn.Parameter(torch.ones(heads, 1, 1) * 1e-4)

        self.pool = nn.AdaptiveAvgPool2d(output_size=pool_size)

        self.local_dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def attention(self, query, key, value):
        dots = (query @ key.transpose(-1, -2)) * self.scale
        attn = torch.softmax(dots, dim=-1)
        return attn @ value, attn

    def forward(self, x):
        B, C, H, W = x.shape

        x = x + self.local_dwconv(x)
        w = self.proj(x)
        rep = self.pool(w)

        w = rearrange(w, 'b (h d) H W -> b h (H W) d', h=self.heads, d=self.dim_head)
        rep = rearrange(rep, 'b (h d) pH pW -> b h (pH pW) d', h=self.heads, d=self.dim_head)

        rep_delta, attn = self.attention(rep, w, w)
        rep = rep + self.step_rep * rep_delta

        x_delta, _ = self.attention(rep, rep, rep)
        x_delta = attn.transpose(-1, -2) @ x_delta

        x_delta = self.step_x * x_delta
        x_delta = rearrange(x_delta, 'b h (H W) d -> b (h d) H W', H=H, W=W)

        return self.to_out(x_delta)

class CBSA_Block(nn.Module):
    def __init__(self, ch_in, ch_out, dim, heads=4, dim_head=16):
        super().__init__()
        self.align = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()
        self.norm1 = nn.GroupNorm(8, ch_out)
        self.attn = CBSA_2D(dim=ch_out, heads=heads, dim_head=dim_head)

        self.norm2 = nn.GroupNorm(8, ch_out)
        self.ffn = nn.Sequential(
            nn.Conv2d(ch_out, ch_out * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=1)
        )

    def forward(self, x):
        x = self.align(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# =====================================================================
# 5. 主网络 U_Net
# =====================================================================
class U_Net(nn.Module):
    def __init__(self, dim=32):
        super(U_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = CBSA_Block(3, dim, dim=dim)
        self.Conv2 = CBSA_Block(dim, dim * 2, dim=dim * 2)
        self.Conv3 = CBSA_Block(dim * 2, dim * 4, dim=dim * 4)
        self.Conv4 = CBSA_Block(dim * 4, dim * 8, dim=dim * 8)
        self.Conv5 = CBSA_Block(dim * 8, dim * 16, dim=dim * 16)

        self.Up5 = UpConv(dim * 16, dim * 8)
        self.Up_conv5 = CBSA_Block(dim * 8, dim * 8, dim=dim * 8)
        self.Up4 = UpConv(dim * 8, dim * 4)
        self.Up_conv4 = CBSA_Block(dim * 4, dim * 4, dim=dim * 4)
        self.Up3 = UpConv(dim * 4, dim * 2)
        self.Up_conv3 = CBSA_Block(dim * 2, dim * 2, dim=dim * 2)
        self.Up2 = UpConv(dim * 2, dim)
        self.Up_conv2 = CBSA_Block(dim, dim, dim=dim)

        self.adjust_channel4 = nn.Conv2d(dim * 8, dim * 8, 1)
        self.adjust_channel3 = nn.Conv2d(dim * 4, dim * 4, 1)
        self.adjust_channel2 = nn.Conv2d(dim * 2, dim * 2, 1)
        self.adjust_channel1 = nn.Conv2d(dim, dim, 1)

        self.skip5 = W_RCP_SFM(dim * 8)
        self.skip4 = W_RCP_SFM(dim * 4)
        self.skip3 = W_RCP_SFM(dim * 2)
        self.skip2 = SFM(dim)

        self.Final_conv = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoding
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        # Decoding
        d5 = self.Up_conv5(self.skip5(self.Up5(x5), x4) + self.adjust_channel4(x4))
        d4 = self.Up_conv4(self.skip4(self.Up4(d5), x3) + self.adjust_channel3(x3))
        d3 = self.Up_conv3(self.skip3(self.Up3(d4), x2) + self.adjust_channel2(x2))
        d2 = self.Up_conv2(self.skip2(self.Up2(d3), x1) + self.adjust_channel1(x1))

        output = self.Final_conv(d2) + x

        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = U_Net(dim=16).to(device)
    input_tensor = torch.randn(1, 3, 720, 480).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print("Output size:", output.size())