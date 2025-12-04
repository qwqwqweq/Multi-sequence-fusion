import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):
        n, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

class ConvNeXtABlock3D(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.ln = LayerNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath3D(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.ln(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x * self.gamma.view(1, -1, 1, 1, 1)
        x = shortcut + self.drop_path(x)
        return x

class DropPath3D(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ConvNeXtAStage3D(nn.Module):
    def __init__(self, in_ch, out_ch, depth, drop_path=0.0):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2)
        ) if in_ch != out_ch else nn.Identity()
        self.blocks = nn.Sequential(*[
            ConvNeXtABlock3D(out_ch, drop_path=drop_path) for _ in range(depth)
        ])
    def forward(self, x):
        if not isinstance(self.downsample, nn.Identity):
            n, c, d, h, w = x.shape
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = self.downsample[0](x)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = self.downsample[1](x)
        x = self.blocks(x)
        return x

class FeatureAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.shared = nn.Sequential(
            nn.Conv3d(in_channels, max(in_channels // 16, 32), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(in_channels // 16, 32), in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.shared(self.avg_pool(x)) + self.shared(self.max_pool(x)))

class CrossModalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x, y):
        b, c, d, h, w = x.size()
        q = self.query_conv(x).view(b, -1, d*h*w).permute(0, 2, 1)
        k = self.key_conv(y).view(b, -1, d*h*w)
        att = torch.bmm(q, k)
        att = F.softmax(att, dim=-1)
        v = self.value_conv(y).view(b, -1, d*h*w)
        out = torch.bmm(v, att.permute(0, 2, 1)).view(b, c, d, h, w)
        return self.gamma * out + x

class DualPathway3DConvNeXtAAttentionFusion(nn.Module):
    def __init__(self, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), num_classes=3, drop_path_rate=0.0):
        super().__init__()
        self.stem_t1 = nn.Sequential(
            nn.Conv3d(1, dims[0], kernel_size=4, stride=4),
            LayerNorm3d(dims[0])
        )
        self.stem_t2 = nn.Sequential(
            nn.Conv3d(1, dims[0], kernel_size=4, stride=4),
            LayerNorm3d(dims[0])
        )
        dp_rates = [drop_path_rate for _ in range(sum(depths))]
        cur = 0
        self.stages_t1 = nn.ModuleList()
        self.stages_t2 = nn.ModuleList()
        in_ch = dims[0]
        for i in range(4):
            out_ch = dims[i]
            depth = depths[i]
            self.stages_t1.append(ConvNeXtAStage3D(in_ch, out_ch, depth, drop_path=dp_rates[cur]))
            self.stages_t2.append(ConvNeXtAStage3D(in_ch, out_ch, depth, drop_path=dp_rates[cur]))
            in_ch = out_ch
            cur += depth
        final_channels = dims[-1]
        self.feature_attention_t1 = FeatureAttention(final_channels)
        self.feature_attention_t2 = FeatureAttention(final_channels)
        self.cross_attention_t1 = CrossModalAttention(final_channels)
        self.cross_attention_t2 = CrossModalAttention(final_channels)
        self.fusion_conv = nn.Conv3d(2 * final_channels, final_channels, kernel_size=1)
        self.fusion_bn = nn.BatchNorm3d(final_channels)
        self.reduce_conv1 = nn.Conv3d(final_channels, final_channels // 2, kernel_size=1)
        self.reduce_bn1 = nn.BatchNorm3d(final_channels // 2)
        self.reduce_conv2 = nn.Conv3d(final_channels, final_channels // 2, kernel_size=1)
        self.reduce_bn2 = nn.BatchNorm3d(final_channels // 2)
        self.fusion_attention = FeatureAttention(final_channels)
        self.reduce_attention1 = FeatureAttention(final_channels // 2)
        self.reduce_attention2 = FeatureAttention(final_channels // 2)
        self.final_bn = nn.BatchNorm3d(final_channels // 2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Sequential(
            nn.Linear(final_channels // 2, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward_single(self, x, pathway):
        x = self.stem_t1(x) if pathway == "t1" else self.stem_t2(x)
        stages = self.stages_t1 if pathway == "t1" else self.stages_t2
        for s in stages:
            x = s(x)
        return x

    def forward(self, t1, t2):
        x1 = self.forward_single(t1, "t1")
        x2 = self.forward_single(t2, "t2")
        x1 = self.feature_attention_t1(x1)
        x2 = self.feature_attention_t2(x2)
        a1 = self.cross_attention_t1(x1, x2)
        a2 = self.cross_attention_t2(x2, x1)
        sum_features = a1 + a2
        sum_avg = F.silu(self.reduce_bn1(self.reduce_conv1(sum_features))) / 2
        sum_avg = self.reduce_attention1(sum_avg)
        mult_features = a1 * a2
        mult = F.silu(self.reduce_bn2(self.reduce_conv2(mult_features)))
        mult = self.reduce_attention2(mult)
        concat = torch.cat([a1, a2], dim=1)
        concat = F.silu(self.fusion_bn(self.fusion_conv(concat)))
        concat = self.fusion_attention(concat)
        final = (sum_avg + mult + concat[:, :sum_avg.size(1), ...]) / 3
        final = self.final_bn(final)
        out = self.avgpool(final)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def ConvNeXtA3D_Tiny_AttentionFusion(num_classes=3):
    return DualPathway3DConvNeXtAAttentionFusion(depths=(3,3,9,3), dims=(96,192,384,768), num_classes=num_classes)

def ConvNeXtA3D_Small_AttentionFusion(num_classes=3):
    return DualPathway3DConvNeXtAAttentionFusion(depths=(3,3,27,3), dims=(96,192,384,768), num_classes=num_classes)
