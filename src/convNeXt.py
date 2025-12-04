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

class DualPathway3DConvNeXtA(nn.Module):
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
        self.fusion_conv = nn.Conv3d(2 * final_channels, final_channels, kernel_size=1)
        self.fusion_bn = nn.BatchNorm3d(final_channels)
        self.reduce_conv1 = nn.Conv3d(final_channels, final_channels // 2, kernel_size=1)
        self.reduce_bn1 = nn.BatchNorm3d(final_channels // 2)
        self.reduce_conv2 = nn.Conv3d(final_channels, final_channels // 2, kernel_size=1)
        self.reduce_bn2 = nn.BatchNorm3d(final_channels // 2)
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
        sum_features = x1 + x2
        sum_avg = F.relu(self.reduce_bn1(self.reduce_conv1(sum_features)), inplace=True) / 2
        mult_features = x1 * x2
        mult = F.relu(self.reduce_bn2(self.reduce_conv2(mult_features)), inplace=True)
        concat = torch.cat([x1, x2], dim=1)
        concat = F.relu(self.fusion_bn(self.fusion_conv(concat)), inplace=True)
        final = (sum_avg + mult + concat[:, :sum_avg.size(1), ...]) / 3
        final = self.final_bn(final)
        out = self.avgpool(final)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def ConvNeXtA3D_Tiny_Fusion(num_classes=3):
    return DualPathway3DConvNeXtA(depths=(3,3,9,3), dims=(96,192,384,768), num_classes=num_classes)

def ConvNeXtA3D_Small_Fusion(num_classes=3):
    return DualPathway3DConvNeXtA(depths=(3,3,27,3), dims=(96,192,384,768), num_classes=num_classes)
