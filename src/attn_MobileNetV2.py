import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_act_3d(in_ch, out_ch, k=3, s=1, p=1, groups=1):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU6(inplace=True),
    )

class InvertedResidual3D(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn_act_3d(inp, hidden_dim, k=1, s=1, p=0))
        layers.append(conv_bn_act_3d(hidden_dim, hidden_dim, k=3, s=stride, p=1, groups=hidden_dim))
        layers.append(nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm3d(oup))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_3D_Backbone(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        input_channel = int(32 * width_mult)
        self.stem = conv_bn_act_3d(1, input_channel, k=3, s=2, p=1)
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        layers = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual3D(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.final_channels = input_channel

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
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

class DualPathway3DMobileNetV2AttentionFusion(nn.Module):
    def __init__(self, num_classes=3, width_mult=1.0):
        super().__init__()
        self.backbone_t1 = MobileNetV2_3D_Backbone(width_mult)
        self.backbone_t2 = MobileNetV2_3D_Backbone(width_mult)
        final_channels = self.backbone_t1.final_channels
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, t1, t2):
        x1 = self.backbone_t1(t1)
        x2 = self.backbone_t2(t2)
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

def MobileNetV2_3D_AttentionFusion(num_classes=3, width_mult=1.0):
    return DualPathway3DMobileNetV2AttentionFusion(num_classes=num_classes, width_mult=width_mult)
