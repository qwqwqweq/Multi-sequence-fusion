import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SplitAttentionConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, radix=2, reduction_factor=4):
        super().__init__()
        self.radix = radix
        self.out_channels = out_channels
        self.split_channels = out_channels * radix
        self.conv = nn.Conv3d(in_channels, self.split_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(self.split_channels)
        inter_channels = max(out_channels // reduction_factor, 32)
        self.fc1 = nn.Conv3d(self.split_channels, inter_channels, 1, groups=radix, bias=False)
        self.bn1 = nn.BatchNorm3d(inter_channels)
        self.fc2 = nn.Conv3d(inter_channels, self.split_channels, 1, groups=radix, bias=False)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = F.relu(x, inplace=True)
        splits = torch.chunk(x, self.radix, dim=1)
        gap = sum(splits)
        att = self.fc2(F.relu(self.bn1(self.fc1(gap)), inplace=True))
        att = att.view(att.size(0), self.radix, self.out_channels, *att.shape[2:])
        att = F.softmax(att, dim=1)
        att = [att[:, i] for i in range(self.radix)]
        out = sum([s * a for s, a in zip(splits, att)])
        return out

class ResNeStBottleneck3D(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, radix=2, reduction_factor=4):
        super().__init__()
        group_width = planes
        self.conv1 = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(group_width)
        self.conv2 = SplitAttentionConv3d(group_width, group_width, kernel_size=3, stride=stride, padding=1, radix=radix, reduction_factor=reduction_factor)
        self.bn2 = nn.BatchNorm3d(group_width)
        self.conv3 = nn.Conv3d(group_width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class DualPathway3DResNeStAttentionFusion(nn.Module):
    def __init__(self, layers=(2,2,2,2), radix=2, num_classes=3):
        super().__init__()
        self.inplanes = 64
        self.conv1_t1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_t1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1_t1 = self._make_layer(64, layers[0], stride=1, radix=radix)
        self.layer2_t1 = self._make_layer(128, layers[1], stride=2, radix=radix)
        self.layer3_t1 = self._make_layer(256, layers[2], stride=2, radix=radix)
        self.layer4_t1 = self._make_layer(512, layers[3], stride=2, radix=radix)

        self.inplanes = 64
        self.conv1_t2 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_t2 = nn.BatchNorm3d(64)
        self.layer1_t2 = self._make_layer(64, layers[0], stride=1, radix=radix)
        self.layer2_t2 = self._make_layer(128, layers[1], stride=2, radix=radix)
        self.layer3_t2 = self._make_layer(256, layers[2], stride=2, radix=radix)
        self.layer4_t2 = self._make_layer(512, layers[3], stride=2, radix=radix)

        final_channels = 512 * ResNeStBottleneck3D.expansion
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
        self._initialize()

    def _make_layer(self, planes, blocks, stride, radix):
        layers = []
        layers.append(ResNeStBottleneck3D(self.inplanes, planes, stride=stride, radix=radix))
        self.inplanes = planes * ResNeStBottleneck3D.expansion
        for _ in range(1, blocks):
            layers.append(ResNeStBottleneck3D(self.inplanes, planes, stride=1, radix=radix))
        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_path(self, x, t):
        if t == 1:
            x = self.relu(self.bn1_t1(self.conv1_t1(x)))
            x = self.pool(x)
            x = self.layer1_t1(x)
            x = self.layer2_t1(x)
            x = self.layer3_t1(x)
            x = self.layer4_t1(x)
        else:
            x = self.relu(self.bn1_t2(self.conv1_t2(x)))
            x = self.pool(x)
            x = self.layer1_t2(x)
            x = self.layer2_t2(x)
            x = self.layer3_t2(x)
            x = self.layer4_t2(x)
        return x

    def forward(self, t1, t2):
        x1 = self.forward_path(t1, 1)
        x2 = self.forward_path(t2, 2)
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

def ResNeSt3D18AttentionFusion(num_classes=3, radix=2):
    return DualPathway3DResNeStAttentionFusion(layers=(2,2,2,2), radix=radix, num_classes=num_classes)

def ResNeSt3D50AttentionFusion(num_classes=3, radix=2):
    return DualPathway3DResNeStAttentionFusion(layers=(3,4,6,3), radix=radix, num_classes=num_classes)
# if __name__ == '__main__':
#     pass
