import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    def __init__(self, in_channels):
        super(FeatureAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 16, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class CrossModalAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossModalAttention, self).__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        batch_size, C, D, H, W = x.size()
        
        query = self.query_conv(x).view(batch_size, -1, D*H*W).permute(0, 2, 1)
        key = self.key_conv(y).view(batch_size, -1, D*H*W)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        
        value = self.value_conv(y).view(batch_size, -1, D*H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)
        
        return self.gamma * out + x

class MBConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=6):
        super(MBConvBlock3D, self).__init__()
        
        self.stride = stride
        hidden_dim = in_channels * expansion_factor
        
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expansion_factor != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size, stride=stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        layers.append(SELayer3D(hidden_dim))

        layers.extend([
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class DualPathway3DEfficientNet(nn.Module):
    def __init__(self, num_classes=3):
        super(DualPathway3DEfficientNet, self).__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True)
        )

        self.stages = nn.ModuleList([
            self._make_stage(32, 16, 1),
            self._make_stage(16, 24, 2),
            self._make_stage(24, 40, 2),
            self._make_stage(40, 80, 2),
            self._make_stage(80, 112, 1),
            self._make_stage(112, 192, 2),
            self._make_stage(192, 320, 1),
        ])

        self.init_conv_t2 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True)
        )
        
        self.stages_t2 = nn.ModuleList([
            self._make_stage(32, 16, 1),
            self._make_stage(16, 24, 2),
            self._make_stage(24, 40, 2),
            self._make_stage(40, 80, 2),
            self._make_stage(80, 112, 1),
            self._make_stage(112, 192, 2),
            self._make_stage(192, 320, 1),
        ])

        final_channels = 320

        self.feature_attention = FeatureAttention(final_channels)
        self.cross_modal_attention = CrossModalAttention(final_channels)
        
        self.fusion_conv = nn.Conv3d(2 * final_channels, final_channels, kernel_size=1, bias=True)
        self.fusion_bn = nn.BatchNorm3d(final_channels)
        self.fusion_attention = FeatureAttention(final_channels)
        
        self.reduce_conv1 = nn.Conv3d(final_channels, final_channels // 2, kernel_size=1, bias=True)
        self.reduce_bn1 = nn.BatchNorm3d(final_channels // 2)
        self.reduce_attention1 = FeatureAttention(final_channels // 2)
        
        self.reduce_conv2 = nn.Conv3d(final_channels, final_channels // 2, kernel_size=1, bias=True)
        self.reduce_bn2 = nn.BatchNorm3d(final_channels // 2)
        self.reduce_attention2 = FeatureAttention(final_channels // 2)
        
        self.final_bn = nn.BatchNorm3d(final_channels // 2)
        
        self.dropout = nn.Dropout(p=0.4)
        self.spatial_dropout = nn.Dropout3d(p=0.2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(final_channels // 2, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes, bias=True)
        )

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, stride):
        layers = []
        layers.append(MBConvBlock3D(in_channels, out_channels, stride=stride))
        layers.append(MBConvBlock3D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_single_pathway(self, x, pathway="t1"):
        if pathway == "t1":
            x = self.init_conv(x)
            for stage in self.stages:
                x = stage(x)
        else:
            x = self.init_conv_t2(x)
            for stage in self.stages_t2:
                x = stage(x)
        return x

    def forward(self, t1, t2):
        if t1.size() != t2.size():
            raise ValueError("Input size mismatch between T1 and T2 images")

        t1_features = self.forward_single_pathway(t1, "t1")
        t2_features = self.forward_single_pathway(t2, "t2")
        
        t1_features = self.feature_attention(t1_features)
        t2_features = self.feature_attention(t2_features)
        
        t1_enhanced = self.cross_modal_attention(t1_features, t2_features)
        t2_enhanced = self.cross_modal_attention(t2_features, t1_features)
        
        sum_features = t1_enhanced + t2_enhanced
        sum_features = self.spatial_dropout(sum_features)
        sum_avg = F.silu(self.reduce_bn1(self.reduce_conv1(sum_features))) / 2
        sum_avg = self.reduce_attention1(sum_avg)
        
        mult_features = t1_enhanced * t2_enhanced
        mult_features = self.spatial_dropout(mult_features)
        mult = F.silu(self.reduce_bn2(self.reduce_conv2(mult_features)))
        mult = self.reduce_attention2(mult)
        
        concat = torch.cat([t1_enhanced, t2_enhanced], dim=1)
        concat = F.silu(self.fusion_bn(self.fusion_conv(concat)))
        concat = self.fusion_attention(concat)
        concat = self.spatial_dropout(concat)
        
        final_features = (sum_avg + mult + concat[:, :sum_avg.size(1), ...]) / 3
        final_features = self.final_bn(final_features)
        
        out = self.avgpool(final_features)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

def EfficientNet3D(num_classes=3):
    return DualPathway3DEfficientNet(num_classes)

# Example usage
if __name__ == "__main__":
    # Create model instance
    model = EfficientNet3D(num_classes=3)
    
    # Create sample input tensors (batch_size, channels, depth, height, width)
    batch_size = 2
    channels = 1
    size = 64
    t1 = torch.randn(batch_size, channels, size, size, size)
    t2 = torch.randn(batch_size, channels, size, size, size)
    
    # Forward pass
    output = model(t1, t2)
    
    # Print model summary
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
