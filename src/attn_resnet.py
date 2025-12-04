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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion *
                              planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class DualPathway3DResNet(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super(DualPathway3DResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.in_planes = 64 
        self.conv1_t2 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_t2 = nn.BatchNorm3d(64)
        self.layer1_t2 = self._make_layer(block, 64, layers[0])
        self.layer2_t2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_t2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_t2 = self._make_layer(block, 512, layers[3], stride=2)

        final_channels = 512 * block.expansion
    
        self.feature_attention_t1 = FeatureAttention(final_channels)
        self.feature_attention_t2 = FeatureAttention(final_channels)
        self.cross_attention_t1 = CrossModalAttention(final_channels)
        self.cross_attention_t2 = CrossModalAttention(final_channels)
        
        self.reduce_attention1 = FeatureAttention(final_channels // 2)  # For sum_avg
        self.reduce_attention2 = FeatureAttention(final_channels // 2)  # For mult
        self.fusion_attention = FeatureAttention(final_channels)        # For concat
        
        self.spatial_dropout = nn.Dropout3d(p=0.1)
        self.final_bn = nn.BatchNorm3d(final_channels // 2)
        
        self.fusion_conv = nn.Conv3d(2 * final_channels, final_channels, kernel_size=1)
        self.fusion_bn = nn.BatchNorm3d(final_channels)
        
        self.reduce_conv1 = nn.Conv3d(final_channels, final_channels // 2, kernel_size=1)
        self.reduce_bn1 = nn.BatchNorm3d(final_channels // 2)
        self.reduce_conv2 = nn.Conv3d(final_channels, final_channels // 2, kernel_size=1)
        self.reduce_bn2 = nn.BatchNorm3d(final_channels // 2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(final_channels // 2, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_single_pathway(self, x, pathway="t1"):
        if pathway == "t1":
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:  
            x = self.relu(self.bn1_t2(self.conv1_t2(x)))
            x = self.maxpool(x)
            x = self.layer1_t2(x)
            x = self.layer2_t2(x)
            x = self.layer3_t2(x)
            x = self.layer4_t2(x)
        return x

    def forward(self, t1, t2):
        if t1.size() != t2.size():
            raise ValueError("Input size mismatch between T1 and T2 images")

        # Get features from both pathways
        t1_features = self.forward_single_pathway(t1, "t1")
        t2_features = self.forward_single_pathway(t2, "t2")

        # Apply Feature Attention
        t1_features = self.feature_attention_t1(t1_features)
        t2_features = self.feature_attention_t2(t2_features)

        # Apply Cross-Modal Attention
        t1_attended = self.cross_attention_t1(t1_features, t2_features)
        t2_attended = self.cross_attention_t2(t2_features, t1_features)

        # 1. Sum fusion with attention
        sum_features = t1_attended + t2_attended
        sum_features = self.spatial_dropout(sum_features)
        sum_avg = F.silu(self.reduce_bn1(self.reduce_conv1(sum_features))) / 2
        sum_avg = self.reduce_attention1(sum_avg)

        mult_features = t1_attended * t2_attended
        mult_features = self.spatial_dropout(mult_features)
        mult = F.silu(self.reduce_bn2(self.reduce_conv2(mult_features)))
        mult = self.reduce_attention2(mult)

        concat = torch.cat([t1_attended, t2_attended], dim=1)
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

def ResNet3D18(num_classes=3):
    return DualPathway3DResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet3D34(num_classes=3):
    return DualPathway3DResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet3D50(num_classes=3):
    return DualPathway3DResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# Example usage
if __name__ == "__main__":
    model = ResNet3D18(num_classes=3)
    
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
