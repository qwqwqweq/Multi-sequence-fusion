import torch
import torch.nn as nn
import torch.nn.functional as F

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

        t1_features = self.forward_single_pathway(t1, "t1")
        t2_features = self.forward_single_pathway(t2, "t2")

        sum_features = t1_features + t2_features
        sum_avg = self.relu(self.reduce_bn1(self.reduce_conv1(sum_features))) / 2

        mult_features = t1_features * t2_features
        mult = self.relu(self.reduce_bn2(self.reduce_conv2(mult_features)))

        concat = torch.cat([t1_features, t2_features], dim=1)
        concat = self.relu(self.fusion_bn(self.fusion_conv(concat)))

        final_features = (sum_avg + mult + concat[:, :sum_avg.size(1), ...]) / 3

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
    # Create model instance
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
