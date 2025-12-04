import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(4 * growth_rate)
        self.conv2 = nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        return self.pool(x)

class DualPathway3DDenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, num_classes=3):
        super(DualPathway3DDenseNet, self).__init__()

        # First convolution for both pathways
        self.features_t1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.features_t2 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock for both pathways
        num_features = num_init_features
        self.dense_blocks_t1 = nn.ModuleList()
        self.trans_blocks_t1 = nn.ModuleList()
        self.dense_blocks_t2 = nn.ModuleList()
        self.trans_blocks_t2 = nn.ModuleList()
        
        # T1 pathway
        num_features_t1 = num_init_features
        self.dense_blocks_t1 = nn.ModuleList()
        self.trans_blocks_t1 = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block_t1 = DenseBlock(num_features_t1, num_layers, growth_rate)
            self.dense_blocks_t1.append(block_t1)
            num_features_t1 = num_features_t1 + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans_t1 = TransitionLayer(num_features_t1, num_features_t1 // 2)
                self.trans_blocks_t1.append(trans_t1)
                num_features_t1 = num_features_t1 // 2
                
        # T2 pathway
        num_features_t2 = num_init_features
        self.dense_blocks_t2 = nn.ModuleList()
        self.trans_blocks_t2 = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block_t2 = DenseBlock(num_features_t2, num_layers, growth_rate)
            self.dense_blocks_t2.append(block_t2)
            num_features_t2 = num_features_t2 + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans_t2 = TransitionLayer(num_features_t2, num_features_t2 // 2)
                self.trans_blocks_t2.append(trans_t2)
                num_features_t2 = num_features_t2 // 2
        
        final_features = num_features_t1

        # for i, num_layers in enumerate(block_config):
        #     # T1 pathway
        #     block_t1 = DenseBlock(num_features, num_layers, growth_rate)
        #     self.dense_blocks_t1.append(block_t1)
        #     num_features = num_features + num_layers * growth_rate
        #     if i != len(block_config) - 1:
        #         trans_t1 = TransitionLayer(num_features, num_features // 2)
        #         self.trans_blocks_t1.append(trans_t1)
        #         num_features = num_features // 2

        #     # Reset num_features for T2 pathway
        #     num_features = num_init_features
        #     # T2 pathway
        #     block_t2 = DenseBlock(num_features, num_layers, growth_rate)
        #     self.dense_blocks_t2.append(block_t2)
        #     num_features = num_features + num_layers * growth_rate
        #     if i != len(block_config) - 1:
        #         trans_t2 = TransitionLayer(num_features, num_features // 2)
        #         self.trans_blocks_t2.append(trans_t2)
        #         num_features = num_features // 2

        # final_features = num_features

        # Fusion and classification layers
        self.fusion_conv = nn.Conv3d(2 * final_features, final_features, kernel_size=1)
        self.fusion_bn = nn.BatchNorm3d(final_features)
        
        self.reduce_conv1 = nn.Conv3d(final_features, final_features // 2, kernel_size=1)
        self.reduce_bn1 = nn.BatchNorm3d(final_features // 2)
        self.reduce_conv2 = nn.Conv3d(final_features, final_features // 2, kernel_size=1)
        self.reduce_bn2 = nn.BatchNorm3d(final_features // 2)

        self.norm_final = nn.BatchNorm3d(final_features // 2)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(final_features // 2, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

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
        features = self.features_t1 if pathway == "t1" else self.features_t2
        dense_blocks = self.dense_blocks_t1 if pathway == "t1" else self.dense_blocks_t2
        trans_blocks = self.trans_blocks_t1 if pathway == "t1" else self.trans_blocks_t2

        x = features(x)
        
        for i in range(len(dense_blocks)):
            x = dense_blocks[i](x)
            if i != len(dense_blocks) - 1:
                x = trans_blocks[i](x)
        
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

def DenseNet121(num_classes=3):
    return DualPathway3DDenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes)

def DenseNet169(num_classes=3):
    return DualPathway3DDenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=num_classes)

def DenseNet201(num_classes=3):
    return DualPathway3DDenseNet(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=num_classes)

# Example usage
if __name__ == "__main__":
    # Create model instance
    model = DenseNet121(num_classes=3)
    
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
