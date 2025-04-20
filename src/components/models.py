import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Simple CNN with 2 convolutional layers and 2 fully connected layers"""
    def __init__(self, num_classes=11):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32x3 -> 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32x32 -> 16x16x32
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16x32 -> 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16x16x64 -> 8x8x64
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 8x8x64 -> 4096
            nn.Linear(64 * 8 * 8, 128),  # 4096 -> 128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # 128 -> 11
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MediumCNN(nn.Module):
    """Medium complexity CNN with 3 convolutional layers and batch normalization"""
    def __init__(self, num_classes=11):
        super(MediumCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # 32x32x3 -> 32x32x64
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32x64 -> 16x16x64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 16x16x64 -> 16x16x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16x128 -> 8x8x128
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 8x8x128 -> 8x8x256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2) # 8x8x256 -> 4x4x256
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), # 4x4x256 -> 4096
            nn.Linear(256 * 4 * 4, 512), # 4096 -> 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes) # 512 -> 11
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # first conv layer changes the number of channels and extracts features
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # second conv layer refines those features without changing dimensions
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential() # identity (no transformation)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class ComplexCNN(nn.Module):
    """Complex CNN with proper residual connections"""
    def __init__(self, num_classes=11):
        super(ComplexCNN, self).__init__()
        
        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # 32x32x3 -> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with proper skip connections
        self.res_block1 = ResidualBlock(64, 128) # 32x32x64 -> 32x32x128
        self.res_block2 = ResidualBlock(128, 256) # 32x32x128 -> 32x32x256
        self.res_block3 = ResidualBlock(256, 512) # 32x32x256 -> 32x32x512
        
        # Final layers
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Reduces to 1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(512, num_classes) # 512 -> 11
        )
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Final classification
        x = self.final(x)
        return x


class DenseNet(nn.Module):
    """DenseNet architecture with proper dense connections and bottleneck layers"""
    def __init__(self, num_classes=11):
        super(DenseNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # 32x32x3 -> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True) # saves memory by not creating a new tensor (overwrites the input)
        )
        
        # Dense blocks with proper growth rate
        self.dense1 = self._make_dense_block(64, 32)
        self.dense2 = self._make_dense_block(48, 32)
        self.dense3 = self._make_dense_block(40, 32)
        
        # Transition layers
        self.trans1 = self._make_transition(96)
        self.trans2 = self._make_transition(80)
        
        # Final layers
        self.final = nn.Sequential(
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), # Reduces to 1x1x72
            nn.Flatten(), # 1x1x72 -> 72
            nn.Linear(72, num_classes) # 72 -> 11
        )
    
    def _make_dense_block(self, in_channels, growth_rate):
        """Creates a dense block with bottleneck layer"""
        return nn.Sequential(
            # Bottleneck layer
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1),
            
            # Main convolution
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)
        )
    
    def _make_transition(self, in_channels):
        """Creates a transition layer to reduce dimensions"""
        return nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.AvgPool2d(2)
        )
    
    def forward(self, x):
        # Initial convolution
        features = self.conv1(x) # 32x32x3 -> 32x32x64
        
        # First dense block
        new_features = self.dense1(features) # 32x32x64 -> 32x32x32 
        features = torch.cat([features, new_features], 1)  # 64 + 32 = 96 channels
        features = self.trans1(features)  # 96 -> 48 channels
        
        # Second dense block
        new_features = self.dense2(features) # 32x32x48 -> 32x32x32
        features = torch.cat([features, new_features], 1)  # 48 + 32 = 80 channels
        features = self.trans2(features)  # 80 -> 40 channels
        
        # Third dense block
        new_features = self.dense3(features) # 32x32x40 -> 32x32x32
        features = torch.cat([features, new_features], 1)  # 40 + 32 = 72 channels
        
        # Final classification
        x = self.final(features) 
        return x


class VGG16FineTune(nn.Module):
    """Fine-tuned VGG16 model"""
    def __init__(self, num_classes=11):
        super(VGG16FineTune, self).__init__()
        self.net = models.vgg16(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Reduces to 1x1x512
        self.net.classifier = nn.Sequential(
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(512, num_classes) # 512 -> 11
        )
    
    def forward(self, x):
        return self.net(x)
    
    

    
    
        
        