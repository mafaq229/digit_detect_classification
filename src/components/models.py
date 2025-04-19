import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Simple CNN with 2 convolutional layers and 2 fully connected layers"""
    def __init__(self, num_classes=11):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
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
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ComplexCNN(nn.Module):
    """Complex CNN with residual connections and multiple feature maps"""
    def __init__(self, num_classes=11):
        super(ComplexCNN, self).__init__()
        
        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128)
        self.res_block2 = self._make_residual_block(128, 256)
        self.res_block3 = self._make_residual_block(256, 512)
        
        # Final layers
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x) + x
        x = self.res_block2(x) + x
        x = self.res_block3(x) + x
        x = self.final(x)
        return x


class DenseNet(nn.Module):
    """DenseNet inspired architecture with dense connections"""
    def __init__(self, num_classes=11):
        super(DenseNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Dense blocks
        self.dense1 = self._make_dense_block(64, 32)
        self.dense2 = self._make_dense_block(160, 32)
        self.dense3 = self._make_dense_block(256, 32)
        
        # Transition layers
        self.trans1 = self._make_transition(160)
        self.trans2 = self._make_transition(256)
        
        # Final layers
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(288, num_classes)
        )
    
    def _make_dense_block(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        )
    
    def _make_transition(self, in_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.AvgPool2d(2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.dense1(x)
        x = torch.cat([x, x1], 1)
        x = self.trans1(x)
        
        x2 = self.dense2(x)
        x = torch.cat([x, x2], 1)
        x = self.trans2(x)
        
        x3 = self.dense3(x)
        x = torch.cat([x, x3], 1)
        x = self.final(x)
        return x


class VGG16FineTune(nn.Module):
    """Fine-tuned VGG16 model"""
    def __init__(self, num_classes=11):
        super(VGG16FineTune, self).__init__()
        self.net = models.vgg16(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.net.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
    
    

    
    
        
        