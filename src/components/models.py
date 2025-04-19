import numpy as np
import cv2
import torch
from torch import nn
from torchvision import transforms, models


inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class VGG16FineTune(nn.Module):
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
    
    

    
    
        
        