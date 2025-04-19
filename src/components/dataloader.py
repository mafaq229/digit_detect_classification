from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


class SVHNFull(Dataset):
    """
    PyTorch Dataset class for the organized SVHN dataset.
    This class reads from the organized dataset structure where images are stored in
    class-specific folders (0-9 for digits, 10 for background).
    
    Args:
        root_dir (str): Path to the organized dataset directory
        transform (callable, optional): Optional transform to be applied on a sample
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all samples
        self.samples = []
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
                
            class_label = int(class_dir.name)
            for img_path in class_dir.glob("*.png"):
                self.samples.append((img_path, class_label))
        
        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")
        print("Class distribution:")
        for i in range(11):
            num_samples = sum(1 for _, lbl in self.samples if lbl == i)
            print(f"Class {i}: {num_samples} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label



class RandomGaussianNoise:
    """
    Add random gaussian noise to the image.
    
    Args:
        std (float): Standard deviation for the noise
        prob (float): Probability of applying noise to an image
    """
    def __init__(self, std=0.05, prob=0.2):
        self.std = std
        self.prob = prob
        
    def __call__(self, tensor):
        if random.random() > self.prob:
            return tensor
            
        # Generate noise
        noise = torch.randn(tensor.size()) * self.std
        # Apply noise and clamp to valid range
        return torch.clamp(tensor + noise, 0., 1.)
    
    def __repr__(self):
        return self.__class__.__name__ + '(std={0}, prob={1})'.format(self.std, self.prob)


class RandomScale:
    """Randomly scale the image while maintaining aspect ratio"""
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range
        
    def __call__(self, img):
        scale = random.uniform(*self.scale_range)
        new_size = [int(dim * scale) for dim in img.size]
        return transforms.functional.resize(img, new_size)
    
    def __repr__(self):
        return self.__class__.__name__ + '(scale_range={0})'.format(self.scale_range)


train_transform = transforms.Compose([
    RandomScale(scale_range=(0.7, 1.3)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomAutocontrast(),
    RandomGaussianNoise(std=0.05, prob=0.05),  # Apply to 5% of images
    
    # Standard transforms
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test transforms (minimal augmentation)
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

