"""This file contains the code for the preprocess step of the pipeline."""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from scipy.io import loadmat
from PIL import Image
import os
import pickle


# load the matlab file (SVHN dataset)
def load_svhn(path):
    """Load the SVHN dataset from the given path."""
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X, y


# custom dataset class for the SVHN dataset
class SVHNDataset(Dataset):
    """Custom dataset class for the SVHN dataset."""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[:, :, :, idx]
        # Convert to PIL Image (H, W, C)
        image = Image.fromarray(image.transpose(1, 0, 2))
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
# Data augmentation and normalization pipeline
transform = transforms.Compose([
    transforms.RandomRotation(20),  # Random rotation between -20 to 20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Gaussian blur for noise robustness
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

def create_dataloaders(mat_path, batch_size=64, shuffle=True, save_artifact=None):
    """Create dataloaders for the SVHN dataset and optionally save as artifact."""
    X, y = load_svhn(mat_path)
    dataset = SVHNDataset(X, y, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    if save_artifact:
        # Create artifacts directory if it doesn't exist
        os.makedirs('artifacts', exist_ok=True)
        
        # Save dataset metadata
        artifact_path = os.path.join('artifacts', save_artifact)
        metadata = {
            'images': X,
            'labels': y,
            'batch_size': batch_size,
            'shuffle': shuffle
        }
        
        with open(artifact_path, 'wb') as f:
            pickle.dump(metadata, f)
            
    return dataloader


if __name__ == "__main__":
    train_mat_path = "data/compressed_dataset_32x32/train_32x32.mat"
    test_mat_path = "data/compressed_dataset_32x32/test_32x32.mat"
    
    train_dataloader = create_dataloaders(train_mat_path, save_artifact="train_dataloader.pkl")
    test_dataloader = create_dataloaders(test_mat_path, save_artifact="test_dataloader.pkl", shuffle=False)
    # Example of loading from artifact
    # artifact_path = "artifacts/dataset_metadata.pkl"
    # loaded_dataloader = load_dataloader_from_artifact(artifact_path)
    