import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import Stream

from src.components.dataloader import SVHN, train_transform, test_transform
from src.components.models import SimpleCNN, MediumCNN, ComplexCNN, DenseNet, VGG16FineTune

class PrefetchDataLoader:
    """DataLoader wrapper that prefetches data to GPU"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.stream = Stream()
        
    def __iter__(self):
        for batch in self.dataloader:
            with torch.cuda.stream(self.stream):
                batch = [item.cuda(non_blocking=True) for item in batch]
            yield batch
    
    def __len__(self):
        return len(self.dataloader)


class ModelTrainer:
    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        
        # Enable DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
    def get_dataloaders(self, batch_size=64, val_split=0.1):
        # Load full training dataset
        full_dataset = SVHN(root_dir="artifacts/data/train", transform=train_transform, cache_size=10000)
        
        # Split into train and validation
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create dataloaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Use multiple workers
            pin_memory=True,  # Enable pinned memory
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=2  # Prefetch batches
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Wrap with prefetch loader if using CUDA
        if self.device == "cuda":
            train_loader = PrefetchDataLoader(train_loader)
            val_loader = PrefetchDataLoader(val_loader)
        
        return train_loader, val_loader
        
    def evaluate(self):
        """
        Evaluate the model on the test set
            
        Returns:
            dict: Dictionary containing test metrics
        """
        # Load test dataset with caching
        test_dataset = SVHN(root_dir="artifacts/data/test", transform=test_transform, cache_size=10000)
        
        # Create test loader with optimized settings
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Wrap with prefetch loader if using CUDA
        if self.device == "cuda":
            test_loader = PrefetchDataLoader(test_loader)
        
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        criterion = nn.CrossEntropyLoss()
        
        # Class-wise accuracy
        class_correct = [0] * 11  # 11 classes (0-9 digits + background)
        class_total = [0] * 11
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                # Images and labels are already on the correct device due to PrefetchDataLoader
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                # Calculate class-wise accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        
        # Calculate class-wise accuracy
        class_acc = [0.0] * 11
        for i in range(11):
            if class_total[i] > 0:
                class_acc[i] = class_correct[i] / class_total[i]
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'class_accuracy': class_acc,
            'class_distribution': class_total
        }
        
        return results
        
    def train(self, num_epochs, save_dir="artifacts/models", batch_size=64, learning_rate=0.001, patience=5, val_split=0.1):
        # Create output directory with model name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model.__class__.__name__
        output_dir = os.path.join(save_dir, f"{model_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get dataloaders
        train_loader, val_loader = self.get_dataloaders(batch_size, val_split)
        
        # Training configuration
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} train", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / total
            train_acc = correct / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} val", leave=False):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_path = os.path.join(output_dir, "best_model.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"Best model saved at epoch {epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Save training details
        training_details = {
            'model_name': model_name,
            'timestamp': timestamp,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'patience': patience,
            'val_split': val_split,
            'best_val_loss': best_val_loss,
            'training_history': history
        }
        
        # Save details to JSON
        with open(os.path.join(output_dir, 'training_details.json'), 'w') as f:
            json.dump(training_details, f, indent=4)
        
        # Plot and save training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        plt.close()
        
        print(f"Training complete")
        print(f"Model saved in: {output_dir}")
        
        return output_dir

if __name__ == "__main__":
    # Example usage with different models
    models = {
        # 'SimpleCNN': SimpleCNN(),
        # 'MediumCNN': MediumCNN(),
        'ComplexCNN': ComplexCNN(),
        'DenseNet': DenseNet(),
        # 'VGG16FineTune': VGG16FineTune()
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}")
        trainer = ModelTrainer(model)
        print("TRAINING DEVICE: ", trainer.device)
        model_dir = trainer.train(
            num_epochs=20,
            batch_size=64,
            learning_rate=0.001,
            patience=3
        )
        
        # Evaluate on test set
        print(f"\nEvaluating {name} on test set")
        test_results = trainer.evaluate()
        results[name] = test_results
        
        # Save test results
        with open(os.path.join(model_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # Print class-wise accuracy
        print("\nClass-wise accuracy:")
        for i, acc in enumerate(test_results['class_accuracy']):
            print(f"Class {i}: {acc:.4f}")
    
    # Print final comparison
    print("\nFinal Model Comparison:")
    print("Model\t\tTest Accuracy")
    print("-" * 30)
    for name, result in results.items():
        print(f"{name}\t{result['test_accuracy']:.4f}")

