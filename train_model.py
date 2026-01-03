"""
Plant Disease Detection Model Training Script - Fixed Version

This script handles the complete training process for the plant disease detection model.

Usage:
    python train_model.py [--force] [--epochs] [--finetune-epochs]

Arguments:
   --force             Force retraining even if a trained model already exists
   --epochs            Number of initial training epochs (default: 10)
   --finetune-epochs   Number of fine-tuning epochs (default: 5)
   --batch-size        Batch size for training (default: 32)
"""

import os
import sys
import argparse
import time
import random
import json
import urllib.request
import zipfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data_utils import (
    PlantVillageDataset,
    create_data_loaders,
    validate_dataset,
    get_data_transforms,
    get_dataset_statistics
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--force', action='store_true', help='Force retraining even if model exists')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--finetune-epochs', type=int, default=5, help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    return parser.parse_args()

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    """Determine the best available device for training"""
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def download_dataset_instructions(base_dir='dataset'):
    """
    Provide instructions for downloading the PlantVillage dataset.
    """
    plantvillage_dir = os.path.join(base_dir, 'PlantVillage')
    
    # Check if dataset already exists using data_utils function
    if validate_dataset(plantvillage_dir, image_type="color"):
        print(f"PlantVillage dataset already exists and is valid at {plantvillage_dir}")
        return True
    
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Kaggle dataset information
    kaggle_dataset_url = "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
    
    print("\n" + "="*80)
    print("PLANTVILLAGE DATASET REQUIRED")
    print("="*80)
    print(f"The PlantVillage dataset needs to be downloaded from Kaggle.")
    print(f"\n Manual Download Instructions:")
    print(f"1. Visit: {kaggle_dataset_url}")
    print("2. Click 'Download' to get plantvillage-dataset.zip")
    print(f"3. Extract the ZIP file to: {plantvillage_dir}")
    print("   (Make sure the 'color' folder is directly inside PlantVillage/)")
    
    print(f"\nExpected directory structure after extraction:")
    print(f"   {plantvillage_dir}/")
    print(f"   └── color/")
    print(f"       ├── Apple___Apple_scab/")
    print(f"       ├── Apple___Black_rot/")
    print(f"       ├── Apple___Cedar_apple_rust/")
    print(f"       └── ... (more disease folders)")
    print("="*80)
    
    return False

def create_efficientnet_model(num_classes):
    """Create and initialize the EfficientNet model"""
    # Load a pre-trained EfficientNet-B0 model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze the feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use tqdm for progress bar
    train_loader_tqdm = tqdm(train_loader, desc='Training')
    
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        train_loader_tqdm.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader.sampler)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # No gradient calculation during validation
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader.sampler)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def get_current_lr(optimizer):
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    """Train and evaluate the model"""
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")
        print(f"Current Learning Rate: {get_current_lr(optimizer):.6f}")
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save model with metadata
            checkpoint = {
                'model_class': model.__class__.__name__,
                'state_dict': model.state_dict(),
                'num_classes': model.classifier[1].out_features,
                'epoch': epoch + 1,
                'best_acc': best_val_acc
            }
            torch.save(checkpoint, 'best_model.pth')
            print(f"Model saved with validation accuracy: {val_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load the best model weights
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, history

def setup_fine_tuning(model, unfreeze_layers=6):
    """Setup model for fine-tuning"""
    print("\n=== Setting up Fine-tuning ===")
    
    # Count total feature layers
    total_feature_layers = len(list(model.features.children()))
    print(f"Total feature layers: {total_feature_layers}")
    
    # Calculate which layers to unfreeze (last few layers)
    layers_to_unfreeze = min(unfreeze_layers, total_feature_layers)
    unfreeze_from = total_feature_layers - layers_to_unfreeze
    
    print(f"Unfreezing last {layers_to_unfreeze} feature layers (from layer {unfreeze_from})")
    
    # Unfreeze the selected layers
    unfrozen_params = 0
    for i, layer in enumerate(model.features.children()):
        if i >= unfreeze_from:
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()
            print(f"  Unfroze layer {i}: {layer.__class__.__name__}")
    
    # Print parameter summary after unfreezing
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter Summary after unfreezing:")
    print(f"• Total parameters: {total_params:,}")
    print(f"• Trainable parameters: {trainable_params:,}")
    print(f"• Newly unfrozen parameters: {unfrozen_params:,}")
    print(f"• Percentage trainable: {100 * trainable_params / total_params:.1f}%")
    
    return model

def create_fine_tuning_optimizer(model, lr=0.0001):
    """Create optimizer for fine-tuning with lower learning rate."""
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Create a more aggressive scheduler for fine-tuning
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode='min', factor=0.5, patience=2
    )
    
    print(f"Fine-tuning optimizer created with learning rate: {lr}")
    return optimizer_ft, scheduler_ft

def fine_tune_model(model, train_loader, val_loader, criterion, device, num_epochs=5):
    """Fine-tune the model"""
    print(f"\n=== Starting Fine-tuning for {num_epochs} epochs ===")
    
    # Setup fine-tuning
    model_ft = setup_fine_tuning(model, unfreeze_layers=6)
    
    # Create optimizer with lower learning rate
    optimizer_ft, scheduler_ft = create_fine_tuning_optimizer(model_ft, lr=0.0001)
    
    # Train the fine-tuned model
    model_ft, history_ft = train_model(
        model_ft, train_loader, val_loader, criterion, optimizer_ft, scheduler_ft, device, num_epochs
    )
    
    return model_ft, history_ft

def plot_training_history(history):
    """Plot and save the training history"""
    plt.figure(figsize=(15, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='s')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved to training_history.png")

def save_model_metadata(class_to_idx, classes):
    """Save model metadata for later use in the application"""
    metadata = {
        'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_classes': len(classes),
        'classes': classes,
        'training_info': {
            'framework': f"PyTorch {torch.__version__}",
            'model': 'EfficientNet-B0',
            'image_size': '224x224'
        }
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("Model metadata saved to model_metadata.json")

def create_class_indices_compatibility():
    """
    Create class_indices.npy file for compatibility with app.py.
    This ensures the web app can load the model correctly.
    """
    if os.path.exists('model_metadata.json'):
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        classes = metadata.get('classes', [])
        if classes:
            # Create class_indices dictionary (same as data_utils creates)
            class_indices = {class_name: i for i, class_name in enumerate(classes)}
            
            # Save class indices
            np.save('class_indices.npy', class_indices)
            print(f"Created class_indices.npy with {len(class_indices)} classes for app compatibility")
            return class_indices
    
    print("Warning: Could not create class_indices.npy - model_metadata.json not found")
    return None

def main():
    """Main function to handle the training process."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    set_seeds(42)
    print("Random seeds set for reproducibility")
    
    # Check if model already exists
    if os.path.exists('best_model.pth') and not args.force:
        print("Trained model already exists at 'best_model.pth'")
        print("Use --force to retrain, or run the web app with: python app.py")
        return
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    if device.type == 'mps':
        print("Apple Silicon GPU acceleration enabled")
    elif device.type == 'cuda':
        print("NVIDIA GPU acceleration enabled")
    else:
        print("Using CPU (consider using GPU for faster training)")
    
    # Dataset validation
    base_dir = 'dataset/PlantVillage'
    print(f"\nChecking dataset at: {base_dir}")
    
    if not download_dataset_instructions('dataset'):
        print("Dataset not found. Please download it first and rerun the script.")
        return
    
    # Get dataset statistics
    print("\nAnalyzing dataset...")
    stats = get_dataset_statistics(base_dir, image_type="color")
    if 'error' not in stats:
        print(f"Found {stats['total_classes']} classes with {stats['total_images']} total images")
        print(f"Plant types: {len(stats['plants'])}")
        print(f"Disease types: {len(stats['diseases'])}")

        # Show some examples
        print(f"\nSample plants: {', '.join(stats['plants'][:5])}")
        if len(stats['plants']) > 5:
            print(f"    ... and {len(stats['plants']) - 5} more")
    else:
        print(f"Error analyzing dataset: {stats['error']}")
        return
    
    # Create data loaders
    print("\nPreparing data loaders...")
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = args.batch_size
    
    try:
        train_loader, val_loader, class_to_idx, class_names = create_data_loaders(
            base_dir,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            batch_size=BATCH_SIZE,
            image_type="color",
            test_size=0.2,
            random_state=42
        )
        
        num_classes = len(class_names)
        print(f"Data loaders created successfully!")
        print(f"Number of classes: {num_classes}")
        print(f"Batch size: {BATCH_SIZE}")
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # Create model
    print("\nInitializing EfficientNet-B0 model...")
    model = create_efficientnet_model(num_classes)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    print(f"Initial learning rate: {get_current_lr(optimizer)}")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs
    )
    
    print("\nStarting fine-tuning...")
    model_ft, history_ft = fine_tune_model(model, train_loader, val_loader, criterion, device, num_epochs=args.finetune_epochs)
    
    # Plot training history
    print("\nCreating training visualizations...")
    plot_training_history(history)
    
    save_model_metadata(class_to_idx, class_names)
    
    # Create compatibility files for the web app
    create_class_indices_compatibility()
    
    print("\nTraining completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()