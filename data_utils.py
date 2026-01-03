"""
Common data utilities for Plant Disease Detection project.

This module contains shared classes and functions for dataset handling,
data loading, and preprocessing that can be used across different
components of the project.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


class PlantVillageDataset(Dataset):
    """
    Custom Dataset class for PlantVillage dataset.
    
    Supports multiple image types (color, grayscale, segmented) and handles
    the directory structure of the PlantVillage dataset.
    """
    
    def __init__(self, root_dir, image_type="color", transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Path to the PlantVillage dataset directory
            image_type (str): Type of images to use ('color', 'grayscale', 'segmented')
            transform (torchvision.transforms): Transformations to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Navigate to the specific image type directory
        self.image_type_dir = os.path.join(root_dir, image_type)
        if not os.path.isdir(self.image_type_dir):
            raise ValueError(f"Image type directory '{image_type}' not found in {root_dir}")
        
        # Get disease classes
        self.classes = [d for d in os.listdir(self.image_type_dir) 
                       if os.path.isdir(os.path.join(self.image_type_dir, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(self.classes))}
        
        self.image_paths = []
        self.labels = []
        
        # Collect all image paths and their labels
        for class_name in self.classes:
            class_dir = os.path.join(self.image_type_dir, class_name)
            if os.path.isdir(class_dir):
                class_idx = self.class_to_idx[class_name]
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        """Return the list of class names."""
        return self.classes
    
    def get_class_to_idx(self):
        """Return the class to index mapping."""
        return self.class_to_idx


def get_data_transforms(img_height=224, img_width=224, for_training=True):
    """
    Get data transformations for training or validation.
    
    Args:
        img_height (int): Target image height
        img_width (int): Target image width
        for_training (bool): Whether to include data augmentation for training
        
    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    if for_training:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Validation transforms without augmentation
        return transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def create_data_loaders(dataset_dir, img_height=224, img_width=224, 
                       batch_size=32, image_type="color", test_size=0.2, 
                       random_state=42):
    """
    Create training and validation data loaders.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        img_height (int): Target image height
        img_width (int): Target image width
        batch_size (int): Batch size for data loaders
        image_type (str): Type of images to use
        test_size (float): Fraction of data to use for validation
        random_state (int): Random state for reproducible splits
        
    Returns:
        tuple: (train_loader, val_loader, class_to_idx, class_names)
    """
    # Get transforms
    train_transforms = get_data_transforms(img_height, img_width, for_training=True)
    val_transforms = get_data_transforms(img_height, img_width, for_training=False)
    
    # Create full dataset for splitting
    full_dataset = PlantVillageDataset(dataset_dir, image_type=image_type, transform=val_transforms)
    
    # Split into train and validation indices
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, 
        stratify=full_dataset.labels
    )
    
    # Create datasets with appropriate transforms
    train_dataset = PlantVillageDataset(dataset_dir, image_type=image_type, transform=train_transforms)
    val_dataset = PlantVillageDataset(dataset_dir, image_type=image_type, transform=val_transforms)
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    
    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Number of training samples: {len(train_indices)}")
    print(f"Number of validation samples: {len(val_indices)}")
    
    return train_loader, val_loader, full_dataset.class_to_idx, full_dataset.classes


def validate_dataset(dataset_dir, image_type="color"):
    """
    Validate the dataset directory structure and contents.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        image_type (str): Type of images to validate
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return False
    
    image_type_dir = os.path.join(dataset_dir, image_type)
    if not os.path.exists(image_type_dir):
        print(f"Image type directory not found: {image_type_dir}")
        return False
    
    # Check for class directories
    class_dirs = [d for d in os.listdir(image_type_dir) 
                 if os.path.isdir(os.path.join(image_type_dir, d))]
    
    if len(class_dirs) == 0:
        print(f"No class directories found in {image_type_dir}")
        return False
    
    # Check for images in class directories
    total_images = 0
    for class_dir in class_dirs:
        class_path = os.path.join(image_type_dir, class_dir)
        images = [f for f in os.listdir(class_path) 
                 if os.path.isfile(os.path.join(class_path, f)) and 
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images += len(images)
    
    print(f"Found {len(class_dirs)} classes with {total_images} total images")
    return True


def visualize_dataset_samples(dataset_dir, image_type="color", num_classes=5, 
                             num_samples_per_class=3, figsize=(15, 10)):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        image_type (str): Type of images to visualize
        num_classes (int): Number of classes to display
        num_samples_per_class (int): Number of samples per class
        figsize (tuple): Figure size for the plot
    """
    image_type_dir = os.path.join(dataset_dir, image_type)
    
    if not os.path.exists(image_type_dir):
        print(f"Image type directory not found: {image_type_dir}")
        return
    
    # Get class directories
    class_dirs = [d for d in os.listdir(image_type_dir) 
                 if os.path.isdir(os.path.join(image_type_dir, d))]
    
    # Select random classes
    selected_classes = random.sample(class_dirs, min(num_classes, len(class_dirs)))
    
    plt.figure(figsize=figsize)
    
    for i, class_name in enumerate(selected_classes):
        class_path = os.path.join(image_type_dir, class_name)
        images = [f for f in os.listdir(class_path) 
                 if os.path.isfile(os.path.join(class_path, f)) and 
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            continue
        
        # Select random samples
        selected_images = random.sample(images, min(num_samples_per_class, len(images)))
        
        for j, img_name in enumerate(selected_images):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path)
                
                # Calculate subplot position
                pos = i * num_samples_per_class + j + 1
                plt.subplot(num_classes, num_samples_per_class, pos)
                plt.imshow(img)
                
                # Extract disease name from class name
                disease_name = class_name.split('___')[-1] if '___' in class_name else class_name
                plt.title(disease_name, fontsize=10)
                plt.axis('off')
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    plt.tight_layout()
    plt.show()


def get_dataset_statistics(dataset_dir, image_type="color"):
    """
    Get comprehensive statistics about the dataset.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        image_type (str): Type of images to analyze
        
    Returns:
        dict: Dictionary containing dataset statistics
    """
    image_type_dir = os.path.join(dataset_dir, image_type)
    
    if not os.path.exists(image_type_dir):
        return {"error": f"Image type directory not found: {image_type_dir}"}
    
    class_dirs = [d for d in os.listdir(image_type_dir) 
                 if os.path.isdir(os.path.join(image_type_dir, d))]
    
    stats = {
        "total_classes": len(class_dirs),
        "total_images": 0,
        "class_distribution": {},
        "plants": set(),
        "diseases": set()
    }
    
    for class_dir in class_dirs:
        class_path = os.path.join(image_type_dir, class_dir)
        images = [f for f in os.listdir(class_path) 
                 if os.path.isfile(os.path.join(class_path, f)) and 
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        num_images = len(images)
        stats["total_images"] += num_images
        stats["class_distribution"][class_dir] = num_images
        
        # Extract plant and disease names
        if '___' in class_dir:
            plant, disease = class_dir.split('___', 1)
            stats["plants"].add(plant)
            stats["diseases"].add(disease)
    
    # Convert sets to lists for JSON serialization
    stats["plants"] = sorted(list(stats["plants"]))
    stats["diseases"] = sorted(list(stats["diseases"]))
    
    return stats