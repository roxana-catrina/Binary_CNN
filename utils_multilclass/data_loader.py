import os

import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Data preprocessing and augmentation for training (IMPROVED)
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize bigger first
    transforms.RandomCrop((224, 224)),  # Then random crop
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),  # Add vertical flip
    transforms.RandomRotation(20),  # More rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Affine transformations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data preprocessing for testing (no augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloaders(batch_size=32, num_workers=0):
    """
    Returns train and test dataloaders for multiclass classification
    """
    # Get the base directory (Binary CNN folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(base_dir, 'data', 'multiclass')

    train_path = os.path.join(data_dir, 'Training')
    test_path = os.path.join(data_dir, 'Testing')

    # Check if directories exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Testing directory not found: {test_path}")

    # Create datasets
    train_dataset = ImageFolder(train_path, transform=train_transforms)
    test_dataset = ImageFolder(test_path, transform=test_transforms)

    print(f"Found {len(train_dataset)} training images")
    print(f"Found {len(test_dataset)} testing images")
    print(f"Classes: {train_dataset.classes}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders(batch_size=16, num_workers=2)

    # Get train_dataset for class names
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(base_dir, 'data', 'multiclass')
    train_path = os.path.join(data_dir, 'Training')
    train_dataset = ImageFolder(train_path, transform=train_transforms)

    # Load a batch of images and labels for visualization
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Convert images to numpy arrays and denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = (images.numpy().transpose((0, 2, 3, 1)) * std + mean).clip(0, 1)

    # Create a grid of images
    num_images = len(images)
    rows = int(np.ceil(num_images / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(15, 15))

    # Plot images with labels
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.set_title(f'Label: {train_dataset.classes[labels[i]]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
