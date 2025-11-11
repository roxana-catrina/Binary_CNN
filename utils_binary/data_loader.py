import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Define data directory and class labels
data_dir = "data/binary"  # Update this path as needed
CLA_label = {0: "No Tumor", 1: "Tumor"}  # Binary classification labels

# Transformări pentru antrenament (cu augmentări)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformări pentru test (fără augmentări)
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Păstrează pentru compatibilitate
transform = train_transform


def load_and_visualize():
    """Load data and display sample images"""
    train_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    print(train_set.classes)
    print(train_set.class_to_idx)

    # Vizualizare
    figure = plt.figure(figsize=(10, 10))
    cols, rows = 4, 4
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_set), size=(1,)).item()
        img, label = train_set[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(CLA_label[label])
        plt.axis("off")
        img_np = img.numpy().transpose((1, 2, 0))
        img_valid_range = np.clip(img_np, 0, 1)
        plt.imshow(img_valid_range)
    plt.suptitle('Brain Images', y=0.95)
    plt.savefig('brain_images_sample.png')
    plt.close()

    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    for key, value in {'Training data': train_loader, "Validation data": test_loader}.items():
        for X, y in value:
            print(f"{key}:")
            print(f"Shape of X : {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}\n")
            break


def get_dataloaders(batch_size=64, num_workers=0):
    """
    Return (train_loader, test_loader). Use num_workers=0 on Windows for safety.
    """
    train_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
