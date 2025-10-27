import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
   ]
)


data_dir = os.path.join(os.path.dirname(__file__), "..", "data","raw")
train_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
print(train_set.transform)
test_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
print(test_set.transform)

print(train_set.classes)
print(train_set.class_to_idx)

# Visualiztion some images from Train Set
CLA_label = {
    0: 'no_tumor',
    1: 'tumor'
}

figure = plt.figure(figsize=(10, 10))
cols, rows = 4, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_set), size=(1,)).item()
    img, label = train_set[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(CLA_label[label])
    plt.axis("off")
    img_np = img.numpy().transpose((1, 2, 0))
    # Clip pixel values to [0, 1]
    img_valid_range = np.clip(img_np, 0, 1)
    plt.imshow(img_valid_range)
    plt.suptitle('Brain Images', y=0.95)
plt.show()