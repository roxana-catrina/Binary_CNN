import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MULTICLASS_DIR = os.path.join(DATA_DIR, 'multiclass')

# Data directories
TRAIN_DIR = os.path.join(MULTICLASS_DIR, 'Training')
TEST_DIR = os.path.join(MULTICLASS_DIR, 'Testing')

# Class names (based on the dataset structure)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)

# Create main directory
os.makedirs(MULTICLASS_DIR, exist_ok=True)

# Image preprocessing parameters
IMG_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model save path
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'Brain_Tumor_Multiclass_model.pt')
WEIGHTS_SAVE_PATH = os.path.join(BASE_DIR, 'weights_multiclass.pt')

print(f"Multiclass config loaded!")
print(f"Device: {DEVICE}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Classes: {CLASS_NAMES}")

