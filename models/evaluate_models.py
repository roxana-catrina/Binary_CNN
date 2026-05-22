import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from texttable import Texttable

# --- Add project paths to sys.path (similar to api_server.py) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import model classes
import models
import models.CNN_TUMOR
import models.CNN_TUMOR_MULTICLASS
import models.HYBRID_MODEL
import utils_binary
import utils_binary.findConv2dOutShape

# Create module aliases for loading the binary model
sys.modules['Binary_CNN'] = sys.modules[__name__]
sys.modules['Binary_CNN.models'] = models
sys.modules['Binary_CNN.models.CNN_TUMOR'] = models.CNN_TUMOR
from models.HYBRID_MODEL import HybridTumorClassifier

# --- Configuration ---
# Adjust paths to be relative to the project root, not the script's location
PROJECT_ROOT_DIR = os.path.dirname(PROJECT_ROOT)
BINARY_MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, 'Brain_Tumor_model.pt')
HYBRID_MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, 'best_hybrid_model_hybrid_concat.pth')
BINARY_TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'binary', 'train')
BINARY_TEST_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'binary', 'test')
MULTICLASS_TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'multiclass', 'Training')
MULTICLASS_TEST_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'multiclass', 'Testing')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Labels ---
BINARY_CLASS_LABELS = ['no_tumor', 'tumor']
HYBRID_LABELS = ['glioma', 'meningioma', 'pituitary']

# --- Image Transformations ---
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

hybrid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Custom Dataset Class ---
class EvaluationDataset(Dataset):
    def __init__(self, binary_data_dir, multiclass_data_dir, transform):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.multiclass_labels = []

        # --- Load Binary Test Data ('no_tumor' and 'tumor') ---
        # Assumes binary_data_dir points to 'data/binary/test'
        for label in BINARY_CLASS_LABELS:
            class_path = os.path.join(binary_data_dir, label)
            if not os.path.isdir(class_path):
                print(f"Warning: Directory not found for binary label '{label}': {class_path}")
                continue

            for filename in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, filename))
                self.labels.append(label)
                # For binary 'no_tumor' images, multiclass label is 'no_tumor'
                # For binary 'tumor' images, we will determine the type later
                self.multiclass_labels.append('no_tumor' if label == 'no_tumor' else 'tumor')

        # --- Load Multiclass Test Data to get specific tumor types ---
        # This helps map a 'tumor' image to its specific type (glioma, etc.)
        # Assumes multiclass_data_dir points to 'data/multiclass/Testing'
        for label in HYBRID_LABELS:
            class_path = os.path.join(multiclass_data_dir, label)
            if not os.path.isdir(class_path):
                print(f"Warning: Directory not found for multiclass label '{label}': {class_path}")
                continue

            for filename in os.listdir(class_path):
                full_path = os.path.join(class_path, filename)
                # Find if this image is already in our list from the binary 'tumor' folder
                try:
                    idx = self.image_paths.index(full_path)
                    # If found, update its multiclass label from 'tumor' to the specific type
                    self.multiclass_labels[idx] = label
                except ValueError:
                    # If not found, it means it's a tumor image not present in the binary set.
                    # We add it to the evaluation set.
                    self.image_paths.append(full_path)
                    self.labels.append('tumor') # It's a tumor
                    self.multiclass_labels.append(label) # We know its specific type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        binary_label = self.labels[idx]
        multiclass_label = self.multiclass_labels[idx]

        # Apply appropriate transform based on what we are evaluating
        tensor = self.transform(image)

        return tensor, binary_label, multiclass_label, img_path


def count_files_in_subdirs(directory):
    """Counts all files in the immediate subdirectories of a given directory."""
    total_count = 0
    if not os.path.isdir(directory):
        return 0
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            total_count += len([name for name in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, name))])
    return total_count

def display_dataset_summary():
    """Displays a summary of the dataset counts, assuming an 80/20 train/validation split."""
    # Get total counts from the training and testing directories
    binary_train_total = count_files_in_subdirs(BINARY_TRAIN_DATA_DIR)
    binary_test_count = count_files_in_subdirs(BINARY_TEST_DATA_DIR)
    multiclass_train_total = count_files_in_subdirs(MULTICLASS_TRAIN_DATA_DIR)
    multiclass_test_count = count_files_in_subdirs(MULTICLASS_TEST_DATA_DIR)

    # Calculate train and validation counts based on an 80/20 split
    binary_train_count = int(binary_train_total * 0.8)
    binary_val_count = binary_train_total - binary_train_count
    multiclass_train_count = int(multiclass_train_total * 0.8)
    multiclass_val_count = multiclass_train_total - multiclass_train_count

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t', 'i', 'i', 'i'])
    table.set_cols_align(['l', 'r', 'r', 'r'])
    table.add_rows([
        ['Dataset Type', 'Training Images', 'Validation Images', 'Testing Images'],
        ['Binary (Tumor/No-Tumor)', binary_train_count, binary_val_count, binary_test_count],
        ['Multiclass (Tumor Types)', multiclass_train_count, multiclass_val_count, multiclass_test_count]
    ])

    print("\n--- Dataset Summary ---")
    print(table.draw())
    print("")


# --- Evaluation Function ---
def evaluate_models():
    # 0. Display dataset summary
    display_dataset_summary()

    # 1. Load Models
    device = torch.device(DEVICE)

    # Load binary model
    binary_model = torch.load(BINARY_MODEL_PATH, map_location=device, weights_only=False)
    binary_model.to(device)
    binary_model.eval()
    print("Binary model loaded.")

    # Load hybrid model
    hybrid_model = HybridTumorClassifier(num_classes=3, input_size=224, fusion_type='concat')
    checkpoint = torch.load(HYBRID_MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        hybrid_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        hybrid_model.load_state_dict(checkpoint)
    hybrid_model.to(device)
    hybrid_model.eval()
    print("Hybrid model loaded.")

    # 2. Prepare Data
    # For binary model
    full_dataset = EvaluationDataset(BINARY_TEST_DATA_DIR, MULTICLASS_TEST_DATA_DIR, test_transform)
    data_loader = DataLoader(full_dataset, batch_size=16, shuffle=False)

    # 3. Run Predictions
    binary_true_labels = []
    binary_pred_labels = []

    hybrid_true_labels = []
    hybrid_pred_labels = []

    with torch.no_grad():
        for images, bin_labels, multi_labels, paths in data_loader:
            images = images.to(device)

            # --- Binary Prediction ---
            binary_outputs = binary_model(images)
            _, predicted_indices = torch.max(binary_outputs, 1)

            predicted_binary_labels = [BINARY_CLASS_LABELS[i] for i in predicted_indices.cpu().numpy()]
            binary_true_labels.extend(bin_labels)
            binary_pred_labels.extend(predicted_binary_labels)

            # --- Hybrid Prediction (only for images predicted as 'tumor') ---
            for i, pred_label in enumerate(predicted_binary_labels):
                # We evaluate the hybrid model if the *true* label is a tumor type
                true_multiclass_label = multi_labels[i]
                if true_multiclass_label in HYBRID_LABELS:
                    hybrid_true_labels.append(true_multiclass_label)

                    # If the binary model also predicted 'tumor', we check the hybrid model's output
                    if pred_label == 'tumor':
                        # Get the single image, transform and predict
                        image = Image.open(paths[i]).convert('RGB')
                        hybrid_tensor = hybrid_transform(image).unsqueeze(0).to(device)
                        hybrid_output = hybrid_model(hybrid_tensor)
                        _, hybrid_pred_index = torch.max(hybrid_output, 1)
                        hybrid_pred_labels.append(HYBRID_LABELS[hybrid_pred_index.item()])
                    else:
                        # If binary model predicted 'no_tumor' for a real tumor, it's a hybrid model misclassification
                        # We can count this as a wrong prediction, e.g., by appending a placeholder
                        hybrid_pred_labels.append('no_tumor_predicted')


    # 4. Calculate and Print Metrics
    print("\n--- Binary Classification Metrics (Tumor vs. No Tumor) ---")
    print(f"Accuracy: {accuracy_score(binary_true_labels, binary_pred_labels):.4f}")
    print(classification_report(binary_true_labels, binary_pred_labels, labels=BINARY_CLASS_LABELS))

    print("\n--- Hybrid Model Metrics (Tumor Type Classification) ---")
    print(f"Evaluated on {len(hybrid_true_labels)} total tumor samples from the test set.")
    if len(hybrid_true_labels) > 0:
        # Adjust labels for report to include cases where binary model failed
        report_labels = HYBRID_LABELS + ['no_tumor_predicted']

        print(f"Accuracy: {accuracy_score(hybrid_true_labels, hybrid_pred_labels):.4f}")
        print(classification_report(hybrid_true_labels, hybrid_pred_labels, labels=report_labels, zero_division=0))

        # Confusion Matrix for Hybrid Model
        cm = confusion_matrix(hybrid_true_labels, hybrid_pred_labels, labels=report_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=report_labels, yticklabels=report_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Hybrid Model Confusion Matrix')
        plt.savefig(os.path.join(PROJECT_ROOT_DIR, 'hybrid_confusion_matrix.png'))
        print(f"\nSaved hybrid model confusion matrix to '{os.path.join(PROJECT_ROOT_DIR, 'hybrid_confusion_matrix.png')}'")


if __name__ == '__main__':
    # Simplified check
    if not os.path.isdir(BINARY_TEST_DATA_DIR):
        print(f"ERROR: Binary test data directory not found at '{BINARY_TEST_DATA_DIR}'")
    elif not os.path.isdir(MULTICLASS_TEST_DATA_DIR):
        print(f"ERROR: Multiclass test data directory not found at '{MULTICLASS_TEST_DATA_DIR}'")
    else:
        try:
            import texttable
        except ImportError:
            print("ERROR: The 'texttable' package is required. Please install it by running: pip install texttable")
        else:
            evaluate_models()
