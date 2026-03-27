# Binary CNN – Brain Tumor Detection

A PyTorch-based Convolutional Neural Network (CNN) project for brain tumor detection and classification from MRI images. The repository contains multiple model architectures, training scripts, a REST API server, and DICOM medical-image support.

## Models

### 1. Binary CNN (`models/CNN_TUMOR.py`)
Detects whether an MRI scan contains a tumor or not (binary classification).

**Architecture:**
- 4 Convolutional layers (8 → 16 → 32 → 64 filters), each followed by ReLU and MaxPool2d
- Fully connected layers: `num_flatten → 100 → 2`
- Dropout (rate = 0.25) for regularization
- Output: `log_softmax` (2 classes: *tumor* / *no_tumor*)

**Input:** 3 × 256 × 256 RGB image

### 2. Multiclass CNN (`models/CNN_TUMOR_MULTICLASS.py`)
Classifies the type of tumor detected (multiclass classification).

**Architecture (`TumorClassifier`):**
- 4 convolutional blocks (3→32→64→128→256 channels), each with:
  - Two Conv2d layers + BatchNorm2d + ReLU
  - MaxPool2d + Dropout2d (0.2 → 0.2 → 0.3 → 0.3)
- Classifier head: `feature_size → 512 → 256 → 128 → num_classes`
- Output: logits (3 classes: *glioma*, *meningioma*, *pituitary*)

**Input:** 3 × 224 × 224 RGB image

### 3. Hybrid Models (`models/HYBRID_MODEL.py`)
Combines the custom CNN with a pre-trained ResNet18 backbone using one of four fusion strategies:

| Strategy | Description |
|----------|-------------|
| `hybrid_concat` | Concatenate feature vectors from both branches, pass through a shared classifier |
| `hybrid_add` | Project both branches to the same dimension and sum them (with residual connection) |
| `hybrid_attention` | Learn attention weights to blend the two branches |
| `ensemble` | Average the raw class predictions from each branch independently |

## Repository Structure

```
Binary_CNN/
├── models/
│   ├── CNN_TUMOR.py              # Binary CNN
│   ├── CNN_TUMOR_MULTICLASS.py   # Multiclass CNN
│   └── HYBRID_MODEL.py           # Hybrid CNN + ResNet18 models
├── config_binary/
│   └── config.py                 # Hyperparameters and data paths for binary model
├── config_multiclass/
│   └── config.py                 # Hyperparameters for multiclass model
├── utils_binary/
│   ├── data_loader.py            # Data loading & augmentation (binary)
│   ├── confusion_matrix.py       # Confusion matrix visualization
│   ├── findConv2dOutShape.py     # Helper to compute Conv2d output shape
│   └── ture_and_pred.py          # Collect predictions from a dataloader
├── utils_multilclass/
│   └── data_loader.py            # Data loading & augmentation (multiclass)
├── data/
│   ├── binary/                   # Raw + split images: tumor / no_tumor
│   └── multiclass/               # Training / Testing split: glioma / meningioma / pituitary
├── train_binary.py               # Train binary CNN (60 epochs, Adam, NLLLoss)
├── train_multiclass.py           # Train multiclass CNN (50 epochs, Adam, CrossEntropyLoss)
├── train_hybrid.py               # Train hybrid models
├── inference.py                  # Single-image inference (JPG, PNG, DICOM)
├── api_server.py                 # Flask REST API server
├── dicom_processor.py            # DICOM file processor
├── test_api.py                   # API endpoint tests
├── test_model_loading.py         # Model loading / forward-pass tests
├── test_hybrid_models.py         # Hybrid model tests
├── test_dicom_integration.py     # DICOM integration tests
└── test_java_backend.py          # Java backend integration tests
```

## Dataset

| Task | Classes | Split |
|------|---------|-------|
| Binary detection | tumor, no_tumor | 80% train / 20% test |
| Multiclass classification | glioma, meningioma, pituitary | pre-split Training / Testing folders |

Place raw images under:
- `data/binary/tumor/` and `data/binary/no_tumor/`
- `data/multiclass/Training/<class>/` and `data/multiclass/Testing/<class>/`

## Training

```bash
# Binary model
python train_binary.py

# Multiclass model
python train_multiclass.py

# Hybrid model (change MODEL_TYPE inside the script: hybrid_concat / hybrid_add / hybrid_attention / ensemble)
python train_hybrid.py
```

Trained weights are saved as:
- `weights.pt` / `Brain_Tumor_model.pt` – binary model
- `best_model_multiclass.pth` – multiclass model
- `best_hybrid_model_hybrid_concat.pth` – hybrid model

## Inference

```bash
python inference.py
```

Supports `.jpg`, `.png`, and DICOM (`.dcm`) files. Edit the file path and model selection inside `inference.py`.

## REST API

```bash
python api_server.py
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/predict` | POST | Predict from uploaded image file |
| `/api/predict-multipart` | POST | Predict from multipart form upload |
| `/api/predict-base64` | POST | Predict from base64-encoded image |
| `/api/models/status` | GET | Show loaded model status |

## Tests

```bash
python -m pytest test_model_loading.py
python -m pytest test_hybrid_models.py
python -m pytest test_api.py
python -m pytest test_dicom_integration.py
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Flask
- scikit-learn
- seaborn / matplotlib
- tqdm
- pydicom (for DICOM support)
- torchsummary
