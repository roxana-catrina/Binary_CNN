import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

# Kaggle dataset
KAGGLE_DATASET = 'preetviradiya/brian-tumor-dataset'

# Raw data directories
RAW_TUMOR_DIR = os.path.join(RAW_DIR, 'tumor')
RAW_NO_TUMOR_DIR = os.path.join(RAW_DIR, 'no_tumor')

# Create directories if they don't exist
os.makedirs(RAW_TUMOR_DIR, exist_ok=True)
os.makedirs(RAW_NO_TUMOR_DIR, exist_ok=True)

