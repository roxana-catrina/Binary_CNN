import os
import sys
import zipfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_binary.config import KAGGLE_DATASET, DATA_DIR, RAW_TUMOR_DIR, RAW_NO_TUMOR_DIR

def download_kaggle_dataset():
    """Download dataset from Kaggle"""
    try:
        import kaggle
        print(f"Downloading dataset: {KAGGLE_DATASET}")
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)
        print("Download completed!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAsigură-te că:")
        print("1. Ai instalat kaggle: pip install kaggle")
        print("2. Ai configurat API key-ul în ~/.kaggle/kaggle.json")
        print("3. Ai acceptat terms & conditions pe pagina dataset-ului")
        sys.exit(1)

def organize_data():
    """Organize images into tumor/no_tumor folders"""
    print("\nOrganizing data...")

    # Find the extracted folder
    extracted_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and f != 'binary']

    if not extracted_folders:
        print("No extracted data found!")
        return

    # Look for yes and no folders
    for folder in extracted_folders:
        folder_path = os.path.join(DATA_DIR, folder)

        # Check for common folder structures
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                dir_lower = dir_name.lower()
                source_dir = os.path.join(root, dir_name)

                if 'yes' in dir_lower or 'tumor' in dir_lower:
                    print(f"Moving tumor images from {source_dir}")
                    move_images(source_dir, RAW_TUMOR_DIR)
                elif 'no' in dir_lower or 'normal' in dir_lower or 'healthy' in dir_lower:
                    print(f"Moving no tumor images from {source_dir}")
                    move_images(source_dir, RAW_NO_TUMOR_DIR)

        # Clean up extracted folder
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    print(f"\nData organized successfully!")
    print(f"Tumor images: {len(os.listdir(RAW_TUMOR_DIR))}")
    print(f"No tumor images: {len(os.listdir(RAW_NO_TUMOR_DIR))}")

def move_images(source_dir, dest_dir):
    """Move images from source to destination"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                dest_path = os.path.join(dest_dir, filename)
                # Handle duplicate filenames
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                    counter += 1
                shutil.copy2(file_path, dest_path)

if __name__ == "__main__":
    print("Starting data download and organization...")
    download_kaggle_dataset()
    organize_data()
    print("\nProcess completed!")

