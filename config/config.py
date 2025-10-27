import os
import shutil
import random
import hashlib

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

tumor_count = len([f for f in os.listdir(RAW_TUMOR_DIR) if os.path.isfile(os.path.join(RAW_TUMOR_DIR, f))])
no_tumor_count = len([f for f in os.listdir(RAW_NO_TUMOR_DIR) if os.path.isfile(os.path.join(RAW_NO_TUMOR_DIR, f))])

print(f"Tumor images: {tumor_count}")
print(f"No tumor images: {no_tumor_count}")
print(f"Total images: {tumor_count + no_tumor_count}")


# Train and test directories
TRAIN_DIR = os.path.join(RAW_DIR, 'train')
TEST_DIR = os.path.join(RAW_DIR, 'test')

TRAIN_TUMOR_DIR = os.path.join(TRAIN_DIR, 'tumor')
TRAIN_NO_TUMOR_DIR = os.path.join(TRAIN_DIR, 'no_tumor')
TEST_TUMOR_DIR = os.path.join(TEST_DIR, 'tumor')
TEST_NO_TUMOR_DIR = os.path.join(TEST_DIR, 'no_tumor')

# Create directories
for dir_path in [RAW_TUMOR_DIR, RAW_NO_TUMOR_DIR, TRAIN_TUMOR_DIR, TRAIN_NO_TUMOR_DIR, TEST_TUMOR_DIR,
                 TEST_NO_TUMOR_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def file_md5(path, chunk_size=8192):
    """Return MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def load_hashes_from_dir(dir_path):
    """Load MD5 hashes of all files in a directory."""
    hashes = set()
    if not os.path.exists(dir_path):
        return hashes
    for fname in os.listdir(dir_path):
        p = os.path.join(dir_path, fname)
        if os.path.isfile(p):
            try:
                hashes.add(file_md5(p))
            except Exception:
                pass
    return hashes


def split_images(source_dir, train_dir, test_dir, split_ratio=0.8):
    """Split images from source directory into train and test directories, avoiding duplicates."""
    # Ensure dirs exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Clear existing files in train and test directories (optional: keep if you want incremental runs)
    for directory in [train_dir, test_dir]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(images)
    print(f"Total images in {source_dir}: {len(images)}")

    if len(images) == 0:
        return 0, 0

    split_index = int(len(images) * split_ratio)
    # avoid putting all images into train (leave at least one for test if possible)
    if split_index >= len(images) and len(images) > 1:
        split_index = len(images) - 1

    train_images = images[:split_index]
    test_images = images[split_index:]

    # Copy to train
    for img in train_images:
        src = os.path.join(source_dir, img)
        dst = os.path.join(train_dir, img)
        shutil.copy2(src, dst)

    # Copy to test
    for img in test_images:
        src = os.path.join(source_dir, img)
        dst = os.path.join(test_dir, img)
        shutil.copy2(src, dst)

    return len(train_images), len(test_images)

# Split tumor images
train_tumor, test_tumor = split_images(RAW_TUMOR_DIR, TRAIN_TUMOR_DIR, TEST_TUMOR_DIR)
print(f"Tumor - Train: {train_tumor}, Test: {test_tumor}")

# Split no_tumor images
train_no_tumor, test_no_tumor = split_images(RAW_NO_TUMOR_DIR, TRAIN_NO_TUMOR_DIR, TEST_NO_TUMOR_DIR)
print(f"No Tumor - Train: {train_no_tumor}, Test: {test_no_tumor}")