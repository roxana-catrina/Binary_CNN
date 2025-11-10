import os
import sys
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_multiclass.config import KAGGLE_DATASET, MULTICLASS_DIR, TRAIN_DIR, TEST_DIR, CLASS_NAMES

def check_kaggle_setup():
    """Check if Kaggle is properly configured"""
    try:
        import kaggle
        print("[OK] Kaggle package is installed")
        return True
    except ImportError:
        print("[ERROR] Kaggle package not found!")
        print("Please install it: pip install kaggle")
        return False

def check_kaggle_credentials():
    """Check if kaggle.json exists"""
    kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.name == 'nt':  # Windows
        kaggle_config = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")

    if os.path.exists(kaggle_config):
        print(f"[OK] Kaggle credentials found at: {kaggle_config}")
        return True
    else:
        print(f"[ERROR] Kaggle credentials not found!")
        print(f"Expected location: {kaggle_config}")
        print("\nPlease:")
        print("1. Go to https://www.kaggle.com/")
        print("2. Login and go to Account settings")
        print("3. Scroll to API section and click 'Create New API Token'")
        print(f"4. Save the downloaded kaggle.json to: {kaggle_config}")
        return False

def download_kaggle_dataset():
    """Download multiclass brain tumor dataset from Kaggle"""
    try:
        import kaggle
        print(f"\nDownloading dataset: {KAGGLE_DATASET}")
        print(f"Destination: {MULTICLASS_DIR}")
        print("This may take several minutes depending on your internet speed...")
        print("")

        # Download and unzip directly to multiclass directory
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=MULTICLASS_DIR, unzip=True)

        print("\nDownload completed!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nMake sure that:")
        print("1. You have accepted terms & conditions on the dataset page")
        print(f"   Dataset URL: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print("2. Your internet connection is working")
        print("3. You have enough disk space (~500MB needed)")
        return False

def verify_dataset_structure():
    """Verify that the dataset was extracted correctly"""
    print("\nVerifying dataset structure...")

    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory not found at {TRAIN_DIR}")
        return False

    if not os.path.exists(TEST_DIR):
        print(f"Error: Testing directory not found at {TEST_DIR}")
        return False

    # Check training classes
    print("\nTraining set:")
    train_total = 0
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
            print(f"  {class_name}: {count} images")
            train_total += count
        else:
            print(f"  Warning: {class_name} directory not found")

    print(f"Total training images: {train_total}")

    # Check testing classes
    print("\nTesting set:")
    test_total = 0
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TEST_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
            print(f"  {class_name}: {count} images")
            test_total += count
        else:
            print(f"  Warning: {class_name} directory not found")

    print(f"Total testing images: {test_total}")
    print(f"\nGrand total: {train_total + test_total} images")

    return True

def main():
    print("=" * 60)
    print("MULTICLASS BRAIN TUMOR DATASET DOWNLOADER")
    print("=" * 60)
    print("")

    # Check Kaggle setup
    if not check_kaggle_setup():
        sys.exit(1)

    if not check_kaggle_credentials():
        sys.exit(1)

    print("")

    # Check if dataset already exists
    if os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR):
        print("\n[INFO] Dataset directories already exist!")

        # Count existing files
        train_count = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
        test_count = sum([len(files) for r, d, files in os.walk(TEST_DIR)])

        if train_count > 0 or test_count > 0:
            print(f"Found {train_count} training images and {test_count} testing images")
            response = input("\nDo you want to re-download? This will remove existing data. (yes/no): ").lower()
            if response not in ['yes', 'y']:
                print("\nSkipping download. Verifying existing dataset...")
                verify_dataset_structure()
                return
            else:
                print("\nCleaning existing data...")
                if os.path.exists(MULTICLASS_DIR):
                    for item in os.listdir(MULTICLASS_DIR):
                        item_path = os.path.join(MULTICLASS_DIR, item)
                        try:
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                        except Exception as e:
                            print(f"Warning: Could not remove {item_path}: {e}")

    # Download dataset
    success = download_kaggle_dataset()

    if success:
        # Verify structure
        if verify_dataset_structure():
            print("\n" + "=" * 60)
            print("✓ DATASET DOWNLOAD COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"\nYou can now train your model using:")
            print(f"  python train_multiclass.py")
            print(f"\nData locations:")
            print(f"  Training: {TRAIN_DIR}")
            print(f"  Testing: {TEST_DIR}")
        else:
            print("\n[WARNING] Dataset downloaded but structure verification failed")
            print("Please check the data manually")
    else:
        print("\n[ERROR] Dataset download failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Accept the dataset terms: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        print("2. Check your internet connection")
        print("3. Try downloading manually from Kaggle and extracting to data/multiclass/")
        sys.exit(1)

if __name__ == "__main__":
    main()

