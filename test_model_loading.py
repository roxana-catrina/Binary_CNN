"""
Test script to verify model loading works correctly
"""
import torch
import sys
import os

# Add models directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.CNN_TUMOR_MULTICLASS import TumorClassifier

BINARY_MODEL_PATH = 'Brain_Tumor_model.pt'
MULTICLASS_MODEL_PATH = 'best_model_multiclass.pth'
DEVICE = 'cpu'  # Force CPU for testing

def test_binary_model():
    """Test loading binary model"""
    print("\n=== Testing Binary Model Loading ===")
    try:
        device = torch.device(DEVICE)

        if not os.path.exists(BINARY_MODEL_PATH):
            print(f"❌ Binary model not found: {BINARY_MODEL_PATH}")
            return False

        binary_model = torch.load(BINARY_MODEL_PATH, map_location=device, weights_only=False)
        binary_model.to(device)
        binary_model.eval()
        print(f"✅ Binary model loaded successfully on {DEVICE}")
        print(f"   Model type: {type(binary_model)}")
        return True
    except Exception as e:
        print(f"❌ Error loading binary model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiclass_model():
    """Test loading multiclass model"""
    print("\n=== Testing Multiclass Model Loading ===")
    try:
        device = torch.device(DEVICE)

        if not os.path.exists(MULTICLASS_MODEL_PATH):
            print(f"❌ Multiclass model not found: {MULTICLASS_MODEL_PATH}")
            return False

        # Load the saved file
        loaded_data = torch.load(MULTICLASS_MODEL_PATH, map_location=device, weights_only=False)
        print(f"   Loaded data type: {type(loaded_data)}")

        # Check what we got
        if isinstance(loaded_data, dict):
            print(f"   It's a dictionary with keys: {loaded_data.keys() if hasattr(loaded_data, 'keys') else 'N/A'}")

            # Try to instantiate model and load state dict
            multiclass_model = TumorClassifier(num_classes=4, input_size=224)
            print(f"✅ TumorClassifier instantiated successfully")

            if 'state_dict' in loaded_data:
                print("   Loading from checkpoint with 'state_dict' key")
                multiclass_model.load_state_dict(loaded_data['state_dict'])
            else:
                print("   Loading state dict directly")
                multiclass_model.load_state_dict(loaded_data)

            multiclass_model.to(device)
            multiclass_model.eval()
            print(f"✅ Multiclass model loaded successfully on {DEVICE}")

            # Test a forward pass
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                output = multiclass_model(dummy_input)
            print(f"✅ Forward pass successful, output shape: {output.shape}")

            return True
        else:
            print(f"   Unexpected data type, attempting to use directly")
            multiclass_model = loaded_data
            multiclass_model.to(device)
            multiclass_model.eval()
            print(f"✅ Multiclass model loaded successfully on {DEVICE}")
            return True

    except Exception as e:
        print(f"❌ Error loading multiclass model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)

    binary_ok = test_binary_model()
    multiclass_ok = test_multiclass_model()

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Binary Model: {'✅ PASSED' if binary_ok else '❌ FAILED'}")
    print(f"Multiclass Model: {'✅ PASSED' if multiclass_ok else '❌ FAILED'}")

    if binary_ok and multiclass_ok:
        print("\n🎉 All models loaded successfully!")
        print("✅ The API server should work correctly now.")
    else:
        print("\n⚠️ Some models failed to load.")

