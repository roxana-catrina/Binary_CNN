"""
Flask REST API server for brain tumor detection
This server accepts image uploads and returns tumor predictions
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from torchvision import transforms
import io
import base64
import os
import logging
import sys

# Add models directory to path
# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Also add parent directory if Binary_CNN is one level up
PARENT_DIR = os.path.dirname(PROJECT_ROOT)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import modules first
import models
import models.CNN_TUMOR
import models.CNN_TUMOR_MULTICLASS
import utils_binary
import utils_binary.findConv2dOutShape

# Create Binary_CNN module alias to fix module loading issue
# This is needed because the binary model was saved with Binary_CNN module references
sys.modules['Binary_CNN'] = sys.modules[__name__]
sys.modules['Binary_CNN.models'] = models
sys.modules['Binary_CNN.models.CNN_TUMOR'] = models.CNN_TUMOR
sys.modules['Binary_CNN.models.CNN_TUMOR_MULTICLASS'] = models.CNN_TUMOR_MULTICLASS
sys.modules['Binary_CNN.utils_binary'] = utils_binary
sys.modules['Binary_CNN.utils_binary.findConv2dOutShape'] = utils_binary.findConv2dOutShape

from models.CNN_TUMOR_MULTICLASS import TumorClassifier
from models.CNN_TUMOR import CNN_TUMOR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Angular frontend

# Model paths
BINARY_MODEL_PATH = 'Brain_Tumor_model.pt'
MULTICLASS_MODEL_PATH = 'best_model_multiclass.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models at startup
binary_model = None
multiclass_model = None

# Image transformations (same as training)
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformations for multiclass model
multiclass_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels for binary model
BINARY_CLASS_LABELS = {
    0: 'no_tumor',
    1: 'tumor'
}

# Class labels for multiclass model (tumor types)
MULTICLASS_LABELS = ['glioma', 'meningioma', 'pituitary']


def load_models():
    """Load both binary and multiclass models"""
    global binary_model, multiclass_model
    try:
        device = torch.device(DEVICE)

        # Load binary model (saved as full model)
        if not os.path.exists(BINARY_MODEL_PATH):
            raise FileNotFoundError(f"Binary model not found: {BINARY_MODEL_PATH}")

        binary_model = torch.load(BINARY_MODEL_PATH, map_location=device, weights_only=False)
        binary_model.to(device)
        binary_model.eval()
        logger.info(f"Binary model loaded successfully on {DEVICE}")

        # Load multiclass model (saved as state dict)
        if not os.path.exists(MULTICLASS_MODEL_PATH):
            raise FileNotFoundError(f"Multiclass model not found: {MULTICLASS_MODEL_PATH}")

        # Instantiate the model architecture
        multiclass_model = TumorClassifier(num_classes=3, input_size=224)

        # Load the state dictionary
        state_dict = torch.load(MULTICLASS_MODEL_PATH, map_location=device, weights_only=False)

        # Handle both cases: full model or state dict
        if isinstance(state_dict, dict) and 'state_dict' not in state_dict:
            # It's a state dict directly
            multiclass_model.load_state_dict(state_dict)
        elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
            # It's a checkpoint with state_dict key
            multiclass_model.load_state_dict(state_dict['state_dict'])
        else:
            # It might be the full model
            multiclass_model = state_dict

        multiclass_model.to(device)
        multiclass_model.eval()
        logger.info(f"Multiclass model loaded successfully on {DEVICE}")

        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def predict_image(image):
    """
    Predict tumor presence in an image and tumor type if present

    Args:
        image: PIL Image object

    Returns:
        dict: Prediction results including tumor type if tumor is detected
    """
    try:
        device = torch.device(DEVICE)

        # Step 1: Binary classification - Check if tumor exists
        binary_model.eval()
        image_tensor = test_transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            binary_outputs = binary_model(image_tensor)
            binary_probabilities = torch.exp(binary_outputs)  # Model returns log_softmax
            predicted_class = torch.argmax(binary_probabilities, dim=1).item()
            binary_confidence = binary_probabilities[0][predicted_class].item()

        logger.info(f"Binary prediction: {BINARY_CLASS_LABELS[predicted_class]} (confidence: {binary_confidence:.4f})")

        result = {
            'success': True,
            'prediction': BINARY_CLASS_LABELS[predicted_class],
            'has_tumor': predicted_class == 1,
            'confidence': float(binary_confidence),
            'probabilities': {
                'no_tumor': float(binary_probabilities[0][0].item()),
                'tumor': float(binary_probabilities[0][1].item())
            }
        }

        # Step 2: If tumor detected, classify tumor type
        if predicted_class == 1:  # Tumor detected
            multiclass_model.eval()
            multiclass_tensor = multiclass_transform(image).unsqueeze(0)
            multiclass_tensor = multiclass_tensor.to(device)

            with torch.no_grad():
                multiclass_outputs = multiclass_model(multiclass_tensor)
                multiclass_probabilities = torch.softmax(multiclass_outputs, dim=1)

                # Get all probabilities for the 3 tumor types
                tumor_type_probs = {
                    label: float(multiclass_probabilities[0][i].item())
                    for i, label in enumerate(MULTICLASS_LABELS)
                }

                # Get the tumor type with highest probability
                tumor_type = max(tumor_type_probs.items(), key=lambda x: x[1])

                logger.info(f"Tumor type prediction: {tumor_type[0]} (confidence: {tumor_type[1]:.4f})")
                logger.info(f"Tumor type probabilities: {tumor_type_probs}")

            # Add tumor type information to result
            result['tumor_type'] = tumor_type[0]
            result['tumor_type_confidence'] = float(tumor_type[1])
            result['tumor_type_probabilities'] = tumor_type_probs

        logger.info(f"Final result: {result['prediction']}" +
                   (f" -> {result.get('tumor_type', 'N/A')}" if result['has_tumor'] else ""))
        logger.info(f"Result details: {result}")
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'binary_model_loaded': binary_model is not None,
        'multiclass_model_loaded': multiclass_model is not None,
        'device': DEVICE
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict tumor from uploaded image

    Expected request:
    - multipart/form-data with 'file' field containing image
    OR
    - JSON with 'image' field containing base64 encoded image

    Returns:
    - JSON with prediction results
    """
    try:
        if binary_model is None or multiclass_model is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded'
            }), 500

        image = None

        # Handle multipart/form-data upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400

            # Read image from uploaded file
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Handle JSON with base64 encoded image
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({
                    'success': False,
                    'error': 'No image data provided'
                }), 400

            # Decode base64 image
            image_data = data['image']
            # Remove data:image/...;base64, prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        else:
            return jsonify({
                'success': False,
                'error': 'Invalid request format. Send file or base64 encoded image'
            }), 400

        # Make prediction
        result = predict_image(image)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Request error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict tumor from multiple uploaded images

    Expected request:
    - multipart/form-data with multiple 'files' fields

    Returns:
    - JSON with array of prediction results
    """
    try:
        if binary_model is None or multiclass_model is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded'
            }), 500

        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400

        files = request.files.getlist('files')
        results = []

        for file in files:
            try:
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                result = predict_image(image)
                result['filename'] = file.filename
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'filename': file.filename,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'results': results
        }), 200

    except Exception as e:
        logger.error(f"Batch request error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Load models on startup
    if not load_models():
        logger.error("Failed to load models. Server will not start.")
        exit(1)

    # Start server
    logger.info("Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
