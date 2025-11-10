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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Angular frontend

# Model path
MODEL_PATH = 'Brain_Tumor_model.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model at startup
model = None

# Image transformations (same as training)
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels
CLASS_LABELS = {
    0: 'no_tumor',
    1: 'tumor'
}


def load_model():
    """Load the trained model"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        device = torch.device(DEVICE)
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {DEVICE}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def predict_image(image):
    """
    Predict tumor presence in an image

    Args:
        image: PIL Image object

    Returns:
        dict: Prediction results
    """
    try:
        model.eval()
        # Preprocess image
        image_tensor = test_transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(torch.device(DEVICE))

        # Make prediction
        with   torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.exp(outputs)  # Model returns log_softmax
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'success': True,
            'prediction': CLASS_LABELS[predicted_class],
            'has_tumor': predicted_class == 1,
            'confidence': float(confidence),
            'probabilities': {
                'no_tumor': float(probabilities[0][0].item()),
                'tumor': float(probabilities[0][1].item())
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
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
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
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
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
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
    # Load model on startup
    if not load_model():
        logger.error("Failed to load model. Server will not start.")
        exit(1)

    # Start server
    logger.info("Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

