
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
import numpy as np
from dicom_processor import DicomProcessor
from tumor_segmentation import process_tumor_segmentation

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
import models.HYBRID_MODEL
import utils_binary
import utils_binary.findConv2dOutShape

# Create Binary_CNN module alias to fix module loading issue
# This is needed because the binary model was saved with Binary_CNN module references
sys.modules['Binary_CNN'] = sys.modules[__name__]
sys.modules['Binary_CNN.models'] = models
sys.modules['Binary_CNN.models.CNN_TUMOR'] = models.CNN_TUMOR
sys.modules['Binary_CNN.models.CNN_TUMOR_MULTICLASS'] = models.CNN_TUMOR_MULTICLASS
sys.modules['Binary_CNN.models.HYBRID_MODEL'] = models.HYBRID_MODEL
sys.modules['Binary_CNN.utils_binary'] = utils_binary
sys.modules['Binary_CNN.utils_binary.findConv2dOutShape'] = utils_binary.findConv2dOutShape

from models.CNN_TUMOR_MULTICLASS import TumorClassifier
from models.HYBRID_MODEL import HybridTumorClassifier
from models.CNN_TUMOR import CNN_TUMOR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Angular frontend

# Model paths
BINARY_MODEL_PATH = 'Brain_Tumor_model.pt'
HYBRID_MODEL_PATH = 'best_hybrid_model_hybrid_concat.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models at startup
binary_model = None
hybrid_model = None
dicom_processor = DicomProcessor()

# Image transformations (same as training)
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformations for hybrid model (same as multiclass: 224x224)
hybrid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels for binary model
BINARY_CLASS_LABELS = {
    0: 'no_tumor',
    1: 'tumor'
}

# Class labels for hybrid model (tumor types)
HYBRID_LABELS = ['glioma', 'meningioma', 'pituitary']


def load_models():
    """Load both binary and hybrid models"""
    global binary_model, hybrid_model
    try:
        device = torch.device(DEVICE)

        # Load binary model (saved as full model)
        if not os.path.exists(BINARY_MODEL_PATH):
            raise FileNotFoundError(f"Binary model not found: {BINARY_MODEL_PATH}")

        binary_model = torch.load(BINARY_MODEL_PATH, map_location=device, weights_only=False)
        binary_model.to(device)
        binary_model.eval()
        logger.info(f"Binary model loaded successfully on {DEVICE}")

        # Load hybrid model (saved as state dict)
        if not os.path.exists(HYBRID_MODEL_PATH):
            raise FileNotFoundError(f"Hybrid model not found: {HYBRID_MODEL_PATH}")

        # Instantiate the model architecture
        hybrid_model = HybridTumorClassifier(num_classes=3, input_size=224, fusion_type='concat')

        # Load the checkpoint
        checkpoint = torch.load(HYBRID_MODEL_PATH, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Training checkpoint format with model_state_dict key
                hybrid_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded hybrid model from training checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
            elif 'state_dict' in checkpoint:
                # Checkpoint with state_dict key
                hybrid_model.load_state_dict(checkpoint['state_dict'])
            else:
                # It's a state dict directly
                hybrid_model.load_state_dict(checkpoint)
        else:
            # It might be the full model
            hybrid_model = checkpoint

        hybrid_model.to(device)
        hybrid_model.eval()
        logger.info(f"Hybrid model loaded successfully on {DEVICE}")

        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def predict_image(image):
    try:
        device = torch.device(DEVICE)
        binary_model.eval()
        image_tensor = test_transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            binary_outputs = binary_model(image_tensor)
            binary_probabilities = torch.exp(binary_outputs)
            predicted_class = torch.argmax(binary_probabilities, dim=1).item()
            binary_confidence = binary_probabilities[0][predicted_class].item()
        logger.info(f"Binary prediction: {BINARY_CLASS_LABELS[predicted_class]} "
                    f"(confidence: {binary_confidence:.4f})")
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
        if predicted_class == 1:  # Tumor detected
            hybrid_model.eval()
            hybrid_tensor = hybrid_transform(image).unsqueeze(0)
            hybrid_tensor = hybrid_tensor.to(device)

            with torch.no_grad():
                hybrid_outputs = hybrid_model(hybrid_tensor)
                hybrid_probabilities = torch.softmax(hybrid_outputs, dim=1)
                tumor_type_probs = {
                    label: float(hybrid_probabilities[0][i].item())
                    for i, label in enumerate(HYBRID_LABELS)
                }
                tumor_type = max(tumor_type_probs.items(), key=lambda x: x[1])
                logger.info(f"Tumor type prediction: {tumor_type[0]}"
                            f" (confidence: {tumor_type[1]:.4f})")
                logger.info(f"Tumor type probabilities: {tumor_type_probs}")
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
        'hybrid_model_loaded': hybrid_model is not None,
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
        if binary_model is None or hybrid_model is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded'
            }), 500

        image = None

        # Handle multipart/form-data upload
        if 'file' in request.files:
            file = request.files['file']

            # Log file details for debugging
            logger.info("="*60)
            logger.info("📥 File Upload Request Received")
            logger.info(f"Filename: {file.filename}")
            logger.info(f"Content-Type: {file.content_type}")
            logger.info(f"Content-Length: {request.content_length}")
            logger.info("="*60)

            if file.filename == '':
                logger.error("❌ No filename provided")
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400

            # Check if it's a DICOM file
            is_dicom = file.filename.lower().endswith(('.dcm', '.dicom'))
            logger.info(f"🔍 File type detection: is_dicom={is_dicom}")

            # IMPORTANT: Reset stream position to beginning (for Spring Boot compatibility)
            try:
                file.stream.seek(0)
                logger.debug("✓ Stream position reset to beginning")
            except Exception as seek_error:
                logger.warning(f"⚠️  Could not seek stream: {seek_error}")

            # Read image bytes from uploaded file
            file_bytes = file.read()
            logger.info(f"📊 File bytes read: {len(file_bytes)} bytes")

            # Validate that we got some data
            if len(file_bytes) == 0:
                logger.error("❌ Empty file received (0 bytes)")
                return jsonify({
                    'success': False,
                    'error': 'Empty file received'
                }), 400

            # Log first bytes for debugging
            if len(file_bytes) >= 16:
                first_bytes_hex = file_bytes[:16].hex()
                logger.info(f"🔬 First 16 bytes (hex): {first_bytes_hex}")

                # Detect format from magic bytes
                if file_bytes.startswith(b'\xff\xd8\xff'):
                    logger.info("✅ Valid JPEG signature detected (FF D8 FF)")
                elif file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                    logger.info("✅ Valid PNG signature detected")
                elif file_bytes.startswith(b'BM'):
                    logger.info("✅ Valid BMP signature detected")
                # Check for DICOM signature at byte 128 (after 128-byte preamble)
                elif len(file_bytes) >= 132 and file_bytes[128:132] == b'DICM':
                    logger.info("✅ Valid DICOM signature detected (DICM at byte 128)")
                else:
                    logger.warning(f"⚠️  Unknown file format signature")

            # Check if file is all zeros (corrupted/empty)
            # BUT: DICOM files have 128 zero bytes as preamble, so we need to check more carefully
            # Check if the entire file is zeros (not just the beginning)
            if len(file_bytes) >= 200:
                # For DICOM, check if bytes 128-132 contain "DICM"
                if file_bytes[128:132] == b'DICM':
                    # Valid DICOM preamble, not an error
                    logger.debug("✓ DICOM preamble (128 zero bytes) detected - this is valid")
                # Check if entire file is zeros (corrupted upload)
                elif file_bytes[:200] == b'\x00' * 200:
                    logger.error("❌ File contains only zero bytes - Spring Boot multipart issue!")
                    logger.error("💡 Java side needs to use ByteArrayResource instead of MultipartFile")
                    return jsonify({
                        'success': False,
                        'error': 'Received file contains only zero bytes. Use ByteArrayResource in Java Spring Boot.'
                    }), 400

            if is_dicom:
                # Process DICOM file
                logger.info("🏥 Processing DICOM file...")
                temp_path = 'temp_dicom_upload.dcm'
                try:
                    # Save to temp file
                    with open(temp_path, 'wb') as f:
                        f.write(file_bytes)
                    logger.info(f"✓ DICOM file saved to {temp_path}")

                    # Read with pydicom
                    logger.info("📖 Reading DICOM with pydicom...")
                    pixel_array = dicom_processor.read_dicom_file(temp_path)

                    if pixel_array is None:
                        logger.error("❌ pydicom failed to read pixel data")
                        return jsonify({
                            'success': False,
                            'error': 'Error reading DICOM file - no pixel data found'
                        }), 400

                    logger.info(f"✅ DICOM pixel array extracted: shape={pixel_array.shape}, dtype={pixel_array.dtype}")

                    # Convert to PIL Image
                    if len(pixel_array.shape) == 2:
                        # Grayscale DICOM
                        logger.info("🎨 Converting grayscale DICOM to RGB...")
                        image = Image.fromarray(pixel_array).convert('RGB')
                    else:
                        # Already RGB/color
                        logger.info("🎨 Converting color DICOM to RGB...")
                        image = Image.fromarray(pixel_array).convert('RGB')

                    logger.info(f"✅ DICOM converted to PIL Image: size={image.size}, mode={image.mode}")

                except Exception as dicom_error:
                    logger.error(f"❌ DICOM processing error: {dicom_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return jsonify({
                        'success': False,
                        'error': f'Error processing DICOM file: {str(dicom_error)}'
                    }), 400
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        logger.debug(f"🗑️  Cleaned up temp file: {temp_path}")
            else:
                # Process normal image (JPG, PNG, etc.)
                logger.info("🖼️  Processing regular image file...")
                try:
                    # Create BytesIO object from bytes
                    img_io = io.BytesIO(file_bytes)

                    # Try to open image
                    try:
                        img = Image.open(img_io)
                        logger.info(f"✅ Image opened: format={img.format}, size={img.size}, mode={img.mode}")
                    except Exception as first_error:
                        logger.warning(f"⚠️  First attempt failed: {first_error}")

                        # Reset stream position
                        img_io.seek(0)

                        # Log magic bytes for debugging
                        header = file_bytes[:16]
                        logger.info(f"🔬 File header (hex): {header.hex()}")

                        # Detect format from magic bytes
                        if file_bytes.startswith(b'\xff\xd8\xff'):
                            logger.info("📷 JPEG format detected from magic bytes")
                        elif file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                            logger.info("🖼️  PNG format detected from magic bytes")
                        elif file_bytes.startswith(b'BM'):
                            logger.info("🎨 BMP format detected from magic bytes")

                        # Try saving to temp file and reopening (fallback)
                        logger.info("🔄 Trying fallback method (temp file)...")
                        temp_img_path = 'temp_upload_img.tmp'
                        try:
                            with open(temp_img_path, 'wb') as f:
                                f.write(file_bytes)
                            logger.debug(f"✓ Saved to temp file: {temp_img_path}")

                            img = Image.open(temp_img_path)
                            logger.info(f"✅ Image opened from temp file: format={img.format}, size={img.size}, mode={img.mode}")

                            # Clean up temp file
                            os.remove(temp_img_path)
                            logger.debug("🗑️  Temp file cleaned up")
                        except Exception as temp_error:
                            if os.path.exists(temp_img_path):
                                os.remove(temp_img_path)
                            logger.error(f"❌ Temp file method also failed: {temp_error}")
                            raise Exception(f"Could not open image: {first_error}, temp file attempt: {temp_error}")

                    # Convert to RGB
                    if img.mode != 'RGB':
                        original_mode = img.mode
                        img = img.convert('RGB')
                        logger.info(f"🎨 Image converted from {original_mode} to RGB")
                    else:
                        logger.info("✅ Image already in RGB mode")

                    image = img
                    logger.info(f"✅ Final image ready: size={image.size}, mode={image.mode}")

                except Exception as img_error:
                    logger.error("="*60)
                    logger.error(f"❌ ERROR OPENING IMAGE: {img_error}")
                    logger.error("="*60)

                    # Save problematic file for debugging
                    temp_debug_path = 'temp_debug_image.bin'
                    with open(temp_debug_path, 'wb') as f:
                        f.write(file_bytes)

                    logger.error(f"📁 Saved problematic file to: {temp_debug_path}")
                    logger.error(f"📊 File size: {len(file_bytes)} bytes")
                    logger.error(f"🔬 First 32 bytes (hex): {file_bytes[:32].hex() if len(file_bytes) >= 32 else file_bytes.hex()}")

                    import traceback
                    logger.error("🔍 Full traceback:")
                    logger.error(traceback.format_exc())

                    return jsonify({
                        'success': False,
                        'error': f'Cannot open image file: {str(img_error)}'
                    }), 400

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

            try:
                image_bytes = base64.b64decode(image_data)
                logger.info(f"Base64 decoded: {len(image_bytes)} bytes")
            except Exception as b64_error:
                return jsonify({
                    'success': False,
                    'error': f'Invalid base64 encoding: {str(b64_error)}'
                }), 400

            # Check if it's DICOM by trying to read it
            is_dicom = False
            try:
                temp_path = 'temp_dicom_base64.dcm'
                with open(temp_path, 'wb') as f:
                    f.write(image_bytes)

                pixel_array = dicom_processor.read_dicom_file(temp_path)

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if pixel_array is not None:
                    is_dicom = True
                    # Convert to PIL Image
                    if len(pixel_array.shape) == 2:
                        image = Image.fromarray(pixel_array).convert('RGB')
                    else:
                        image = Image.fromarray(pixel_array).convert('RGB')
                    logger.info(f"DICOM from base64 loaded: {image.size}, mode: {image.mode}")
            except Exception as dicom_error:
                logger.debug(f"Not a DICOM file: {dicom_error}")
                is_dicom = False

            if not is_dicom:
                # Process as normal image
                try:
                    img_io = io.BytesIO(image_bytes)
                    img = Image.open(img_io)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    image = img
                    logger.info(f"Image from base64 loaded: {image.size}, mode: {image.mode}")
                except Exception as img_error:
                    return jsonify({
                        'success': False,
                        'error': f'Cannot open image from base64: {str(img_error)}'
                    }), 400

        else:
            return jsonify({
                'success': False,
                'error': 'Invalid request format. Send file or base64 encoded image'
            }), 400

        # Verify we have a valid image
        if image is None:
            logger.error("❌ Image is None after processing")
            return jsonify({
                'success': False,
                'error': 'Failed to load image'
            }), 400

        logger.info("="*60)
        logger.info("🤖 Starting AI Prediction")
        logger.info(f"Image ready: size={image.size}, mode={image.mode}")
        logger.info("="*60)

        # Make prediction
        result = predict_image(image)

        if result['success']:
            logger.info("="*60)
            logger.info("✅ PREDICTION SUCCESSFUL")
            logger.info(f"Result: {result.get('prediction')}")
            if result.get('has_tumor'):
                logger.info(f"Tumor Type: {result.get('tumor_type', 'N/A')}")
            logger.info("="*60)
            return jsonify(result), 200
        else:
            logger.error("="*60)
            logger.error("❌ PREDICTION FAILED")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            logger.error("="*60)
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Request error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict-with-segmentation', methods=['POST'])
def predict_with_segmentation():
    """
    Predict tumor and generate segmentation overlay with tumor dimensions.

    Expected request:
    - multipart/form-data with 'file' field containing image
    OR
    - JSON with 'image' field containing base64 encoded image

    Optional query params:
    - threshold: float (0.0-1.0) for CAM threshold (default 0.4)

    Returns:
    - JSON with prediction results + segmentation data:
        - overlay_image_base64: heatmap overlay image
        - contour_image_base64: image with tumor contoured in red
        - dimensions: {width_pixels, height_pixels, width_mm, height_mm, area_mm2, ...}
        - bounding_box: {x, y, width, height}
    """
    try:
        if binary_model is None or hybrid_model is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded'
            }), 500

        # Get threshold from query params
        threshold = float(request.args.get('threshold', 0.4))
        threshold = max(0.1, min(0.9, threshold))  # Clamp between 0.1 and 0.9

        image = None

        # Handle multipart/form-data upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400

            is_dicom = file.filename.lower().endswith(('.dcm', '.dicom'))

            try:
                file.stream.seek(0)
            except Exception:
                pass

            file_bytes = file.read()

            if len(file_bytes) == 0:
                return jsonify({'success': False, 'error': 'Empty file received'}), 400

            if is_dicom:
                temp_path = 'temp_dicom_seg.dcm'
                try:
                    with open(temp_path, 'wb') as f:
                        f.write(file_bytes)
                    pixel_array = dicom_processor.read_dicom_file(temp_path)
                    if pixel_array is None:
                        return jsonify({'success': False, 'error': 'Error reading DICOM file'}), 400
                    image = Image.fromarray(pixel_array).convert('RGB')
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                try:
                    img_io = io.BytesIO(file_bytes)
                    img = Image.open(img_io)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    image = img
                except Exception as img_error:
                    return jsonify({'success': False, 'error': f'Cannot open image: {str(img_error)}'}), 400

        # Handle JSON with base64 encoded image
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'success': False, 'error': 'No image data provided'}), 400

            image_data = data['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            try:
                image_bytes = base64.b64decode(image_data)
            except Exception as b64_error:
                return jsonify({'success': False, 'error': f'Invalid base64: {str(b64_error)}'}), 400

            try:
                img_io = io.BytesIO(image_bytes)
                img = Image.open(img_io)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image = img
            except Exception as img_error:
                return jsonify({'success': False, 'error': f'Cannot open image: {str(img_error)}'}), 400
        else:
            return jsonify({'success': False, 'error': 'Invalid request format'}), 400

        if image is None:
            return jsonify({'success': False, 'error': 'Failed to load image'}), 400

        logger.info(f"🔬 Starting segmentation prediction (threshold={threshold})")

        # Run full segmentation pipeline
        device = torch.device(DEVICE)
        result = process_tumor_segmentation(
            image_pil=image,
            binary_model=binary_model,
            hybrid_model=hybrid_model,
            device=device,
            binary_transform=test_transform,
            hybrid_transform=hybrid_transform,
            threshold=threshold
        )

        if result['success']:
            logger.info(f"✅ Segmentation complete: has_tumor={result['has_tumor']}")
            if result.get('segmentation'):
                dims = result['segmentation'].get('dimensions')
                if dims:
                    logger.info(f"📐 Tumor dimensions: {dims['width_mm']}mm x {dims['height_mm']}mm")
                    logger.info(f"📐 Tumor area: {dims['area_mm2']} mm²")
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Segmentation request error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


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
        if binary_model is None or hybrid_model is None:
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
                # Check if it's a DICOM file
                is_dicom = file.filename.lower().endswith(('.dcm', '.dicom'))

                image_bytes = file.read()

                if is_dicom:
                    # Process DICOM file
                    temp_path = f'temp_dicom_batch_{file.filename}'
                    with open(temp_path, 'wb') as f:
                        f.write(image_bytes)

                    pixel_array = dicom_processor.read_dicom_file(temp_path)

                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    if pixel_array is None:
                        results.append({
                            'success': False,
                            'filename': file.filename,
                            'error': 'Error reading DICOM file'
                        })
                        continue

                    # Convert to PIL Image
                    if len(pixel_array.shape) == 2:
                        image = Image.fromarray(pixel_array).convert('RGB')
                    else:
                        image = Image.fromarray(pixel_array).convert('RGB')
                else:
                    # Process normal image
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
