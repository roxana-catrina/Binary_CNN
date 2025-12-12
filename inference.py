import argparse
import torch
from PIL import Image
from torchvision import transforms
import os
import sys
import numpy as np
from dicom_processor import DicomProcessor

# No need to import CNN_TUMOR since we load the entire model object

# Transformări pentru inferență (aceleași ca la test)
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Labelurile claselor
CLA_label = {
    0: 'no_tumor',
    1: 'tumor'
}


def load_model(model_path, device):
    """Încarcă modelul salvat"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device(device)

    # Încarcă întregul model direct (salvat cu torch.save(model, path))
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    return model



def predict_image(image_path, model_path, device='cpu'):
    """Face predicția pentru o imagine"""
    # Verifică dacă imaginea există
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check if it's a DICOM file
    is_dicom = image_path.lower().endswith(('.dcm', '.dicom'))

    if is_dicom:
        # Process DICOM file
        dicom_processor = DicomProcessor()
        pixel_array = dicom_processor.read_dicom_file(image_path)
        if pixel_array is None:
            raise ValueError('Error reading DICOM file')

        # Convert pixel array to PIL Image
        if len(pixel_array.shape) == 2:
            # Grayscale DICOM
            image = Image.fromarray(pixel_array).convert('RGB')
        else:
            # Already RGB/color
            image = Image.fromarray(pixel_array)
    else:
        # Process normal image (JPG, PNG)
        image = Image.open(image_path).convert('RGB')

    # Apply transformations
    image_tensor = test_transform(image).unsqueeze(0)  # Add batch dimension

    # Încarcă modelul
    device = torch.device(device)
    model = load_model(model_path, device)

    # Face predicția
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.exp(outputs)  # Model returnează log_softmax
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence, probabilities[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Run inference on a brain scan image")
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', required=True, help='Path to saved model (.pth)')
    parser.add_argument('--device', '-d', default='cpu', choices=['cpu', 'cuda'], help='Device to use')

    args = parser.parse_args()

    try:
        predicted_class, confidence, all_probs = predict_image(
            args.image,
            args.model,
            args.device
        )

        print(f"\n{'=' * 50}")
        print(f"Image: {args.image}")
        print(f"{'=' * 50}")
        print(f"\nPrediction: {CLA_label[predicted_class]}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"\nAll probabilities:")
        for class_idx, prob in enumerate(all_probs):
            print(f"  {CLA_label[class_idx]}: {prob * 100:.2f}%")
        print(f"{'=' * 50}\n")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()