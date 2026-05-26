"""
Tumor Segmentation Module
Uses Grad-CAM on the hybrid model to highlight tumor regions and compute approximate dimensions.
Improved version: uses hybrid model for better localization, brain masking, and
focused region selection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

try:
    import cv2
except ImportError:
    cv2 = None


class GradCAM:
    """
    Grad-CAM implementation for tumor localization.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles = [h1, h2]

    def remove_hooks(self):
        """Remove registered hooks to avoid memory leaks."""
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor (1, C, H, W)
            target_class: class index to generate CAM for (None = predicted class)

        Returns:
            cam: numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()

        # Enable gradients for this forward pass
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


class GradCAMPlusPlus:
    """
    Grad-CAM++ for better localization of smaller regions.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles = [h1, h2]

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM++ weighting
        grads = self.gradients  # (1, C, H, W)
        acts = self.activations  # (1, C, H, W)

        # Alpha computation for Grad-CAM++
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        eps = 1e-7

        alpha_numer = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + eps
        alpha = alpha_numer / alpha_denom
        alpha = alpha * torch.relu(grads)  # Only positive gradients

        weights = alpha.sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def get_target_layer_for_binary_model(model):
    """Get the last convolutional layer from the binary CNN model."""
    return model.conv4


def get_target_layer_for_hybrid_model(model):
    """
    Get the last convolutional layer from the hybrid model's custom branch.
    This gives better spatial resolution for localization.
    """
    last_conv = None
    for module in model.custom_features.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def create_brain_mask(image_pil):
    """
    Create a mask of the brain region to exclude background from segmentation.
    Uses Otsu thresholding on the grayscale image.
    """
    if cv2 is None:
        return None

    img_array = np.array(image_pil)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu thresholding to separate brain from background
    _, brain_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up
    kernel = np.ones((7, 7), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)

    # Fill holes - keep only the largest connected component (the brain)
    contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        brain_mask = np.zeros_like(brain_mask)
        cv2.drawContours(brain_mask, [largest], -1, 255, -1)

    # Erode slightly to avoid edge artifacts
    kernel_small = np.ones((3, 3), np.uint8)
    brain_mask = cv2.erode(brain_mask, kernel_small, iterations=2)

    return brain_mask


def generate_tumor_overlay(image_pil, cam, threshold=0.5):
    """
    Generate an overlay image with the tumor region highlighted and contoured.
    Uses brain masking and focused region selection for better accuracy.
    """
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for segmentation. "
            "Install it with: pip install opencv-python"
        )

    img_width, img_height = image_pil.size

    # Resize CAM to match image dimensions
    cam_resized = cv2.resize(cam, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    # Apply brain mask to exclude background activations
    brain_mask = create_brain_mask(image_pil)
    if brain_mask is not None:
        cam_resized = cam_resized * (brain_mask / 255.0)
        # Re-normalize after masking
        if cam_resized.max() > 0:
            cam_resized = cam_resized / cam_resized.max()

    # Use adaptive threshold based on the CAM distribution
    # Focus on the top activations (top 20% of non-zero values)
    non_zero_values = cam_resized[cam_resized > 0.1]
    if len(non_zero_values) > 0:
        adaptive_threshold = max(threshold, np.percentile(non_zero_values, 70))
    else:
        adaptive_threshold = threshold

    # Create binary mask with the adaptive threshold
    binary_mask = (cam_resized >= adaptive_threshold).astype(np.uint8) * 255

    # Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Remove small noise regions
    kernel_small = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert PIL to OpenCV format
    img_cv = np.array(image_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # --- Create heatmap overlay (only in tumor region) ---
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Apply heatmap only inside the brain mask for cleaner visualization
    overlay = img_cv.copy()
    if brain_mask is not None:
        mask_3ch = cv2.merge([brain_mask, brain_mask, brain_mask]) / 255.0
        overlay = (img_cv * (1 - 0.35 * mask_3ch) + heatmap * 0.35 * mask_3ch).astype(np.uint8)
    else:
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    # --- Create contour image ---
    contour_img = img_cv.copy()

    tumor_dimensions = None
    bounding_box = None
    tumor_area_pixels = 0
    tumor_percentage = 0.0

    if contours:
        # Filter contours: keep only significant ones (> 1% of brain area)
        brain_area = cv2.countNonZero(brain_mask) if brain_mask is not None else (img_width * img_height)
        min_contour_area = brain_area * 0.01  # At least 1% of brain

        significant_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

        if not significant_contours:
            # If no significant contours, take the largest one anyway
            significant_contours = [max(contours, key=cv2.contourArea)]

        # Select the contour with the highest mean CAM activation (most likely tumor)
        best_contour = None
        best_mean_activation = 0

        for contour in significant_contours:
            # Create mask for this contour
            contour_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)

            # Compute mean CAM activation inside this contour
            mean_activation = cv2.mean(cam_resized, mask=contour_mask)[0]

            if mean_activation > best_mean_activation:
                best_mean_activation = mean_activation
                best_contour = contour

        if best_contour is not None:
            tumor_area_pixels = cv2.contourArea(best_contour)

            # Total image area
            total_area = img_width * img_height
            tumor_percentage = (tumor_area_pixels / total_area) * 100

            # Bounding box
            x, y, w, h = cv2.boundingRect(best_contour)
            bounding_box = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}

            # Draw filled semi-transparent highlight on the tumor region
            tumor_highlight = contour_img.copy()
            cv2.drawContours(tumor_highlight, [best_contour], -1, (0, 0, 255), -1)
            contour_img = cv2.addWeighted(contour_img, 0.7, tumor_highlight, 0.3, 0)

            # Draw contour (red, thick)
            cv2.drawContours(contour_img, [best_contour], -1, (0, 0, 255), 3)
            cv2.drawContours(overlay, [best_contour], -1, (0, 0, 255), 3)

            # Draw bounding box (green)
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add dimension text
            dim_text = f"{w}x{h} px"
            cv2.putText(contour_img, dim_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(overlay, dim_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Compute approximate real-world dimensions
            fov_mm = 240.0
            pixel_spacing_x = fov_mm / img_width
            pixel_spacing_y = fov_mm / img_height

            tumor_width_mm = w * pixel_spacing_x
            tumor_height_mm = h * pixel_spacing_y
            tumor_area_mm2 = tumor_area_pixels * pixel_spacing_x * pixel_spacing_y

            tumor_dimensions = {
                'width_pixels': int(w),
                'height_pixels': int(h),
                'width_mm': round(tumor_width_mm, 1),
                'height_mm': round(tumor_height_mm, 1),
                'area_pixels': int(tumor_area_pixels),
                'area_mm2': round(tumor_area_mm2, 1),
                'tumor_percentage': round(tumor_percentage, 2),
                'pixel_spacing_mm': round(pixel_spacing_x, 3)
            }

    # Convert back to PIL RGB
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    contour_pil = Image.fromarray(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))

    return {
        'overlay_image': overlay_pil,
        'contour_image': contour_pil,
        'mask': binary_mask,
        'dimensions': tumor_dimensions,
        'bounding_box': bounding_box,
        'tumor_area_pixels': int(tumor_area_pixels),
        'tumor_percentage': round(tumor_percentage, 2)
    }


def pil_to_base64(image_pil, format='PNG'):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image_pil.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def process_tumor_segmentation(image_pil, binary_model, hybrid_model, device,
                                binary_transform, hybrid_transform, threshold=0.5):
    """
    Full pipeline: detect tumor, generate segmentation overlay, compute dimensions.
    Uses the HYBRID model for Grad-CAM (better localization than binary model).
    """
    # Step 1: Binary prediction (tumor / no tumor)
    binary_tensor = binary_transform(image_pil).unsqueeze(0).to(device)

    binary_model.eval()
    with torch.no_grad():
        binary_outputs = binary_model(binary_tensor)
        binary_probabilities = torch.exp(binary_outputs)
        predicted_class = torch.argmax(binary_probabilities, dim=1).item()
        binary_confidence = binary_probabilities[0][predicted_class].item()

    result = {
        'success': True,
        'prediction': 'tumor' if predicted_class == 1 else 'no_tumor',
        'has_tumor': predicted_class == 1,
        'confidence': float(binary_confidence),
        'probabilities': {
            'no_tumor': float(binary_probabilities[0][0].item()),
            'tumor': float(binary_probabilities[0][1].item())
        }
    }

    if predicted_class == 1:  # Tumor detected
        # Step 2: Classify tumor type with hybrid model
        hybrid_tensor = hybrid_transform(image_pil).unsqueeze(0).to(device)

        hybrid_model.eval()
        with torch.no_grad():
            hybrid_outputs = hybrid_model(hybrid_tensor)
            hybrid_probabilities = torch.softmax(hybrid_outputs, dim=1)

        hybrid_labels = ['glioma', 'meningioma', 'pituitary']
        tumor_type_probs = {
            label: float(hybrid_probabilities[0][i].item())
            for i, label in enumerate(hybrid_labels)
        }
        tumor_type = max(tumor_type_probs.items(), key=lambda x: x[1])

        result['tumor_type'] = tumor_type[0]
        result['tumor_type_confidence'] = float(tumor_type[1])
        result['tumor_type_probabilities'] = tumor_type_probs

        # Step 3: Generate Grad-CAM++ from HYBRID model (better localization)
        # The hybrid model learned tumor-type-specific features = better spatial focus
        hybrid_tensor_grad = hybrid_transform(image_pil).unsqueeze(0).to(device)

        target_layer = get_target_layer_for_hybrid_model(hybrid_model)
        grad_cam = GradCAMPlusPlus(hybrid_model, target_layer)

        # Use the predicted tumor type class for Grad-CAM
        predicted_type_idx = hybrid_probabilities.argmax(dim=1).item()
        cam_hybrid = grad_cam.generate(hybrid_tensor_grad, target_class=predicted_type_idx)
        grad_cam.remove_hooks()

        # Also generate from binary model and combine for robustness
        binary_tensor_grad = binary_transform(image_pil).unsqueeze(0).to(device)
        target_layer_binary = get_target_layer_for_binary_model(binary_model)
        grad_cam_binary = GradCAMPlusPlus(binary_model, target_layer_binary)
        cam_binary = grad_cam_binary.generate(binary_tensor_grad, target_class=1)
        grad_cam_binary.remove_hooks()

        # Resize both CAMs to same size and combine
        img_w, img_h = image_pil.size
        cam_hybrid_resized = cv2.resize(cam_hybrid, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        cam_binary_resized = cv2.resize(cam_binary, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

        # Weighted combination: hybrid model gets more weight (better localization)
        # Multiply instead of average — this focuses on regions where BOTH models agree
        cam_combined = cam_hybrid_resized * 0.7 + cam_binary_resized * 0.3

        # Normalize
        if cam_combined.max() > 0:
            cam_combined = cam_combined / cam_combined.max()

        # Step 4: Generate overlay and compute dimensions
        segmentation_result = generate_tumor_overlay(image_pil, cam_combined, threshold=threshold)

        # Step 5: Convert images to base64 for API response
        result['segmentation'] = {
            'overlay_image_base64': pil_to_base64(segmentation_result['overlay_image']),
            'contour_image_base64': pil_to_base64(segmentation_result['contour_image']),
            'dimensions': segmentation_result['dimensions'],
            'bounding_box': segmentation_result['bounding_box'],
            'tumor_area_pixels': segmentation_result['tumor_area_pixels'],
            'tumor_percentage': segmentation_result['tumor_percentage']
        }

    return result
