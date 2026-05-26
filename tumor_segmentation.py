"""
Tumor Segmentation Module
Detects and highlights tumor regions in brain MRI images using
image processing techniques focused on the brain parenchyma (excluding skull).
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None


def create_brain_parenchyma_mask(gray_img):
    """
    Create a mask of brain parenchyma ONLY (excluding skull, scalp, background).
    This is critical to avoid detecting skull edges as tumor.
    """
    h, w = gray_img.shape

    # Step 1: Basic foreground mask (brain + skull)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, foreground = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    # Keep largest component
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(gray_img)
    largest = max(contours, key=cv2.contourArea)
    outer_mask = np.zeros_like(gray_img)
    cv2.drawContours(outer_mask, [largest], -1, 255, -1)

    # Step 2: Erode significantly to remove skull/scalp (typically 5-15 pixels thick)
    # Use percentage-based erosion relative to image size
    erosion_size = max(7, int(min(h, w) * 0.04))  # ~4% of image size
    kernel_erode = np.ones((erosion_size, erosion_size), np.uint8)
    brain_mask = cv2.erode(outer_mask, kernel_erode, iterations=1)

    # Step 3: Clean up
    kernel_clean = np.ones((3, 3), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel_clean)

    return brain_mask


def segment_tumor(image_pil):
    """
    Segment tumor from brain MRI using intensity-based detection
    within the brain parenchyma (skull excluded).

    Strategy:
    1. Create brain parenchyma mask (exclude skull/scalp)
    2. Find bright regions within parenchyma (tumors enhance on MRI)
    3. Use connected components to find the best tumor candidate
    4. Score candidates by: compactness, brightness, distance from edge

    Returns dict with contour, mask, brain_mask or None
    """
    if cv2 is None:
        raise ImportError("opencv-python is required")

    img_array = np.array(image_pil)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Step 1: Get brain parenchyma mask (no skull)
    brain_mask = create_brain_parenchyma_mask(gray)
    brain_area = cv2.countNonZero(brain_mask)

    if brain_area < 100:
        return None

    # Step 2: Analyze intensity distribution within brain parenchyma
    brain_pixels = gray[brain_mask > 0].astype(np.float32)
    median_val = np.median(brain_pixels)
    # Use percentile-based threshold (top 25% brightest pixels in brain)
    p75 = np.percentile(brain_pixels, 75)
    p90 = np.percentile(brain_pixels, 90)

    # Bright region mask: pixels brighter than 75th percentile
    bright_mask = ((gray.astype(np.float32) > p75) &
                   (brain_mask > 0)).astype(np.uint8) * 255

    # Step 3: Morphological cleanup
    # Close small gaps within tumor
    kernel_close = np.ones((7, 7), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_close)
    # Remove small noise spots
    kernel_open = np.ones((5, 5), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_open)

    # Step 4: Find connected components and score them
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright_mask, connectivity=8)

    if num_labels <= 1:
        return None

    # Minimum tumor size: 3% of brain, maximum: 45%
    min_tumor_area = brain_area * 0.03
    max_tumor_area = brain_area * 0.45

    # Compute distance transform from brain edge (to penalize edge regions)
    dist_transform = cv2.distanceTransform(brain_mask, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    if max_dist > 0:
        dist_normalized = dist_transform / max_dist
    else:
        dist_normalized = np.zeros_like(dist_transform, dtype=np.float32)

    best_label = -1
    best_score = -1

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        # Size filter
        if area < min_tumor_area or area > max_tumor_area:
            continue

        # Get component properties
        comp_mask = (labels == label_id).astype(np.uint8)
        cx, cy = centroids[label_id]

        # Score 1: Compactness (circularity) - tumors are roughly round/oval
        comp_contours, _ = cv2.findContours(comp_mask * 255, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
        if not comp_contours:
            continue
        perimeter = cv2.arcLength(comp_contours[0], True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
        else:
            compactness = 0

        # Score 2: Mean intensity (brighter = more likely tumor)
        mean_intensity = cv2.mean(gray, mask=comp_mask)[0]
        intensity_score = mean_intensity / 255.0

        # Score 3: Distance from brain edge (prefer interior regions)
        mean_dist = cv2.mean(dist_normalized, mask=comp_mask)[0]

        # Score 4: Size relative to brain (prefer medium-sized regions)
        size_ratio = area / brain_area
        # Optimal size: 5-25% of brain
        if 0.05 <= size_ratio <= 0.25:
            size_score = 1.0
        elif size_ratio < 0.05:
            size_score = size_ratio / 0.05
        else:
            size_score = max(0, 1.0 - (size_ratio - 0.25) / 0.20)

        # Combined score
        score = (compactness * 0.25 +
                 intensity_score * 0.25 +
                 mean_dist * 0.30 +  # Distance from edge is most important
                 size_score * 0.20)

        if score > best_score:
            best_score = score
            best_label = label_id

    if best_label == -1:
        # No valid component found - try with lower threshold
        # Use pixels above median as candidates
        lower_mask = ((gray.astype(np.float32) > median_val + (p75 - median_val) * 0.5) &
                      (brain_mask > 0)).astype(np.uint8) * 255
        lower_mask = cv2.morphologyEx(lower_mask, cv2.MORPH_CLOSE, kernel_close)
        lower_mask = cv2.morphologyEx(lower_mask, cv2.MORPH_OPEN, kernel_open)

        contours_lower, _ = cv2.findContours(lower_mask, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours_lower
                 if min_tumor_area < cv2.contourArea(c) < max_tumor_area]

        if valid:
            # Pick the one with highest mean distance from edge
            best_contour = None
            best_dist_score = -1
            for c in valid:
                c_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(c_mask, [c], -1, 255, -1)
                mean_d = cv2.mean(dist_normalized, mask=c_mask)[0]
                if mean_d > best_dist_score:
                    best_dist_score = mean_d
                    best_contour = c

            if best_contour is not None:
                # Smooth contour
                epsilon = 0.015 * cv2.arcLength(best_contour, True)
                best_contour = cv2.approxPolyDP(best_contour, epsilon, True)
                return {
                    'contour': best_contour,
                    'mask': lower_mask,
                    'brain_mask': brain_mask
                }

        return None

    # Step 5: Extract the best component as final tumor mask
    final_mask = ((labels == best_label) * 255).astype(np.uint8)

    # Dilate slightly to include tumor edges
    kernel_dilate = np.ones((3, 3), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel_dilate, iterations=1)
    final_mask = cv2.bitwise_and(final_mask, brain_mask)

    # Find final contour
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_contour = max(contours, key=cv2.contourArea)

    # Smooth contour
    epsilon = 0.015 * cv2.arcLength(best_contour, True)
    best_contour = cv2.approxPolyDP(best_contour, epsilon, True)

    # Final size check
    final_area = cv2.contourArea(best_contour)
    if final_area < min_tumor_area:
        return None

    return {
        'contour': best_contour,
        'mask': final_mask,
        'brain_mask': brain_mask
    }


# ============================================================================
# OVERLAY GENERATION
# ============================================================================

def generate_overlay_from_contour(image_pil, contour, brain_mask=None):
    """Generate overlay and contour images with tumor highlighted."""
    img_width, img_height = image_pil.size
    img_cv = np.array(image_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    tumor_area_pixels = cv2.contourArea(contour)
    total_area = img_width * img_height
    tumor_percentage = (tumor_area_pixels / total_area) * 100

    x, y, w, h = cv2.boundingRect(contour)
    bounding_box = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}

    # --- Overlay: semi-transparent red fill + contour ---
    overlay = img_cv.copy()
    tumor_fill = overlay.copy()
    cv2.drawContours(tumor_fill, [contour], -1, (0, 0, 255), -1)
    overlay = cv2.addWeighted(overlay, 0.65, tumor_fill, 0.35, 0)
    cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 3)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    dim_text = f"{w}x{h} px"
    cv2.putText(overlay, dim_text, (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Contour image: red border + light fill ---
    contour_img = img_cv.copy()
    contour_fill = contour_img.copy()
    cv2.drawContours(contour_fill, [contour], -1, (0, 0, 255), -1)
    contour_img = cv2.addWeighted(contour_img, 0.75, contour_fill, 0.25, 0)
    cv2.drawContours(contour_img, [contour], -1, (0, 0, 255), 3)
    cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(contour_img, dim_text, (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Approximate dimensions in mm
    fov_mm = 240.0
    px_x = fov_mm / img_width
    px_y = fov_mm / img_height

    tumor_dimensions = {
        'width_pixels': int(w),
        'height_pixels': int(h),
        'width_mm': round(w * px_x, 1),
        'height_mm': round(h * px_y, 1),
        'area_pixels': int(tumor_area_pixels),
        'area_mm2': round(tumor_area_pixels * px_x * px_y, 1),
        'tumor_percentage': round(tumor_percentage, 2),
        'pixel_spacing_mm': round(px_x, 3)
    }

    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    contour_pil = Image.fromarray(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))

    return {
        'overlay_image': overlay_pil,
        'contour_image': contour_pil,
        'dimensions': tumor_dimensions,
        'bounding_box': bounding_box,
        'tumor_area_pixels': int(tumor_area_pixels),
        'tumor_percentage': round(tumor_percentage, 2)
    }


# ============================================================================
# HEATMAP GENERATION (for the overlay view)
# ============================================================================

def generate_heatmap_overlay(image_pil, brain_mask):
    """
    Generate a heatmap overlay showing intensity distribution within the brain.
    Brighter areas (potential tumor) shown in warm colors.
    """
    img_cv = np.array(image_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    # Create intensity-based heatmap within brain
    brain_pixels = gray[brain_mask > 0].astype(np.float32)
    if len(brain_pixels) == 0:
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    p50 = np.percentile(brain_pixels, 50)
    p95 = np.percentile(brain_pixels, 95)

    # Normalize: map [p50, p95] to [0, 255]
    heatmap_raw = np.zeros_like(gray, dtype=np.float32)
    heatmap_raw = (gray.astype(np.float32) - p50) / max(p95 - p50, 1) * 255
    heatmap_raw = np.clip(heatmap_raw, 0, 255).astype(np.uint8)
    heatmap_raw[brain_mask == 0] = 0

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)

    # Blend with original (only inside brain)
    result = img_cv.copy()
    mask_3ch = cv2.merge([brain_mask, brain_mask, brain_mask]) / 255.0
    result = (img_cv * (1 - 0.4 * mask_3ch) + heatmap_colored * 0.4 * mask_3ch).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def pil_to_base64(image_pil, format='PNG'):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image_pil.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def process_tumor_segmentation(image_pil, binary_model, hybrid_model, device,
                                binary_transform, hybrid_transform, threshold=0.5):
    """
    Full pipeline: detect tumor, segment it, generate overlay, compute dimensions.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for segmentation")

    # Step 1: Binary prediction
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

    if predicted_class == 1:
        # Step 2: Classify tumor type
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

        # Step 3: Segment tumor
        logger.info("🔬 Running tumor segmentation...")
        seg_result = segment_tumor(image_pil)

        if seg_result is not None and seg_result['contour'] is not None:
            logger.info("✅ Segmentation successful")
            contour = seg_result['contour']
            brain_mask = seg_result['brain_mask']

            # Generate contour overlay
            overlay_result = generate_overlay_from_contour(image_pil, contour, brain_mask)

            # Generate heatmap overlay
            heatmap_img = generate_heatmap_overlay(image_pil, brain_mask)

            result['segmentation'] = {
                'overlay_image_base64': pil_to_base64(heatmap_img),
                'contour_image_base64': pil_to_base64(overlay_result['contour_image']),
                'dimensions': overlay_result['dimensions'],
                'bounding_box': overlay_result['bounding_box'],
                'tumor_area_pixels': overlay_result['tumor_area_pixels'],
                'tumor_percentage': overlay_result['tumor_percentage'],
                'method': 'image_processing'
            }
        else:
            logger.warning("⚠️ Segmentation failed - returning empty result")
            result['segmentation'] = _empty_segmentation(image_pil)

    return result


def _empty_segmentation(image_pil):
    """Return empty segmentation when detection fails."""
    return {
        'overlay_image_base64': pil_to_base64(image_pil),
        'contour_image_base64': pil_to_base64(image_pil),
        'dimensions': {
            'width_pixels': 0, 'height_pixels': 0,
            'width_mm': 0, 'height_mm': 0,
            'area_pixels': 0, 'area_mm2': 0,
            'tumor_percentage': 0, 'pixel_spacing_mm': 0
        },
        'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
        'tumor_area_pixels': 0,
        'tumor_percentage': 0,
        'method': 'none'
    }
