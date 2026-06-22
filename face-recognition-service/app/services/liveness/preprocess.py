"""Preprocessing for anti-spoofing model - adapted from facenox/face-antispoof-onnx."""

import cv2
import numpy as np
from typing import Tuple


def crop(img: np.ndarray, bbox: Tuple, bbox_expansion_factor: float = 1.5) -> np.ndarray:
    """
    Extract square face crop from bbox with expansion.
    This is critical for anti-spoofing - needs face + background context.
    
    Args:
        img: RGB image
        bbox: (x, y, x2, y2) bbox coordinates
        bbox_expansion_factor: How much to expand the bbox (1.5 = 50% larger)
    
    Returns:
        Square RGB face crop
    """
    original_height, original_width = img.shape[:2]
    x, y, x2, y2 = bbox
    w = x2 - x
    h = y2 - y
    
    if w <= 0 or h <= 0:
        raise ValueError("Invalid bbox dimensions")
    
    # Square crop centered on face
    max_dim = max(w, h)
    center_x = x + w / 2
    center_y = y + h / 2
    
    x = int(center_x - max_dim * bbox_expansion_factor / 2)
    y = int(center_y - max_dim * bbox_expansion_factor / 2)
    crop_size = int(max_dim * bbox_expansion_factor)
    
    # Clamp to image bounds
    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(original_width, x + crop_size)
    crop_y2 = min(original_height, y + crop_size)
    
    # Calculate padding needed
    top_pad = int(max(0, -y))
    left_pad = int(max(0, -x))
    bottom_pad = int(max(0, (y + crop_size) - original_height))
    right_pad = int(max(0, (x + crop_size) - original_width))
    
    # Crop the region (may be smaller than crop_size if near edge)
    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    else:
        img = np.zeros((0, 0, 3), dtype=img.dtype)
    
    # Apply padding with reflection to get exact crop_size x crop_size
    result = cv2.copyMakeBorder(
        img, top_pad, bottom_pad, left_pad, right_pad,
        cv2.BORDER_REFLECT_101,
    )
    
    if result.shape[0] != crop_size or result.shape[1] != crop_size:
        raise ValueError(f"Crop size mismatch: expected {crop_size}x{crop_size}, got {result.shape[0]}x{result.shape[1]}")
    
    return result


def preprocess(img: np.ndarray, model_img_size: int = 128) -> np.ndarray:
    """
    Resize with letterboxing, normalize to [0,1], convert to CHW.
    Matches the preprocessing used during model training.
    
    Args:
        img: RGB image (face crop, ideally square)
        model_img_size: Target size (default 128)
    
    Returns:
        Preprocessed array: (3, model_img_size, model_img_size)
    """
    new_size = model_img_size
    old_size = img.shape[:2]
    
    # Letterboxing - resize maintaining aspect ratio
    ratio = float(new_size) / max(old_size)
    scaled_shape = tuple([int(x * ratio) for x in old_size])
    
    # Use appropriate interpolation
    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, (scaled_shape[1], scaled_shape[0]), interpolation=interpolation)
    
    # Pad to exact size
    delta_w = new_size - scaled_shape[1]
    delta_h = new_size - scaled_shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    
    # Normalize [0, 1], transpose to CHW
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    
    return img