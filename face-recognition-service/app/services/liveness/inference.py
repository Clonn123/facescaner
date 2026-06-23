"""Inference functions for anti-spoofing model."""

import cv2
import numpy as np
import onnxruntime as ort


def infer(faces, session, input_name, model_img_size=128, batch_size=1):
    """
    Run inference on a batch of face crops.
    
    Args:
        faces: List of RGB face crops (numpy arrays)
        session: ONNX Runtime InferenceSession
        input_name: Input tensor name
        model_img_size: Size to resize faces to (default 128)
        batch_size: Batch size for inference
    
    Returns:
        List of raw logits for each face
    """
    results = []
    
    for i in range(0, len(faces), batch_size):
        batch = faces[i:i+batch_size]
        
        # Preprocess batch
        batch_array = np.zeros((len(batch), 3, model_img_size, model_img_size), dtype=np.float32)
        for j, face in enumerate(batch):
            batch_array[j] = preprocess(face, model_img_size)
        
        # Run inference
        output = session.run(None, {input_name: batch_array})[0]
        results.extend(output)
    
    return results


def process_with_logits(logits, logit_threshold):
    """
    Process raw logits to determine real vs spoof.
    
    Args:
        logits: Raw logits from model [spoof_logit, real_logit]
        logit_threshold: Log-odds threshold for classification
    
    Returns:
        dict with is_real, status, logit_diff
    """
    real_logit = logits[0]
    spoof_logit = logits[1]
    logit_diff = real_logit - spoof_logit
    
    is_real = logit_diff >= logit_threshold
    
    if is_real:
        status = "REAL"
    else:
        status = "SPOOF"
    
    return {
        "is_real": is_real,
        "status": status,
        "logit_diff": float(logit_diff),
        "real_logit": float(real_logit),
        "spoof_logit": float(spoof_logit)
    }


def preprocess(img, model_img_size=128):
    """
    Preprocess a single face crop.
    
    Args:
        img: RGB face crop
        model_img_size: Target size
    
    Returns:
        Preprocessed array (3, model_img_size, model_img_size)
    """
    new_size = model_img_size
    old_size = img.shape[:2]
    
    ratio = float(new_size) / max(old_size)
    scaled_shape = tuple([int(x * ratio) for x in old_size])
    
    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, (scaled_shape[1], scaled_shape[0]), interpolation=interpolation)
    
    delta_w = new_size - scaled_shape[1]
    delta_h = new_size - scaled_shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    
    return img