"""
Anti-spoofing module adapted from facenox/face-antispoof-onnx.
Uses MiniFASNet with correct preprocessing and logit threshold.
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import urllib.request
from app.core.config import get_settings
from app.services.liveness.preprocess import crop
from app.services.liveness.inference import process_with_logits

settings = get_settings()


class AntiSpoofONNX:
    """
    Anti-spoofing on base of MiniFASNet (ONNX).
    Uses the same preprocessing and inference as facenox/demo.py.
    """
    MODEL_URL = "https://raw.githubusercontent.com/facenox/face-antispoof-onnx/main/models/best/98.20/best_model.onnx"
    MODEL_DIR = Path("/app/models/antelopev2")
    MODEL_NAME = "minifasnet.onnx"

    def __init__(self):
        self.model_path = self.MODEL_DIR / self.MODEL_NAME
        self.session = None
        self.is_ready = False
        self.input_name = None
        
        # Logit threshold (corresponds to ~0.5 probability threshold)
        # logit_threshold = np.log(0.5 / (1 - 0.5)) = 0
        self.logit_threshold = 0.0
        
        self._ensure_model()
        self._load_model()

    def _ensure_model(self):
        """Download model if not found locally."""
        if self.model_path.exists():
            return
        print(f"Anti-spoof model not found at {self.model_path}. Downloading...")
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(self.MODEL_URL, str(self.model_path))
            print("MiniFASNet model downloaded successfully.")
        except Exception as e:
            print(f"Failed to download anti-spoof model: {e}")

    def _load_model(self):
        """Load ONNX model with GPU + graph optimization."""
        if not self.model_path.exists():
            return
        
        try:
            # Session options для оптимизации
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 0
            opts.inter_op_num_threads = 0
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.is_ready = True
            print("MiniFASNet anti-spoof model loaded successfully.")
        except Exception as e:
            print(f"Error loading anti-spoof model: {e}")
            # Fallback to CPU
            try:
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                self.is_ready = True
                print("Loaded on CPU (GPU not available).")
            except Exception as e2:
                print(f"Error loading anti-spoof model: {e2}")

    def predict(self, face_crop: np.ndarray, bbox: tuple = None) -> dict:
        """
        Predict liveness for a face crop.
        
        Args:
            face_crop: RGB image containing face (from InsightFace detection)
            bbox: Optional (x, y, x2, y2) bbox - will use full image if not provided
        
        Returns:
            dict with is_real, liveness_score, logit_diff
        """
        if not self.is_ready or self.session is None:
            return {"is_real": False, "liveness_score": 0.0, "logit_diff": 0.0}
        
        try:
            # Convert BGR to RGB if needed
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            else:
                img = face_crop
            
            # Use crop() function with expansion factor 1.5
            if bbox is not None:
                x, y, x2, y2 = bbox
            else:
                # If no bbox provided, use full image
                h, w = img.shape[:2]
                x, y, x2, y2 = 0, 0, w, h
            
            # Make square crop with 1.5x expansion (key to success!)
            face_square = crop(img, (x, y, x2, y2), bbox_expansion_factor=1.5)
            
            # Preprocess for model
            from app.services.liveness.inference import preprocess
            face_input = preprocess(face_square, model_img_size=128)
            face_input = np.expand_dims(face_input, axis=0)
            
            # Run inference
            output = self.session.run(None, {self.input_name: face_input})[0]
            logits = output[0]
            
            # Process with logit threshold
            result = process_with_logits(logits, self.logit_threshold)
            
            # Convert logit_diff to probability-like score for convenience
            # sigmoid(logit_diff) gives probability of being real
            exp_diff = np.exp(result["logit_diff"])
            prob = exp_diff / (1 + exp_diff)
            
            return {
                "is_real": result["is_real"],
                "liveness_score": float(prob),
                "logit_diff": result["logit_diff"],
                "status": result["status"]
            }

        except Exception as e:
            print(f"Error in anti-spoof prediction: {e}")
            import traceback
            traceback.print_exc()
            return {"is_real": False, "liveness_score": 0.0, "logit_diff": 0.0}

    def predict_from_bbox(self, image: np.ndarray, bbox: tuple) -> dict:
        """
        Predict liveness using provided bbox.
        
        Args:
            image: Full BGR image
            bbox: (x, y, x2, y2) face bounding box
        
        Returns:
            dict with is_real, liveness_score, etc.
        """
        return self.predict(image, bbox)

    def predict_with_smoothing(self, face_crop: np.ndarray, bbox: tuple = None, frames_count: int = 3) -> float:
        """
        Average prediction over multiple frames for stability.
        
        Args:
            face_crop: RGB face crop
            bbox: Optional bbox
            frames_count: Number of frames to average
        
        Returns:
            Averaged liveness score (0-1)
        """
        scores = []
        for _ in range(frames_count):
            result = self.predict(face_crop, bbox)
            scores.append(result["liveness_score"])
        
        return float(np.mean(scores))