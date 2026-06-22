import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import urllib.request
from app.core.config import get_settings

settings = get_settings()

class AntiSpoofONNX:
    """
    Anti-spoofing на базе MiniFASNet (ONNX).
    Загружает модель при первом запуске: сначала проверяет локально,
    если нет — скачивает.
    """
    MODEL_URL = "https://raw.githubusercontent.com/facenox/face-antispoof-onnx/main/models/best/98.20/best_model.onnx"
    MODEL_DIR = Path("/app/models/antelopev2")
    MODEL_NAME = "minifasnet.onnx"

    def __init__(self):
        self.model_path = self.MODEL_DIR / self.MODEL_NAME
        self.session = None
        self.is_ready = False
        self._ensure_model()
        self._load_model()

    def _ensure_model(self):
        """Скачивает модель, если её нет локально."""
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
        """Загружает ONNX модель."""
        if not self.model_path.exists():
            return
        
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"]
            )
            self.is_ready = True
            print("MiniFASNet anti-spoof model loaded successfully.")
        except Exception as e:
            print(f"Error loading anti-spoof model: {e}")

    def predict(self, face_crop: np.ndarray) -> float:
        """
        Предсказание для одного кадра.
        Возвращает вероятность класса REAL (0.0 - 1.0).
        
        face_crop — BGR изображение лица (из InsightFace extract_face).
        """
        if not self.is_ready or self.session is None:
            return 0.0

        try:
            # Отладка: выводим размер входного crop
            h, w = face_crop.shape[:2]
            print(f"[DEBUG] predict: input face_crop size: {w}x{h}")
            
            # 1. BGR -> RGB
            face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # 2. Resize с letterboxing (как в оригинальном preprocessing)
            model_size = 128
            h, w = face.shape[:2]
            ratio = model_size / max(h, w)
            scaled_h, scaled_w = int(h * ratio), int(w * ratio)
            face = cv2.resize(face, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            
            # Padding
            delta_w = model_size - scaled_w
            delta_h = model_size - scaled_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            face = cv2.copyMakeBorder(face, top, bottom, left, right,
                                      cv2.BORDER_REFLECT_101)
            
            # 3. Normalize [0, 1], transpose CHW
            face = face.transpose(2, 0, 1).astype(np.float32) / 255.0
            
            # 4. Batch dimension
            face = np.expand_dims(face, axis=0)

            # Inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: face})
            
            # Отладка: выводим raw logits
            logits = outputs[0][0]
            print(f"[DEBUG] predict: raw logits = {logits}")
            
            # Softmax на logits
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            print(f"[DEBUG] predict: probs = {probs}")
            
            # Класс 0 = spoof, класс 1 = real
            return float(probs[1]) 

        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def predict_with_smoothing(self, face_crop: np.ndarray, frames_count: int = 3) -> float:
        """
        Усреднение предсказания по N кадрам (имитация видеопотока для статики).
        """
        if not self.is_ready:
            print("[DEBUG] AntiSpoof: model not ready, returning 0.0")
            return 0.0
        
        # Проверка размера
        h, w = face_crop.shape[:2]
        if h == 0 or w == 0:
            print("[DEBUG] AntiSpoof: empty crop!")
            return 0.0
        
        scores = []
        for i in range(frames_count):
            score = self.predict(face_crop)
            scores.append(score)
            print(f"[DEBUG] AntiSpoof frame {i}: score={score:.4f}")
        
        result = float(np.mean(scores))
        print(f"[DEBUG] AntiSpoof final score (mean): {result:.4f}")
        return result