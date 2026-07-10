import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from typing import Optional, List
from app.core.config import get_settings
import zipfile
import shutil

settings = get_settings()


class FaceDetector:
    """Детектор лиц на основе SCRFD через InsightFace."""

    def __init__(self):
        self.model_dir = Path("/app/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Скачиваем и распаковываем модели вручную
        self._download_and_extract_models()
        
        # Создаём FaceAnalysis с поддержкой OpenVINO
        try:
            self.app = FaceAnalysis(
                name="antelopev2",
                root="/app",
                providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            )
            print("FaceDetector: using OpenVINO", flush=True)
        except Exception:
            self.app = FaceAnalysis(
                name="antelopev2",
                root="/app",
                providers=["CPUExecutionProvider"]
            )
            print("FaceDetector: using CPU (OpenVINO not available)")
        self.app.prepare(ctx_id=0, det_size=(160, 160))
        self.is_ready = False

    def _download_and_extract_models(self):
        """Скачивание и распаковка моделей вручную."""
        model_path = self.model_dir / "antelopev2"
        zip_path = self.model_dir / "antelopev2.zip"
        
        # Если модели уже есть в правильном месте - выходим
        if model_path.exists() and any(model_path.iterdir()):
            det_model = model_path / "scrfd_10g_bnkps.onnx"
            if det_model.exists():
                print(f"Models found at {model_path}")
                return
        
        # Скачиваем если нет
        if not zip_path.exists():
            import urllib.request
            url = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
            print(f"Downloading models from {url}...")
            urllib.request.urlretrieve(url, str(zip_path))
            print("Downloaded")
        
        # Распаковываем
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(self.model_dir))
        
        # После распаковки zip создаёт вложенную структуру — исправляем
        wrong_path = self.model_dir / "models" / "antelopev2"
        if wrong_path.exists() and any(wrong_path.iterdir()):
            inner_path = wrong_path / "antelopev2"
            if inner_path.exists() and any(inner_path.iterdir()):
                for item in inner_path.iterdir():
                    shutil.move(str(item), str(wrong_path / item.name))
                shutil.rmtree(inner_path)
            
            correct_path = self.model_dir / "antelopev2"
            if correct_path.exists():
                shutil.rmtree(correct_path)
            wrong_path.rename(correct_path)
            
        print(f"Models extracted to {self.model_dir / 'antelopev2'}")

    def initialize(self) -> bool:
        """Инициализация детектора."""
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            self.app.get(test_img)
            self.is_ready = True
            return True
        except Exception as e:
            print(f"Error initializing face detector: {e}")
            return False

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """Детекция лиц на изображении."""
        if not self.is_ready:
            raise RuntimeError("Face detector not initialized")

        faces = self.app.get(image)
        return [
            {
                "bbox": face.bbox.tolist(),
                "kps": face.kps.tolist(),
                "det_score": float(face.det_score),
                "landmarks": face.landmark_68_68.tolist() if hasattr(face, 'landmark_68_68') and face.landmark_68_68 is not None else None
            }
            for face in faces
            if face.det_score >= settings.FACE_DETECTION_THRESHOLD
        ]

    def extract_face(self, image: np.ndarray, face_info: dict) -> Optional[np.ndarray]:
        """Извлечение области лица из изображения."""
        x1, y1, x2, y2 = map(int, face_info["bbox"])
        
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        return image[y1:y2, x1:x2]

    def assess_quality(self, face_image: np.ndarray) -> dict:
        """Оценка качества лица."""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 100.0)
        brightness = float(np.mean(gray)) / 255.0

        if sharpness > 0.7 and brightness > 0.3 and brightness < 0.8:
            quality = "excellent"
        elif sharpness > 0.5 and brightness > 0.2 and brightness < 0.9:
            quality = "good"
        elif sharpness > 0.3:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "sharpness": round(sharpness, 3),
            "brightness": round(brightness, 3),
            "size": (face_image.shape[1], face_image.shape[0]),
            "quality": quality
        }

    def get_model_info(self) -> str:
        return "SCRFD (InsightFace AntelopeV2)"
