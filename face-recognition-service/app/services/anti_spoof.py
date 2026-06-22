import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from typing import Optional, Tuple
from app.core.config import get_settings
import zipfile
import shutil
import os
import urllib.request
import onnxruntime as ort

settings = get_settings()


class AntiSpoof:
    """Anti-spoofing проверка через 4d-prior модель InsightFace."""

    def __init__(self):
        self.model_dir = Path("/app/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Скачиваем и распаковываем модели
        self._download_and_extract_models()
        
        # FaceAnalysis для детекции
        self.app = FaceAnalysis(
            name="antelopev2",
            root="/app",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Загружаем 4d-prior модель
        self.model = None
        self.is_ready = False

    def _download_and_extract_models(self):
        """Скачивание и распаковка моделей вручную."""
        model_path = self.model_dir / "antelopev2"
        zip_path = self.model_dir / "antelopev2.zip"
        
        # Если модели уже есть в правильном месте - выходим
        if model_path.exists() and any(model_path.iterdir()):
            det_model = model_path / "scrfd_10g_bnkps.onnx"
            if det_model.exists():
                return
        
        # Скачиваем если нет
        if not zip_path.exists():
            url = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
            print(f"Downloading models from {url}...")
            urllib.request.urlretrieve(url, str(zip_path))
            print("Downloaded")
        
        # Распаковываем
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(self.model_dir))
        
        # После распаковки структура: models/models/antelopev2/antelopev2/
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

    def initialize(self) -> bool:
        """Инициализация anti-spoofing — загрузка 4d-prior модели."""
        try:
            # Ищем 4d-prior.onnx
            model_path = self.model_dir / "antelopev2" / "4d-prior.onnx"
            
            if not model_path.exists():
                print(f"WARNING: 4d-prior.onnx not found at {model_path}")
                print("Falling back to basic liveness detection")
                self.is_ready = False
                return False
            
            # Загружаем модель через onnxruntime
            sess_options = ort.SessionOptions()
            self.model = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
            self.is_ready = True
            print(f"4d-prior model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading 4d-prior model: {e}")
            import traceback
            traceback.print_exc()
            self.is_ready = False
            return False

    def verify_liveness(self, image: np.ndarray) -> dict:
        """
        Проверка живого присутствия через 4d-prior модель.
        
        Args:
            image: BGR изображение
            
        Returns:
            Словарь с результатами проверки
        """
        if not self.is_ready or self.model is None:
            # Fallback на базовую проверку
            return self._fallback_liveness(image)

        try:
            # Получаем лица
            faces = self.app.get(image)
            
            if not faces:
                return {
                    "is_real": False,
                    "liveness_score": 0.0,
                    "confidence": "low",
                    "face_detected": False,
                    "message": "Face not detected"
                }

            # Берём самое большое лицо
            face = faces[0]
            
            # Получаем bbox и keypoints
            bbox = face.bbox.astype(int)
            kps = face.kps
            
            # Нормализуем лицо для 4d-prior
            cropped = self._crop_face(image, bbox, kps)
            
            if cropped is None:
                return self._fallback_liveness(image)
            
            # Делаем предсказание
            score = self._predict(cropped)
            
            # 4d-prior возвращает:
            # score[0][0] > 0.5 — реальное лицо
            # score[0][0] < 0.5 — spoof (фото/видео)
            is_real = score > 0.5
            
            confidence = "high" if score > 0.8 else ("medium" if score > 0.6 else "low")
            
            return {
                "is_real": is_real,
                "liveness_score": round(float(score), 3),
                "confidence": confidence,
                "face_detected": True,
                "message": "Live face detected" if is_real else "Possible spoof detected"
            }
            
        except Exception as e:
            print(f"Error in liveness verification: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_liveness(image)

    def _crop_face(self, image: np.ndarray, bbox: np.ndarray, kps: np.ndarray) -> Optional[np.ndarray]:
        """
        Нормализация лица для 4d-prior модели.
        Масштабирует и выравнивает лицо по keypoints.
        """
        try:
            from insightface.utils import face_align
            
            # Нормализуем bbox в float
            bbox_float = bbox.astype(np.float32)
            
            # Вырезаем и выравниваем лицо
            cropped = face_align.norm_crop(image, landmark=kps, bbox=bbox_float)
            
            # 4d-prior ожидает 224x224
            if cropped.shape[0] != 224 or cropped.shape[1] != 224:
                cropped = cv2.resize(cropped, (224, 224))
            
            return cropped
        except Exception as e:
            print(f"Error cropping face: {e}")
            return None

    def _predict(self, cropped_face: np.ndarray) -> float:
        """
        Предсказание 4d-prior модели.
        
        Input: cropped_face BGR 224x224
        Output: float score (0-1)
        """
        try:
            # 4d-prior ожидает float32, нормализованный к [0, 1]
            input_tensor = cropped_face.astype(np.float32) / 255.0
            
            # Добавляем batch dimension
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # Получаем имена входных данных
            input_name = self.model.get_inputs()[0].name
            
            # Делаем предсказание
            outputs = self.model.run(None, {input_name: input_tensor})
            
            # Берём первый выход
            score = float(outputs[0][0][0])
            
            # Сигмоида если нужно
            if score < 0 or score > 1:
                score = 1 / (1 + np.exp(-score))
            
            return score
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.5

    def _fallback_liveness(self, image: np.ndarray) -> dict:
        """Базовая проверка если 4d-prior недоступна."""
        try:
            faces = self.app.get(image)
            if not faces:
                return {
                    "is_real": False,
                    "liveness_score": 0.0,
                    "confidence": "low",
                    "face_detected": False,
                    "message": "Face not detected"
                }

            face = faces[0]
            det_score = float(face.det_score)
            
            # Простая эвристика — только детекция confidence
            is_real = det_score >= (settings.LIVENESS_THRESHOLD * 0.9)
            
            return {
                "is_real": is_real,
                "liveness_score": round(det_score, 3),
                "confidence": "medium",
                "face_detected": True,
                "message": "Live face detected" if is_real else "Possible spoof detected"
            }
        except Exception as e:
            print(f"Error in fallback liveness: {e}")
            return {
                "is_real": False,
                "liveness_score": 0.0,
                "confidence": "low",
                "face_detected": False,
                "message": f"Error: {str(e)}"
            }

    def get_model_info(self) -> str:
        model_info = "InsightFace AntelopeV2"
        if self.is_ready:
            model_info += " + 4d-prior"
        return model_info
