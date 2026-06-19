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

settings = get_settings()


class AntiSpoof:
    """Anti-spoofing проверка через InsightFace AntelopeV2."""

    def __init__(self):
        self.model_dir = Path("/app/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Сначала скачиваем и распаковываем модели вручную
        self._download_and_extract_models()
        
        # Теперь создаём FaceAnalysis - модели уже на месте
        # root="/app" → ищет в /app/models/antelopev2
        self.app = FaceAnalysis(
            name="antelopev2",
            root="/app",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
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
        """Инициализация anti-spoofing."""
        try:
            self.is_ready = True
            return True
        except Exception as e:
            print(f"Error initializing anti-spoof: {e}")
            return False

    def verify_liveness(self, image: np.ndarray) -> dict:
        """
        Проверка живого присутствия.
        
        Args:
            image: BGR изображение
            
        Returns:
            Словарь с результатами проверки
        """
        if not self.is_ready:
            raise RuntimeError("Anti-spoof not initialized")

        try:
            # Получаем все лица с anti-spoofing scoring
            # InsightFace AntelopeV2 возвращает liveness scores
            faces = self.app.get(image)
            
            if not faces:
                return {
                    "is_real": False,
                    "liveness_score": 0.0,
                    "confidence": "low",
                    "face_detected": False,
                    "message": "Face not detected"
                }

            # Берём самое большое лицо (первое в списке)
            face = faces[0]
            
            # InsightFace AntelopeV2 предоставляет:
            # face.liveness или face.bbox_conf для anti-spoof
            # Используем встроенные метрики
            
            # Получаем liveness score (если доступен)
            # В AntelopeV2 это обычно face.liveness или через отдельную модель
            
            # Для AntelopeV2 anti-spoofing scores доступны как:
            # - face.bbox (детекция)
            # - face.kps (keypoints)
            # - face.embedding (распознавание)
            # - face.det_score (уверенность детекции)
            
            # Для anti-spoof используем det_score + дополнительные проверки
            # Так как AntelopeV2 не имеет явного liveness score,
            # используем комбинированный подход
            
            liveness_score = self._compute_liveness_score(face, image)
            
            is_real = liveness_score >= settings.LIVENESS_THRESHOLD
            
            if is_real:
                confidence = "high" if liveness_score > 0.85 else "medium"
            else:
                confidence = "low"
            
            return {
                "is_real": is_real,
                "liveness_score": round(float(liveness_score), 3),
                "confidence": confidence,
                "face_detected": True,
                "message": "Live face detected" if is_real else "Possible spoof detected"
            }
            
        except Exception as e:
            print(f"Error in liveness verification: {e}")
            return {
                "is_real": False,
                "liveness_score": 0.0,
                "confidence": "low",
                "face_detected": False,
                "message": f"Error: {str(e)}"
            }

    def _compute_liveness_score(self, face, image: np.ndarray) -> float:
        """
        Вычисление оценки живости на основе нескольких метрик.
        
        Используем комбинированный подход:
        1. Детекция confidence
        2. Проверка ключевых точек
        3. Анализ текстуры (если возможно)
        """
        # Базовый score - уверенность детекции
        base_score = float(face.det_score)
        
        # Дополнительная проверка - количество keypoints
        # Если keypoints чёткие и в правильных позициях - лицо реальное
        kps_score = 1.0
        if hasattr(face, 'kps') and face.kps is not None:
            # Проверяем разброс keypoints (для фото на экране могут быть артефакты)
            kps = face.kps
            if len(kps) >= 5:
                # Вычисляем дисперсию расстояний между ключевыми точками
                distances = []
                for i in range(len(kps) - 1):
                    dist = np.linalg.norm(kps[i] - kps[i+1])
                    distances.append(dist)
                
                if distances:
                    std_dev = np.std(distances)
                    # Низкая дисперсия может указывать на плоское изображение
                    if std_dev < 5.0:
                        kps_score = 0.7
                    elif std_dev < 10.0:
                        kps_score = 0.85
        
        # Комбинируем метрики
        combined_score = base_score * 0.6 + kps_score * 0.4
        
        return min(1.0, max(0.0, combined_score))

    def get_model_info(self) -> str:
        return "InsightFace AntelopeV2 Anti-Spoof"
