"""
Wrapper для anti-spoof модуля.
Совместим с старым API (verify_liveness), использует AntiSpoofONNX.
"""
import numpy as np
from app.core.config import get_settings
from app.services.liveness import AntiSpoofONNX

settings = get_settings()


class AntiSpoof:
    """
    Wrapper для AntiSpoofONNX с методом verify_liveness
    для совместимости с app/api/liveness.py.
    """

    def __init__(self):
        self._spoof = AntiSpoofONNX()

    def initialize(self):
        """Инициализация (пустая для совместимости)."""
        pass

    def verify_liveness(self, image: np.ndarray) -> dict:
        """
        Проверка живого присутствия.

        Args:
            image: BGR изображение с лицом (crop)

        Returns:
            dict с полями:
                - is_real: bool
                - liveness_score: float (0.0 - 1.0)
                - confidence: str (high/medium/low)
                - face_detected: bool
        """
        if not self._spoof.is_ready:
            return {
                "is_real": False,
                "liveness_score": 0.0,
                "confidence": "low",
                "face_detected": True,
            }

        # Отладка: выводим размер crop
        h, w = image.shape[:2]
        print(f"[DEBUG] AntiSpoof crop size: {w}x{h}")

        # Усреднение по 3 кадрам для стабильности
        liveness_score = self._spoof.predict_with_smoothing(image, frames_count=3)

        # Отладка: выводим score
        print(f"[DEBUG] AntiSpoof score: {liveness_score:.4f}, threshold: {settings.LIVENESS_THRESHOLD}")

        is_real = liveness_score >= settings.LIVENESS_THRESHOLD

        if liveness_score >= 0.8:
            confidence = "high"
        elif liveness_score >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "is_real": is_real,
            "liveness_score": float(liveness_score),
            "confidence": confidence,
            "face_detected": True,
        }

    def get_model_info(self) -> str:
        """Информация о модели."""
        if self._spoof.is_ready:
            return "minifasnet-onnx (loaded)"
        return "minifasnet-onnx (not loaded)"
