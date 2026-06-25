import numpy as np
from typing import List, Optional, Tuple
from insightface.app import FaceAnalysis
from pathlib import Path
from app.services.face_detector import FaceDetector


class FaceRecognizer:
    """Распознавание лиц на основе ArcFace через InsightFace."""

    def __init__(self, detector: FaceDetector):
        self.detector = detector
        self.threshold = 0.35  # Cosine similarity threshold

    def generate_embedding(self, image: np.ndarray, face_info: dict = None) -> Optional[np.ndarray]:
        """
        Генерация embedding для лица.
        
        Args:
            image: BGR изображение (полный кадр)
            face_info: Информация о лице (из detect_faces). Если None — детектим заново.
            
        Returns:
            Numpy array embedding или None
        """
        try:
            if face_info is not None:
                faces = self.detector.app.get(image)
                if not faces:
                    return None
                # Находим лицо с максимальным IoU с переданным bbox
                target_bbox = face_info["bbox"]
                best_face = None
                best_iou = -1
                for face in faces:
                    fb = face.bbox.tolist()
                    score = self._iou(target_bbox, fb)
                    if score > best_iou:
                        best_iou = score
                        best_face = face
                if best_face is None or best_iou < 0.3:
                    return None
                emb = best_face.embedding
            else:
                faces = self.detector.app.get(image)
                if not faces:
                    return None
                emb = faces[0].embedding
            
            # Нормализуем embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                return emb / norm
            return emb
        except Exception as e:
            print(f"Error generating embedding: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _iou(self, box_a, box_b) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Вычисление косинусного сходства между двумя embedding.
        
        Args:
            emb1: Первый embedding
            emb2: Второй embedding
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        emb1 = emb1.astype(np.float64)
        emb2 = emb2.astype(np.float64)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def find_best_match(self, query_embedding: np.ndarray, 
                       candidates: List[Tuple[str, np.ndarray]]) -> Optional[Tuple[str, float]]:
        """
        Поиск лучшего совпадения среди кандидатов.
        
        Args:
            query_embedding: Запрос embedding
            candidates: Список кортежей (employee_id, embedding)
            
        Returns:
            Кортеж (employee_id, similarity) или None если нет совпадений
        """
        best_match = None
        best_similarity = -1.0
        
        for employee_id, candidate_emb in candidates:
            similarity = self.compute_similarity(query_embedding, candidate_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = employee_id
        
        if best_match and best_similarity >= self.threshold:
            return (best_match, best_similarity)
        
        return None

    def average_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Усреднение нескольких embedding.
        
        Args:
            embeddings: Список embedding
            
        Returns:
            Усреднённый embedding
        """
        if not embeddings:
            return np.zeros(512)
        
        return np.mean(embeddings, axis=0)

    def get_confidence_label(self, similarity: float) -> str:
        """Получить метку уверенности."""
        if similarity >= 0.6:
            return "high"
        elif similarity >= self.threshold:
            return "medium"
        else:
            return "low"

