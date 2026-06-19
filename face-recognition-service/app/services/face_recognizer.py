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
            face_info: Информация о лице (из detect_faces). Если None — берётся первое лицо.
            
        Returns:
            Numpy array embedding или None
        """
        try:
            # Используем detector.app, который уже загружен
            faces = self.detector.app.get(image)
            if faces:
                emb = faces[0].embedding
                # Нормализуем embedding
                norm = np.linalg.norm(emb)
                if norm > 0:
                    return emb / norm
                return emb
            return None
        except Exception as e:
            print(f"Error generating embedding: {e}")
            import traceback
            traceback.print_exc()
            return None

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
