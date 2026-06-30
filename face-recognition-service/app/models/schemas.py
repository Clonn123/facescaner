from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class FaceQuality(str, Enum):
    """Уровень качества лица."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class FaceQualityResponse(BaseModel):
    """Ответ с оценкой качества лица."""
    sharpness: float = Field(..., ge=0.0, le=1.0)    # резкость (0-1)
    brightness: float = Field(..., ge=0.0, le=1.0)   # яркость (0-1)
    size: tuple                                        # размер (ширина, высота)
    quality: FaceQuality                               # общая оценка


class FaceDetection(BaseModel):
    """Обнаруженное лицо."""
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")
    kps: List[List[float]]                             # ключевые точки лица
    det_score: float                                   # уверенность детекции
    landmarks: Optional[List[List[float]]] = None


class RegisterUserRequest(BaseModel):
    """Запрос на регистрацию пользователя."""
    user_id: str = Field(..., description="ID пользователя")
    name: Optional[str] = Field(None, description="Имя пользователя")
    image_base64: Optional[str] = Field(None, description="Base64 фото (одно)")
    images_base64: Optional[List[str]] = Field(None, description="Base64 фото (пачка, до 10)")


class RegisterUserResponse(BaseModel):
    """Ответ на регистрацию пользователя."""
    user_id: str
    name: Optional[str]
    faces_registered: int       # количество зарегистрированных лиц
    embedding_dimensions: int   # размерность embedding
    message: str


class RecognizeRequest(BaseModel):
    """Запрос на распознавание."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL изображения")
    bbox: Optional[List[float]] = Field(None, description="[x1, y1, x2, y2] конкретного лица")
    camera_url: Optional[str] = Field(None, description="URL камеры (RTSP или 'local') для определения двери")


class RecognizeResponse(BaseModel):
    """Ответ на распознавание."""
    recognized: bool                                     # распознан или нет
    user_id: Optional[str] = None                        # ID пользователя
    name: Optional[str] = None                           # имя
    similarity: Optional[float] = None                   # cosine similarity
    confidence: str                                      # high/medium/low
    face_detected: bool = False                          # было ли лицо
    liveness_checked: bool = False                       # проверялся ли liveness
    is_live: Optional[bool] = None                       # реальное лицо или spoof
    liveness_score: Optional[float] = None               # оценка liveness
    processing_time_ms: float                            # время обработки


class LivenessVerifyRequest(BaseModel):
    """Запрос на проверку живого присутствия."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL изображения")


class LivenessVerifyResponse(BaseModel):
    """Ответ на проверку живого присутствия."""
    is_real: bool                    # реальное лицо или spoof
    liveness_score: float            # оценка (0-1)
    confidence: str                  # high/medium/low
    face_detected: bool = False
    processing_time_ms: float


class UserResponse(BaseModel):
    """Информация о пользователе."""
    user_id: str
    name: Optional[str]
    faces_registered: int
    last_auth_time: Optional[datetime] = None     # время последней авторизации
    last_auth_device: Optional[str] = None        # устройство последней авторизации
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Ответ проверки здоровья сервиса."""
    status: str
    version: str
    face_detector: str
    database: str
    model_info: dict
