from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Конфигурация приложения."""

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/face_recognition"

    # Face Recognition thresholds
    FACE_RECOGNITION_THRESHOLD: float = 0.35
    FACE_DETECTION_THRESHOLD: float = 0.5
    LIVENESS_THRESHOLD: float = 0.5

    # Face processing
    MIN_FACE_SIZE: int = 80
    MAX_EMBEDDING_DIM: int = 512
    EMBEDDING_AVG_METHOD: str = "mean"  # mean or median

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Door control
    DOOR_ENABLED: bool = False

    # CORS
    CORS_ORIGINS: str = "http://localhost,http://localhost:3000,http://localhost:8080"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Получить кэшированные настройки."""
    return Settings()
