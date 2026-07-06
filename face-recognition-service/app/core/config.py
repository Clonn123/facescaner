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
    BACKEND_API_BASE_URL: str = "http://localhost:3000"
    HR_API_KEY: str = ""
    CAMERA_URL: str = "local"

    # Camera workers
    CAMERAS_ENABLED: bool = False

    # Camera worker pipeline
    # Как часто запускать детекцию лиц (каждые N кадров)
    DETECT_INTERVAL: int = 20
    # Как часто запускать anti-spoof проверку (каждые N кадров)
    SPOOF_INTERVAL: int = 3
    # Сколько подряд REAL ответов нужно для запуска распознавания
    REAL_WINDOW_SIZE: int = 4
    # Размер окна для определения движения (кадров)
    MOTION_WINDOW: int = 3
    # Порог движения: ниже этого значения = статичное лицо (фото/экран)
    MOTION_THRESHOLD: float = 1.0
    # Задержка между повторными попытками распознавания при ошибке (сек)
    RECOGNIZE_RETRY_DELAY: int = 5
    # Как часто обновлять список кандидатов из БД (сек)
    CANDIDATES_REFRESH: int = 30

    # CORS
    CORS_ORIGINS: str = "http://localhost,http://localhost:3000,http://localhost:8080"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # postgres:// → postgresql://
        if self.DATABASE_URL.startswith("postgres://"):
            self.DATABASE_URL = self.DATABASE_URL.replace("postgres://", "postgresql://", 1)
        # Добавляем asyncpg драйвер если нет
        if self.DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in self.DATABASE_URL:
            self.DATABASE_URL = self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)


@lru_cache()
def get_settings() -> Settings:
    """Получить кэшированные настройки."""
    return Settings()
