from sqlalchemy import Column, String, Float, Integer, DateTime, ARRAY, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class UserBiometric(Base):
    """Биометрия пользователя — хранит embedding лица и метаданные."""
    __tablename__ = "user_biometrics"

    # Уникальный ID пользователя (основной ключ)
    user_id = Column(UUID(as_uuid=False), primary_key=True, index=True)
    name = Column(String, nullable=True)
    
    # Embedding лица (512 измерений для ArcFace)
    embedding = Column(ARRAY(Float), nullable=False)
    embedding_dim = Column(Integer, default=512)
    
    # Метаданные
    faces_registered = Column(Integer, default=1)          # количество зарегистрированных фото
    last_auth_time = Column(DateTime(timezone=True), server_default=func.now())    # время последней авторизации
    last_auth_device = Column(UUID(as_uuid=False), server_default=func.gen_random_uuid())       # устройство последней авторизации
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DoorAccessDoor(Base):
    """Дверь — связывает камеру с дверью для door access."""
    __tablename__ = "door_access_door"

    id = Column(UUID(as_uuid=False), primary_key=True, index=True)
    name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    related_camera = Column(Text, nullable=True)  # RTSP URL камеры или "local"
