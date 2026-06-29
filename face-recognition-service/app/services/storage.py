import uuid
import numpy as np
from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.models.database_models import UserBiometric
from app.models.schemas import UserResponse


class StorageService:
    """Сервис хранения биометрических данных пользователей."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def register_user(self, user_id: str, name: Optional[str] = None) -> UserBiometric:
        """
        Регистрация или обновление пользователя.
        Если пользователь уже существует — обновляет имя и возвращает его.
        """
        stmt = select(UserBiometric).where(UserBiometric.user_id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if user:
            # Обновляем имя если нужно
            if name:
                user.name = name
            await self.db.commit()
            await self.db.refresh(user)
            return user
        
        # Создаём нового пользователя с пустым embedding
        user = UserBiometric(
            user_id=user_id,
            name=name,
            embedding=[0.0] * 512,  # инициализируем нулями
            faces_registered=0
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def update_embedding(self, user_id: str, embedding: np.ndarray,
                               faces_count: int = 1) -> Optional[UserBiometric]:
        """
        Обновление embedding пользователя.
        
        Args:
            user_id: ID пользователя
            embedding: Массив embedding (512 измерений)
            faces_count: Количество зарегистрированных лиц
        """
        stmt = select(UserBiometric).where(UserBiometric.user_id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        # Сохраняем embedding как массив floats
        user.embedding = embedding.tolist()
        user.embedding_dim = len(embedding)
        user.faces_registered = faces_count
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def add_face_to_user(self, user_id: str,
                                new_embedding: np.ndarray) -> Optional[UserBiometric]:
        """
        Добавление нового face embedding к существующему пользователю.
        Усредняет новый embedding с существующим.
        
        Args:
            user_id: ID пользователя
            new_embedding: Новый embedding
        """
        stmt = select(UserBiometric).where(UserBiometric.user_id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user or not user.embedding:
            return None
        
        # Получаем существующий embedding как numpy массив
        existing_emb = np.array(user.embedding, dtype=np.float32)
        
        # Усредняем два embedding
        averaged = (existing_emb + new_embedding) / 2.0
        
        user.embedding = averaged.tolist()
        user.faces_registered += 1
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def get_all_users_for_recognition(self) -> List[Tuple[str, np.ndarray]]:
        """
        Получение всех пользователей для распознавания.
        
        Returns:
            Список кортежей (user_id, embedding)
        """
        stmt = select(UserBiometric.user_id, UserBiometric.embedding).where(
            UserBiometric.embedding.isnot(None)
        )
        result = await self.db.execute(stmt)
        rows = result.all()
        
        candidates = []
        for user_id, embedding_list in rows:
            if embedding_list:
                embedding = np.array(embedding_list)
                candidates.append((user_id, embedding))
        
        return candidates

    async def get_user(self, user_id: str) -> Optional[UserResponse]:
        """Получение информации о пользователе."""
        stmt = select(UserBiometric).where(UserBiometric.user_id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        return UserResponse(
            user_id=user.user_id,
            name=user.name,
            faces_registered=user.faces_registered,
            last_auth_time=user.last_auth_time,
            last_auth_device=user.last_auth_device,
            created_at=user.created_at,
            updated_at=user.updated_at
        )

    async def delete_user(self, user_id: str) -> bool:
        """Удаление пользователя."""
        stmt = select(UserBiometric).where(UserBiometric.user_id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            return False
        
        await self.db.delete(user)
        await self.db.commit()
        return True

    async def update_auth_info(self, user_id: str, device: Optional[str] = None):
        """
        Обновление информации об авторизации.
        Вызывается при успешном распознавании.
        
        Args:
            user_id: ID пользователя
            device: Идентификатор устройства (опционально)
        """
        stmt = select(UserBiometric).where(UserBiometric.user_id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if user:
            user.last_auth_time = datetime.utcnow()
            if device:
                user.last_auth_device = device
            await self.db.commit()
