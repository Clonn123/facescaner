"""
Кеш дверей и кандидатов в оперативке.
При старте загружает door_access_door из БД, обновляет каждые 15 минут.
Кеш кандидатов обновляется каждую минуту.
"""
import asyncio
import numpy as np
from typing import List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.database_models import DoorAccessDoor
from app.core.config import get_settings

settings = get_settings()

CACHE_TTL = 900  # 15 минут
CANDIDATES_TTL = 60  # 1 минута

_door_cache: dict[str, str] = {}  # camera_url -> door_id
_candidates_cache: List[Tuple[str, np.ndarray]] = []
_candidates_time: float = 0.0


async def refresh_door_cache(db: AsyncSession) -> int:
    """Загружает door_access_door из БД в память."""
    global _door_cache
    stmt = select(DoorAccessDoor).where(DoorAccessDoor.is_active == True)
    result = await db.execute(stmt)
    doors = result.scalars().all()

    mapping = {}
    for door in doors:
        if door.related_camera:
            mapping[door.related_camera] = door.id

    _door_cache = mapping
    print(f"[DoorCache] Refreshed: {len(mapping)} doors cached", flush=True)
    return len(mapping)


async def get_door_id_by_camera(camera_url: str) -> str | None:
    """Ищет door_id по camera_url в памяти."""
    return _door_cache.get(camera_url)


async def refresh_candidates_cache(db: AsyncSession) -> int:
    """Загружает кандидатов (user_id, embedding) из БД в память."""
    global _candidates_cache, _candidates_time
    import time
    from app.models.database_models import UserBiometric

    stmt = select(UserBiometric.user_id, UserBiometric.embedding).where(
        UserBiometric.embedding.isnot(None)
    )
    result = await db.execute(stmt)
    rows = result.all()

    candidates = []
    for user_id, embedding_list in rows:
        if embedding_list:
            embedding = np.array(embedding_list)
            candidates.append((user_id, embedding))

    _candidates_cache = candidates
    _candidates_time = time.time()
    print(f"[CandidatesCache] Refreshed: {len(candidates)} users cached", flush=True)
    return len(candidates)


async def get_candidates(db: AsyncSession) -> List[Tuple[str, np.ndarray]]:
    """Получить кандидатов из кеша или обновить."""
    import time
    now = time.time()

    if _candidates_cache and (now - _candidates_time) < CANDIDATES_TTL:
        return _candidates_cache

    await refresh_candidates_cache(db)
    return _candidates_cache


async def start_cache_refresh_loop():
    """Фоновая задача — обновляет кеш дверей каждые 15 минут, кандидатов каждую минуту."""
    from app.core.database import async_session_factory

    candidates_counter = 0

    while True:
        try:
            async with async_session_factory() as db:
                # Двери — каждые 15 минут
                await refresh_door_cache(db)

                # Кандидаты — каждую минуту
                candidates_counter += 1
                if candidates_counter >= 1:  # каждую итерацию (60 сек)
                    await refresh_candidates_cache(db)
                    candidates_counter = 0
        except Exception as e:
            print(f"[Cache] Refresh error: {e}", flush=True)
        await asyncio.sleep(CACHE_TTL)
