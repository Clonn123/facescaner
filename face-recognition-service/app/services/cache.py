"""
Кеш дверей в оперативке.
При старте загружает door_access_door из БД, обновляет каждые 15 минут.
"""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.database_models import DoorAccessDoor
from app.core.config import get_settings

settings = get_settings()

CACHE_TTL = 900  # 15 минут

_door_cache: dict[str, str] = {}  # camera_url -> door_id


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
    print(f"[DoorCache] Refreshed: {len(mapping)} doors cached")
    return len(mapping)


async def get_door_id_by_camera(camera_url: str) -> str | None:
    """Ищет door_id по camera_url в памяти."""
    return _door_cache.get(camera_url)


async def start_cache_refresh_loop():
    """Фоновая задача — обновляет кеш каждые 15 минут."""
    from app.core.database import async_session_factory

    while True:
        try:
            async with async_session_factory() as db:
                await refresh_door_cache(db)
        except Exception as e:
            print(f"[DoorCache] Refresh error: {e}")
        await asyncio.sleep(CACHE_TTL)
