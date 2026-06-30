from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

from app.core.config import get_settings
from app.core.database import init_db, close_db
from app.api import users, recognize, liveness, camera
from app.models.schemas import HealthResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    print("Initializing Face Recognition Service...")
    await init_db()
    print("Database initialized")
    
    # Прогрев моделей: загружаем сразу, чтобы первый запрос не ждал
    try:
        from app.services.model_singletons import warmup
        warmup()
    except Exception as e:
        print(f"  Model loading error: {e}")
    
    # Кеш дверей: загружаем + запускаем фоновое обновление
    cache_task = None
    if settings.HR_API_KEY:
        try:
            from app.services.cache import refresh_door_cache, start_cache_refresh_loop
            from app.core.database import async_session_factory
            async with async_session_factory() as db:
                await refresh_door_cache(db)
            cache_task = asyncio.create_task(start_cache_refresh_loop())
            print("Door cache started")
        except Exception as e:
            print(f"Door cache init error: {e}")
    
    print("Service ready")
    yield
    
    if cache_task:
        cache_task.cancel()
    await close_db()
    print("Face Recognition Service stopped")


app = FastAPI(
    title="Face Recognition Service",
    description="Сервис распознавания лиц для HR-модуля",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение маршрутов
app.include_router(users.router, prefix="/api/v1")
app.include_router(recognize.router, prefix="/api/v1")
app.include_router(liveness.router, prefix="/api/v1")
app.include_router(camera.router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка работоспособности сервиса."""
    from app.services.model_singletons import get_detector, get_anti_spoof
    
    detector = get_detector()
    anti_spoof = get_anti_spoof()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        face_detector=detector.get_model_info(),
        database="connected",
        model_info={
            "detector": detector.get_model_info(),
            "anti_spoof": anti_spoof.get_model_info()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
