import asyncio
import base64
import cv2
import json
import time
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory
from app.services.model_singletons import get_detector, get_anti_spoof
from app.services.face_recognizer import FaceRecognizer
from app.models.database_models import UserBiometric
from app.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/ws", tags=["WebSocket"])


def get_face_recognizer(detector=Depends(get_detector)):
    """Singleton распознавателя лиц."""
    if not hasattr(get_face_recognizer, "_recognizer"):
        get_face_recognizer._recognizer = FaceRecognizer(detector)
    return get_face_recognizer._recognizer


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Декодирование изображения из base64."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


async def get_all_users(db: AsyncSession):
    """Получение всех пользователей для распознавания."""
    stmt = select(UserBiometric.user_id, UserBiometric.name, UserBiometric.embedding).where(
        UserBiometric.embedding.isnot(None)
    )
    result = await db.execute(stmt)
    rows = result.all()
    
    candidates = []
    for user_id, name, embedding_list in rows:
        if embedding_list:
            embedding = np.array(embedding_list)
            candidates.append((user_id, name, embedding))
    
    return candidates


@router.websocket("/camera")
async def camera_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint для постоянного видеопотока.
    
    Клиент отправляет: {"type": "frame", "image_base64": "..."}
    Сервер отвечает: {"type": "status", ...}
    """
    await websocket.accept()
    
    detector = get_detector()
    recognizer = get_face_recognizer(detector)
    anti_spoof = get_anti_spoof()
    
    # Буфер для temporal smoothing (5 кадров)
    liveness_buffer = []
    BUFFER_SIZE = 5
    
    # Кэш пользователей (обновляется раз в 60 секунд)
    cached_employees = None
    cache_ttl = 0
    cache_lock = asyncio.Lock()
    
    async def get_cached_employees():
        """Получение кэшированного списка пользователей."""
        nonlocal cached_employees, cache_ttl
        async with cache_lock:
            if cached_employees is None or time.time() > cache_ttl:
                async with async_session_factory() as db:
                    cached_employees = await get_all_users(db)
                    cache_ttl = time.time() + 60  # Кэш на 60 секунд
            return cached_employees
    
    try:
        while True:
            data = await websocket.receive_text()
            
            message = json.loads(data)
            
            # Принимаем только кадры
            if message.get("type") != "frame":
                continue
            
            start_time = time.time()
            
            # Декодируем изображение
            image = decode_base64_image(message["image_base64"])
            
            # 1. Детекция лица
            faces = detector.detect_faces(image)
            
            if not faces:
                await websocket.send_json({
                    "type": "status",
                    "face_detected": False,
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
                continue
            
            face_info = faces[0]
            
            # 2. Anti-spoof проверка (crop с 1.5x расширением)
            x1, y1, x2, y2 = map(int, face_info["bbox"])
            
            result = anti_spoof.predict_from_bbox(image, (x1, y1, x2, y2))
            liveness_score = result["liveness_score"]
            is_live = result["is_real"]
            
            print(f"[LIVENESS] score={liveness_score:.3f} is_real={is_live}")
            
            # Добавляем в буфер для сглаживания
            liveness_buffer.append(liveness_score)
            
            # Temporal smoothing: усредняем по последним N кадрам
            if len(liveness_buffer) > BUFFER_SIZE:
                liveness_buffer.pop(0)
            avg_liveness = float(np.mean(liveness_buffer))
            
            is_live = avg_liveness >= settings.LIVENESS_THRESHOLD
            
            if not is_live:
                await websocket.send_json({
                    "type": "status",
                    "face_detected": True,
                    "liveness_checked": True,
                    "is_live": False,
                    "liveness_score": avg_liveness,
                    "confidence": "low",
                    "recognized": False,
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
                continue
            
            # 3. Генерация embedding
            embedding = recognizer.generate_embedding(image, face_info)
            if embedding is None:
                continue
            
            # 4. Поиск среди пользователей
            candidates = await get_cached_employees()
            
            if not candidates:
                await websocket.send_json({
                    "type": "status",
                    "face_detected": True,
                    "liveness_checked": True,
                    "is_live": True,
                    "liveness_score": avg_liveness,
                    "confidence": "low",
                    "recognized": False,
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
                continue
            
            # 5. Поиск лучшего совпадения
            match = recognizer.find_best_match(embedding, candidates)
            
            if match:
                user_id, name, similarity = match
                
                await websocket.send_json({
                    "type": "status",
                    "face_detected": True,
                    "liveness_checked": True,
                    "is_live": True,
                    "liveness_score": avg_liveness,
                    "recognized": True,
                    "user_id": user_id,
                    "name": name,
                    "similarity": float(similarity),
                    "confidence": recognizer.get_confidence_label(similarity),
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
            else:
                await websocket.send_json({
                    "type": "status",
                    "face_detected": True,
                    "liveness_checked": True,
                    "is_live": True,
                    "liveness_score": avg_liveness,
                    "recognized": False,
                    "confidence": "low",
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()
