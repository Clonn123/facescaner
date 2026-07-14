import base64
import cv2
import uuid
import json
import asyncio
import numpy as np
import time
import urllib.request
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services.model_singletons import get_detector, get_anti_spoof
from app.services.face_recognizer import FaceRecognizer
from app.services.storage import StorageService
from app.services.cache import get_door_id_by_camera
from app.models.schemas import RecognizeRequest, RecognizeResponse
from app.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/recognize", tags=["Recognition"])


def get_face_recognizer(detector=Depends(get_detector)):
    """Singleton распознавателя лиц."""
    if not hasattr(get_face_recognizer, "_recognizer"):
        get_face_recognizer._recognizer = FaceRecognizer(detector)
    return get_face_recognizer._recognizer


def decode_image(request: RecognizeRequest) -> Optional[np.ndarray]:
    """Декодирование изображения из base64."""
    if request.image_base64:
        # Удаляем data URL prefix если есть
        if "," in request.image_base64:
            request.image_base64 = request.image_base64.split(",")[1]
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return None


@router.post("/", response_model=RecognizeResponse)
async def recognize_face(
    request: RecognizeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Распознавание лица.
    
    Пайплайн:
    1. Детекция лица (или использование переданного bbox)
    2. Anti-spoof проверка (реальное лицо или фото/video)
    3. Генерация embedding
    4. Поиск лучшего совпадения среди зарегистрированных
    5. Обновление времени последней авторизации
    """
    start_time = time.time()
    
    detector = get_detector()
    recognizer = get_face_recognizer(detector)
    anti_spoof = get_anti_spoof()
    storage = StorageService(db)
    
    # Декодирование изображения
    t0 = time.time()
    image = decode_image(request)
    if image is None:
        raise HTTPException(status_code=400, detail="No image provided")
    print(f"[Timing] decode: {(time.time()-t0)*1000:.0f}ms")
    
    # 1. Детекция лица (или используем переданный bbox)
    t0 = time.time()
    if request.bbox:
        face_info = {
            "bbox": request.bbox,
            "kps": [],
            "det_score": 1.0,
            "landmarks": None
        }
        faces = [face_info]
    else:
        faces = detector.detect_faces(image)
    print(f"[Timing] detect: {(time.time()-t0)*1000:.0f}ms, found {len(faces)}")
    
    if not faces:
        return RecognizeResponse(
            recognized=False,
            face_detected=False,
            confidence="low",
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    face_info = faces[0]
    x1, y1, x2, y2 = map(int, face_info["bbox"])
    
    # 2. Anti-spoof проверка
    t0 = time.time()
    result = anti_spoof.predict_from_bbox(image, (x1, y1, x2, y2))
    liveness_score = result["liveness_score"]
    is_live = result["is_real"]
    print(f"[Timing] anti-spoof: {(time.time()-t0)*1000:.0f}ms, live={is_live}, score={liveness_score:.3f}")
    
    if not is_live:
        return RecognizeResponse(
            recognized=False,
            face_detected=True,
            liveness_checked=True,
            is_live=False,
            liveness_score=liveness_score,
            confidence="low",
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    # 3. Генерация embedding
    t0 = time.time()
    embedding = recognizer.generate_embedding(image, face_info)
    print(f"[Timing] embedding: {(time.time()-t0)*1000:.0f}ms")
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to generate face embedding")
    
    # 4. Поиск среди зарегистрированных пользователей
    t0 = time.time()
    candidates = await storage.get_all_users_for_recognition()
    print(f"[Timing] db candidates: {(time.time()-t0)*1000:.0f}ms, count={len(candidates)}")
    
    if not candidates:
        return RecognizeResponse(
            recognized=False,
            face_detected=True,
            liveness_checked=True,
            is_live=True,
            liveness_score=liveness_score,
            confidence="low",
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    # 5. Поиск лучшего совпадения
    t0 = time.time()
    match = recognizer.find_best_match(embedding, candidates)
    print(f"[Timing] match: {(time.time()-t0)*1000:.0f}ms")
    
    if match:
        user_id, similarity = match
        
        # Обновляем время последней авторизации
        await storage.update_auth_info(user_id)
        
        user = await storage.get_user(user_id)

        # === Door Access: ищем дверь в кеше → POST на бэкенд ===
        print(f"[DoorAccess] camera_url={request.camera_url}, HR_API_KEY={'SET' if settings.HR_API_KEY else 'EMPTY'}")
        if request.camera_url and settings.HR_API_KEY:
            try:
                door_id = await get_door_id_by_camera(request.camera_url)

                if door_id:
                    _, jpeg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    photo_bytes = jpeg.tobytes()

                    MAX_DOOR_RETRIES = 5
                    DOOR_RETRY_DELAY = 1.0

                    for attempt in range(MAX_DOOR_RETRIES):
                        boundary = uuid.uuid4().hex
                        body = b""
                        for field, value in [("userId", user_id), ("doorId", door_id)]:
                            body += f"--{boundary}\r\n".encode()
                            body += f'Content-Disposition: form-data; name="{field}"\r\n\r\n'.encode()
                            body += f"{value}\r\n".encode()
                        body += f"--{boundary}\r\n".encode()
                        body += b'Content-Disposition: form-data; name="photo"; filename="frame.jpg"\r\n'
                        body += b"Content-Type: image/jpeg\r\n\r\n"
                        body += photo_bytes
                        body += f"\r\n--{boundary}--\r\n".encode()

                        req = urllib.request.Request(
                            f"{settings.BACKEND_API_BASE_URL}/door-access/open-door",
                            data=body,
                            headers={
                                "Content-Type": f"multipart/form-data; boundary={boundary}",
                                "hr-api-key": settings.HR_API_KEY,
                            },
                        )
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            resp_data = resp.read().decode()
                            print(f"[DoorAccess] Attempt {attempt+1}: door={door_id}, user={user_id}, status={resp.status}, body={resp_data}")

                            try:
                                resp_json = json.loads(resp_data)
                            except Exception:
                                resp_json = {}

                            if resp_json.get("cooldown"):
                                if attempt < MAX_DOOR_RETRIES - 1:
                                    print(f"[DoorAccess] Cooldown active, retrying in {DOOR_RETRY_DELAY}s...")
                                    await asyncio.sleep(DOOR_RETRY_DELAY)
                                    continue
                                else:
                                    print(f"[DoorAccess] Cooldown still active after {MAX_DOOR_RETRIES} attempts, giving up")
                            break
                else:
                    print(f"[DoorAccess] No door found for camera_url={request.camera_url}")
            except Exception as e:
                print(f"[DoorAccess] Error: {e}")
        
        print(f"[Timing] total: {(time.time()-start_time)*1000:.0f}ms")
        return RecognizeResponse(
            recognized=user is not None,
            user_id=user_id,
            name=user.name if user else None,
            similarity=float(similarity),
            confidence=recognizer.get_confidence_label(similarity),
            face_detected=True,
            liveness_checked=True,
            is_live=True,
            liveness_score=liveness_score,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    # Пользователь не найден
    return RecognizeResponse(
        recognized=False,
        face_detected=True,
        liveness_checked=True,
        is_live=True,
        liveness_score=liveness_score,
        confidence="low",
        processing_time_ms=(time.time() - start_time) * 1000
    )
