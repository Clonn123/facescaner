import base64
import cv2
import numpy as np
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.model_singletons import get_detector, get_anti_spoof
from app.services.face_recognizer import FaceRecognizer
from app.services.storage import StorageService
from app.models.schemas import RecognizeRequest, RecognizeResponse
from app.models.database_models import Employee
from app.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/recognize", tags=["Recognition"])


def get_face_recognizer(detector=Depends(get_detector)):
    if not hasattr(get_face_recognizer, "_recognizer"):
        get_face_recognizer._recognizer = FaceRecognizer(detector)
    return get_face_recognizer._recognizer


def decode_image(request: RecognizeRequest) -> Optional[np.ndarray]:
    """Декодирование изображения из запроса."""
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
    
    - **image_base64**: Base64 encoded изображение
    - **image_url**: URL изображения (альтернатива)
    
    Возвращает информацию о распознанном сотруднике или сообщает об отсутствии совпадений.
    """
    start_time = time.time()
    
    detector = get_detector()
    recognizer = get_face_recognizer(detector)
    anti_spoof = get_anti_spoof()
    storage = StorageService(db)
    
    # Декодирование изображения
    image = decode_image(request)
    
    if image is None:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # 1. Детекция лица (или используем переданный bbox)
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
    
    if not faces:
        return RecognizeResponse(
            recognized=False,
            face_detected=False,
            confidence="low",
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    # Берём лицо (переданное или первое из детекции)
    face_info = faces[0]
    
    # 2. Anti-spoof using bbox
    x1, y1, x2, y2 = map(int, face_info["bbox"])
    
    result = anti_spoof.predict_from_bbox(image, (x1, y1, x2, y2))
    liveness_score = result["liveness_score"]
    is_live = result["is_real"]
    
    print(f"[LIVENESS] score={liveness_score:.3f} is_real={is_live}")
    
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
    
    # 3. Aligned face для recognition
    face_image = detector.extract_face(image, face_info)
    
    # 4. Генерация embedding (передаём face_info чтобы не детектить заново)
    embedding = recognizer.generate_embedding(image, face_info)
    if embedding is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate face embedding"
        )
        
    # 4. Поиск среди зарегистрированных сотрудников
    candidates = await storage.get_all_employees_for_recognition()
    
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
    match = recognizer.find_best_match(embedding, candidates)
    
    if match:
        employee_id, similarity = match
        
        # Получаем информацию о сотруднике
        stmt = select(Employee).where(Employee.employee_id == employee_id)
        result = await db.execute(stmt)
        employee = result.scalar_one_or_none()
        
        # Логирование
        await storage.log_liveness(
            employee_id=employee_id,
            account_id=employee.account_id if employee else None,
            liveness_score=liveness_score,
            is_real=True,
            source="api"
        )
        
        return RecognizeResponse(
            recognized=True,
            employee_id=employee_id,
            account_id=employee.account_id if employee else None,
            name=employee.name if employee else None,
            similarity=float(similarity),
            confidence=recognizer.get_confidence_label(similarity),
            face_detected=True,
            liveness_checked=True,
            is_live=True,
            liveness_score=liveness_score,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    # Сотрудник не найден
    await storage.log_liveness(
        employee_id=None,
        account_id=None,
        liveness_score=liveness_score,
        is_real=True,
        source="api"
    )
    
    return RecognizeResponse(
        recognized=False,
        face_detected=True,
        liveness_checked=True,
        is_live=True,
        liveness_score=liveness_score,
        confidence="low",
        processing_time_ms=(time.time() - start_time) * 1000
    )