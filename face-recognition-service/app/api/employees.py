import base64
import io
import cv2
import numpy as np
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.model_singletons import get_detector, get_anti_spoof
from app.services.face_recognizer import FaceRecognizer
from app.services.storage import StorageService
from app.models.schemas import (
    RegisterEmployeeRequest,
    RegisterEmployeeResponse,
    EmployeeResponse
)
from app.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/employees", tags=["Employees"])

def get_face_recognizer(detector=Depends(get_detector)):
    if not hasattr(get_face_recognizer, "_recognizer"):
        get_face_recognizer._recognizer = FaceRecognizer(detector)
    return get_face_recognizer._recognizer

def decode_image(image_base64: str) -> Optional[np.ndarray]:
    """Декодирование изображения из base64."""
    if not image_base64:
        return None
    
    # Удаляем data URL prefix если есть
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    
    image_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@router.post("/register", response_model=RegisterEmployeeResponse)
async def register_employee(
    request: RegisterEmployeeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Регистрация сотрудника.
    
    - **account_id**: Уникальный ID аккаунта в HR системе
    - **name**: Имя сотрудника (необязательно)
    - **description**: Описание (необязательно)
    - **image_base64**: Base64 фото (одно)
    - **images_base64**: Base64 фото (пачка, до 10) — усредняется в один embedding
    """
    detector = get_detector()
    recognizer = get_face_recognizer(detector)
    storage = StorageService(db)
    
    # Создаём запись сотрудника
    employee = await storage.register_employee(
        account_id=request.account_id,
        name=request.name,
        description=request.description
    )
    
    # Собираем все фото: image_base64 + images_base64
    raw_images = []
    if request.image_base64:
        raw_images.append(request.image_base64)
    if request.images_base64:
        raw_images.extend(request.images_base64)
    
    if not raw_images:
        return RegisterEmployeeResponse(
            employee_id=employee.employee_id,
            account_id=request.account_id,
            name=request.name,
            faces_registered=0,
            embedding_dimensions=0,
            message="Employee registered. Upload photos to complete registration."
        )
    
    # Генерируем embedding из каждого фото
    embeddings = []
    errors = []
    for i, img_b64 in enumerate(raw_images):
        image = decode_image(img_b64)
        if image is None:
            errors.append(f"photo {i+1}: invalid format")
            continue
        
        faces = detector.detect_faces(image)
        if not faces:
            errors.append(f"photo {i+1}: no face detected")
            continue
        
        face_info = faces[0]
        face_image = detector.extract_face(image, face_info)
        
        quality = detector.assess_quality(face_image)
        if quality["quality"] in ["poor"]:
            errors.append(f"photo {i+1}: quality too low ({quality['quality']})")
            continue
        
        embedding = recognizer.generate_embedding(image, face_info)
        if embedding is None:
            errors.append(f"photo {i+1}: embedding failed")
            continue
        
        embeddings.append(embedding)
    
    if not embeddings:
        raise HTTPException(
            status_code=400,
            detail=f"No valid photos. Errors: {'; '.join(errors)}"
        )
    
    # Усредняем embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
    
    # Сохраняем
    faces_count = employee.faces_registered + len(embeddings)
    await storage.update_embedding(
        employee_id=employee.employee_id,
        embedding=avg_embedding,
        faces_count=faces_count
    )
    
    employee = await storage.get_employee(employee.employee_id)
    
    msg_parts = [f"{len(embeddings)} photos processed"]
    if errors:
        msg_parts.append(f"{len(errors)} skipped: {'; '.join(errors[:3])}")
    
    return RegisterEmployeeResponse(
        employee_id=employee.employee_id,
        account_id=request.account_id,
        name=request.name,
        faces_registered=employee.faces_registered,
        embedding_dimensions=len(avg_embedding),
        message=". ".join(msg_parts)
    )

@router.get("/{employee_id}", response_model=EmployeeResponse)
async def get_employee(employee_id: str, db: AsyncSession = Depends(get_db)):
    """Получение информации о сотруднике."""
    storage = StorageService(db)
    employee = await storage.get_employee(employee_id)
    
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    return employee

@router.delete("/{employee_id}")
async def delete_employee(employee_id: str, db: AsyncSession = Depends(get_db)):
    """Удаление сотрудника."""
    storage = StorageService(db)
    deleted = await storage.delete_employee(employee_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    return {"message": "Employee deleted successfully"}