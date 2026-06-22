import base64
import io
import cv2
import numpy as np
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.face_detector import FaceDetector
from app.services.face_recognizer import FaceRecognizer
from app.services.liveness.anti_spoof_onnx import AntiSpoofONNX
from app.services.storage import StorageService
from app.models.schemas import (
    RegisterEmployeeRequest,
    RegisterEmployeeResponse,
    EmployeeResponse
)
from app.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/employees", tags=["Employees"])


def get_face_detector():
    """Глобальный экземпляр детектора."""
    if not hasattr(get_face_detector, "_detector"):
        detector = FaceDetector()
        detector.initialize()
        get_face_detector._detector = detector
    return get_face_detector._detector


def get_anti_spoof():
    """Глобальный экземпляр anti-spoof (singleton)."""
    if not hasattr(get_anti_spoof, "_spoof"):
        spoof = AntiSpoofONNX()
        get_anti_spoof._spoof = spoof
    return get_anti_spoof._spoof

def get_face_recognizer(detector=Depends(get_face_detector)):
    """Глобальный экземпляр распознавателя."""
    if not hasattr(get_face_recognizer, "_recognizer"):
        recognizer = FaceRecognizer(detector)
        get_face_recognizer._recognizer = recognizer
    return get_face_recognizer._recognizer


def get_anti_spoof():
    """Глобальный экземпляр anti-spoof (singleton)."""
    if not hasattr(get_anti_spoof, "_spoof"):
        spoof = AntiSpoofONNX()
        get_anti_spoof._spoof = spoof
    return get_anti_spoof._spoof


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
    - **image_base64**: Base64 фото для генерации embedding (необязательно)
    """
    detector = get_face_detector()
    recognizer = get_face_recognizer(detector)
    anti_spoof = get_anti_spoof()
    storage = StorageService(db)
    
    # Создаём запись сотрудника
    employee = await storage.register_employee(
        account_id=request.account_id,
        name=request.name,
        description=request.description
    )
    
    # Если предоставлено фото, генерируем embedding
    image = decode_image(request.image_base64)
    
    if image is not None:
        # Детекция лица
        faces = detector.detect_faces(image)
        if not faces:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the uploaded photo"
            )
        
        # Берём первое лицо
        face_info = faces[0]
        face_image = detector.extract_face(image, face_info)
        
        # Оценка качества
        quality = detector.assess_quality(face_image)
        if quality["quality"] in ["poor"]:
            raise HTTPException(
                status_code=400,
                detail=f"Photo quality is too low: {quality['quality']}"
            )
        
        # Anti-spoofing НЕ используется для регистрации
        # Anti-spoof нужен только для real-time camera pipeline
        
        # Генерация embedding
        embedding = recognizer.generate_embedding(image, face_info)
        if embedding is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate face embedding"
            )
        
        # Сохраняем embedding
        # Если у сотрудника уже есть embedding — усредняем
        if employee.embedding and any(e != 0.0 for e in employee.embedding):
            await storage.add_face_to_employee(
                employee_id=employee.employee_id,
                new_embedding=embedding
            )
        else:
            await storage.update_embedding(
                employee_id=employee.employee_id,
                embedding=embedding,
                faces_count=1
            )
        
        # Обновляем employee для ответа
        employee = await storage.get_employee(employee.employee_id)
        
        return RegisterEmployeeResponse(
            employee_id=employee.employee_id,
            account_id=request.account_id,
            name=request.name,
            faces_registered=employee.faces_registered,
            embedding_dimensions=len(embedding),
            message="Employee registered successfully" if employee.faces_registered == 1 
                    else f"Face added to existing employee (total: {employee.faces_registered} faces)"
        )
    
    # Если фото не предоставлено, возвращаем только информацию о регистрации
    return RegisterEmployeeResponse(
        employee_id=employee.employee_id,
        account_id=request.account_id,
        name=request.name,
        faces_registered=0,
        embedding_dimensions=0,
        message="Employee registered. Please upload a photo to complete registration."
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