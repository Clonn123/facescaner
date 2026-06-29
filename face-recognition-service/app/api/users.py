import base64
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
    RegisterUserRequest,
    RegisterUserResponse,
    UserResponse
)
from app.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/users", tags=["Users"])


def get_face_recognizer(detector=Depends(get_detector)):
    """Singleton распознавателя лиц."""
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


@router.post("/register", response_model=RegisterUserResponse)
async def register_user(
    request: RegisterUserRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Регистрация пользователя.
    
    - **user_id**: Уникальный ID пользователя
    - **name**: Имя (необязательно)
    - **image_base64**: Base64 фото (одно)
    - **images_base64**: Base64 фото (пачка, до 10) — усредняется в один embedding
    """
    detector = get_detector()
    recognizer = get_face_recognizer(detector)
    storage = StorageService(db)
    
    # Создаём запись пользователя
    user = await storage.register_user(
        user_id=request.user_id,
        name=request.name
    )
    
    # Собираем все фото: image_base64 + images_base64
    raw_images = []
    if request.image_base64:
        raw_images.append(request.image_base64)
    if request.images_base64:
        raw_images.extend(request.images_base64)
    
    # Если фото нет — просто регистрируем без embedding
    if not raw_images:
        return RegisterUserResponse(
            user_id=user.user_id,
            name=request.name,
            faces_registered=0,
            embedding_dimensions=0,
            message="User registered. Upload photos to complete registration."
        )
    
    # Генерируем embedding из каждого фото
    embeddings = []
    errors = []
    for i, img_b64 in enumerate(raw_images):
        image = decode_image(img_b64)
        if image is None:
            errors.append(f"photo {i+1}: invalid format")
            continue
        
        # Детекция лица
        faces = detector.detect_faces(image)
        if not faces:
            errors.append(f"photo {i+1}: no face detected")
            continue
        
        face_info = faces[0]
        face_image = detector.extract_face(image, face_info)
        
        # Проверка качества
        quality = detector.assess_quality(face_image)
        if quality["quality"] in ["poor"]:
            errors.append(f"photo {i+1}: quality too low ({quality['quality']})")
            continue
        
        # Генерация embedding
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
    
    # Усредняем embeddings и нормализуем
    avg_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
    
    # Сохраняем embedding
    faces_count = user.faces_registered + len(embeddings)
    await storage.update_embedding(
        user_id=user.user_id,
        embedding=avg_embedding,
        faces_count=faces_count
    )
    
    user = await storage.get_user(user.user_id)
    
    msg_parts = [f"{len(embeddings)} photos processed"]
    if errors:
        msg_parts.append(f"{len(errors)} skipped: {'; '.join(errors[:3])}")
    
    return RegisterUserResponse(
        user_id=user.user_id,
        name=request.name,
        faces_registered=user.faces_registered,
        embedding_dimensions=len(avg_embedding),
        message=". ".join(msg_parts)
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    """Получение информации о пользователе."""
    storage = StorageService(db)
    user = await storage.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@router.delete("/{user_id}")
async def delete_user(user_id: str, db: AsyncSession = Depends(get_db)):
    """Удаление пользователя."""
    storage = StorageService(db)
    deleted = await storage.delete_user(user_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User deleted successfully"}
