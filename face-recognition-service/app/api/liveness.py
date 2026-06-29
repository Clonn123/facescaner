import base64
import cv2
import numpy as np
import time
from fastapi import APIRouter, HTTPException

from app.services.model_singletons import get_anti_spoof
from app.models.schemas import LivenessVerifyRequest, LivenessVerifyResponse

router = APIRouter(prefix="/liveness", tags=["Liveness"])


def decode_image(image_base64: str) -> np.ndarray:
    """Декодирование изображения из base64."""
    # Удаляем data URL prefix если есть
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    
    image_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@router.post("/verify", response_model=LivenessVerifyResponse)
async def verify_liveness(request: LivenessVerifyRequest):
    """
    Проверка живого присутствия.
    
    Определяет, является ли лицо реальным или это фото/видео на экране.
    
    - **image_base64**: Base64 encoded изображение
    
    Возвращает liveness_score и is_real.
    """
    start_time = time.time()
    
    anti_spoof = get_anti_spoof()
    
    # Декодирование изображения
    if request.image_base64:
        image = decode_image(request.image_base64)
    else:
        raise HTTPException(status_code=400, detail="No image provided")
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Проверка живого присутствия
    result = anti_spoof.predict(image)
    
    return LivenessVerifyResponse(
        is_real=result["is_real"],
        liveness_score=result["liveness_score"],
        confidence="high" if result["liveness_score"] > 0.8 else "medium" if result["liveness_score"] > 0.5 else "low",
        face_detected=True,
        processing_time_ms=(time.time() - start_time) * 1000
    )
