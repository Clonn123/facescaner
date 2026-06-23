"""
Local test script - оптимизированный вариант.
Детекция лица только раз в 10 кадров.
Anti-spoof раз в 5 кадров.
Между ними — используем bbox от предыдущего кадра (без детекции!).
"""

import sys
import time
import json
import base64
import urllib.request
import cv2
import numpy as np

sys.path.insert(0, '.')

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow not installed. Russian text will show as '?'.")
    print("Install: pip install Pillow")
    Image = None


def put_text_unicode(img, text, pos, font_path=None, font_size=16, color=(255, 255, 255)):
    """Рисует Unicode-текст (в т.ч. русский) на OpenCV BGR-изображении через Pillow."""
    if Image is None:
        return
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path or "arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    b, g, r = color
    draw.text(pos, text, font=font, fill=(r, g, b))
    img[:] = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

from app.services.face_detector import FaceDetector
from app.services.liveness.anti_spoof_onnx import AntiSpoofONNX


def main():
    print("Initializing models...")
    
    detector = FaceDetector()
    detector.initialize()
    print("✓ Face detector ready")
    
    anti_spoof = AntiSpoofONNX()
    if anti_spoof.is_ready:
        print("✓ Anti-spoof model ready")
    else:
        print("✗ Anti-spoof model NOT ready")
    
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera opened. Press 'q' to quit.\n")
    
    # Состояние трекинга
    current_bbox = None  # [x, y, x2, y2] последнего найденного лица
    frame_count = 0
    
    # Сглаживание: скользящее окно из 3 предсказаний
    REAL_WINDOW_SIZE = 3
    prediction_history = []  # список is_real за последние N предсказаний
    
    # Распознавание через API (один раз на сессию REAL)
    RECOGNIZE_API = "http://localhost:8000/api/v1/recognize/"
    recognition_result = None
    was_real_before = False  # для отслеживания перехода SPOOF→REAL
    
    # Интервалы
    DETECT_INTERVAL = 10       # Полная детекция раз в 10 кадров
    SPOOF_INTERVAL = 5         # Anti-spoof раз в 5 кадров
    
    fps_count = 0
    fps_time = time.time()
    window_name = "Anti-Spoof Test"
    
    while True:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # === ОПТИМИЗАЦИЯ: детекция только раз в N кадров ===
        need_detect = (frame_count % DETECT_INTERVAL == 0)
        
        if need_detect or current_bbox is None:
            # Полная детекция (медленно, но только раз в 10 кадров)
            faces = detector.detect_faces(frame)
            
            if faces:
                current_bbox = faces[0]["bbox"]
                print(f"[{time.strftime('%H:%M:%S')}] Full detection: {current_bbox}")
            else:
                current_bbox = None
                print("[No face detected]")
        
        # === Используем bbox для текущего кадра ===
        x, y, x2, y2 = 0, 0, 0, 0
        smoothed_real = False
        real_count = 0
        
        if current_bbox is not None:
            x, y, x2, y2 = map(int, current_bbox)
            
            if x < 0: x = 0
            if y < 0: y = 0
            if x2 > w: x2 = w
            if y2 > h: y2 = h
            
            # === Anti-spoof раз в 5 кадров ===
            if frame_count % SPOOF_INTERVAL == 0:
                try:
                    result = anti_spoof.predict(frame, (x, y, x2, y2))
                    prediction_history.append(result['is_real'])
                    if len(prediction_history) > REAL_WINDOW_SIZE:
                        prediction_history.pop(0)
                    
                    print(f"[{time.strftime('%H:%M:%S')}] {'REAL' if result['is_real'] else 'SPOOF'} "
                          f"(score={result['liveness_score']:.3f})")
                except Exception as e:
                    print(f"Anti-spoof error: {e}")
            
            # Сглаживание (на каждом кадре, из истории)
            smoothed_real = (len(prediction_history) == REAL_WINDOW_SIZE
                             and all(prediction_history))
            real_count = sum(prediction_history)
            
            # === Триггер распознавания: один раз при переходе SPOOF→REAL ===
            if smoothed_real and not was_real_before:
                was_real_before = True
                try:
                    _, jpeg = cv2.imencode('.jpg', frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    b64 = base64.b64encode(jpeg).decode('utf-8')
                    req = urllib.request.Request(
                        RECOGNIZE_API,
                        data=json.dumps({"image_base64": b64}).encode(),
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        recognition_result = json.loads(resp.read())
                    print(f"[RECOGNIZE] {recognition_result}")
                except Exception as e:
                    print(f"[RECOGNIZE] API error: {e}")
                    recognition_result = {"error": str(e)}
            elif not smoothed_real:
                was_real_before = False
                recognition_result = None
            
            # === Отрисовка на каждом кадре (без мигания) ===
            color = (0, 255, 0) if smoothed_real else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            
            label = f"{'REAL' if smoothed_real else 'SPOOF'} ({real_count}/{REAL_WINDOW_SIZE})"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if recognition_result:
                r = recognition_result
                if "error" in r:
                    cv2.putText(frame, "API error", (x, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                elif r.get("recognized"):
                    name = r.get('name', '?') or r.get('account_id', '?')
                    put_text_unicode(frame, name, (x, y2 + 20), font_size=18, color=(0, 255, 0))
                    sim = r.get("similarity", 0)
                    cv2.putText(frame, f"match: {sim:.2f}", (x, y2 + 42),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, "Not recognized", (x, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            was_real_before = False
            recognition_result = None
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        fps_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Время обработки
        proc_time = (time.time() - frame_start) * 1000
        cv2.putText(frame, f"Time: {proc_time:.0f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest completed.")


if __name__ == "__main__":
    main()