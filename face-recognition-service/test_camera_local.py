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
from app.services.face_recognizer import FaceRecognizer
from app.services.liveness.anti_spoof_onnx import AntiSpoofONNX


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def main():
    print("Initializing models...")
    
    detector = FaceDetector()
    detector.initialize()
    print("✓ Face detector ready")

    recognizer = FaceRecognizer(detector)
    print("✓ Face recognizer ready")
    
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
    
    # Состояние (per-face)
    tracks = {}  # track_id -> {...}
    next_track_id = 0
    frame_count = 0

    # Сглаживание: скользящее окно из предсказаний
    REAL_WINDOW_SIZE = 5

    # Motion
    MOTION_WINDOW = 3
    MOTION_THRESHOLD = 1.4

    # Распознавание
    RECOGNIZE_API = "http://localhost:8000/api/v1/recognize/"

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
        
        # === Детекция раз в N кадров ===
        need_detect = (frame_count % DETECT_INTERVAL == 0)

        if need_detect or not tracks:
            faces = detector.detect_faces(frame)
            new_bboxes = [f["bbox"] for f in faces]

            matched = set()
            track_to_new = {}
            FACE_MATCH_THRESHOLD = 0.4

            for tid, tr in list(tracks.items()):
                best_j = -1
                best_iou = 0.3
                for j, nb in enumerate(new_bboxes):
                    if j in matched:
                        continue
                    score = iou(tr["bbox"], nb)
                    if score > best_iou:
                        best_iou = score
                        best_j = j
                if best_j >= 0:
                    track_to_new[tid] = best_j

            for tid, best_j in list(track_to_new.items()):
                tr = tracks[tid]
                if tr.get("embedding") is not None:
                    emb = recognizer.generate_embedding(frame)
                    if emb is not None:
                        sim = recognizer.compute_similarity(tr["embedding"], emb)
                        if sim < FACE_MATCH_THRESHOLD:
                            continue
                tr["bbox"] = new_bboxes[best_j]
                tr["last_seen"] = frame_count
                matched.add(best_j)

            for j, nb in enumerate(new_bboxes):
                if j not in matched:
                    emb = recognizer.generate_embedding(frame)
                    tracks[next_track_id] = {
                        "bbox": nb,
                        "last_seen": frame_count,
                        "embedding": emb,
                        "prediction_history": [],
                        "motion_history": [],
                        "prev_face_gray": None,
                        "was_real_before": False,
                        "recognition_result": None,
                        "is_static": False,
                    }
                    next_track_id += 1

            to_del = [tid for tid, tr in tracks.items() if frame_count - tr["last_seen"] > 30]
            for tid in to_del:
                del tracks[tid]

        # === Обработка каждого лица ===
        for tid, tr in list(tracks.items()):
            if frame_count - tr["last_seen"] > DETECT_INTERVAL:
                continue

            x, y, x2, y2 = map(int, tr["bbox"])
            if x < 0: x = 0
            if y < 0: y = 0
            if x2 > w: x2 = w
            if y2 > h: y2 = h

            if x2 - x < 10 or y2 - y < 10:
                continue

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            # === Anti-spoof раз в 5 кадров ===
            if frame_count % SPOOF_INTERVAL == 0:
                try:
                    result = anti_spoof.predict(frame, (x, y, x2, y2))

                    face_roi = frame[y:y2, x:x2]
                    if face_roi.size > 0:
                        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        face_roi_gray = cv2.resize(face_roi_gray, (64, 64))

                        if tr["prev_face_gray"] is not None:
                            diff = np.mean(np.abs(face_roi_gray.astype(float) - tr["prev_face_gray"].astype(float)))
                            tr["motion_history"].append(diff)
                            if len(tr["motion_history"]) > MOTION_WINDOW:
                                tr["motion_history"].pop(0)
                            tr["is_static"] = (len(tr["motion_history"]) == MOTION_WINDOW
                                               and all(m < MOTION_THRESHOLD for m in tr["motion_history"]))
                        tr["prev_face_gray"] = face_roi_gray

                    is_real = result['is_real'] and not tr["is_static"]
                    tr["prediction_history"].append(is_real)
                    if len(tr["prediction_history"]) > REAL_WINDOW_SIZE:
                        tr["prediction_history"].pop(0)

                    print(f"[Track {tid}] {'REAL' if is_real else 'SPOOF'} "
                          f"(score={result['liveness_score']:.3f}, motion={np.mean(tr['motion_history']) if tr['motion_history'] else 0:.2f})")
                except Exception as e:
                    print(f"[Track {tid}] Anti-spoof error: {e}")

            smoothed_real = (len(tr["prediction_history"]) == REAL_WINDOW_SIZE
                             and sum(tr["prediction_history"]) >= REAL_WINDOW_SIZE - 1)
            real_count = sum(tr["prediction_history"])

            if smoothed_real and not tr["was_real_before"]:
                tr["was_real_before"] = True
                try:
                    _, jpeg = cv2.imencode('.jpg', frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    b64 = base64.b64encode(jpeg).decode('utf-8')
                    req = urllib.request.Request(
                        RECOGNIZE_API,
                        data=json.dumps({"image_base64": b64}).encode(),
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        tr["recognition_result"] = json.loads(resp.read())
                    print(f"[Track {tid}] RECOGNIZE: {tr['recognition_result']}")
                except urllib.request.HTTPError as e:
                    body = e.read().decode()
                    print(f"[Track {tid}] RECOGNIZE HTTP {e.code}: {body}")
                    tr["recognition_result"] = {"error": f"HTTP {e.code}"}
                except urllib.request.URLError as e:
                    print(f"[Track {tid}] RECOGNIZE not reachable: {e}")
                    tr["recognition_result"] = {"error": "API not running"}
                except Exception as e:
                    print(f"[Track {tid}] RECOGNIZE error: {e}")
                    tr["recognition_result"] = {"error": str(e)[:30]}
            elif not smoothed_real:
                tr["was_real_before"] = False
                tr["recognition_result"] = None

            color = (0, 255, 0) if smoothed_real else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

            label = f"#{tid} {'REAL' if smoothed_real else 'SPOOF'} ({real_count}/{REAL_WINDOW_SIZE})"
            if tr["is_static"]:
                label += " STATIC"
                color = (0, 165, 255)
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if tr["recognition_result"]:
                r = tr["recognition_result"]
                err = r.get("error", "")
                if err:
                    cv2.putText(frame, f"API: {err}", (x, y2 + 16),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                elif r.get("recognized"):
                    name = r.get('name', '?') or r.get('account_id', '?')
                    put_text_unicode(frame, name, (x, y2 + 16), font_size=16, color=(0, 255, 0))
                    sim = r.get("similarity", 0)
                    cv2.putText(frame, f"match: {sim:.2f}", (x, y2 + 36),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, "Not recognized", (x, y2 + 16),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # FPS
        fps_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f} | Faces: {len(tracks)}", (10, 60),
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