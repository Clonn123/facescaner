"""
Door camera test (RTSP stream).
Детекция лица → anti-spoof → распознавание через API.

Usage:
    python test_camera_door.py
    python test_camera_door.py --url rtsp://admin:eqwew@150.150.150.229/cam/realmonitor?channel=1&subtype=1
"""

import sys
import time
import json
import base64
import argparse
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


RTSP_URL = "rtsp://admin:eqwew@150.150.150.229/cam/realmonitor?channel=1&subtype=1"
RECOGNIZE_API = "http://localhost:8000/api/v1/recognize/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=RTSP_URL, help="RTSP stream URL")
    parser.add_argument("--no-display", action="store_true", help="Run without showing window (headless)")
    args = parser.parse_args()

    print("Initializing models...")

    detector = FaceDetector()
    detector.initialize()
    print("✓ Face detector ready")

    anti_spoof = AntiSpoofONNX()
    if anti_spoof.is_ready:
        print("✓ Anti-spoof model ready")
    else:
        print("✗ Anti-spoof model NOT ready")

    print(f"\nConnecting to RTSP: {args.url}")
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Cannot connect to RTSP stream")
        print("Check: URL, network, camera credentials")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Connected ({w}x{h} @ {fps_cam:.0f} fps)")

    # Состояние
    current_bbox = None
    frame_count = 0

    # Сглаживание
    REAL_WINDOW_SIZE = 5
    prediction_history = []

    # Распознавание
    recognition_result = None
    was_real_before = False

    # Motion
    prev_face_gray = None
    motion_history = []
    MOTION_WINDOW = 3
    MOTION_THRESHOLD = 1.4
    is_static = False

    # Интервалы
    DETECT_INTERVAL = 10
    SPOOF_INTERVAL = 5

    fps_count = 0
    fps_time = time.time()
    window_name = "Door Camera"

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[!] Frame drop, reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print("Cannot reconnect")
                break
            continue

        h, w = frame.shape[:2]

        # Детекция раз в N кадров
        need_detect = (frame_count % DETECT_INTERVAL == 0)

        if need_detect or current_bbox is None:
            faces = detector.detect_faces(frame)

            if faces:
                current_bbox = faces[0]["bbox"]
            else:
                current_bbox = None

        x, y, x2, y2 = 0, 0, 0, 0
        smoothed_real = False
        real_count = 0

        if current_bbox is not None:
            x, y, x2, y2 = map(int, current_bbox)
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Anti-spoof раз в N кадров
            if frame_count % SPOOF_INTERVAL == 0:
                try:
                    result = anti_spoof.predict(frame, (x, y, x2, y2))

                    face_roi_gray = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
                    face_roi_gray = cv2.resize(face_roi_gray, (64, 64))

                    if prev_face_gray is not None:
                        diff = np.mean(np.abs(face_roi_gray.astype(float) - prev_face_gray.astype(float)))
                        motion_history.append(diff)
                        if len(motion_history) > MOTION_WINDOW:
                            motion_history.pop(0)
                        is_static = (len(motion_history) == MOTION_WINDOW
                                     and all(m < MOTION_THRESHOLD for m in motion_history))
                    prev_face_gray = face_roi_gray

                    is_real = result['is_real'] and not is_static
                    prediction_history.append(is_real)
                    if len(prediction_history) > REAL_WINDOW_SIZE:
                        prediction_history.pop(0)

                    motion_val = np.mean(motion_history) if motion_history else 0
                    print(f"[{time.strftime('%H:%M:%S')}] {'REAL' if is_real else 'SPOOF'} "
                          f"(score={result['liveness_score']:.3f}, motion={motion_val:.2f})")
                except Exception as e:
                    print(f"Anti-spoof error: {e}")

            # Сглаживание: 4 из 5 REAL (допускаем 1 выброс)
            smoothed_real = (len(prediction_history) == REAL_WINDOW_SIZE
                             and sum(prediction_history) >= REAL_WINDOW_SIZE - 1)
            real_count = sum(prediction_history)

            # Триггер распознавания: один раз при переходе SPOOF→REAL
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
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        recognition_result = json.loads(resp.read())
                    print(f"[RECOGNIZE] {recognition_result}")
                except urllib.request.HTTPError as e:
                    body = e.read().decode()
                    print(f"[RECOGNIZE] HTTP {e.code}: {body}")
                    recognition_result = {"error": f"HTTP {e.code}"}
                except urllib.request.URLError as e:
                    print(f"[RECOGNIZE] API not reachable: {e}")
                    recognition_result = {"error": "API not running"}
                except Exception as e:
                    print(f"[RECOGNIZE] Error: {e}")
                    recognition_result = {"error": str(e)[:30]}
            elif not smoothed_real:
                was_real_before = False
                recognition_result = None

            # Отрисовка
            color = (0, 255, 0) if smoothed_real else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

            label = f"{'REAL' if smoothed_real else 'SPOOF'} ({real_count}/{REAL_WINDOW_SIZE})"
            if is_static:
                label += " STATIC"
                color = (0, 165, 255)
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if recognition_result:
                r = recognition_result
                err = r.get("error", "")
                if err:
                    cv2.putText(frame, f"API: {err}", (x, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
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
            is_static = False
            motion_history.clear()
            prediction_history.clear()
            prev_face_gray = None

        # FPS
        fps_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if not args.no_display:
            # Масштабируем под экран если кадр слишком большой
            dh, dw = frame.shape[:2]
            max_w, max_h = 1280, 720
            if dw > max_w or dh > max_h:
                scale = min(max_w / dw, max_h / dh)
                frame_show = cv2.resize(frame, (int(dw * scale), int(dh * scale)))
            else:
                frame_show = frame
            cv2.imshow(window_name, frame_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
