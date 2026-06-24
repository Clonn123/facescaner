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
from app.services.face_recognizer import FaceRecognizer
from app.services.liveness.anti_spoof_onnx import AntiSpoofONNX


RTSP_URL = "rtsp://admin:eqwew@150.150.150.229/cam/realmonitor?channel=1&subtype=1"
RECOGNIZE_API = "http://localhost:8000/api/v1/recognize/"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=RTSP_URL, help="RTSP stream URL")
    parser.add_argument("--no-display", action="store_true", help="Run without showing window (headless)")
    args = parser.parse_args()

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

    # Состояние (per-face)
    tracks = {}  # track_id -> {...}
    next_track_id = 0
    frame_count = 0

    # Сглаживание
    REAL_WINDOW_SIZE = 5

    # Motion
    MOTION_WINDOW = 3
    MOTION_THRESHOLD = 1.4

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

        # Обработка каждого лица
        for tid, tr in list(tracks.items()):
            if frame_count - tr["last_seen"] > DETECT_INTERVAL:
                continue

            x, y, x2, y2 = map(int, tr["bbox"])
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 - x < 10 or y2 - y < 10:
                continue

            # Anti-spoof раз в N кадров
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

                    motion_val = np.mean(tr["motion_history"]) if tr["motion_history"] else 0
                    print(f"[Track {tid}] {'REAL' if is_real else 'SPOOF'} "
                          f"(score={result['liveness_score']:.3f}, motion={motion_val:.2f})")
                except Exception as e:
                    print(f"[Track {tid}] Anti-spoof error: {e}")

            # Сглаживание: 4 из 5 REAL (допускаем 1 выброс)
            smoothed_real = (len(tr["prediction_history"]) == REAL_WINDOW_SIZE
                             and sum(tr["prediction_history"]) >= REAL_WINDOW_SIZE - 1)
            real_count = sum(tr["prediction_history"])

            # Триггер распознавания: один раз при переходе SPOOF→REAL
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

            # Отрисовка
            color = (0, 255, 0) if smoothed_real else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

            label = f"#{tid} {'REAL' if smoothed_real else 'SPOOF'} ({real_count}/{REAL_WINDOW_SIZE})"
            if tr["is_static"]:
                label += " STATIC"
                color = (0, 165, 255)
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if tr["recognition_result"]:
                r = tr["recognition_result"]
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

        # FPS
        fps_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f} | Faces: {len(tracks)}", (10, 30),
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
