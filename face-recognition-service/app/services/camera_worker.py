"""
Воркер обработки камеры.
Подключается к RTSP, обрабатывает кадры параллельно.
Пайплайн идентичен test_camera_local.py.
"""

import asyncio
import time
import uuid
import json
import urllib.request
import cv2
import numpy as np

from app.services.rtsp_connection import RTSPConnection
from app.services.model_singletons import get_detector, get_anti_spoof
from app.services.face_recognizer import FaceRecognizer
from app.services.liveness.flicker_detector import FlickerDetector
from app.services.cache import get_door_id_by_camera
from app.core.config import get_settings

settings = get_settings()


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


class CameraWorker:
    """Обработка кадров с одной RTSP камеры."""

    # Общий lock для ONNX моделей — one call at a time
    _model_lock = asyncio.Lock()

    def __init__(self, door_id: str, door_name: str, camera_url: str):
        self.door_id = door_id
        self.door_name = door_name
        self.camera_url = camera_url
        self.tag = f"[Worker:{door_name}]"

        self.rtsp = RTSPConnection(camera_url)
        self.detector = get_detector()
        self.anti_spoof = get_anti_spoof()
        self.recognizer = FaceRecognizer(self.detector)

        self.tracks = {}
        self.next_track_id = 0
        self.frame_count = 0

    async def run(self):
        """Главный цикл: подключение → чтение кадров → обработка."""
        if self.camera_url == "local":
            print(f"{self.tag} Skipping RTSP worker for local camera", flush=True)
            return

        print(f"{self.tag} Starting (camera={self.camera_url}, door={self.door_id})", flush=True)

        while True:
            if not self.rtsp.connect():
                print(f"{self.tag} Cannot connect to {self.camera_url}", flush=True)
                await self.rtsp.wait_and_reconnect()
                continue

            print(f"{self.tag} Connected to {self.camera_url}", flush=True)
            consecutive_errors = 0

            while True:
                ret, frame = await self.rtsp.read()
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors > 30:
                        print(f"{self.tag} Too many errors, reconnecting...", flush=True)
                        break
                    await asyncio.sleep(0.01)
                    continue

                consecutive_errors = 0
                await self.process_frame(frame)
                self.frame_count += 1

            self.rtsp.release()
            await self.rtsp.wait_and_reconnect()

    async def process_frame(self, frame):
        """Пайплайн обработки кадра."""
        frame_start = time.time()
        h, w = frame.shape[:2]
        tracks = self.tracks
        fc = self.frame_count

        need_detect = (fc % settings.DETECT_INTERVAL == 0)

        if need_detect or not tracks:
            async with self._model_lock:
                faces = self.detector.detect_faces(frame)
            new_bboxes = [f["bbox"] for f in faces]
            new_scores = [f["det_score"] for f in faces]

            matched = set()
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
                    new_score = new_scores[best_j]
                    old_score = tr.get("avg_det_score", new_score)
                    tr["bbox"] = new_bboxes[best_j]
                    tr["last_seen"] = fc
                    tr["avg_det_score"] = old_score * 0.7 + new_score * 0.3

                    if tr["was_real_before"] and old_score - new_score > 0.25:
                        tr["prediction_history"] = []
                        tr["motion_history"] = []
                        tr["prev_face_gray"] = None
                        tr["was_real_before"] = False
                        tr["recognition_result"] = None
                        tr["is_static"] = False
                        tr["flicker_detector"].reset()
                        tr["flicker_score"] = 0.0
                        tr["last_recognize_time"] = 0.0
                        tr["_force_spoof"] = True

                    matched.add(best_j)

            for j, nb in enumerate(new_bboxes):
                if j not in matched:
                    tracks[self.next_track_id] = {
                        "bbox": nb,
                        "last_seen": fc,
                        "prediction_history": [],
                        "motion_history": [],
                        "prev_face_gray": None,
                        "was_real_before": False,
                        "recognition_result": None,
                        "is_static": False,
                        "avg_det_score": new_scores[j],
                        "_force_spoof": False,
                        "flicker_detector": FlickerDetector(window_size=30, min_samples=15),
                        "flicker_score": 0.0,
                        "last_recognize_time": 0.0,
                        "door_opened": False,
                    }
                    self.next_track_id += 1

            to_del = [tid for tid, tr in tracks.items() if fc - tr["last_seen"] > 30]
            for tid in to_del:
                del tracks[tid]

        for idx, tr in enumerate(list(tracks.values())):
            if fc - tr["last_seen"] > settings.DETECT_INTERVAL:
                continue

            x, y, x2, y2 = map(int, tr["bbox"])
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 - x < 10 or y2 - y < 10:
                continue

            face_roi = frame[y:y2, x:x2]
            frame_time_ms = (time.time() - frame_start) * 1000

            if face_roi.size > 0 and frame_time_ms < 50:
                face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_roi_gray = cv2.resize(face_roi_gray, (64, 64))

                if tr["prev_face_gray"] is not None:
                    diff = np.mean(np.abs(face_roi_gray.astype(float) - tr["prev_face_gray"].astype(float)))
                    tr["motion_history"].append(diff)
                    if len(tr["motion_history"]) > settings.MOTION_WINDOW:
                        tr["motion_history"].pop(0)
                    tr["is_static"] = (
                        len(tr["motion_history"]) == settings.MOTION_WINDOW
                        and all(m < settings.MOTION_THRESHOLD for m in tr["motion_history"])
                    )
                tr["prev_face_gray"] = face_roi_gray
            elif face_roi.size > 0:
                tr["prev_face_gray"] = None

            tr["flicker_score"] = tr["flicker_detector"].update(face_roi)

            force = tr.pop("_force_spoof", False)
            if fc % settings.SPOOF_INTERVAL == 0 or force:
                try:
                    async with self._model_lock:
                        result = self.anti_spoof.predict(face_roi)
                    is_real = result["is_real"] and not tr["is_static"] and tr["flicker_score"] < 0.5
                    tr["prediction_history"].append(is_real)
                    if len(tr["prediction_history"]) > settings.REAL_WINDOW_SIZE:
                        tr["prediction_history"].pop(0)

                    motion_val = np.mean(tr["motion_history"]) if tr["motion_history"] else 0
                    print(f"{self.tag} {'REAL' if is_real else 'SPOOF'} "
                          f"(score={result['liveness_score']:.3f}, motion={motion_val:.2f}, "
                          f"flicker={tr['flicker_score']:.3f})")
                except Exception as e:
                    print(f"{self.tag} Anti-spoof error: {e}")

            smoothed_real = (
                len(tr["prediction_history"]) == settings.REAL_WINDOW_SIZE
                and sum(tr["prediction_history"]) >= settings.REAL_WINDOW_SIZE - 1
            )

            need_retry = False
            if tr["recognition_result"]:
                r = tr["recognition_result"]
                is_error = r.get("error")
                is_not_recognized = not r.get("recognized")
                time_since = time.time() - tr["last_recognize_time"]
                if (is_error or is_not_recognized) and time_since > settings.RECOGNIZE_RETRY_DELAY:
                    need_retry = True

            if (smoothed_real and not tr["was_real_before"]) or need_retry:
                tr["was_real_before"] = True
                tr["last_recognize_time"] = time.time()
                tr["recognition_result"] = None
                await self.recognize_and_open_door(frame, tr, x, y, x2, y2)
            elif not smoothed_real:
                tr["was_real_before"] = False
                tr["recognition_result"] = None

    async def recognize_and_open_door(self, frame, tr, x, y, x2, y2):
        """Распознавание + отправка на бэкенд."""
        recognizer = self.recognizer

        face_info = {
            "bbox": [float(x), float(y), float(x2), float(y2)],
            "kps": [],
            "det_score": 1.0,
            "landmarks": None,
        }

        async with self._model_lock:
            embedding = recognizer.generate_embedding(frame, face_info)
        if embedding is None:
            return

        candidates = await self._get_candidates()
        if not candidates:
            return

        match = recognizer.find_best_match(embedding, candidates)
        if not match:
            tr["recognition_result"] = {"recognized": False}
            print(f"{self.tag} Not recognized")
            return

        user_id, similarity = match
        tr["recognition_result"] = {
            "recognized": True,
            "user_id": user_id,
            "similarity": float(similarity),
        }
        print(f"{self.tag} Recognized: user={user_id}, similarity={similarity:.3f}")

        if not tr.get("door_opened"):
            opened = await self._open_door(user_id, frame, tr)
            if opened:
                tr["door_opened"] = True

    async def _get_candidates(self):
        """Получить кандидатов из глобального кеша."""
        from app.core.database import async_session_factory
        from app.services.cache import get_candidates

        async with async_session_factory() as db:
            return await get_candidates(db)

    async def _open_door(self, user_id: str, frame: np.ndarray, track: dict) -> bool:
        """POST на бэкенд: door_id + user_id + photo. Returns True if door opened."""
        if not settings.HR_API_KEY:
            return False

        door_id = await get_door_id_by_camera(self.camera_url)
        if not door_id:
            print(f"{self.tag} No door found for camera_url={self.camera_url}")
            return False

        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        photo_bytes = jpeg.tobytes()

        for attempt in range(settings.MAX_DOOR_RETRIES):
            if track not in self.tracks.values():
                print(f"{self.tag} Face track gone, stopping door retry")
                return False

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

            def send_request():
                req = urllib.request.Request(
                    f"{settings.BACKEND_API_BASE_URL}/door-access/open-door",
                    data=body,
                    headers={
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                        "hr-api-key": settings.HR_API_KEY,
                    },
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.read().decode()

            try:
                resp_data = await asyncio.to_thread(send_request)
                print(f"{self.tag} Door attempt {attempt+1}: door={door_id}, user={user_id}, resp={resp_data}")

                try:
                    resp_json = json.loads(resp_data)
                except Exception:
                    resp_json = {}

                data = resp_json.get("data", {})
                if data.get("cooldown"):
                    if attempt < settings.MAX_DOOR_RETRIES - 1:
                        print(f"{self.tag} Cooldown active, retrying in {settings.DOOR_RETRY_DELAY}s...")
                        await asyncio.sleep(settings.DOOR_RETRY_DELAY)
                        continue
                    else:
                        print(f"{self.tag} Cooldown still active after {settings.MAX_DOOR_RETRIES} attempts")
                        return False
                return data.get("success", False)
            except Exception as e:
                print(f"{self.tag} Door open error: {e}")
                return False

        return False
