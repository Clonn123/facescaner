"""
Local test script - directly uses InsightFace + anti-spoof like demo.py
No WebSocket, no base64 encoding - pure local processing.
"""

import sys
import time
import cv2
import numpy as np

# Add app to path
sys.path.insert(0, '.')

from app.services.face_detector import FaceDetector
from app.services.face_recognizer import FaceRecognizer
from app.services.liveness.anti_spoof_onnx import AntiSpoofONNX
from app.services.liveness.preprocess import crop


def main():
    print("Initializing models...")
    
    # Initialize detectors
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
    
    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera opened. Press 'q' to quit.\n")
    
    fps_count = 0
    fps_time = time.time()
    window_name = "Anti-Spoof Test"
    spoof_frame_interval = 3  # Anti-spoof раз в 3 кадра
    
    while True:
        frame_start = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip horizontally (selfie view)
        t0 = time.time()
        frame = cv2.flip(frame, 1)
        t_flip = (time.time() - t0) * 1000
        
        # Convert to RGB
        t0 = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_rgb = (time.time() - t0) * 1000
        
        # Detect faces - используем прямой вызов app.get
        t0 = time.time()
        faces = detector.app.get(frame)
        t_detect = (time.time() - t0) * 1000
        
        # Фильтруем по порогу
        threshold = 0.5
        faces = [f for f in faces if f.det_score >= threshold]
        
        # Anti-spoof только раз в N кадров
        t0 = time.time()
        if faces and fps_count % spoof_frame_interval == 0:
            for face in faces:
                bbox = face.bbox
                x, y, x2, y2 = map(int, bbox)
                
                # Draw bbox on frame
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                
                # Anti-spoof с crop expansion
                try:
                    face_crop = crop(frame_rgb, (x, y, x2, y2), bbox_expansion_factor=1.5)
                    
                    result = anti_spoof.predict(face_crop)
                    
                    label = f"{'REAL' if result['is_real'] else 'SPOOF'}: {result['liveness_score']:.2f}"
                    color = (0, 255, 0) if result['is_real'] else (0, 0, 255)
                    
                    cv2.putText(frame, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    print(f"[{time.strftime('%H:%M:%S')}] {'REAL' if result['is_real'] else 'SPOOF'} "
                          f"(score={result['liveness_score']:.3f}, logit={result['logit_diff']:.2f})")
                    
                except Exception as e:
                    print(f"Anti-spoof error: {e}")
        elif faces:
            # Anti-spoof не делаем, просто рисуем bbox
            for face in faces:
                bbox = face.bbox
                x, y, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        t_spoof = (time.time() - t0) * 1000
        
        if not faces:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS counter
        fps_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Processing time
        proc_time = (time.time() - frame_start) * 1000
        cv2.putText(frame, f"Time: {proc_time:.0f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS breakdown
        cv2.putText(frame, f"Flip: {t_flip:.0f}ms RGB: {t_rgb:.0f}ms", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Detect: {t_detect:.0f}ms Spoof: {t_spoof:.0f}ms", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show frame - с ограничением частоты обновления
        cv2.imshow(window_name, frame)
        
        # Quit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest completed.")


if __name__ == "__main__":
    main()