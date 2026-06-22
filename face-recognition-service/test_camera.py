"""
Local test script for WebSocket camera anti-spoofing.
Connects to server and sends frames from webcam for real-time detection.
"""

import cv2
import json
import time
import asyncio
import base64
import websockets

SERVER_URL = "ws://localhost:8000/ws/camera"


async def test_camera():
    """Test anti-spoofing with real-time camera."""
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print("Connecting to server...")
    
    try:
        async with websockets.connect(SERVER_URL) as ws:
            print("Connected to server!")
            print("Press 'q' to quit\n")
            
            frame_count = 0
            fps_history = []
            last_send_time = time.time()
            send_interval = 0.2  # 5 FPS (каждые 200ms)
            
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Отправляем каждые send_interval секунд
                now = time.time()
                elapsed = now - last_send_time
                
                if elapsed >= send_interval:
                    last_send_time = now
                    frame_count += 1
                    
                    # Encode to JPEG and convert to base64
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    base64_frame = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send to server
                    message = json.dumps({
                        "type": "frame",
                        "image_base64": base64_frame
                    })
                    
                    await ws.send(message)
                    
                    # Receive response
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(response)
                    except asyncio.TimeoutError:
                        data = {}
                    
                    # Calculate FPS
                    fps_history.append(time.time())
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                    avg_fps = frame_count / (fps_history[-1] - fps_history[0]) if len(fps_history) > 1 else 0
                    
                    # Print result
                    status = ""
                    if data.get("face_detected"):
                        if data.get("liveness_checked"):
                            if data.get("is_live"):
                                status = "✅ LIVE"
                            else:
                                status = "❌ SPOOF"
                        else:
                            status = "👁 FACE"
                    else:
                        status = "⏳ NO FACE"
                    
                    # Get recognition info
                    recog_info = ""
                    if data.get("recognized"):
                        recog_info = f"  → {data.get('name', 'Unknown')} (sim={data.get('similarity', 0):.2f})"
                    
                    print(f"\rFPS: {avg_fps:.1f} | {status} | Score: {data.get('liveness_score', 0):.3f} | Time: {data.get('processing_time_ms', 0):.0f}ms{recog_info}", end="", flush=True)
                    
                    # Show frame with overlay
                    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Draw status overlay
                    y_offset = 60
                    cv2.putText(frame, f"Face: {'Yes' if data.get('face_detected') else 'No'}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 30
                
                    if data.get("liveness_checked"):
                        color = (0, 255, 0) if data.get("is_live") else (0, 0, 255)
                        text = f"Liveness: {'REAL' if data.get('is_live') else 'SPOOF'}"
                        cv2.putText(frame, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 30
                        cv2.putText(frame, f"Score: {data.get('liveness_score', 0):.3f}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_offset += 30
                    
                    if data.get("recognized"):
                        cv2.putText(frame, f"ID: {data.get('employee_id', '')}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 30
                        cv2.putText(frame, f"Name: {data.get('name', '')}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow("Anti-Spoof Test", frame)
                    
                    # Check for quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    # Показываем кадр без обновления статуса
                    cv2.imshow("Anti-Spoof Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nTest completed")


if __name__ == "__main__":
    asyncio.run(test_camera())
