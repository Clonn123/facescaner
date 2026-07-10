"""
Shared ML model singletons.
Loaded once at startup, reused by all endpoints.
"""

from app.services.face_detector import FaceDetector
from app.services.liveness.anti_spoof_onnx import AntiSpoofONNX
import numpy as np

_detector = None
_anti_spoof = None


def get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector()
        _detector.initialize()
    return _detector


def get_anti_spoof() -> AntiSpoofONNX:
    global _anti_spoof
    if _anti_spoof is None:
        _anti_spoof = AntiSpoofONNX()
    return _anti_spoof


def warmup():
    """Pre-load all models at startup."""
    print("Warming up models...", flush=True)
    det = get_detector()
    # Detector warmup on real frame size
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    det.app.get(test_img)
    print("  Face detector ready", flush=True)

    spoof = get_anti_spoof()
    # Anti-spoof warmup on face crop size
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    spoof.predict(dummy)
    providers = spoof.session.get_providers() if spoof.session else "none"
    print(f"  Anti-spoof ready (providers: {providers})", flush=True)
