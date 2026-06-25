"""
Flicker-based screen detector.

Phone screens use PWM for brightness -> flickers at specific frequency.
Captured at 30fps, this creates periodic brightness patterns.
Real faces have random (non-periodic) brightness variation.

High-pass filter removes slow brightness changes from head movement,
leaving only high-frequency flicker signal.
"""

import numpy as np
from collections import deque


class FlickerDetector:

    def __init__(self, window_size=30, min_samples=15):
        self.window_size = window_size
        self.min_samples = min_samples
        self.brightness_history = deque(maxlen=window_size)

    def update(self, face_roi_bgr: np.ndarray) -> float:
        if face_roi_bgr is None or face_roi_bgr.size == 0:
            return 0.0

        import cv2
        hsv = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2HSV)
        mean_v = float(np.mean(hsv[:, :, 2]))
        self.brightness_history.append(mean_v)

        if len(self.brightness_history) < self.min_samples:
            return 0.0

        return self._compute_flicker_score()

    def _highpass_filter(self, signal: np.ndarray, cutoff: int = 5) -> np.ndarray:
        if len(signal) < cutoff * 2:
            return signal - np.mean(signal)
        kernel = np.ones(cutoff) / cutoff
        smoothed = np.convolve(signal, kernel, mode='same')
        high_freq = signal - smoothed
        high_freq = high_freq - np.mean(high_freq)
        return high_freq

    def _compute_flicker_score(self) -> float:
        signal = np.array(self.brightness_history, dtype=float)

        # Raw signal variance — if brightness barely changes, no flicker
        raw_std = np.std(signal)
        if raw_std < 2.0:
            return 0.0

        # High-pass filter
        high_freq = self._highpass_filter(signal, cutoff=5)

        hf_std = np.std(high_freq)
        # High-freq variance too low — just camera noise, not flicker
        if hf_std < 1.0:
            return 0.0

        # Normalize
        normalized = high_freq / hf_std

        n = len(normalized)

        # Autocorrelation via FFT
        fft = np.fft.rfft(normalized)
        acf = np.fft.irfft(fft * np.conj(fft))
        acf = acf / acf[0] if acf[0] != 0 else acf

        # Check lags 1..n//4 (skip very short lags = noise)
        check_lags = acf[2:n // 4 + 1]
        if len(check_lags) == 0:
            return 0.0

        max_ac = np.max(np.abs(check_lags))

        if max_ac > 0.5:
            score = min(1.0, (max_ac - 0.3) / 0.4)
        else:
            score = 0.0

        return min(1.0, score)

    def reset(self):
        self.brightness_history.clear()
