import asyncio
import queue
import threading
import time
import cv2
import numpy as np


class RTSPConnection:
    """RTSP подключение с отдельным потоком чтения кадров."""

    def __init__(self, url: str, initial_delay: float = 5.0, max_delay: float = 60.0, queue_size: int = 1):
        self.url = url
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.delay = initial_delay
        self.cap = None
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected = False

    def _reader_loop(self):
        """Поток: читает кадры из RTSP в очередь."""
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._connected = False
                break
            # Берём последний кадр, старые выбрасываем
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)
            self._connected = True

    def connect(self) -> bool:
        self.release()
        self.cap = cv2.VideoCapture(self.url)
        if self.cap.isOpened():
            self.delay = self.initial_delay
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
            return True
        return False

    async def read(self) -> tuple[bool, np.ndarray | None]:
        """Чтение кадра из очереди (без блокировки event loop)."""
        if not self._connected and self._queue.empty():
            return False, None
        try:
            frame = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._queue.get(timeout=0.5)
            )
            return True, frame
        except queue.Empty:
            return self._connected, None

    async def wait_and_reconnect(self) -> bool:
        print(f"[RTSP] {self.url} reconnecting in {self.delay:.0f}s...")
        await asyncio.sleep(self.delay)
        self.delay = min(self.delay * 2, self.max_delay)
        return self.connect()

    def release(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._connected = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
