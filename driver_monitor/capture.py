"""Frame source: threaded webcam (always latest frame) or synchronous video-file replay."""
import threading
import time
import cv2


class FrameSource:
    def __init__(self, source="0", width=640, height=480, mirror=True, max_width=960):
        self.is_file = not str(source).isdigit()
        self.cap = cv2.VideoCapture(source if self.is_file else int(source))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        self.max_width = max_width
        self.mirror = mirror and not self.is_file
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._idx = 0
        self._stop = False
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._lock = threading.Lock()
            self._new = threading.Event()
            self._frame = None
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        # Keep reading so the driver buffer never fills: processing always sees the newest frame.
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = (f, time.monotonic())
            self._new.set()

    def read(self):
        """Return (ok, frame_bgr, t_seconds)."""
        if self.is_file:
            ok, f = self.cap.read()
            if not ok:
                return False, None, 0.0
            t = self._idx / self.fps
            self._idx += 1
        else:
            if not self._new.wait(1.0):
                return False, None, 0.0
            with self._lock:
                f, t = self._frame
            self._new.clear()
            if self.mirror:
                f = cv2.flip(f, 1)
        h, w = f.shape[:2]
        if w > self.max_width:
            f = cv2.resize(f, (self.max_width, int(h * self.max_width / w)))
        return True, f, t

    def release(self):
        self._stop = True
        if not self.is_file:
            self._thread.join(timeout=1.0)
        self.cap.release()
