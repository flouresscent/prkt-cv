import cv2
import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VideoStream")

class VideoStream:
    def __init__(self, source_url, apply_clahe=True, reconnect_delay=5):
        """
        :param source_url: str — URL камеры (RTSP/HTTP/файл)
        :param apply_clahe: bool — применять ли CLAHE для улучшения качества
        :param reconnect_delay: int — пауза между попытками переподключения при ошибке
        """
        self.source_url = source_url
        self.apply_clahe = apply_clahe
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.connected = False
        self.last_read_success = time.time()

    def _connect(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source_url)
        self.connected = self.cap.isOpened()
        if self.connected:
            logger.info(f"[VideoStream] Connected to {self.source_url}")
        else:
            logger.warning(f"[VideoStream] Failed to connect to {self.source_url}")

    def _read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self._connect()
            if not self.connected:
                return None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("[VideoStream] Frame read failed, attempting reconnect...")
            self.connected = False
            self._connect()
            time.sleep(self.reconnect_delay)
            return None

        self.last_read_success = time.time()

        if self.apply_clahe:
            frame = self._apply_clahe(frame)

        return frame

    def _apply_clahe(self, frame):
        """Повышение контрастности с помощью CLAHE"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    async def get_frame(self):
        """Асинхронно получить кадр (через executor)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_frame)

    def release(self):
        """Очистка ресурсов"""
        if self.cap:
            self.cap.release()
        logger.info(f"[VideoStream] Released stream {self.source_url}")