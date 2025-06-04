import os
import json
import cv2
import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

event_logger = logging.getLogger("EventLogger")
log_path = Path("logs/slot_events.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
event_logger.setLevel(logging.INFO)
event_logger.addHandler(handler)


def log_slot_event(cam_id, slot_id, old_status, new_status, frame, roi_coords):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    status_str = f"{'Free' if old_status else 'Occupied'} -> {'Free' if new_status else 'Occupied'}"

    event_dir = Path(f"logs/events/{slot_id}/{timestamp}")
    event_dir.mkdir(parents=True, exist_ok=True)

    frame_path = event_dir / "frame.jpg"
    roi_path = event_dir / "roi.jpg"
    meta_path = event_dir / "meta.json"

    # Сохраняем кадр
    cv2.imwrite(str(frame_path), frame)

    # Вырезаем ROI
    x1, y1, x2, y2 = roi_coords
    roi_img = frame[y1:y2, x1:x2]
    cv2.imwrite(str(roi_path), roi_img)

    # JSON с метаданными
    meta = {
        "timestamp": timestamp,
        "slot_id": slot_id,
        "camera_id": cam_id,
        "old_status": "Free" if old_status else "Occupied",
        "new_status": "Free" if new_status else "Occupied",
        "frame_path": str(frame_path),
        "roi_path": str(roi_path),
        "roi_coords": [x1, y1, x2, y2],
        "image_size": frame.shape[:2],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    event_logger.info(f"[{cam_id}] {slot_id} status changed: {status_str} — saved to {event_dir}")
