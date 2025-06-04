import json
import os
from pathlib import Path
import logging

from core.utils import compute_iou

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ZoneManager")

class ZoneManager:
    def __init__(self, zones_dir="config/zones", iou_threshold=0.5):
        """
        :param zones_dir: папка с json-файлами зон парковки для каждой камеры
        :param iou_threshold: порог для определения занятости (IoU)
        """
        self.zones_dir = Path(zones_dir)
        self.zones_dir.mkdir(parents=True, exist_ok=True)
        self.iou_threshold = iou_threshold
        self.zone_map = {}  # cam_id → {slot_id: [x1, y1, x2, y2]}
        self.trust_map = {}  # (cam_id, slot_id) → trust

    def load_zones(self, cam_id):
        path = self.zones_dir / f"{cam_id}.json"
        if not path.exists():
            logger.warning(f"[ZoneManager] Zones for {cam_id} not found")
            self.zone_map[cam_id] = {}
            return

        with open(path, "r", encoding="utf-8") as f:
            zones = json.load(f)
            self.zone_map[cam_id] = zones

            # 📥 Заполняем trust_map из файла
            for slot_id, data in zones.items():
                trust = data.get("trust", 0.5)  # если trust не указан — берём 0.5
                self.trust_map[(cam_id, slot_id)] = trust

        logger.info(f"[ZoneManager] Loaded {len(self.zone_map[cam_id])} zones for {cam_id}")

    def save_zones(self, cam_id):
        path = self.zones_dir / f"{cam_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.zone_map.get(cam_id, {}), f, indent=2)
        logger.info(f"[ZoneManager] Saved zones for {cam_id}")

    def get_zones(self, cam_id):
        return self.zone_map.get(cam_id, {})

    def set_zones(self, cam_id, zones_dict):
        self.zone_map[cam_id] = zones_dict
        self.save_zones(cam_id)

    def get_trust(self, cam_id, slot_id):
        trust = self.trust_map.get((cam_id, slot_id))
        if trust is None:
            logger.warning(f"[Trust] Не указан trust для {cam_id} → {slot_id}, используется по умолчанию 0.5")
            trust = 0.5
        return trust

    def is_occupied(self, slot_box, detections):
        """
        Проверка, занята ли зона на основе списка детекций.
        :param slot_box: [x1, y1, x2, y2]
        :param detections: список [(x1, y1, x2, y2, cls_id, conf), ...]
        :return: bool — занято/свободно
        """
        return any(compute_iou(slot_box, det[:4]) >= self.iou_threshold for det in detections)

    def analyze_occupancy(self, cam_id, detections):
        """
        Анализ занятости всех зон по детекциям объектов.
        :return: словарь {slot_id: True (свободно) / False (занято)}
        """
        zones = self.get_zones(cam_id)
        return {
            slot_id: not self.is_occupied(slot_data["coords"], detections)
            for slot_id, slot_data in zones.items()
        }
