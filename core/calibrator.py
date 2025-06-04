import json
import logging

from pathlib import Path

from core.detector import ObjectDetector
from core.visualizer import draw_parking_zones

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Calibrator")


class ParkingCalibrator:
    def __init__(self, detector, output_dir="config/zones"):
        """
        :param detector: экземпляр ObjectDetector
        :param output_dir: куда сохранять json-файл с зонами
        """
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(self, frame, cam_id, save=True, visualize=False):
        """
        Находит парковочные зоны по кадру и сохраняет их.
        :param frame: изображение (numpy.ndarray)
        :param cam_id: идентификатор камеры
        :param save: сохранить ли результат в JSON
        :param visualize: показать ли окно с результатом
        :return: словарь зон {slot_N: [x1, y1, x2, y2]}
        """
        detections = self.detector.detect(frame)
        zones = {f"slot_{idx}": [x1, y1, x2, y2] for idx, (x1, y1, x2, y2, *_rest) in enumerate(detections, 1)}
        logger.info(f"[Calibrator] Found {len(zones)} slots on camera {cam_id}")

        if save:
            path = self.output_dir / f"{cam_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(zones, f, indent=2)
            logger.info(f"[Calibrator] Saved zones to {path}")

        if visualize:
            vis = draw_parking_zones(frame.copy(), zones, {k: False for k in zones})
            import cv2
            cv2.imshow(f"Calibration [{cam_id}]", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return zones