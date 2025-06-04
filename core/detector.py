import logging
from ultralytics import YOLO

# COCO-классы для транспорта
DEFAULT_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

class ObjectDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.3, allowed_classes=None):
        """
        :param model_path: путь к весам YOLOv8 (например, 'yolov8s.pt')
        :param conf_threshold: минимальный порог уверенности
        :param allowed_classes: словарь {int: str} — допустимые классы объектов
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.allowed_classes = allowed_classes or DEFAULT_CLASSES

        logging.info(f"[Detector] YOLOv8 model loaded: {model_path}")
        logging.info(f"[Detector] Allowed classes: {self.allowed_classes}")

    def detect(self, frame):
        """
        Детектирует объекты на кадре и фильтрует по классам транспорта.

        :param frame: numpy.ndarray — изображение
        :return: список детекций в формате:
                 [(x1, y1, x2, y2, class_id, confidence), ...]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence < self.conf_threshold:
                continue

            if class_id not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, class_id, confidence))

        logging.debug(f"[Detector] {len(detections)} valid objects detected")
        return detections