import cv2
import numpy as np

def draw_parking_zones(frame, zones, occupancy_status):
    """
    Отображает зоны парковки и их статус поверх кадра.
    
    :param frame: исходный кадр (numpy.ndarray)
    :param zones: словарь {slot_id: [x1, y1, x2, y2]}
    :param occupancy_status: словарь {slot_id: bool} — True: свободно, False: занято
    :return: аннотированный кадр
    """
    for slot_id, zone_data in zones.items():
        x1, y1, x2, y2 = zone_data["coords"]
        is_free = occupancy_status.get(slot_id, True)
        color = (0, 255, 0) if is_free else (0, 0, 255)  # зелёный / красный
        label = f"{slot_id} {'Free' if is_free else 'Occupied'}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


def draw_detections(frame, detections, class_map=None):
    """
    Отображает bounding box-ы объектов (машин и т.п.) на кадре.

    :param frame: исходный кадр
    :param detections: список [(x1, y1, x2, y2, class_id, conf)]
    :param class_map: словарь {class_id: class_name}
    :return: кадр с отрисованными bbox
    """
    for (x1, y1, x2, y2, cls_id, conf) in detections:
        label = f"{class_map.get(cls_id, str(cls_id))} {conf:.2f}" if class_map else f"{cls_id} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return frame


def draw_status_window(zones, status_map, window_name="Parking Status"):
    rows = []
    for slot_id, zone_data in zones.items():
        is_free = status_map.get(slot_id, True)
        status_text = "Free" if is_free else "Occupied"
        label = f"{slot_id}: {'Free' if is_free else 'Occupied'}"
        rows.append((label, is_free))

    width = 300
    height = 30 * len(rows) + 20
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    for i, (text, is_free) in enumerate(rows):
        color = (0, 128, 0) if is_free else (0, 0, 255)
        cv2.putText(image, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow(window_name, image)