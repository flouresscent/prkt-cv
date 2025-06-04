import cv2
import json

from pathlib import Path
from omegaconf import OmegaConf
from core.detector import ObjectDetector

"""
Управление работой программы:
    [ЛКМ] — нарисовать зону
    [Перетаскивание] — переместить существующую зону
    [ПКМ] — удалить зону
    [Ctrl+Z] — отмена последней добавленной зоны
    [S] — сохранить в config/zones/{camX}.json
    [Ctrl+R] — сбросить все зоны
    [Колесо мыши] — масштабирование
    [ESC] — выйти
"""

class ManualSlotMarker:
    def __init__(self, cam_id, config_path="config/config.yaml"):
        self.cfg = OmegaConf.load(config_path)
        self.cam_id = cam_id
        self.rectangles = []
        self.dragging = False
        self.drag_idx = None
        self.drag_offset = (0, 0)
        self.scale = 1.0
        self.offset = [0, 0]
        self.display = None
        self.temp_x = 0 
        self.temp_y = 0

        self.output_dir = Path("config/zones")
        self.image_dir = Path("annotated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def print_instructions(self):
        print("""
        [INFO]
            Управление работой программы:
            [ЛКМ] — нарисовать зону
            [Перетаскивание] — переместить существующую зону
            [ПКМ] — удалить зону
            [Ctrl+Z] — отмена последней добавленной зоны
            [S] — сохранить в config/zones/{camX}.json
            [Ctrl+R] — сбросить все зоны
            [Колесо мыши] — масштабирование
            [ESC] — выйти
        """)


    def run(self):
        self.print_instructions()
        if self.cam_id not in self.cfg.cameras and self.cam_id not in self.cfg.get("test_videos", {}):
            print(f"❌ Камера или видео '{self.cam_id}' не найдена в config.yaml")
            return

        # Определяем источник в зависимости от режима
        mode = self.cfg.get("mode", "live")
        if mode == "video":
            stream_url = self.cfg.test_videos[self.cam_id]
        else:
            stream_url = self.cfg.cameras[self.cam_id].url

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"❌ Не удалось открыть поток: {stream_url}")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("❌ Не удалось получить кадр.")
            return

        self.orig_frame = self.apply_clahe(frame)

        self.rectangles = self.get_yolo_detections()

        cv2.namedWindow("Manual Marking")
        cv2.setMouseCallback("Manual Marking", self.mouse_callback)

        while True:
            frame = self.get_scaled_frame()
            self.display = frame.copy()

            for idx, (x1, y1, x2, y2) in enumerate(self.rectangles, 1):
                cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.display, f"slot_{idx}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if getattr(self, 'drawing', False):
                cv2.rectangle(self.display, (self.ix, self.iy), (self.temp_x, self.temp_y), (255, 0, 0), 1)

            cv2.imshow("Manual Marking", self.display)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("s"): # S
                self.save()
                break
            elif key == 18:     # Ctrl+R
                self.rectangles.clear()
                print("🔄 Все зоны сброшены")
            elif key == 26:     # Ctrl+Z
                if self.rectangles:
                    removed = self.rectangles.pop()
                    print(f"↩️ Удалён bbox: {removed}")
            elif key == 27:     # ESC
                break

    cv2.destroyAllWindows()

    def get_scaled_frame(self):
        h, w = self.orig_frame.shape[:2]
        scaled = cv2.resize(self.orig_frame, None, fx=self.scale, fy=self.scale)
        return scaled

    def apply_clahe(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def get_yolo_detections(self):
        print("🤖 Предобработка: YOLOv8 детекция...")
        detector = ObjectDetector(self.cfg.model.path, conf_threshold=self.cfg.model.conf_threshold)
        dets = detector.detect(self.orig_frame)
        rects = [[x1, y1, x2, y2] for x1, y1, x2, y2, *_ in dets]
        print(f"✅ Добавлено {len(rects)} bbox от YOLO")
        return rects

    def save(self):
        zones = {}
        print("\n💬 Укажите реальные ID и trust для каждого парковочного места:")
        for i, box in enumerate(self.rectangles):
            slot_name = f"slot_{i+1}"
            while True:
                real_id = input(f"Введите ID для {slot_name}: ").strip()
                if real_id:
                    break
                print("❌ ID не может быть пустым.")

            while True:
                trust_input = input(f"Введите trust (0.0–1.0) для {slot_name}: ").strip()
                try:
                    trust = float(trust_input)
                    if 0.0 <= trust <= 1.0:
                        break
                    else:
                        print("❌ Trust должен быть от 0.0 до 1.0")
                except ValueError:
                    print("❌ Введите корректное число (например, 0.8)")

            zones[real_id] = {
                "name": slot_name,
                "coords": box,
                "trust": trust
            }

        json_path = self.output_dir / f"{self.cam_id}.json"
        img_path = self.image_dir / f"{self.cam_id}_marked_manual.jpg"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(zones, f, indent=2, ensure_ascii=False)

        cv2.imwrite(str(img_path), self.display)

        print(f"💾 Зоны с ID и trust сохранены в {json_path}")
        print(f"🖼️ Изображение сохранено в {img_path}")


    def mouse_callback(self, event, x, y, flags, param):
        x, y = int(x / self.scale), int(y / self.scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, (x1, y1, x2, y2) in enumerate(self.rectangles):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.dragging = True
                    self.drag_idx = idx
                    self.drag_offset = (x - x1, y - y1)
                    return
            self.ix, self.iy = x, y
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if getattr(self, 'drawing', False):
                self.temp_x, self.temp_y = x, y
            elif self.dragging and self.drag_idx is not None:
                dx, dy = self.drag_offset
                w = self.rectangles[self.drag_idx][2] - self.rectangles[self.drag_idx][0]
                h = self.rectangles[self.drag_idx][3] - self.rectangles[self.drag_idx][1]
                new_x1 = x - dx
                new_y1 = y - dy
                self.rectangles[self.drag_idx] = [new_x1, new_y1, new_x1 + w, new_y1 + h]

        elif event == cv2.EVENT_LBUTTONUP:
            if getattr(self, 'drawing', False):
                self.drawing = False
                self.rectangles.append([self.ix, self.iy, x, y])
            elif self.dragging:
                self.dragging = False
                self.drag_idx = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Удалить bbox под курсором
            for idx, (x1, y1, x2, y2) in enumerate(self.rectangles):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    removed = self.rectangles.pop(idx)
                    print(f"❌ Удалён bbox: {removed}")
                    break

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.scale *= 1.1
            else:
                self.scale /= 1.1
            self.scale = max(0.1, min(5.0, self.scale))


if __name__ == "__main__":
    cfg = OmegaConf.load("config/config.yaml")
    mode = cfg.get("mode", "live")
    cam_list = list(cfg.test_videos.keys()) if mode == "video" else list(cfg.cameras.keys())

    output_dir = Path("config/zones")

    for cam_id in cam_list:
        zone_file = output_dir / f"{cam_id}.json"

        if zone_file.exists():
            overwrite = input(f"⚠️ Разметка для '{cam_id}' уже существует. Перезаписать? [y/n]: ").strip().lower()
            if overwrite != "y":
                print(f"⏭️ Пропуск камеры {cam_id}")
                continue

        print(f"\n📷 Начало разметки для {cam_id} ({'видео' if mode == 'video' else 'камеры'})")
        marker = ManualSlotMarker(cam_id)
        marker.run()

        choice = input("Продолжить разметку следующей камеры? [y/n]: ").strip().lower()
        if choice == "n":
            break

    print("\n✅ Разметка завершена для всех выбранных камер.")


