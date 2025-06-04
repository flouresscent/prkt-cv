import time
from collections import defaultdict, deque

class OccupancyAnalyzer:
    def __init__(self, zone_manager, window_seconds=2.0, min_confirmations=3):
        """
        :param zone_manager: экземпляр ZoneManager
        :param window_seconds: окно времени для фильтрации флуктуаций
        :param min_confirmations: сколько раз подряд должен наблюдаться статус, чтобы он был принят
        """
        self.zone_manager = zone_manager
        self.history = defaultdict(lambda: defaultdict(deque))  # cam_id → slot_id → deque[(timestamp, is_free)]
        self.window_seconds = window_seconds
        self.min_confirmations = min_confirmations
        self.latest_stable_status = defaultdict(dict)  # cam_id → slot_id → is_free

    def analyze(self, cam_id, detections):
        current_time = time.time()
        raw_status = self.zone_manager.analyze_occupancy(cam_id, detections)

        stable_status = {}
        for slot_id, is_free in raw_status.items():
            hist = self.history[cam_id][slot_id]
            hist.append((current_time, is_free))

            # Очистка старых записей за пределами окна
            while hist and (current_time - hist[0][0] > self.window_seconds):
                hist.popleft()

            # Проверка, стабилен ли статус
            states = [status for _, status in hist]
            if len(states) >= self.min_confirmations and all(s == is_free for s in states[-self.min_confirmations:]):
                self.latest_stable_status[cam_id][slot_id] = is_free

            # Возвращаем последнее стабильное значение
            stable_status[slot_id] = self.latest_stable_status[cam_id].get(slot_id, True)

        return stable_status

    def get_latest_status(self, cam_id):
        return self.latest_stable_status.get(cam_id, {})

    def clear_history(self):
        self.history.clear()
        self.latest_stable_status.clear()
