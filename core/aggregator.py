import logging
from collections import defaultdict

logger = logging.getLogger("GlobalAggregator")

class GlobalAggregator:
    def __init__(self, zone_manager):
        """
        :param zone_manager: экземпляр ZoneManager с доступом к координатам и trust_score
        """
        self.zone_manager = zone_manager
        self.status_reports = defaultdict(dict)  # slot_id → {cam_id: bool}

    def update(self, cam_id, slot_statuses):
        """
        Обновляет статус мест с одной камеры.

        :param cam_id: str — идентификатор камеры
        :param slot_statuses: dict {slot_id: bool (True — свободно, False — занято)}
        """
        for slot_id, is_free in slot_statuses.items():
            self.status_reports[slot_id][cam_id] = is_free

    def get_aggregated_status(self):
        """
        Возвращает финальный статус всех слотов с учётом trust_scores.

        :return: dict {slot_id: bool}
        """
        aggregated = {}
        for slot_id, reports in self.status_reports.items():
            weighted_sum = 0.0
            total_weight = 0.0

            for cam_id, is_free in reports.items():
                weight = self.zone_manager.get_trust(cam_id, slot_id)
                total_weight += weight
                weighted_sum += weight * (1.0 if is_free else 0.0)

            if total_weight == 0:
                logger.warning(f"[Aggregator] Нет данных доверия для {slot_id}, принимаем по умолчанию: занято")
                aggregated[slot_id] = False
            else:
                average = weighted_sum / total_weight
                aggregated[slot_id] = average >= 0.5  # >= 0.5 → свободно, иначе занято

        return aggregated

    def clear(self):
        """Очищает собранные отчёты — вызывать перед новым циклом."""
        self.status_reports.clear()