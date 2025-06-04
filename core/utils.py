import random
import colorsys


def compute_iou(boxA, boxB):
    """
    Вычисляет IoU между двумя прямоугольниками.
    :param boxA: (x1, y1, x2, y2)
    :param boxB: (x1, y1, x2, y2)
    :return: float [0, 1]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter_area / float(areaA + areaB - inter_area)


def expand_box(x1, y1, x2, y2, scale=1.1):
    """
    Увеличивает размеры бокса на scale, сохраняя центр.
    """
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    x1n, y1n = int(cx - w / 2), int(cy - h / 2)
    x2n, y2n = int(cx + w / 2), int(cy + h / 2)
    return [x1n, y1n, x2n, y2n]


def generate_distinct_colors(n):
    """
    Генерирует n визуально различимых RGB цветов.
    :return: список (R, G, B)
    """
    hsv = [(i / n, 1, 1) for i in range(n)]
    rgb = list(map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
    random.shuffle(rgb)
    return rgb


def clamp(val, min_val, max_val):
    """Ограничивает значение в пределах [min_val, max_val]."""
    return max(min_val, min(val, max_val))