import asyncio
import cv2
import logging
import signal

from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
from colorama import Fore, Style, init
init(autoreset=True)

from core.video_stream import VideoStream
from core.detector import ObjectDetector
from core.zone_manager import ZoneManager
from core.occupancy_analyzer import OccupancyAnalyzer
from core.visualizer import draw_parking_zones, draw_detections
from core.dashboard import StatusDashboard
from core.aggregator import GlobalAggregator
from core.display_board import DisplayBoard
from core.event_logger import log_slot_event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")


LOG_DIR = Path("logs")
IMG_LOG_DIR = Path("logs/images")
STATUS_LOG_FILE = LOG_DIR / "status_changes.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)
IMG_LOG_DIR.mkdir(parents=True, exist_ok=True)


async def print_aggregated_status_periodically(aggregator, stop_event, interval=5):
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        print(f"\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} Aggregated Parking Status (live):")
        aggregated = aggregator.get_aggregated_status()
        for slot_id, is_free in aggregated.items():
            color = Fore.GREEN if is_free else Fore.RED
            status = "Free" if is_free else "Occupied"
            print(f"{slot_id}: {color}{status}{Style.RESET_ALL}")


import threading

def render_display_loop(stop_event, display_board):
    while not stop_event.is_set():
        display_board.render()

        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            stop_event.set()
            break
        elif key == ord("d"):
            display_board.next_page()
        elif key == ord("a"):
            display_board.prev_page()


async def process_camera(cam_id, cam_cfg, detector, zone_manager, analyzer, aggregator,
                         mode, stop_event, display_board, test_video_path=None):
    logger.info(f"[{cam_id}] Starting in mode: {mode}")

    source = test_video_path if mode == "video" else cam_cfg.url
    stream = VideoStream(source)
    zone_manager.load_zones(cam_id)
    zones = zone_manager.get_zones(cam_id)

    output_path = Path(f"tests/output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{cam_id}_annotated.avi")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    # dashboard = StatusDashboard()
    last_statuses = {}

    while not stop_event.is_set():
        frame = await stream.get_frame()
        if frame is None:
            await asyncio.sleep(0.05)
            continue

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))

        detections = detector.detect(frame)
        status = analyzer.analyze(cam_id, detections)

        aggregator.update(cam_id, status)

        frame = draw_detections(frame, detections)
        frame = draw_parking_zones(frame, zones, status)
        # dashboard.update({slot_id: status[slot_id] for slot_id in zones})

        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for slot_id, is_free in status.items():
            prev = last_statuses.get(slot_id)
            color = Fore.GREEN if is_free else Fore.RED
            state = "Free" if is_free else "Occupied"
            if prev is None:
                print(f"[INIT]  {slot_id}: {color}{state}{Style.RESET_ALL}")
            elif prev != is_free:
                print(f"[UPDATE] {slot_id}: {color}{state}{Style.RESET_ALL}")

                # логируем событие с сохранением ROI и JSON
                zone = zones.get(slot_id)
                if zone and "coords" in zone:
                    log_slot_event(cam_id, slot_id, prev, is_free, frame.copy(), zone["coords"])

            last_statuses[slot_id] = is_free

        writer.write(frame)
        if display_board:
            display_board.update_frame(cam_id, frame)

        if mode == "video" and stream.cap.get(cv2.CAP_PROP_POS_FRAMES) >= stream.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            logger.info(f"[{cam_id}] End of test video")
            break

    stream.release()
    if writer:
        writer.release()
    # dashboard.close()


async def main():
    cfg = OmegaConf.load("config/config.yaml")
    mode = cfg.get("mode", "live")

    if mode == "video" and "weather" in cfg:
        weather = cfg.weather
        for cam_id in cfg.test_videos:
            template = cfg.test_videos[cam_id]
            cfg.test_videos[cam_id] = template.format(weather=weather)
        print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Using video directory for weather: {weather}")

    detector = ObjectDetector(cfg.model.path, conf_threshold=cfg.model.conf_threshold)
    zone_manager = ZoneManager(iou_threshold=cfg.logic.iou_threshold)

    filter_cfg = cfg.logic.get("filter", {})
    window_seconds = filter_cfg.get("window_seconds", 2.0)
    min_confirmations = filter_cfg.get("min_confirmations", 3)
    analyzer = OccupancyAnalyzer(zone_manager,
                                window_seconds=window_seconds,
                                min_confirmations=min_confirmations)

    aggregator = GlobalAggregator(zone_manager)
    display_board = DisplayBoard(width=1280, height=720, max_columns=2) if cfg.get("show_display", True) else None

    stop_event = asyncio.Event()

    def handle_exit(*args):
        logger.info("Received exit signal. Stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    tasks = []

    cam_sources = (
        cfg.test_videos.items() if mode == "video" else cfg.cameras.items()
    )

    for cam_id, cam_cfg in cam_sources:
        test_video = cam_cfg if mode == "video" else None
        camera_cfg = {} if mode == "video" else cam_cfg

        task = process_camera(
            cam_id,
            camera_cfg,
            detector,
            zone_manager,
            analyzer,
            aggregator,
            mode,
            stop_event,
            display_board,
            test_video_path=test_video
        )
        tasks.append(task)

    tasks.append(print_aggregated_status_periodically(aggregator, stop_event, interval=5))
    # tasks.append(render_display_loop(stop_event, display_board))  # Одно окно

    if cfg.get("show_display", True):
        render_thread = threading.Thread(target=render_display_loop, args=(stop_event, display_board))
        render_thread.start()
    else:
        render_thread = None

    await asyncio.gather(*tasks, return_exceptions=True)
    if render_thread:
        render_thread.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Interrupted by user. Exiting.")
