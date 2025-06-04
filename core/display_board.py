import cv2
import numpy as np
from math import ceil
import asyncio

class DisplayBoard:
    def __init__(self, width=1280, height=720, max_columns=2, tiles_per_page=4):
        self.frames = {}  # cam_id -> frame
        self.width = width
        self.height = height
        self.max_columns = max_columns
        self.tiles_per_page = tiles_per_page
        self.current_page = 0
        self.lock = asyncio.Lock()

    def render(self):
        def sort_key(cam_id):
            return int(''.join(filter(str.isdigit, cam_id)) or 0)

        cams = sorted(self.frames.keys(), key=sort_key)
        N = len(cams)
        if N == 0:
            return
        total_pages = ceil(N / self.tiles_per_page)

        start_idx = self.current_page * self.tiles_per_page
        end_idx = min(start_idx + self.tiles_per_page, N)
        current_cams = cams[start_idx:end_idx]

        cols = self.max_columns
        rows = self.tiles_per_page // self.max_columns

        tile_w = self.width // cols
        tile_h = self.height // rows

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for idx, cam_id in enumerate(current_cams):
            row = idx // cols
            col = idx % cols

            frame = self.frames.get(cam_id)
            if frame is None:
                continue

            resized = cv2.resize(frame, (tile_w, tile_h))

            label = f"{cam_id}"
            cv2.putText(resized, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(resized, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w
            canvas[y1:y2, x1:x2] = resized

        cv2.putText(canvas, f"Page {self.current_page + 1}/{total_pages}",
                    (10, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("All Cameras", canvas)


    def update_frame(self, cam_id, frame):
        self.frames[cam_id] = frame

    def next_page(self):
        total_pages = ceil(len(self.frames) / self.tiles_per_page)
        if self.current_page < total_pages - 1:
            self.current_page += 1

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1