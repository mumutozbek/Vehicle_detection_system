import cv2
import numpy as np
from supervision.tools.detections import Detections

class Visualizer:
    @staticmethod
    def draw_line_counter(frame: np.ndarray, start_point: tuple, end_point: tuple, count: int):
        cv2.line(frame, start_point, end_point, (0, 255, 0), 4)
        cv2.putText(
            frame,
            f"Vehicles: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return frame 