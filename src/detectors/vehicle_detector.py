import numpy as np
import supervision as sv
from ultralytics import YOLO
import logging
import cv2
from supervision.tools.detections import Detections
from supervision.tools.line_counter import LineCounter
from supervision.draw.annotator import BoxAnnotator, LabelAnnotator
from supervision.geometry import Point
from ..config.settings import VEHICLE_CLASSES, MODEL_PATH

class VehicleDetector:
    def __init__(self, source_path: str, target_path: str):
        self.source_path = source_path
        self.target_path = target_path
        self._init_model()
        self._init_annotators()
        self._init_video_info()
        self._init_line_counter()
        
    def _init_model(self):
        self.model = YOLO(MODEL_PATH)
        
    def _init_annotators(self):
        self.box_annotator = BoxAnnotator(thickness=4)
        self.label_annotator = LabelAnnotator(text_thickness=4, text_scale=2)
        
    def _init_video_info(self):
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {self.source_path}")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
    def _init_line_counter(self):
        self.line_counter = LineCounter(
            start=Point(0, int(self.height * 0.7)),
            end=Point(self.width, int(self.height * 0.7))
        )

    # Rest of the VehicleDetector implementation... 