import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vehicle_tracking.log'),
        logging.StreamHandler()
    ]
)

class VehicleTrackingSystem:
    def __init__(self, source_path: str, target_path: str):
        self.source_path = source_path
        self.target_path = target_path
        
        # Initialize video info
        self.video_info = sv.VideoInfo.from_video_path(source_path)
        logging.info(f"Video Info: {self.video_info}")
        
        # Initialize YOLO model
        self.model = YOLO("yolov8x.pt")
        
        # Initialize ByteTracker
        self.byte_tracker = sv.ByteTrack()
        
        # Initialize line counter with proper position
        self.line_start = sv.Point(0, int(self.video_info.height * 0.5))
        self.line_end = sv.Point(self.video_info.width, int(self.video_info.height * 0.5))
        self.line_zone = sv.LineZone(start=self.line_start, end=self.line_end)
        
        # Initialize trace annotator
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=30
        )
        
        # Vehicle classes (now we'll treat all vehicle classes as cars)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        try:
            # Run detection
            results = self.model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter for all vehicle types but label them as cars
            mask = np.array([
                int(class_id) in self.vehicle_classes  # Include all vehicle types
                for class_id in detections.class_id
            ])
            detections = detections[mask]
            
            # Update tracking
            tracked_detections = self.byte_tracker.update_with_detections(detections)
            
            # Update line counter
            self.line_zone.trigger(detections=tracked_detections)
            
            # Prepare frame for annotation
            annotated_frame = frame.copy()
            
            # Draw trace paths
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame,
                detections=tracked_detections
            )
            
            # Draw boxes and labels (all as cars)
            for i, (xyxy, confidence, _) in enumerate(zip(
                tracked_detections.xyxy,
                tracked_detections.confidence,
                tracked_detections.class_id
            )):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Car {confidence:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # Draw line counter with improved visibility
            line_color = (0, 255, 255)  # Yellow
            cv2.line(
                annotated_frame,
                (int(self.line_start.x), int(self.line_start.y)),
                (int(self.line_end.x), int(self.line_start.y)),
                line_color,
                4  # Thicker line
            )
            
            # Add direction indicators with better visibility
            mid_x = (self.line_start.x + self.line_end.x) // 2
            
            # Draw direction arrows
            arrow_length = 50
            arrow_color = (0, 255, 255)  # Yellow
            
            # IN arrow
            cv2.arrowedLine(
                annotated_frame,
                (int(mid_x - 150), int(self.line_start.y - 40)),
                (int(mid_x - 50), int(self.line_start.y - 40)),
                arrow_color,
                3,
                tipLength=0.3
            )
            
            # OUT arrow
            cv2.arrowedLine(
                annotated_frame,
                (int(mid_x + 150), int(self.line_start.y - 40)),
                (int(mid_x + 50), int(self.line_start.y - 40)),
                arrow_color,
                3,
                tipLength=0.3
            )
            
            # Add text with background for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            
            # IN text
            in_text = "IN"
            (text_width, text_height), _ = cv2.getTextSize(in_text, font, font_scale, thickness)
            cv2.rectangle(
                annotated_frame,
                (int(mid_x - 150 - 10), int(self.line_start.y - 40 - text_height - 10)),
                (int(mid_x - 150 + text_width + 10), int(self.line_start.y - 40 + 10)),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                annotated_frame,
                in_text,
                (int(mid_x - 150), int(self.line_start.y - 40)),
                font,
                font_scale,
                arrow_color,
                thickness
            )
            
            # OUT text
            out_text = "OUT"
            (text_width, text_height), _ = cv2.getTextSize(out_text, font, font_scale, thickness)
            cv2.rectangle(
                annotated_frame,
                (int(mid_x + 150 - text_width - 10), int(self.line_start.y - 40 - text_height - 10)),
                (int(mid_x + 150 + 10), int(self.line_start.y - 40 + 10)),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                annotated_frame,
                out_text,
                (int(mid_x + 150 - text_width), int(self.line_start.y - 40)),
                font,
                font_scale,
                arrow_color,
                thickness
            )
            
            # In the process_frame method, update the count overlay with better design
            # Add count overlay with improved design
            # Create background for counts
            overlay_height = 130
            overlay_width = 250
            overlay = annotated_frame[10:10+overlay_height, 10:10+overlay_width].copy()
            cv2.rectangle(
                annotated_frame,
                (10, 10),
                (10 + overlay_width, 10 + overlay_height),
                (0, 0, 0),
                -1
            )
            cv2.rectangle(
                annotated_frame,
                (10, 10),
                (10 + overlay_width, 10 + overlay_height),
                (0, 255, 255),  # Yellow border
                2
            )

            # Add counts with improved styling
            # IN count
            cv2.putText(
                annotated_frame,
                "Cars IN:",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),  # Yellow text
                2
            )
            cv2.putText(
                annotated_frame,
                str(self.line_zone.in_count),
                (160, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),  # Green number
                2
            )

            # OUT count
            cv2.putText(
                annotated_frame,
                "Cars OUT:",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),  # Yellow text
                2
            )
            cv2.putText(
                annotated_frame,
                str(self.line_zone.out_count),
                (160, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),  # Green number
                2
            )

            # Total count
            cv2.putText(
                annotated_frame,
                "TOTAL:",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),  # Yellow text
                2
            )
            cv2.putText(
                annotated_frame,
                str(self.line_zone.in_count + self.line_zone.out_count),
                (160, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),  # Green number
                2
            )
            
            return annotated_frame
            
        except Exception as e:
            logging.error(f"Error processing frame {frame_number}: {str(e)}")
            return frame
    
    def process_video(self):
        try:
            def callback(frame: np.ndarray, frame_number: int) -> np.ndarray:
                if frame_number % 30 == 0:
                    logging.info(f"Processing frame {frame_number}")
                return self.process_frame(frame, frame_number)
            
            sv.process_video(
                source_path=self.source_path,
                target_path=self.target_path,
                callback=callback
            )
            
            logging.info("Video processing completed successfully")
            
        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            raise

def main():
    try:
        # Initialize the tracking system
        tracker = VehicleTrackingSystem(
            source_path='/Users/mustafaumutozbek/Documents/finance_analysis/factory_analysis/ml-depth-pro/data/parking_test.mp4',
            target_path='/Users/mustafaumutozbek/Documents/finance_analysis/factory_analysis/ml-depth-pro/data/output_tracking.mp4'
        )
        
        print("Starting vehicle tracking...")
        tracker.process_video()
        print("\nProcessing completed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Check vehicle_tracking.log for details")

if __name__ == "__main__":
    main()