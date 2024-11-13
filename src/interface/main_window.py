from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QProgressBar,
                            QStatusBar, QGroupBox, QApplication)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from app_parking_management import VehicleTrackingSystem
import os
import time
import sys

class VideoProcessingThread(QThread):
    frame_processed = pyqtSignal(np.ndarray)
    progress_updated = pyqtSignal(int)
    counts_updated = pyqtSignal(dict)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, source_path, target_path):
        super().__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.is_running = True
        self.tracker = VehicleTrackingSystem(source_path, target_path)
        self.video_writer = None
        
        # Initialize counts
        self.in_count = 0
        self.out_count = 0

    def run(self):
        try:
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                raise ValueError("Could not open video source")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Initialize video writer with platform-specific codec
            if sys.platform == 'darwin':  # macOS
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            else:  # Windows/Linux
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.target_path), exist_ok=True)

            # Initialize video writer
            self.video_writer = cv2.VideoWriter(
                self.target_path,
                fourcc,
                fps,
                (width, height)
            )

            if not self.video_writer.isOpened():
                raise ValueError("Failed to initialize video writer")

            frame_number = 0
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame = self.tracker.process_frame(frame, frame_number)
                
                # Update counts
                if hasattr(self.tracker, 'line_zone'):
                    self.in_count = self.tracker.line_zone.in_count
                    self.out_count = self.tracker.line_zone.out_count
                    
                    self.counts_updated.emit({
                        'in': self.in_count,
                        'out': self.out_count,
                        'total': self.in_count + self.out_count
                    })
                
                # Save frame
                try:
                    if self.video_writer is not None and self.video_writer.isOpened():
                        self.video_writer.write(processed_frame)
                except Exception as e:
                    self.error_occurred.emit(f"Error saving frame: {str(e)}")
                
                # Emit frame for display
                self.frame_processed.emit(processed_frame)
                
                frame_number += 1
                progress = int((frame_number / total_frames) * 100)
                self.progress_updated.emit(progress)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            # Clean up resources
            if 'cap' in locals():
                cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"Video saved to: {self.target_path}")
            self.finished.emit()

    def stop(self):
        self.is_running = False
        # Ensure video writer is properly closed
        if self.video_writer is not None:
            self.video_writer.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Tracking System")
        
        # Set the default save path
        self.default_save_path = "/Users/mustafaumutozbek/Documents/finance_analysis/factory_analysis/ml-depth-pro/data"
        
        # Get screen size
        screen = QApplication.primaryScreen().geometry()
        # Use smaller percentage of screen for better fit
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))
        
        # Ensure minimum size
        self.setMinimumSize(1024, 768)
        
        # Center the window
        self.center_window()
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
            QProgressBar {
                border: 2px solid #666;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0d6efd;
            }
        """)
        self.setup_ui()
        self.video_thread = None
        self.start_time = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_processing_time)

    def center_window(self):
        # Center window on screen
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.primaryScreen().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Adjust spacing and margins based on window size
        window_size = self.size()
        margin = int(window_size.width() * 0.02)  # 2% of window width
        spacing = int(margin / 2)
        
        layout.setSpacing(spacing)
        layout.setContentsMargins(margin, margin, margin, margin)

        # Create title
        title_label = QLabel("Vehicle Tracking System")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: white;
            padding: 10px;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Video display with frame
        video_frame = QGroupBox()
        video_frame.setStyleSheet("""
            QGroupBox {
                border: 2px solid #444;
                border-radius: 10px;
                padding: 10px;
                background-color: #1e1e1e;
            }
        """)
        video_layout = QVBoxLayout(video_frame)
        
        self.video_display = QLabel()
        self.video_display.setMinimumSize(1280, 720)
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setStyleSheet("border: none;")
        video_layout.addWidget(self.video_display)

        # Controls panel
        controls_frame = QGroupBox("Controls")
        controls_frame.setStyleSheet("""
            QGroupBox {
                border: 2px solid #444;
                border-radius: 10px;
                padding: 15px;
                background-color: #1e1e1e;
                color: white;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setSpacing(20)  # Increased spacing
        controls_layout.setContentsMargins(20, 20, 20, 20)  # Increased margins

        # Buttons
        self.select_file_btn = QPushButton("Select Video")
        self.start_btn = QPushButton("Start Processing")
        self.stop_btn = QPushButton("Stop")
        
        self.select_file_btn.clicked.connect(self.select_video_file)
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        
        # Make buttons larger
        button_style = """
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 12px 24px;  # Increased padding
                border-radius: 6px;
                font-size: 16px;     # Increased font size
                min-width: 150px;    # Increased minimum width
                min-height: 40px;    # Set minimum height
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """

        # Update button styles
        self.select_file_btn.setStyleSheet(button_style)
        self.start_btn.setStyleSheet(button_style)
        self.stop_btn.setStyleSheet(button_style)

        # Ensure buttons are visible with proper spacing
        controls_layout.addWidget(self.select_file_btn)
        controls_layout.addSpacing(20)  # Add space between buttons
        controls_layout.addWidget(self.start_btn)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()

        # Statistics panel
        stats_frame = QGroupBox("Statistics")
        stats_frame.setStyleSheet("""
            QGroupBox {
                border: 2px solid #444;
                border-radius: 10px;
                padding: 15px;
                background-color: #1e1e1e;
                color: white;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        stats_layout = QHBoxLayout(stats_frame)
        
        # Count labels with improved styling
        self.in_count_label = QLabel("Cars IN: 0")
        self.out_count_label = QLabel("Cars OUT: 0")
        self.total_count_label = QLabel("Total Cars: 0")
        
        for label in [self.in_count_label, self.out_count_label, self.total_count_label]:
            label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    border: 2px solid #0d6efd;
                    border-radius: 5px;
                    background-color: #212529;
                    min-width: 150px;
                    text-align: center;
                }
            """)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        stats_layout.addWidget(self.in_count_label)
        stats_layout.addWidget(self.out_count_label)
        stats_layout.addWidget(self.total_count_label)

        # Progress section
        progress_frame = QGroupBox("Progress")
        progress_frame.setStyleSheet("""
            QGroupBox {
                border: 2px solid #444;
                border-radius: 10px;
                padding: 15px;
                background-color: #1e1e1e;
                color: white;
                font-size: 14px;
            }
        """)
        progress_layout = QVBoxLayout(progress_frame)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(20)
        self.processing_time = QLabel("Processing Time: 00:00")
        self.processing_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.processing_time)

        # Add all components to main layout
        layout.addWidget(title_label)
        layout.addWidget(video_frame)
        layout.addWidget(controls_frame)
        layout.addWidget(stats_frame)
        layout.addWidget(progress_frame)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #1e1e1e;
                color: white;
                padding: 5px;
                font-size: 13px;
            }
        """)
        self.status_bar.showMessage("Ready")

        # Adjust video display size
        video_width = int(self.width() * 0.7)  # Reduced from 0.8
        video_height = int(video_width * 9/16)
        self.video_display.setMinimumSize(video_width, video_height)
        self.video_display.setMaximumHeight(int(self.height() * 0.7))  # Limit maximum height

        # Make statistics more prominent
        stats_style = """
            QLabel {
                font-size: 18px;     # Increased font size
                font-weight: bold;
                padding: 15px;       # Increased padding
                border: 2px solid #0d6efd;
                border-radius: 8px;
                background-color: #212529;
                min-width: 180px;    # Increased minimum width
                text-align: center;
            }
        """
        
        for label in [self.in_count_label, self.out_count_label, self.total_count_label]:
            label.setStyleSheet(stats_style)

        # Adjust layout spacing
        layout.setSpacing(15)  # Increased spacing between elements
        for frame in [video_frame, controls_frame, stats_frame, progress_frame]:
            frame.layout().setContentsMargins(15, 15, 15, 15)
            frame.layout().setSpacing(15)

    def select_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi);;All Files (*)"
        )
        if file_name:
            # Create output filename with timestamp in the specified directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            
            # Use the default save path
            self.source_path = file_name
            self.target_path = os.path.join(
                self.default_save_path,
                f"{base_name}_processed_{timestamp}.mp4"
            )
            
            # Create directory if it doesn't exist
            os.makedirs(self.default_save_path, exist_ok=True)
            
            self.start_btn.setEnabled(True)
            self.status_bar.showMessage(f"Selected: {file_name}")
            self.status_bar.showMessage(f"Output will be saved to: {self.target_path}")

    def start_processing(self):
        if hasattr(self, 'source_path'):
            try:
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.select_file_btn.setEnabled(False)
                
                # Create output path
                output_dir = os.path.dirname(self.target_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Initialize processing thread
                self.video_thread = VideoProcessingThread(self.source_path, self.target_path)
                self.video_thread.frame_processed.connect(self.update_frame)
                self.video_thread.progress_updated.connect(self.update_progress)
                self.video_thread.counts_updated.connect(self.update_counts)
                self.video_thread.error_occurred.connect(self.handle_error)
                self.video_thread.finished.connect(self.processing_finished)
                
                self.video_thread.start()
                self.start_time = time.time()
                self.timer.start(1000)
                
                self.status_bar.showMessage(f"Processing video... Output will be saved to: {self.target_path}")
                
            except Exception as e:
                self.handle_error(f"Failed to start processing: {str(e)}")

    def stop_processing(self):
        if self.video_thread:
            self.video_thread.stop()
            self.stop_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            self.select_file_btn.setEnabled(True)
            self.status_bar.showMessage("Processing stopped. Partial results saved.")

    def update_frame(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]
            bytes_per_line = 3 * w
            
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, 
                           QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            scaled_pixmap = pixmap.scaled(
                self.video_display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.video_display.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.handle_error(f"Frame update error: {str(e)}")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_processing_time(self):
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.processing_time.setText(f"Processing Time: {minutes:02d}:{seconds:02d}")

    def processing_finished(self):
        self.timer.stop()
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        
        if os.path.exists(self.target_path):
            file_size = os.path.getsize(self.target_path) / (1024 * 1024)  # Size in MB
            self.status_bar.showMessage(
                f"Processing completed. Saved to: {self.target_path} (Size: {file_size:.1f} MB)"
            )
        else:
            self.status_bar.showMessage("Processing completed but file not saved successfully")

    def handle_error(self, error_message):
        self.status_bar.showMessage(f"Error: {error_message}")
        self.stop_processing()

    def update_counts(self, counts):
        self.in_count_label.setText(f"Cars IN: {counts['in']}")
        self.out_count_label.setText(f"Cars OUT: {counts['out']}")
        self.total_count_label.setText(f"Total Cars: {counts['total']}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'video_display'):
            # Adjust video display size while maintaining aspect ratio
            window_size = self.size()
            video_width = int(window_size.width() * 0.7)  # Reduced from 0.8
            video_height = int(video_width * 9/16)
            max_height = int(window_size.height() * 0.7)
            
            if video_height > max_height:
                video_height = max_height
                video_width = int(video_height * 16/9)
                
            self.video_display.setMinimumSize(video_width, video_height) 