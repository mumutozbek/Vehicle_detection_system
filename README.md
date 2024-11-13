# Vehicle Tracking System

A computer vision system for detecting and counting vehicles in video feeds using YOLO and computer vision techniques.

## Features
- Real-time vehicle detection and tracking
- Accurate IN/OUT vehicle counting
- Modern dark-themed GUI interface
- Real-time statistics and progress tracking
- Automatic video saving with timestamps
- Multi-platform support (Windows, macOS, Linux)

## Demo
![Vehicle Tracking Demo](demo.gif)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/mumutozbek/vehicle-tracking-system.git
cd vehicle-tracking-system
```

2. Create and activate virtual environment:
```bash
# For macOS/Linux
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python main.py
```

## Project Structure
```
vehicle-tracking-system/
├── src/
│   ├── interface/        # GUI components
│   │   └── main_window.py
│   ├── config/          # Configuration files
│   │   └── settings.py
│   ├── utils/           # Utility functions
│   │   └── visualization.py
│   └── detectors/       # Detection models
│       └── vehicle_detector.py
├── data/
│   ├── input/           # Place input videos here
│   └── output/          # Processed videos saved here
├── models/              # YOLO models (downloaded automatically)
├── logs/               # Application logs
├── app_parking_management.py
├── main.py
└── requirements.txt
```


https://github.com/user-attachments/assets/cea62682-fea9-403a-90c4-75eb1f1109c1


## Usage Guide

1. Start the Application:
   - Run `python main.py`
   - The modern dark-themed interface will appear

2. Load Video:
   - Click "Select Video" button
   - Choose your input video file
   - Supported formats: .mp4, .avi

3. Process Video:
   - Click "Start Processing" to begin analysis
   - Monitor real-time statistics:
     - IN/OUT vehicle counts
     - Total vehicles
     - Processing progress
     - Elapsed time

4. Output:
   - Processed videos are automatically saved with timestamps
   - Default save location: `data/output/`
   - Format: `original_name_processed_YYYYMMDD_HHMMSS.mp4`

## Features Details

### Detection & Tracking
- Uses YOLOv8 for accurate vehicle detection
- ByteTrack algorithm for robust vehicle tracking
- All vehicles classified as "Car" for simplicity

### Counting System
- Bidirectional counting (IN/OUT)
- Clear visual indicators for counting line
- Direction arrows for better understanding
- Real-time count updates

### User Interface
- Modern dark theme
- Real-time video preview
- Progress tracking
- Processing time display
- Status updates
- Error handling

### Video Processing
- Automatic codec selection for different platforms
- Progress tracking
- Error recovery
- Resource cleanup

## Requirements
- Python 3.8+
- PyQt6
- OpenCV
- Ultralytics YOLO
- Supervision
- NumPy

## Known Issues
- Some video codecs might not be supported on certain platforms
- High resolution videos might require more processing power

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Author
[Mustafa Umut Ozbek](https://github.com/mumutozbek)

## License
MIT License

## Acknowledgments
- YOLOv8 (9,11 etc.) for object detection
- ByteTrack for object tracking
- Supervision for video processing utilities
- PyQt6 for the user interface
