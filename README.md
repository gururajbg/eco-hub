# E-Waste Detection Application

This application provides both a desktop GUI and web interface for detecting e-waste items using a YOLO model.

## Setup Instructions

1. Install Python 3.8 or higher
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the YOLO model file (`best.pt`) in the `public/models/` directory

## Running the Application

### Desktop GUI Version
To start the desktop application, run:
```bash
python object_detection_gui.py
```

### Web Version
To start the web application, run:
```bash
python app.py
```
Then open your web browser and navigate to:
```
http://localhost:5000
```

## Features

- Real-time object detection using YOLO model
- Webcam support
- Display of detection results with bounding boxes and confidence scores
- Available as both desktop GUI and web interface
- Simple and intuitive interface

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Verify that the model file exists in the correct location
3. Check if your webcam is properly connected and accessible
4. Ensure you have sufficient system resources (CPU/GPU) for running the model
5. For web version, make sure no other application is using port 5000

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- PyQt5 (for desktop version)
- Flask (for web version)
- YOLO model file
