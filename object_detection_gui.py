import sys
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("E-Waste Object Detection")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create label for video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Load YOLO model
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='public/models/best.pt')
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        # Set up timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Run detection
            results = self.model(frame)
            
            # Draw detections
            frame_with_detections = results.render()[0]
            
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale image to fit label
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Update label
            self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_()) 