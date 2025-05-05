import cv2
import time
import threading
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import socket
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Object Detection with Web Server')
parser.add_argument('--port', type=int, default=5001, help='Port for the web server')
parser.add_argument('--model1', type=str, default="./model1.pt", help='Path to the first YOLO model')
parser.add_argument('--model2', type=str, default="./model2.pt", help='Path to the second YOLO model')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
args = parser.parse_args()

# Load models
model_s = YOLO(args.model1)
model_m = YOLO(args.model2)

# Define colors for each class (Total: 8 classes)
CLASS_COLORS = {
    'Mobile': (255, 0, 0),           # Blue
    'PCB': (0, 255, 0),              # Green
    'Phone_Battery': (0, 0, 255),    # Red
    'Remote': (255, 255, 0),         # Cyan
    'Adapter': (255, 0, 255),        # Magenta
    'Headset': (0, 255, 255),        # Yellow
    'Keyboard': (128, 0, 128),       # Purple
    'Mouse': (0, 128, 255),          # Orange
}

# Confidence threshold
CONF_THRESHOLD = args.conf

# Global variable for the latest processed frame
global_frame = None
frame_lock = threading.Lock()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def get_local_ip():
    """Get the local IP address of the server"""
    try:
        # Create a socket connection to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return "127.0.0.1"

def process_webcam():
    """Process webcam feed and perform object detection"""
    global global_frame
    
    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # For FPS calculation
    prev_time = time.time()
    fps = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start time for FPS
        current_time = time.time()
        
        # Inference with both models
        results_s = model_s(frame)[0]
        results_m = model_m(frame)[0]
        
        # Prepare annotated frame
        annotated_frame = frame.copy()
        
        # Collect boxes from both models
        detections = []
        for results, model in [(results_s, model_s), (results_m, model_m)]:
            for box in results.boxes:
                if box.conf.item() >= CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = model.names[cls]
                    color = CLASS_COLORS.get(class_name, (255, 255, 255))  # Default white
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'conf': conf,
                        'class_name': class_name,
                        'color': color
                    })
        
        # Draw all detections
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class_name']}: {det['conf']:.2f}"
            color = det['color']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Calculate FPS
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Draw FPS on frame
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Update the global frame with the annotated frame
        with frame_lock:
            global_frame = annotated_frame.copy()
        
    cap.release()

def generate_frames():
    """Generate frames for streaming"""
    global global_frame
    
    while True:
        # Ensure we have a frame to send
        with frame_lock:
            if global_frame is None:
                continue
            frame_to_send = global_frame.copy()
        
        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame_to_send)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the format expected by Flask's Response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Add a small delay to control streaming rate
        time.sleep(0.03)  # ~30 FPS

@app.route('/api/video_feed')
def video_feed():
    """Route for the video feed"""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/classes')
def get_classes():
    """Get the list of detection classes and their colors"""
    # Convert BGR colors to RGB for frontend
    classes = {name: {'color': (c[2], c[1], c[0])} for name, c in CLASS_COLORS.items()}
    return jsonify(classes)

if __name__ == "__main__":
    # Start webcam processing in a separate thread
    webcam_thread = threading.Thread(target=process_webcam, daemon=True)
    webcam_thread.start()
    
    # Print access information
    server_ip = get_local_ip()
    print(f"Object Detection Server starting up!")
    print(f"API endpoints available at: http://{server_ip}:{args.port}")
    print(f"Video feed: http://{server_ip}:{args.port}/api/video_feed")
    print(f"Classes: http://{server_ip}:{args.port}/api/classes")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)