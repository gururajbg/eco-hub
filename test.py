import cv2
import time
from ultralytics import YOLO

# Load models
model_s = YOLO("./model1.pt")
model_m = YOLO("./model2.pt")

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
CONF_THRESHOLD = 0.5

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# For FPS calculation
prev_time = time.time()

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

    # Display frame
    cv2.imshow('Multi-Model Detection (Accurate)', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()