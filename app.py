from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the YOLO model
model = YOLO('public/models/best.pt')

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        # Get image data from request
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run detection
        results = model(frame)
        
        # Get detection results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': int(cls)
                })

        return jsonify({
            'success': True,
            'detections': detections
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 