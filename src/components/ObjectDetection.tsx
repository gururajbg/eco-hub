import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

interface DetectionResult {
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
}

const ObjectDetection: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detections, setDetections] = useState<DetectionResult[]>([]);

  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Load the converted model
        const modelUrl = '/models/model.json';
        const loadedModel = await tf.loadGraphModel(modelUrl);
        
        // Warm up the model
        const dummyInput = tf.zeros([1, 640, 640, 3]);
        await loadedModel.predict(dummyInput);
        dummyInput.dispose();
        
        setModel(loadedModel);
        setIsLoading(false);
      } catch (error) {
        console.error('Error loading model:', error);
        setError('Error loading model. Please make sure the model is properly converted and placed in the public/models directory.');
        setIsLoading(false);
      }
    };

    loadModel();
  }, []);

  useEffect(() => {
    if (!model || !videoRef.current || !canvasRef.current) return;

    const detectObjects = async () => {
      if (!videoRef.current || !canvasRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (!ctx) return;

      try {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Preprocess the image
        const input = tf.browser.fromPixels(video)
          .resizeNearestNeighbor([640, 640])
          .toFloat()
          .div(255.0)
          .expandDims(0);

        // Run inference
        const predictions = await model.predict(input) as tf.Tensor;
        const results = await predictions.array();

        // Process detections
        const processedDetections: DetectionResult[] = [];
        const gridSize = 20;
        const cellSize = 640 / gridSize;

        for (let i = 0; i < gridSize; i++) {
          for (let j = 0; j < gridSize; j++) {
            const confidence = results[0][i][j][0];
            if (confidence > 0.3) {
              processedDetections.push({
                class: 'object',
                confidence: confidence,
                bbox: [
                  j * cellSize,
                  i * cellSize,
                  (j + 1) * cellSize,
                  (i + 1) * cellSize
                ]
              });
            }
          }
        }

        setDetections(processedDetections);

        // Draw detections on canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        processedDetections.forEach(detection => {
          const [x1, y1, x2, y2] = detection.bbox;
          ctx.strokeStyle = '#00FF00';
          ctx.lineWidth = 2;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.fillStyle = '#00FF00';
          ctx.fillText(`${detection.class} (${(detection.confidence * 100).toFixed(2)}%)`, x1, y1 - 5);
        });

        // Clean up
        input.dispose();
        predictions.dispose();

        // Continue detection loop
        requestAnimationFrame(detectObjects);
      } catch (error) {
        console.error('Error during detection:', error);
        setError('Error during object detection. Please try again.');
      }
    };

    // Start video stream
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            detectObjects();
          };
        }
      })
      .catch(err => {
        console.error('Error accessing camera:', err);
        setError('Error accessing camera. Please make sure you have granted camera permissions.');
      });

    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [model]);

  return (
    <div className="relative">
      {isLoading && <div className="text-center p-4">Loading model...</div>}
      {error && (
        <div className="text-red-500 p-4 text-center">
          {error}
          <p className="mt-2 text-sm">
            Please make sure you have:
            <br />
            1. Run the convert_model.py script to convert your PyTorch model
            <br />
            2. Placed the converted model files in the public/models directory
          </p>
        </div>
      )}
      <video
        ref={videoRef}
        className="w-full h-auto"
        autoPlay
        muted
        playsInline
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full"
      />
      <div className="mt-4">
        <h3 className="text-lg font-semibold">Detections:</h3>
        <ul>
          {detections.map((detection, index) => (
            <li key={index}>
              {detection.class}: {(detection.confidence * 100).toFixed(2)}%
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default ObjectDetection; 