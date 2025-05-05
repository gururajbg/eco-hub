import React, { useState, useRef, useEffect } from 'react';
import { Box, Button, Typography, Paper } from '@mui/material';
import { detectObjects } from '../api/detection';

const EwasteDetection: React.FC = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startDetection = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsDetecting(true);
        detectObjects();
      }
    } catch (err) {
      console.error('Error accessing webcam:', err);
    }
  };

  const stopDetection = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsDetecting(false);
  };

  const detectObjects = async () => {
    if (!videoRef.current || !canvasRef.current || !isDetecting) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      // Get image data from canvas
      const imageData = canvas.toDataURL('image/jpeg');
      
      // Send to API for detection
      const response = await detectObjects(imageData);
      
      // Clear previous drawings
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Draw detection boxes
      response.detections.forEach((detection: any) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const confidence = detection.confidence;
        const className = detection.class;

        // Draw bounding box
        context.strokeStyle = '#00FF00';
        context.lineWidth = 2;
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Draw label
        context.fillStyle = '#00FF00';
        context.font = '16px Arial';
        context.fillText(
          `Class ${className} (${(confidence * 100).toFixed(1)}%)`,
          x1,
          y1 - 5
        );
      });
    } catch (error) {
      console.error('Error in detection:', error);
    }

    // Continue detection loop
    requestAnimationFrame(detectObjects);
  };

  return (
    <Paper elevation={3} sx={{ p: 3, maxWidth: 800, mx: 'auto', mt: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        E-Waste Detection
      </Typography>
      
      <Box sx={{ position: 'relative', width: '100%', mb: 2 }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ width: '100%', display: isDetecting ? 'block' : 'none' }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            display: isDetecting ? 'block' : 'none'
          }}
        />
      </Box>

      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={startDetection}
          disabled={isDetecting}
        >
          Start Detection
        </Button>
        <Button
          variant="contained"
          color="secondary"
          onClick={stopDetection}
          disabled={!isDetecting}
        >
          Stop Detection
        </Button>
      </Box>
    </Paper>
  );
};

export default EwasteDetection; 