import React, { useState, useRef, useEffect } from "react";
import { Camera, Upload, Maximize, X, Save, RefreshCw } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface DetectedObject {
  label: string;
  confidence: number;
  box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

interface ModelData {
  format: string;
  data: string;
  input_shape: number[];
  output_shape: number[];
}

const ObjectDetection: React.FC = () => {
  const { toast } = useToast();
  const [mode, setMode] = useState<"webcam" | "upload" | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const modelRef = useRef<any>(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsModelLoading(true);
        
        // Load the model data
        const response = await fetch('/models/model.json');
        const modelData: ModelData = await response.json();
        
        if (modelData.format !== 'base64') {
          throw new Error('Unsupported model format');
        }
        
        // Store the model data
        modelRef.current = modelData;
        
        toast({ title: "Model Loaded Successfully", duration: 3000 });
      } catch (error) {
        console.error("Error loading object detection model:", error);
        toast({ 
          title: "Model Loading Failed", 
          description: error instanceof Error ? error.message : "Unknown error occurred",
          variant: "destructive", 
          duration: 10000 
        });
      } finally {
        setIsModelLoading(false);
      }
    };

    loadModel();

    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [toast]);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        mediaStreamRef.current = stream;
      }
      setMode("webcam");
    } catch (error) {
      console.error("Error accessing webcam:", error);
      toast({ 
        title: "Webcam Access Error", 
        description: "Unable to access the webcam. Please check permissions and try again.",
        variant: "destructive",
        duration: 5000 
      });
    }
  };

  const stopWebcam = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setMode(null);
    setDetectedObjects([]);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      setMode("upload");
      setDetectedObjects([]);
    }
  };

  const preprocessImage = async (imageElement: HTMLImageElement | HTMLVideoElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Convert to Float32Array
    const float32Data = new Float32Array(imageData.data.length / 4);
    for (let i = 0; i < imageData.data.length; i += 4) {
      float32Data[i / 4] = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3 / 255.0;
    }
    
    return float32Data;
  };

  const processPredictions = async (predictions: onnx.Tensor, imageWidth: number, imageHeight: number): Promise<DetectedObject[]> => {
    const boxes = await predictions.array() as number[][];
    const detectedObjects: DetectedObject[] = [];

    boxes.forEach((box: number[]) => {
      const [x1, y1, x2, y2, confidence, classId] = box;
      if (confidence > 0.5) {
        detectedObjects.push({
          label: `Class ${classId}`,
          confidence: confidence,
          box: {
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1
          }
        });
      }
    });

    return detectedObjects;
  };

  const detectObjects = async () => {
    if (!modelRef.current) {
      toast({ title: "Model Not Ready", variant: "destructive", duration: 3000 });
      return;
    }

    try {
      setIsDetecting(true);
      let imageElement: HTMLImageElement | HTMLVideoElement | null = null;

      if (mode === "webcam" && videoRef.current) {
        imageElement = videoRef.current;
      } else if (mode === "upload" && imageUrl) {
        const img = new Image();
        img.src = imageUrl;
        await new Promise((resolve) => {
          img.onload = resolve;
        });
        imageElement = img;
      }

      if (!imageElement) return;

      const tensor = await preprocessImage(imageElement);
      if (!tensor) return;

      // TODO: Implement YOLOv12 inference using the model data
      // This will require implementing the YOLOv12 detection logic
      // For now, we'll just show a placeholder
      const results: DetectedObject[] = [];
      setDetectedObjects(results);
      drawBoundingBoxes(results, imageElement.width, imageElement.height);
      
      toast({ title: "Detection completed", description: "YOLOv12 inference will be implemented soon" });
    } catch (error) {
      console.error("Error detecting objects:", error);
      toast({ title: "Detection Error", variant: "destructive", duration: 5000 });
    } finally {
      setIsDetecting(false);
    }
  };

  const drawBoundingBoxes = (objects: DetectedObject[], imageWidth: number, imageHeight: number) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!ctx || !canvas) return;

    canvas.width = imageWidth;
    canvas.height = imageHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    objects.forEach(obj => {
      const { box, label, confidence } = obj;
      const x = box.x * imageWidth;
      const y = box.y * imageHeight;
      const width = box.width * imageWidth;
      const height = box.height * imageHeight;

      // Draw bounding box
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // Draw label background
      ctx.fillStyle = '#00FF00';
      const textWidth = ctx.measureText(`${label} ${(confidence * 100).toFixed(1)}%`).width;
      ctx.fillRect(x, y - 20, textWidth + 10, 20);

      // Draw label text
      ctx.fillStyle = '#000000';
      ctx.font = '16px Arial';
      ctx.fillText(`${label} ${(confidence * 100).toFixed(1)}%`, x + 5, y - 5);
    });
  };

  const saveDetectionResult = () => {
    if (canvasRef.current) {
      const dataUrl = canvasRef.current.toDataURL('image/jpeg');
      const a = document.createElement('a');
      a.href = dataUrl;
      a.download = 'e-waste-detection.jpg';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-grow bg-gray-50 dark:bg-eco-green-medium/10 p-6">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center mb-8 animate-in fade-in slide-in-from-top duration-500">
            <Maximize className="h-10 w-10 mr-3 text-gray-700 dark:text-gray-200" />
            <h1 className="text-3xl font-bold text-eco-green-dark dark:text-eco-green-light">
              E-Waste Object Detection
            </h1>
          </div>

          {isModelLoading ? (
            <div className="flex flex-col items-center justify-center p-10 bg-white dark:bg-eco-green-dark rounded-lg shadow-md animate-pulse">
              <RefreshCw className="h-16 w-16 text-eco-green-medium mb-4 animate-spin" />
              <p className="text-xl text-gray-700 dark:text-gray-200">
                Loading object detection model...
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                This may take a moment depending on your connection.
              </p>
            </div>
          ) : (
            <>
              {!mode && (
                <div className="grid md:grid-cols-2 gap-8 animate-in fade-in slide-in-from-bottom duration-500">
                  <div 
                    className="bg-white dark:bg-eco-green-dark rounded-lg shadow-md p-8 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 hover:shadow-lg hover:scale-105"
                    onClick={startWebcam}
                  >
                    <Camera className="h-20 w-20 text-eco-green-medium mb-4" />
                    <h2 className="text-xl font-semibold text-eco-green-dark dark:text-eco-green-light mb-2">
                      Use Webcam
                    </h2>
                    <p className="text-gray-600 dark:text-gray-300 text-center">
                      Detect e-waste objects in real-time using your camera
                    </p>
                  </div>
                  
                  <div className="bg-white dark:bg-eco-green-dark rounded-lg shadow-md p-8 flex flex-col items-center justify-center relative transition-all duration-300 hover:shadow-lg hover:scale-105">
                    <Upload className="h-20 w-20 text-eco-blue-dark dark:text-eco-blue-light mb-4" />
                    <h2 className="text-xl font-semibold text-eco-green-dark dark:text-eco-green-light mb-2">
                      Upload Image
                    </h2>
                    <p className="text-gray-600 dark:text-gray-300 text-center">
                      Upload an image to detect e-waste objects
                    </p>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                  </div>
                </div>
              )}
              
              {mode && (
                <div className="bg-white dark:bg-eco-green-dark rounded-lg shadow-md p-6 animate-in fade-in slide-in-from-bottom duration-500">
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-semibold text-eco-green-dark dark:text-eco-green-light">
                      {mode === "webcam" ? "Webcam Detection" : "Image Detection"}
                    </h2>
                    <button
                      onClick={stopWebcam}
                      className="text-gray-500 hover:text-red-500 transition-colors"
                    >
                      <X className="h-6 w-6" />
                    </button>
                  </div>
                  
                  <div className="flex flex-col md:flex-row gap-6">
                    <div className="flex-1 relative">
                      {mode === "webcam" && (
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          className="w-full rounded-lg border border-gray-300 dark:border-eco-green-medium"
                        />
                      )}
                      
                      {mode === "upload" && imageUrl && (
                        <img
                          src={imageUrl}
                          alt="Uploaded"
                          className="w-full rounded-lg border border-gray-300 dark:border-eco-green-medium"
                        />
                      )}
                      
                      <canvas
                        ref={canvasRef}
                        className="absolute top-0 left-0 w-full h-full rounded-lg"
                      />
                    </div>
                    
                    <div className="flex-1 flex flex-col">
                      <div className="mb-4 flex gap-2">
                        <button
                          onClick={detectObjects}
                          disabled={isDetecting}
                          className="flex-1 bg-eco-green-medium hover:bg-eco-green-dark text-white py-2 px-4 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                          {isDetecting ? (
                            <>
                              <RefreshCw className="h-5 w-5 animate-spin" />
                              <span>Detecting...</span>
                            </>
                          ) : (
                            <>
                              <Camera className="h-5 w-5" />
                              <span>Detect Objects</span>
                            </>
                          )}
                        </button>
                        
                        {detectedObjects.length > 0 && (
                          <button
                            onClick={saveDetectionResult}
                            className="bg-eco-blue-dark hover:bg-eco-blue-dark/80 text-white py-2 px-4 rounded-md transition-colors"
                          >
                            <Save className="h-5 w-5" />
                          </button>
                        )}
                      </div>
                      
                      <div className="flex-1 overflow-y-auto bg-gray-50 dark:bg-eco-green-dark/50 rounded-lg p-4">
                        <h3 className="text-lg font-medium text-eco-green-dark dark:text-white mb-3">
                          Detected Objects ({detectedObjects.length})
                        </h3>
                        
                        {detectedObjects.length > 0 ? (
                          <div className="space-y-2">
                            {detectedObjects.map((obj, index) => (
                              <div
                                key={index}
                                className="bg-white dark:bg-eco-green-dark/80 rounded p-3 shadow-sm"
                              >
                                <div className="flex justify-between items-center">
                                  <span className="font-medium text-eco-green-dark dark:text-eco-green-light">
                                    {obj.label}
                                  </span>
                                  <span className="text-sm bg-eco-green-medium/20 text-eco-green-dark dark:text-white px-2 py-1 rounded">
                                    {Math.round(obj.confidence * 100)}%
                                  </span>
                                </div>
                                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                  Location: x:{Math.round(obj.box.x * 100)}% y:{Math.round(obj.box.y * 100)}%
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-gray-500 dark:text-gray-400 text-center py-4">
                            No objects detected yet.
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
      
      <footer className="bg-eco-green-dark text-white py-6 animate-in fade-in slide-in-from-bottom duration-500">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; {new Date().getFullYear()} Eco-Doc Hub. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default ObjectDetection;
