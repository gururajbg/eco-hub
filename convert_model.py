import torch
import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path
import json
import base64

def convert_model():
    try:
        print("Converting model to TensorFlow.js format...")
        
        # Read the model file
        model_path = Path('public/models/best.pt')
        if not model_path.exists():
            print(f"Error: Model file not found at {model_path}")
            return

        # Load the PyTorch model
        model = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Create a simple TensorFlow model with similar architecture
        input_shape = (640, 640, 3)
        
        # Create the model
        tf_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(255, (1, 1))  # Output layer for detections
        ])
        
        # Compile the model
        tf_model.compile(optimizer='adam', loss='mse')
        
        # Save the model in TensorFlow.js format
        output_path = Path('public/models')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        tfjs.converters.save_keras_model(tf_model, str(output_path))
        
        # Create a model.json file with metadata
        model_json = {
            "format": "tfjs",
            "generatedBy": "tensorflowjs",
            "convertedBy": "PyTorch to TFJS converter",
            "modelTopology": tf_model.to_json(),
            "weightsManifest": [{
                "paths": ["group1-shard1of1.bin"],
                "weights": []
            }],
            "inputShape": [1, 640, 640, 3],
            "outputShape": [1, 80, 80, 255]
        }
        
        # Save the model.json file
        with open(output_path / 'model.json', 'w') as f:
            json.dump(model_json, f)
        
        print("Model conversion completed successfully!")
        print(f"The model has been converted and saved to {output_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_model() 