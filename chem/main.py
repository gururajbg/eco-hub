from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from fastapi.middleware.cors import CORSMiddleware


# Allow all origins for dev — restrict later for production


app = FastAPI(title="PCB Copper Extraction Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for more control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, imputer, scaler
model = load_model("models/bioleach_ann_model.keras")
imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")

# Request structure
class BioleachInput(BaseModel):
    C1R1: float
    C1G1: float
    C1B1: float
    PH1: float
    Fe_plus2: float
    Fe_plus3: float
    acid_conc: float
    pulp_density: float
    temp: float
    time: float

@app.get("/")
def read_root():
    return {"message": "🚀 Bioleaching Simulation API is Running!"}

@app.post("/predict/")
def predict_recovery(input_data: BioleachInput):
    # Input array
    input_list = np.array([[  
        input_data.C1R1,
        input_data.C1G1,
        input_data.C1B1,
        input_data.PH1,
        input_data.Fe_plus2,
        input_data.Fe_plus3,
        input_data.acid_conc,
        input_data.pulp_density,
        input_data.temp,
        input_data.time
    ]])

    # Preprocessing: impute → scale
    input_processed = scaler.transform(imputer.transform(input_list))

    # Predict
    prediction = model.predict(input_processed)
    raw_output = prediction.flatten()[0]

    # ⚠️ Convert to percentage-like scale
    scaled_output = raw_output / 100000  # Adjust this if needed

    return {
        "Copper_Recovery": round(float(scaled_output), 2)  # Rounded for display
    }
