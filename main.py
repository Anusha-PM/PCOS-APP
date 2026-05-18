from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# ── Load scaler & model ───────────────────────────────────────────────────────

scaler = joblib.load("pcos_scaler_1_.pkl")
model = joblib.load("pcos_final_stacking_model.pkl")

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="PCOS Early Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Input schema ──────────────────────────────────────────────────────────────

class PCOSInput(BaseModel):
    age: int
    weight: float
    height: float
    weight_gain: int
    excess_hair_growth: int
    skin_darkening: int
    hair_loss: int
    pimples_acne: int
    cycle_type: int
    cycle_length: float
    fast_food: int
    exercise: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "PCOS Detection API is running 🚀"}


@app.post("/predict")
def predict(data: PCOSInput):
    # Auto-calculate BMI
    height_m = data.height / 100
    bmi = round(data.weight / (height_m ** 2), 2)

    # Feature order matching training:
    # Age, BMI, Menstrual_Irregularity, Weight_Gain, Facial_Hair,
    # Skin_Darkening, Hair_Loss, Acne, Fast_Food, Exercise
    features = np.array([[
        data.age,
        bmi,
        data.cycle_type,
        data.weight_gain,
        data.excess_hair_growth,
        data.skin_darkening,
        data.hair_loss,
        data.pimples_acne,
        data.fast_food,
        data.exercise,
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    confidence = round(float(max(proba)) * 100, 2)

    result = "PCOS Detected" if prediction == 1 else "No PCOS Detected"

    return {
        "prediction": result,
        "bmi": bmi,
        "confidence": f"{confidence}%",
    }
