from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="MamaSafe AI API",
    description="Maternal Health Risk Prediction API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE, "artifacts")

scaler        = joblib.load(os.path.join(ARTIFACTS, "scaler.pkl"))
pca           = joblib.load(os.path.join(ARTIFACTS, "pca.pkl"))
model         = joblib.load(os.path.join(ARTIFACTS, "model.pkl"))
feature_cols  = joblib.load(os.path.join(ARTIFACTS, "features.pkl"))
label_map     = joblib.load(os.path.join(ARTIFACTS, "label_map.pkl"))
explainer     = joblib.load(os.path.join(ARTIFACTS, "shap_explainer.pkl"))
pc_to_feature = joblib.load(os.path.join(ARTIFACTS, "pc_to_feature.pkl"))

# Extended mapping for all possible components
pc_to_feature.update({
    'PC1': 'Blood Pressure',
    'PC2': 'Heart Rate',
    'PC3': 'Body Temperature',
    'PC4': 'Age',
    'PC5': 'Blood Sugar',
    'PC6': 'Blood Pressure & Age',
    'PC7': 'Heart Rate & Blood Sugar',
    'PC8': 'Body Temperature & Blood Pressure',
    'PC9': 'Blood Sugar & Age',
    'PC10': 'Heart Rate & Body Temperature',
})

reverse_label = {v: k for k, v in label_map.items()}

class PatientData(BaseModel):
    age: float          = Field(..., example=28)
    systolic_bp: float  = Field(..., example=120)
    diastolic_bp: float = Field(..., example=80)
    bs: float           = Field(..., example=7.5)
    body_temp: float    = Field(..., example=98.6)
    heart_rate: float   = Field(..., example=76)

def explain_prediction(shap_vals, predicted_class, input_values):
    class_names = ['low risk', 'mid risk', 'high risk']

    sv = np.array(shap_vals).flatten()
    pc_names = [f'PC{i+1}' for i in range(len(sv))]

    contributions = [(pc, float(val)) for pc, val in zip(pc_names, sv)]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    # Clinical thresholds for direction
    thresholds = {
        'age':          {'high': 35},
        'systolic_bp':  {'high': 130},
        'diastolic_bp': {'high': 85},
        'bs':           {'high': 11.0},
        'body_temp':    {'high': 99.5},
        'heart_rate':   {'high': 90},
    }

    # Dominant original feature per PC
    pc_dominant_feature = {
        'PC1': 'systolic_bp',
        'PC2': 'heart_rate',
        'PC3': 'body_temp',
        'PC4': 'age',
        'PC5': 'bs',
        'PC6': 'systolic_bp',
        'PC7': 'heart_rate',
        'PC8': 'body_temp',
        'PC9': 'bs',
        'PC10': 'heart_rate',
    }

    reasons = []
    for pc, value in contributions[:3]:
        feature_label = pc_to_feature.get(pc, pc)
        raw_feature   = pc_dominant_feature.get(pc, None)

        if raw_feature and raw_feature in input_values:
            threshold = thresholds.get(raw_feature, {}).get('high', None)
            actual_val = input_values[raw_feature]
            direction = "elevated" if (threshold and actual_val > threshold) else "normal"
        else:
            direction = "elevated" if value > 0 else "normal"

        impact = "strongly" if abs(value) > 1.0 else "moderately"
        reasons.append(f"{feature_label} is {direction} ({impact} influencing this result)")

    top_two = [pc_to_feature.get(c[0], c[0]) for c in contributions[:2]]
    return {
        "summary": f"Classified as {class_names[predicted_class]} primarily due to: {', '.join(top_two)}",
        "top_factors": reasons
    }

@app.get("/")
def root():
    return {
        "message": "MamaSafe AI API is running",
        "version": "2.0.0",
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: PatientData):
    try:
        input_array = np.array([[
            data.age,
            data.systolic_bp,
            data.diastolic_bp,
            data.bs,
            data.body_temp,
            data.heart_rate
        ]])

        scaled  = scaler.transform(input_array)
        reduced = pca.transform(scaled)

        prediction    = model.predict(reduced)[0]
        probabilities = model.predict_proba(reduced)[0]
        risk_label    = reverse_label[int(prediction)]
        confidence    = float(probabilities[int(prediction)])

        shap_vals = explainer.shap_values(reduced)

        # Handle both list and array formats from different SHAP versions
        if isinstance(shap_vals, list):
            sv = shap_vals[int(prediction)][0]
        else:
            sv = shap_vals[0]

        input_values = {
            'age':          data.age,
            'systolic_bp':  data.systolic_bp,
            'diastolic_bp': data.diastolic_bp,
            'bs':           data.bs,
            'body_temp':    data.body_temp,
            'heart_rate':   data.heart_rate,
}
        explanation = explain_prediction(sv, int(prediction), input_values)

        advice = {
            "low risk":  "Continue routine prenatal care and monitoring.",
            "mid risk":  "Schedule a follow-up visit with a healthcare provider within 48 hours.",
            "high risk": "Seek immediate medical attention at the nearest facility."
        }

        return {
            "risk_level":    risk_label,
            "confidence":    round(confidence * 100, 2),
            "probabilities": {
                "low risk":  round(float(probabilities[0]) * 100, 2),
                "mid risk":  round(float(probabilities[1]) * 100, 2),
                "high risk": round(float(probabilities[2]) * 100, 2)
            },
            "advice":      advice[risk_label],
            "explanation": explanation,
            "input_received": data.dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))