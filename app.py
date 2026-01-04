import os
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# -----------------------
# Config
# -----------------------
MODELS_DIR = os.getenv("MODELS_DIR", "models")
API_KEY = os.getenv("MODEL_API_KEY")  # set in hosting env vars

# -----------------------
# Load artifacts once
# -----------------------
rf_classifier = joblib.load(os.path.join(MODELS_DIR, "rf_classifier_breeding.pkl"))
rf_regressor  = joblib.load(os.path.join(MODELS_DIR, "rf_regressor_score.pkl"))
xgb_classifier = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier_risk.pkl"))

le_recommendation = joblib.load(os.path.join(MODELS_DIR, "label_encoder_recommendation.pkl"))
le_risk = joblib.load(os.path.join(MODELS_DIR, "label_encoder_risk.pkl"))
le_breed = joblib.load(os.path.join(MODELS_DIR, "label_encoder_breed.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

app = FastAPI(title="SNOUTSCAN Breeding Compatibility API", version="1.0.0")


# -----------------------
# Schemas
# -----------------------
class DogProfile(BaseModel):
    breed: str
    age_months: int
    gender: str
    weight_kg: float
    body_size_category: str
    hip_dysplasia_risk: float
    elbow_dysplasia_risk: float
    eye_condition_risk: float
    heart_condition_risk: float
    temperament_score: float
    energy_level: int
    genetic_diversity_coi: float


class PredictRequest(BaseModel):
    dog1: DogProfile
    dog2: DogProfile


def _auth(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def calculate_features(dog1: Dict[str, Any], dog2: Dict[str, Any]) -> pd.DataFrame:
    # Size
    size_map = {"Small": 1, "Medium": 2, "Large": 3}
    size1 = size_map.get(dog1["body_size_category"], 2)
    size2 = size_map.get(dog2["body_size_category"], 2)
    size_diff = abs(size1 - size2)

    w1 = float(dog1["weight_kg"])
    w2 = float(dog2["weight_kg"])
    weight_ratio = max(w1, w2) / max(0.0001, min(w1, w2))

    if size_diff >= 2:
        size_score = max(1.0, 10 - (weight_ratio * 2.5))
    elif size_diff == 1:
        size_score = max(4.0, 10 - (weight_ratio * 1.2))
    else:
        size_score = max(7.0, 10 - (weight_ratio * 0.5))

    # Age
    a1 = int(dog1["age_months"])
    a2 = int(dog2["age_months"])
    age_gap = abs(a1 - a2)
    if age_gap <= 12:
        age_score = 10.0
    elif age_gap <= 24:
        age_score = 8.5
    elif age_gap <= 36:
        age_score = 7.0
    elif age_gap <= 48:
        age_score = 5.5
    else:
        age_score = 3.0

    # Genetic diversity
    coi1 = float(dog1["genetic_diversity_coi"])
    coi2 = float(dog2["genetic_diversity_coi"])
    combined_coi = (coi1 + coi2) / 2
    if dog1["breed"] != dog2["breed"]:
        combined_coi *= 0.7
    if dog1["breed"] == "Mixed" or dog2["breed"] == "Mixed":
        combined_coi *= 0.6
    genetic_score = min(10, max(0, (1 - combined_coi) * 10))

    # Health compatibility
    avg_hip = (float(dog1["hip_dysplasia_risk"]) + float(dog2["hip_dysplasia_risk"])) / 2
    avg_elbow = (float(dog1["elbow_dysplasia_risk"]) + float(dog2["elbow_dysplasia_risk"])) / 2
    avg_eye = (float(dog1["eye_condition_risk"]) + float(dog2["eye_condition_risk"])) / 2
    avg_heart = (float(dog1["heart_condition_risk"]) + float(dog2["heart_condition_risk"])) / 2
    combined_risk = (avg_hip * 0.3 + avg_elbow * 0.2 + avg_eye * 0.25 + avg_heart * 0.25)
    health_score = (1 - combined_risk) * 10

    # Temperament match
    temp_diff = abs(float(dog1["temperament_score"]) - float(dog2["temperament_score"]))
    energy_diff = abs(int(dog1["energy_level"]) - int(dog2["energy_level"]))
    temperament_score = max(5, 10 - (temp_diff * 0.5) - (energy_diff * 1.0))

    # Combined risks (slightly inflated)
    combined_hip = (float(dog1["hip_dysplasia_risk"]) + float(dog2["hip_dysplasia_risk"])) / 2 * 1.1
    combined_eye = (float(dog1["eye_condition_risk"]) + float(dog2["eye_condition_risk"])) / 2 * 1.1
    combined_heart = (float(dog1["heart_condition_risk"]) + float(dog2["heart_condition_risk"])) / 2 * 1.1

    same_gender = 1 if dog1["gender"] == dog2["gender"] else 0
    same_breed = 1 if dog1["breed"] == dog2["breed"] else 0
    has_mixed = 1 if (dog1["breed"] == "Mixed" or dog2["breed"] == "Mixed") else 0

    # Breed encoding (safe fallback)
    breed_pair = f"{dog1['breed']}_x_{dog2['breed']}"
    try:
        breed_encoded = int(le_breed.transform([breed_pair])[0])
    except Exception:
        breed_encoded = 0

    features = {
        "dog_1_age_months": a1,
        "dog_2_age_months": a2,
        "age_gap_months": age_gap,
        "dog_1_weight_kg": w1,
        "dog_2_weight_kg": w2,
        "weight_ratio": round(weight_ratio, 2),
        "size_compatibility_score": round(size_score, 1),
        "age_compatibility_score": age_score,
        "genetic_diversity_score": round(genetic_score, 1),
        "health_compatibility_score": round(health_score, 1),
        "temperament_match_score": round(temperament_score, 1),
        "combined_hip_risk": round(min(1.0, combined_hip), 3),
        "combined_eye_risk": round(min(1.0, combined_eye), 3),
        "combined_heart_risk": round(min(1.0, combined_heart), 3),
        "same_gender": same_gender,
        "same_breed": same_breed,
        "has_mixed": has_mixed,
        "breed_pair_encoded": breed_encoded,
        "dog_1_size_encoded": size1,
        "dog_2_size_encoded": size2,
        "size_difference": size_diff,
        "genetic_health_interaction": round(genetic_score * health_score, 1),
        "size_age_interaction": round(size_score * age_score, 1),
    }

    # Ensure correct feature order and no missing columns
    missing = [f for f in feature_names if f not in features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing engineered features: {missing}")

    X = pd.DataFrame([features], columns=feature_names)
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isnull().any().any():
        bad = X.columns[X.isnull().any()].tolist()
        raise HTTPException(status_code=422, detail=f"Invalid (non-numeric) feature values: {bad}")

    return X


@app.get("/health")
def health():
    return {"status": "ok", "features": len(feature_names)}


@app.post("/predict")
def predict(req: PredictRequest, x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)

    dog1 = req.dog1.model_dump()
    dog2 = req.dog2.model_dump()

    X = calculate_features(dog1, dog2)

    # 1) Recommendation
    rec_proba = rf_classifier.predict_proba(X)[0]
    rec_idx = int(np.argmax(rec_proba))
    recommendation = le_recommendation.inverse_transform([rec_idx])[0]
    rec_conf = {cls: float(rec_proba[i]) for i, cls in enumerate(le_recommendation.classes_)}

    # 2) Score
    score = float(rf_regressor.predict(X)[0])

    # 3) Risk
    risk_proba = xgb_classifier.predict_proba(X)[0]
    risk_idx = int(np.argmax(risk_proba))
    risk_level = le_risk.inverse_transform([risk_idx])[0]
    risk_conf = {cls: float(risk_proba[i]) for i, cls in enumerate(le_risk.classes_)}

    # Warnings (same as your script)
    warnings = []
    if int(X["same_gender"].iloc[0]) == 1:
        warnings.append("Same gender - biologically incompatible")
    if float(X["size_compatibility_score"].iloc[0]) < 5:
        warnings.append("Severe size mismatch - high whelping risk")
    if float(X["combined_hip_risk"].iloc[0]) > 0.35:
        warnings.append("High hip dysplasia risk in offspring")
    if float(X["genetic_diversity_score"].iloc[0]) < 6:
        warnings.append("Low genetic diversity - inbreeding concern")
    if float(X["age_compatibility_score"].iloc[0]) < 6:
        warnings.append("Significant age gap")

    return {
        "breeding_recommended": str(recommendation),
        "recommendation_confidence": rec_conf,
        "compatibility_score": round(score, 1),
        "risk_level": str(risk_level),
        "risk_confidence": risk_conf,
        "warnings": warnings,
    }