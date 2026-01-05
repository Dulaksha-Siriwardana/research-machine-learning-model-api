from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Dog Breeding Compatibility API")

# -----------------------------
# Load models & assets
# -----------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "models")

rf_classifier = joblib.load(os.path.join(MODEL_DIR, "rf_classifier_breeding.pkl"))
rf_regressor  = joblib.load(os.path.join(MODEL_DIR, "rf_regressor_score.pkl"))
xgb_classifier = joblib.load(os.path.join(MODEL_DIR, "xgb_classifier_risk.pkl"))

le_recommendation = joblib.load(os.path.join(MODEL_DIR, "label_encoder_recommendation.pkl"))
le_risk = joblib.load(os.path.join(MODEL_DIR, "label_encoder_risk.pkl"))
le_breed = joblib.load(os.path.join(MODEL_DIR, "label_encoder_breed.pkl"))

feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))


# -----------------------------
# Request schema
# -----------------------------
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
    energy_level: float
    genetic_diversity_coi: float


class PredictRequest(BaseModel):
    dog1: DogProfile
    dog2: DogProfile


# -----------------------------
# Feature engineering (29)
# -----------------------------
def calc_features(d1: dict, d2: dict) -> dict:
    size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
    size1 = size_map.get(d1.get('body_size_category'), 2)
    size2 = size_map.get(d2.get('body_size_category'), 2)

    size_diff = abs(size1 - size2)

    # protect against division by 0
    w_min = max(0.1, min(d1['weight_kg'], d2['weight_kg']))
    w_max = max(d1['weight_kg'], d2['weight_kg'])
    weight_ratio = w_max / w_min

    # Enhanced size scoring
    if weight_ratio > 5.0:
        size_score = 0.0
    elif weight_ratio > 4.0:
        size_score = 1.0
    elif weight_ratio > 3.0:
        size_score = 2.0
    elif size_diff >= 2:
        size_score = max(1.0, 10 - (weight_ratio * 3.0))
    elif size_diff == 1:
        size_score = max(4.0, 10 - (weight_ratio * 1.5))
    else:
        size_score = max(7.0, 10 - (weight_ratio * 0.5))

    #  age scoring with validation
    age_gap = abs(d1['age_months'] - d2['age_months'])
    age1_valid = 18 <= d1['age_months'] <= 96
    age2_valid = 18 <= d2['age_months'] <= 96

    if (not age1_valid) or (not age2_valid):
        age_score = 0.0
    elif age_gap > 48:
        age_score = 0.0
    elif age_gap <= 12:
        age_score = 10.0
    elif age_gap <= 24:
        age_score = 8.5
    elif age_gap <= 36:
        age_score = 6.0
    elif age_gap <= 48:
        age_score = 3.0
    else:
        age_score = 0.0

    # Genetic diversity
    combined_coi = (d1['genetic_diversity_coi'] + d2['genetic_diversity_coi']) / 2
    if d1['breed'] != d2['breed']:
        combined_coi *= 0.7
    if d1['breed'] == 'Mixed' or d2['breed'] == 'Mixed':
        combined_coi *= 0.6
    genetic_score = min(10, max(0, (1 - combined_coi) * 10))

    # Health compatibility
    avg_hip = (d1['hip_dysplasia_risk'] + d2['hip_dysplasia_risk']) / 2
    avg_elbow = (d1['elbow_dysplasia_risk'] + d2['elbow_dysplasia_risk']) / 2
    avg_eye = (d1['eye_condition_risk'] + d2['eye_condition_risk']) / 2
    avg_heart = (d1['heart_condition_risk'] + d2['heart_condition_risk']) / 2
    combined_risk = (avg_hip * 0.3 + avg_elbow * 0.2 + avg_eye * 0.25 + avg_heart * 0.25)
    health_score = (1 - combined_risk) * 10

    # Temperament
    temp_diff = abs(d1['temperament_score'] - d2['temperament_score'])
    energy_diff = abs(d1['energy_level'] - d2['energy_level'])
    temperament_score = max(5, 10 - (temp_diff * 0.5) - (energy_diff * 1.0))

    # Combined risks
    combined_hip = (d1['hip_dysplasia_risk'] + d2['hip_dysplasia_risk']) / 2 * 1.1
    combined_eye = (d1['eye_condition_risk'] + d2['eye_condition_risk']) / 2 * 1.1
    combined_heart = (d1['heart_condition_risk'] + d2['heart_condition_risk']) / 2 * 1.1

    # Same gender
    same_gender = 1 if d1['gender'] == d2['gender'] else 0

    # Breed encoding
    breed_pair = f"{d1['breed']}_x_{d2['breed']}"
    try:
        breed_encoded = int(le_breed.transform([breed_pair])[0])
    except Exception:
        breed_encoded = 0

    # 29 features
    return {
        'dog_1_age_months': d1['age_months'],
        'dog_2_age_months': d2['age_months'],
        'age_gap_months': age_gap,
        'dog_1_weight_kg': float(d1['weight_kg']),
        'dog_2_weight_kg': float(d2['weight_kg']),
        'weight_ratio': round(float(weight_ratio), 2),
        'size_compatibility_score': round(float(size_score), 1),
        'age_compatibility_score': float(age_score),
        'genetic_diversity_score': round(float(genetic_score), 1),
        'health_compatibility_score': round(float(health_score), 1),
        'temperament_match_score': round(float(temperament_score), 1),
        'combined_hip_risk': round(float(min(1.0, combined_hip)), 3),
        'combined_eye_risk': round(float(min(1.0, combined_eye)), 3),
        'combined_heart_risk': round(float(min(1.0, combined_heart)), 3),

        # validation flags
        'dog_1_too_young': 1 if d1['age_months'] < 18 else 0,
        'dog_2_too_young': 1 if d2['age_months'] < 18 else 0,
        'dog_1_too_old': 1 if d1['age_months'] > 96 else 0,
        'dog_2_too_old': 1 if d2['age_months'] > 96 else 0,
        'extreme_age_gap': 1 if age_gap > 48 else 0,
        'extreme_weight_ratio': 1 if weight_ratio > 4.0 else 0,

        'same_gender': same_gender,
        'same_breed': 1 if d1['breed'] == d2['breed'] else 0,
        'has_mixed': 1 if (d1['breed'] == 'Mixed' or d2['breed'] == 'Mixed') else 0,
        'breed_pair_encoded': breed_encoded,
        'dog_1_size_encoded': size1,
        'dog_2_size_encoded': size2,
        'size_difference': size_diff,
        'genetic_health_interaction': round(float(genetic_score * health_score), 1),
        'size_age_interaction': round(float(size_score * age_score), 1),
    }


def make_X(features: dict) -> pd.DataFrame:
    #  reindex by feature_names to avoid KeyError + keep order
    return pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)


# -----------------------------
# Endpoint
# -----------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        d1 = req.dog1.model_dump()
        d2 = req.dog2.model_dump()

        features = calc_features(d1, d2)
        X = make_X(features)

        rec_encoded = rf_classifier.predict(X)[0]
        rec_proba = rf_classifier.predict_proba(X)[0]
        rec_label = le_recommendation.inverse_transform([rec_encoded])[0]

        score = float(rf_regressor.predict(X)[0])

        risk_encoded = xgb_classifier.predict(X)[0]
        risk_proba = xgb_classifier.predict_proba(X)[0]
        risk_label = le_risk.inverse_transform([risk_encoded])[0]

        # map probabilities to labels
        rec_conf = {cls: float(rec_proba[list(le_recommendation.classes_).index(cls)]) for cls in le_recommendation.classes_}
        risk_conf = {cls: float(risk_proba[list(le_risk.classes_).index(cls)]) for cls in le_risk.classes_}

        # Optional warnings from engineered features (kept simple)
        warnings = []
        if features['same_gender'] == 1:
            warnings.append("Same gender - biologically incompatible")
        if features['dog_1_too_young'] == 1 or features['dog_2_too_young'] == 1:
            warnings.append("One dog is below the minimum breeding age (18 months).")
        if features['dog_1_too_old'] == 1 or features['dog_2_too_old'] == 1:
            warnings.append("One dog is above the recommended breeding age range.")
        if features['extreme_age_gap'] == 1:
            warnings.append("Large age gap between the two dogs.")
        if features['extreme_weight_ratio'] == 1:
            warnings.append("Large size mismatch between the two dogs.")
        if features['combined_hip_risk'] > 0.35:
            warnings.append("High hip dysplasia risk in offspring")
        if features['genetic_diversity_score'] < 6:
            warnings.append("Low genetic diversity - possible inbreeding concern")

        return {
            "breeding_recommended": rec_label,
            "recommendation_confidence": rec_conf,
            "compatibility_score": round(score, 1),
            "risk_level": risk_label,
            "risk_confidence": risk_conf,
            "warnings": warnings,
            "model_features_used": len(feature_names),  
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))