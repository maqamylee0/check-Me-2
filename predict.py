import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'model.joblib'

def load_model():
    """Loads the trained model pipeline."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        return None

def predict_risk(input_data):
    """
    Predicts risk level based on input data.
    
    Args:
        input_data (dict): Dictionary containing patient data.
        
    Returns:
        dict: containing 'risk_level' (0, 1, 2), 'probabilities', 'risk_band_label'
    """
    model = load_model()
    if not model:
        return {"error": "Model not found. Please train the model first."}
    
    # Ensure correct column order and missing value handling
    # Expected columns based on training script
    columns = ['age', 'family_history', 'previous_lumps', 'breast_pain', 
               'nipple_discharge', 'skin_dimples', 'lump_size_mm', 
               'symptom_duration_days', 'pregnancy_status', 'hormonal_contraception',
               'region', 'language']
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure all columns exist (fill with defaults if missing, though UI should provide them)
    for col in columns:
        if col not in df.columns:
            if col == 'lump_size_mm':
                df[col] = np.nan
            else:
                df[col] = 0 # Default 0 or appropriate missing value
                
    # Reorder
    df = df[columns]
    
    # Predict
    try:
        probs = model.predict_proba(df)[0]
        prediction = model.predict(df)[0]
        
        # Map to label
        risk_labels = {0: "Green (Low Concern)", 1: "Yellow (Monitor)", 2: "Red (Urgent)"}
        
        return {
            "risk_level": int(prediction),
            "risk_band": risk_labels.get(int(prediction), "Unknown"),
            "probabilities": {
                "Green": round(probs[0], 4),
                "Yellow": round(probs[1], 4),
                "Red": round(probs[2], 4)
            }
        }
    except Exception as e:
        return {"error": str(e)}

def get_recommendations(risk_level):
    """
    Returns rule-based recommendations based on risk level.
    """
    if risk_level == 0: # Green
        return [
            "Monitor breast health routinely.",
            "Continue regular self-exams.",
            "Schedule standard screening mammogram if over 40."
        ]
    elif risk_level == 1: # Yellow
        return [
            "Monitor symptoms for changes over next 2 weeks.",
            "Consult a general practitioner if symptoms persist.",
            "Review lifestyle factors (diet, exercise)."
        ]
    elif risk_level == 2: # Red
        return [
            "URGENT: Schedule an appointment with a specialist immediately.",
            "Mention 'red flag' symptoms like skin dimpling or discharge.",
            "Do not panic, but do not delay clinical assessment."
        ]
    return []
