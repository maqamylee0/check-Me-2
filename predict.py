import joblib
import pandas as pd
import numpy as np
import shap

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
        
        # Calculate local SHAP values
        fitted_preprocessor = model.named_steps['preprocessor']
        
        # We need to manually reconstruct the feature names
        num_features = ['age', 'lump_size_mm', 'symptom_duration_days']
        # cat_features = ['region', 'language']
        bin_features = ['family_history', 'previous_lumps', 'breast_pain', 
                       'nipple_discharge', 'skin_dimples', 
                       'pregnancy_status', 'hormonal_contraception']
                       
        # ohe_cols = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features).tolist()
        all_feature_names = num_features + bin_features
        
        df_transformed = fitted_preprocessor.transform(df)
        
        # Convert sparse matrix to dense if necessary
        if hasattr(df_transformed, "toarray"):
            df_dense = df_transformed.toarray()
        else:
            df_dense = df_transformed
        
        # Get purely the classifier from the pipeline
        classifier = model.named_steps['classifier']
        
        # Differentiate between Logistic Regression and XGBoost
        if type(classifier).__name__ == "LogisticRegression":
            # For linear models, the impact is simply coefficient * feature value
            coefs = classifier.coef_[2] # coefficients for class index 2 ("Red")
            local_shap = coefs * df_dense[0]
        else:
            explainer = shap.Explainer(classifier)
            shap_values = explainer(df_dense)
            
            # SHAP returns a 3D array for multiclass: (samples, features, classes)
            # We look at the SHAP values pushing towards the "Red" class (index 2)
            local_shap = shap_values[0, :, 2].values
        
        # Create a df of feature importances
        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'impact': local_shap,
            'abs_impact': np.abs(local_shap)
        })
        
        # Sort by absolute impact to find top 3 drivers (positive or negative)
        top_drivers = importance_df.sort_values(by='abs_impact', ascending=False).head(3)
        
        reasons = []
        for _, row in top_drivers.iterrows():
            feat = row['feature']
            impact = row['impact']
            direction = "Increased" if impact > 0 else "Decreased"
            reasons.append(f"{feat} ({direction} risk)")
            
        # Map to label
        risk_labels = {0: "Green (Low Concern)", 1: "Yellow (Monitor)", 2: "Red (Urgent)"}
        
        return {
            "risk_level": int(prediction),
            "risk_band": risk_labels.get(int(prediction), "Unknown"),
            "probabilities": {
                "Green": round(probs[0], 4),
                "Yellow": round(probs[1], 4),
                "Red": round(probs[2], 4)
            },
            "top_reasons": reasons
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
