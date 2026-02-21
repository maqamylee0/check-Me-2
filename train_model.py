import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import xgboost as xgb

def load_data(filepath="synthetic_breast_cancer_risk.csv"):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Features and Target
    X = df.drop(columns=['risk_level'])
    y = df['risk_level']
    
    # Identify column types
    numeric_features = ['age', 'lump_size_mm', 'symptom_duration_days']
    categorical_features = ['region', 'language']
    # Binary features are already 0/1, but we can treat them as numeric or categorical.
    # treating them as passthrough numeric for simplicity as they are already 0/1.
    binary_features = ['family_history', 'previous_lumps', 'breast_pain', 'nipple_discharge', 
                       'skin_dimples', 'pregnancy_status', 'hormonal_contraception']
    
    # Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ])

    return X, y, preprocessor

from sklearn.utils import class_weight

def train_baseline_model(X_train, y_train, preprocessor):
    # Logistic Regression with class weights
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced'))])
    clf.fit(X_train, y_train)
    return clf

def train_improved_model(X_train, y_train, preprocessor, sample_weights=None):
    # XGBoost
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', xgb.XGBClassifier(
                              objective='multi:softprob',
                              num_class=3,
                              eval_metric='mlogloss',
                              random_state=42
                          ))])
    
    fit_params = {}
    if sample_weights is not None:
        fit_params['classifier__sample_weight'] = sample_weights
        
    clf.fit(X_train, y_train, **fit_params)
    return clf

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    print(f"--- Evaluation: {model_name} ---")
    print(classification_report(y_test, y_pred, target_names=['Green', 'Yellow', 'Red']))
    
    # ROC AUC for multiclass (weighted or macro)
    try:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        print(f"ROC AUC (Weighted): {roc_auc:.4f}")
    except Exception as e:
        print(f"Could not calculate ROC AUC: {e}")
        
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("-" * 30)
    return y_pred, y_prob

def explain_model(model, X_train, feature_names=None):
    # SHAP explanation for XGBoost
    try:
        model_xgb = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        X_train_transformed = preprocessor.transform(X_train)
        
        explainer = shap.Explainer(model_xgb)
        shap_values = explainer(X_train_transformed)
        
        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_train_transformed, show=False)
        plt.savefig("shap_summary.png", bbox_inches='tight')
        print("SHAP summary plot saved to shap_summary.png")
    except Exception as e:
        print(f"SHAP Error: {e}")

def main():
    print("Loading data...")
    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Compute sample weights for XGBoost
    sample_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    
    print("Training Baseline (Logistic Regression)...")
    baseline_model = train_baseline_model(X_train, y_train, preprocessor)
    evaluate_model(baseline_model, X_test, y_test, "Logistic Regression (Balanced)")
    
    print("Training Improved (XGBoost)...")
    xgb_model = train_improved_model(X_train, y_train, preprocessor, sample_weights)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost (Balanced)")
    
    # Save best model
    joblib.dump(xgb_model, 'model.joblib')
    print("Model saved to model.joblib")
    
    # Interpretability
    print("Generating SHAP explanation...")
    explain_model(xgb_model, X_train[:500]) # Using subset for speed

if __name__ == "__main__":
    main()
