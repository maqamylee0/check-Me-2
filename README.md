# Check Me: AI/ML Risk Triage Demo

This repository contains a demo application for self-screening breast cancer risk triage. It includes a synthetic data generator, a machine learning model training pipeline, and a Streamlit-based interactive demo.

## Project Structure
- **generate_data.py**: Generates the synthetic dataset (**synthetic_breast_cancer_risk.csv**).
- **notebook.ipynb**: Exploratory Data Analysis (EDA), trains the risk assessment models, evaluates them, and saves artifacts (**model.joblib**, **shap_summary.png**).
- **predict.py**: Core prediction logic and recommendation rules.
- **app.py**: Streamlit web interface for the demo.
- **DATA_SOURCES.md**: Documentation of data generation logic.
- **requirements.txt**: Python dependencies.

## Setup Instructions

1. **Create and Activate Virtual Environment** (Recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Data and Train Model**
   Run the following command to create the dataset:
   ```bash
   python generate_data.py
   ```
   Then, execute all cells within **notebook.ipynb** to train and evaluate the models.
   This will create **synthetic_breast_cancer_risk.csv**, **model.joblib**, and **shap_summary.png**.

4. **Run the Demo App**
   ```bash
   streamlit run app.py
   ```
   The opens the app in your default browser.

## Model Details
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Input Features**: Age, Family History, Previous Lumps, Breast Pain, Nipple Discharge, Skin Dimples, Lump Size, Symptom Duration, Pregnancy Status, Hormonal Contraception, Region, Language.
- **Risk Classes**: 
  - **Green**: Low Concern (Routine checks)
  - **Yellow**: Medium Risk (Monitor/Consult GP)
  - **Red**: High Risk (Urgent Specialist Referral)
- **Evaluation**: The model prioritizes Recall for the Red class to minimize missed high-risk cases.

## Disclaimer
This tool is a **demonstration only** and is not a medical device. It uses synthetic data and should not be used for actual diagnosis.
