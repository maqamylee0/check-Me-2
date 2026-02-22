import pandas as pd
import numpy as np
import random

def generate_synthetic_data(num_samples=50000):
    """
    Generates a synthetic dataset for breast cancer risk triage.
    """
    np.random.seed(42)  # For reproducibility

    data = []

    for _ in range(num_samples):
        # --- Independent Variables (Inputs) ---
        
        # Age: Skewed slightly towards older, but covering a range.
        # Beta distribution scaled to 18-90
        age = int(18 + np.random.beta(2, 5) * 72)
        
        # Family History: 0/1. Higher probability if older? Maybe not, usually genetic.
        # Let's say ~15% population rate
        family_history = np.random.choice([0, 1], p=[0.85, 0.15])
        
        # Previous Lumps: Higher prob if older
        prob_prev_lumps = 0.05 + (age / 100) * 0.1
        previous_lumps = np.random.choice([0, 1], p=[1 - prob_prev_lumps, prob_prev_lumps])
        
        # Symptoms (correlated with each other and cancer risk)
        # Base probability of having symptoms
        has_symptoms = np.random.choice([0, 1], p=[0.7, 0.3]) 
        
        breast_pain = 0
        nipple_discharge = 0
        skin_dimples = 0
        lump_size_mm = np.nan
        symptom_duration_days = 0

        if has_symptoms:
            # If symptoms exist, they might be benign or malignant indicators
            breast_pain = np.random.choice([0, 1], p=[0.6, 0.4]) # Pain is common in benign
            nipple_discharge = np.random.choice([0, 1], p=[0.9, 0.1])
            skin_dimples = np.random.choice([0, 1], p=[0.95, 0.05])
            
            # Lump size: If present, usually 10-50mm. 
            # Not everyone with symptoms has a distinct lump.
            has_lump = np.random.choice([0, 1], p=[0.4, 0.6])
            if has_lump:
                lump_size_mm = int(np.random.gamma(shape=2, scale=10)) 
                lump_size_mm = max(5, min(lump_size_mm, 100)) # Clip reasonable range
            
            symptom_duration_days = int(np.random.exponential(scale=30))
            symptom_duration_days = max(1, min(symptom_duration_days, 365))

        # Reproductive Factors
        # Pregnancy status: Only for fertile age ~18-50
        is_pregnant = 0
        if 18 <= age <= 50:
             is_pregnant = np.random.choice([0, 1], p=[0.95, 0.05])
        
        # Hormonal Contraception: Only for relevant age
        hormonal_contraception = 0
        if 15 <= age <= 55:
            hormonal_contraception = np.random.choice([0, 1], p=[0.8, 0.2])

        # Demographics
        region = np.random.choice(['Kigali City', 'Northern Province', 'Southern Province', 'Eastern Province', 'Western Province', 'Other'], p=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16])
        language = np.random.choice(['English', 'Kinyarwanda', 'Kiswahili', 'Other'], p=[0.7, 0.15, 0.05, 0.1])

        # --- Dependent Variable (Target Risk) ---
        # Constructing a "ground truth" risk score to assign labels
        # This simulates the unknown biological reality
        
        risk_score = 0
        
        # Age factor
        if age > 50: risk_score += 2
        elif age > 40: risk_score += 1
        
        # Genetics / History
        if family_history: risk_score += 3
        if previous_lumps: risk_score += 1
        
        # Symptom Severity (Red Flags)
        if skin_dimples: risk_score += 5 # High risk sign
        if nipple_discharge: risk_score += 3
        if has_symptoms and not breast_pain: risk_score += 1 # Painless lumps can be concerning
        if pd.notna(lump_size_mm):
            if lump_size_mm > 20: risk_score += 2
            elif lump_size_mm > 50: risk_score += 4
            
        # Noise
        risk_score += np.random.normal(0, 1)

        # Risk Categories (0=Low/Green, 1=Medium/Yellow, 2=High/Red)
        # We want roughly 60% Green, 30% Yellow, 10% Red
        if risk_score > 6:
            risk_level = 2 # Red
        elif risk_score > 3:
            risk_level = 1 # Yellow
        else:
            risk_level = 0 # Green

        sample = {
            'age': age,
            'family_history': family_history,
            'previous_lumps': previous_lumps,
            'breast_pain': breast_pain,
            'nipple_discharge': nipple_discharge,
            'skin_dimples': skin_dimples,
            'lump_size_mm': lump_size_mm,
            'symptom_duration_days': symptom_duration_days,
            'pregnancy_status': is_pregnant,
            'hormonal_contraception': hormonal_contraception,
            'region': region,
            'language': language,
            'risk_level': risk_level
        }
        data.append(sample)

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data(5000)
    
    # Check distribution
    print("\nRisk Level Distribution:")
    print(df['risk_level'].value_counts(normalize=True))
    
    output_file = "synthetic_breast_cancer_risk.csv"
    df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")
