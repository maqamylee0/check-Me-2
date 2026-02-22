# import streamlit as st
# import predict
# import pandas as pd
# import numpy as np

# st.set_page_config(page_title="Check Me: Risk Triage", page_icon="🩺", layout="wide")

# st.title("Check Me: Self-Screening Risk Triage")
# st.markdown("""
# This demo assesses breast cancer risk based on self-reported inputs.
# **Disclaimer:** This tool is for demonstration purposes only. It is **not** a diagnostic tool.
# """)

# # Sidebar Inputs
# st.sidebar.header("Patient Inputs")

# age = st.sidebar.slider("Age", 18, 90, 30)

# st.sidebar.subheader("History")
# family_history = st.sidebar.checkbox("Family History of Breast Cancer")
# previous_lumps = st.sidebar.checkbox("Previous Benign Lumps")

# st.sidebar.subheader("Current Symptoms")
# symptoms_exist = st.sidebar.checkbox("Any current breast symptoms?")

# breast_pain = False
# nipple_discharge = False
# skin_dimples = False
# lump_size_mm = np.nan
# symptom_duration_days = 0

# if symptoms_exist:
#     st.sidebar.markdown("---")
#     breast_pain = st.sidebar.checkbox("Breast Pain")
#     nipple_discharge = st.sidebar.checkbox("Nipple Discharge")
#     skin_dimples = st.sidebar.checkbox("Skin Dimpling (Peau d'orange)")
    
#     if st.sidebar.checkbox("Palpable Lump?"):
#         lump_size_mm = st.sidebar.number_input("Est. Lump Size (mm)", min_value=1, max_value=100, value=15)
        
#     symptom_duration_days = st.sidebar.number_input("Duration of symptoms (days)", min_value=1, value=7)

# st.sidebar.subheader("Other Factors")
# pregnancy_status = st.sidebar.checkbox("Currently Pregnant")
# hormonal_contraception = st.sidebar.checkbox("Using Hormonal Contraception")

# col1, col2 = st.sidebar.columns(2)
# region = col1.selectbox("Region", ['North', 'South', 'East', 'West', 'Central'])
# language = col2.selectbox("Language", ['English', 'Spanish', 'Mandarin', 'Other'])

# # Prepare Input Data
# input_data = {
#     'age': age,
#     'family_history': 1 if family_history else 0,
#     'previous_lumps': 1 if previous_lumps else 0,
#     'breast_pain': 1 if breast_pain else 0,
#     'nipple_discharge': 1 if nipple_discharge else 0,
#     'skin_dimples': 1 if skin_dimples else 0,
#     'lump_size_mm': lump_size_mm,
#     'symptom_duration_days': symptom_duration_days,
#     'pregnancy_status': 1 if pregnancy_status else 0,
#     'hormonal_contraception': 1 if hormonal_contraception else 0,
#     'region': region,
#     'language': language
# }

# if st.button("Assess Risk"):
#     with st.spinner("Analyzing risk factors..."):
#         result = predict.predict_risk(input_data)
        
#         if "error" in result:
#             st.error(f"Error: {result['error']}")
#         else:
#             # Display Result
#             risk_band = result['risk_band']
#             risk_color = "green"
#             if "Yellow" in risk_band: risk_color = "orange"
#             if "Red" in risk_band: risk_color = "red"
            
#             st.markdown(f"### Assessment Result: <span style='color:{risk_color}'>{risk_band}</span>", unsafe_allow_html=True)
            
#             col_res1, col_res2 = st.columns(2)
            
#             with col_res1:
#                 st.subheader("Probabilities")
#                 probs = result['probabilities']
#                 st.write(probs)
#                 st.progress(float(probs['Red'])) # Show Red probability as a bar
                
#             with col_res2:
#                 st.subheader("Recommendations")
#                 recs = predict.get_recommendations(result['risk_level'])
#                 for rec in recs:
#                     st.info(f"• {rec}")

#             st.markdown("---")
#             st.subheader("Interpretability (Global)")
#             try:
#                 st.image("shap_summary.png", caption="Top Predictors of Risk (Global SHAP)", width=True)
#             except:
#                 st.warning("SHAP summary plot not found. Run training script first.")

# st.markdown("---")
# st.caption("Safety Disclaimer: This application assumes no liability for medical decisions. If you have any concerns, please consult a doctor immediately.")
import streamlit as st
import predict
import numpy as np

st.set_page_config(
    page_title="Check Me Health",
    layout="centered"
)

st.title("Check Me – Breast Health Self-Check")

st.markdown("""
Hello mama/sister   

This tool helps you check your **breast health risk** based on simple questions.  
It **does NOT replace a doctor**. If you are worried, please visit a clinic.

 Early checking saves lives.
""")

st.markdown("---")

st.header(" About You")

age = st.number_input("Your age", min_value=15, max_value=90, value=None, placeholder="Type your age")

family_history = st.checkbox("Someone in your family had breast cancer")
previous_lumps = st.checkbox("You had a breast lump before")

st.markdown("---")
st.header("🩺 How do you feel now?")

symptoms_exist = st.checkbox("Do you feel anything unusual in your breast?")

breast_pain = False
nipple_discharge = False
skin_dimples = False
lump_size_mm = np.nan
symptom_duration_days = 0

if symptoms_exist:
    breast_pain = st.checkbox("Breast pain")
    nipple_discharge = st.checkbox("Liquid from nipple")
    skin_dimples = st.checkbox("Skin looks like orange peel")

    if st.checkbox("You can feel a lump"):
        lump_size_mm = st.number_input("Size of lump (approx in mm)", 1, 100, 15)

    symptom_duration_days = st.number_input(
        "How many days have you noticed this?",
        1, 365, 7
    )

st.markdown("---")
st.header(" Other Information")

pregnancy_status = st.checkbox("You are pregnant")
hormonal_contraception = st.checkbox("Using family planning pills/injection")

region = st.selectbox(
    "Where do you live?",
    ["Kigali City", "Northern Province", "Southern Province",
     "Eastern Province", "Western Province", "Other"],
    index=None,
    placeholder="Select region..."
)

language = st.selectbox(
    "Preferred language",
    ["English", "Kinyarwanda", "French", "Swahili", "Other"],
    index=None,
    placeholder="Select language..."
)

st.markdown("---")

input_data = {
    'age': age,
    'family_history': int(family_history),
    'previous_lumps': int(previous_lumps),
    'breast_pain': int(breast_pain),
    'nipple_discharge': int(nipple_discharge),
    'skin_dimples': int(skin_dimples),
    'lump_size_mm': lump_size_mm,
    'symptom_duration_days': symptom_duration_days,
    'pregnancy_status': int(pregnancy_status),
    'hormonal_contraception': int(hormonal_contraception),
    'region': region,
    'language': language
}

if st.button("🔍 Check My Risk"):
    # Input Validation
    missing_info = []
    
    if age is None:
        missing_info.append("Please enter your age.")
        
    if region is None:
        missing_info.append("Please select a region.")
        
    if language is None:
        missing_info.append("Please select a language.")
    
    if symptoms_exist and not (breast_pain or nipple_discharge or skin_dimples or not np.isnan(lump_size_mm)):
        missing_info.append("You checked 'Do you feel anything unusual' but didn't select any symptoms below it.")
        
    if missing_info:
        for warning in missing_info:
            st.warning(f"⚠️ {warning}")
    else:
        with st.spinner("Checking... please wait"):
            result = predict.predict_risk(input_data)

        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown("---")
            st.header(" Your Result")

            risk_band = result['risk_band']

            if "Green" in risk_band:
                st.success(" Low Risk – Keep checking yourself monthly.")
            elif "Yellow" in risk_band:
                st.warning(" Medium Risk – Please visit a clinic soon.")
            else:
                st.error(" High Risk – Please go to a hospital immediately.")

            st.markdown("### Chance of High Risk")
            st.progress(float(result['probabilities']['Red']))

            st.markdown("### Top Factors Influencing Risk")
            for reason in result.get('top_reasons', []):
                st.write(f"- {reason}")

            st.markdown("---")
            st.header(" What You Should Do")

            recs = predict.get_recommendations(result['risk_level'])
            for rec in recs:
                st.write("•", rec)

st.markdown("---")
st.caption("""
 This tool is only for education.  
If you feel pain, lump, or worry, please visit your nearest health center.
""")