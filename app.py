import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Models
@st.cache_resource
def load_models():
    with open("xgb_model.pkl", "rb") as f1, open("rf_model.pkl", "rb") as f2, open("lr_model.pkl", "rb") as f3:
        return pickle.load(f1), pickle.load(f2), pickle.load(f3)

xgb_model, rf_model, lr_model = load_models()

# Title
st.title("🔍 Loan Default Prediction App")
st.markdown("Enter loan applicant details below to predict if they will **default or not**.")

# Select Model
model_choice = st.selectbox("Select Prediction Model", ["XGBoost", "Random Forest", "Logistic Regression"])

# User Inputs
loan_amnt = st.slider("Loan Amount", 1000, 40000, 15000)
term = st.selectbox("Term (months)", [36, 60])
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
emp_length = st.slider("Employment Length (years)", 0, 20, 5)
home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
annual_inc = st.slider("Annual Income", 10000, 300000, 60000)
purpose = st.selectbox("Purpose", ['credit_card', 'car', 'home_improvement', 'major_purchase', 'debt_consolidation'])
dti = st.slider("DTI (Debt-to-Income Ratio)", 0.0, 40.0, 10.0)
delinq_2yrs = st.slider("Delinquency in 2 years", 0, 10, 0)
fico_range_high = st.slider("FICO Score (High)", 600, 850, 720)
revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, 50.0)
total_acc = st.slider("Total Credit Accounts", 5, 100, 25)

# Preprocess Inputs
def encode_inputs():
    return pd.DataFrame([{
        'loan_amnt': loan_amnt,
        'term': 1 if term == 60 else 0,
        'int_rate': int_rate,
        'grade': ['A','B','C','D','E','F','G'].index(grade),
        'emp_length': emp_length,
        'home_ownership': ['RENT','OWN','MORTGAGE','OTHER'].index(home_ownership),
        'annual_inc': annual_inc,
        'purpose': ['credit_card','car','home_improvement','major_purchase','debt_consolidation'].index(purpose),
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'fico_range_high': fico_range_high,
        'revol_util': revol_util,
        'total_acc': total_acc
    }])

input_df = encode_inputs()

# Predict Button
if st.button("Predict"):
    if model_choice == "XGBoost":
        model = xgb_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = lr_model

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "❌ Will Default" if prediction == 1 else "✅ Will Not Default"
    st.subheader(f"📊 Prediction: {result}")
    st.markdown(f"**Probability of Default:** `{probability:.2f}`")

# SHAP Explainability (optional)
st.markdown("---")
st.subheader("🔍 Model Explainability with SHAP")
st.image("shap_summary_plot.png", caption="SHAP Feature Importance", use_column_width=True)





