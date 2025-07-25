# app.py

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Load the trained XGBoost model
model = XGBClassifier()
model.load_model("xgb_model.json")

# Load cleaned data to get column structure
df = pd.read_csv("cleaned_loan_data.csv")
input_features = df.drop(columns=['loan_status']).columns

# Define categorical options and their label encodings
categorical_options = {
    'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
    'purpose': ['credit_card', 'debt_consolidation', 'educational', 'home_improvement',
                'major_purchase', 'small_business', 'vacation', 'wedding', 'other'],
}

# Page Title
st.title("💰 Loan Default Prediction App")
st.markdown("Enter loan applicant details below to predict if they are likely to default on a loan.")

# Build the user input form
user_input = {}
for col in input_features:
    if col in categorical_options:
        selected_option = st.selectbox(f"{col}", categorical_options[col])
        user_input[col] = categorical_options[col].index(selected_option)  # label encoding
    else:
        user_input[col] = st.number_input(f"{col}", value=0.0)

# Predict Button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ The applicant is **likely to default** on the loan.")
    else:
        st.success("✅ The applicant is **not likely to default** on the loan.")

