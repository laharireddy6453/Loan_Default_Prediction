# Loan_Default_Predictio💸 Loan Default Prediction

A complete end-to-end Machine Learning project that predicts whether a customer will default on their loan. This project combines data preprocessing, EDA, model building (Logistic Regression, Random Forest, XGBoost), and a user-friendly Streamlit interface.

📌 Features

- Cleaned real-world loan dataset
- Preprocessing, encoding, scaling, and PCA
- Multiple ML models with evaluation
- XGBoost model exported to JSON
- Streamlit Web UI for predictions

📁 Project Structure

Loan_Default_Prediction_Project/
│
├── app.py                   # Streamlit UI for prediction
├── eda.py                   # EDA + Preprocessing + Model training
├── model.py                 # ML models: Logistic, RF, XGBoost
├── cleaned_loan_data.csv    # Final cleaned dataset
├── loan_data_sample.csv     # Original raw data
├── xgb_model.json           # Trained XGBoost model
├── README.md                # Project overview

🚀 How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the Streamlit app:
   streamlit run app.py

   App runs at: http://localhost:8502

🧠 Models Used

- Logistic Regression
- Random Forest
- XGBoost (Best performance)

Evaluation metrics:
- Accuracy
- Precision, Recall
- Confusion Matrix
- ROC-AUC Curve

🌐 Streamlit UI

- Enter user inputs like income, FICO score, employment length, etc.
- Instantly get prediction: Default or Not
- Visual model performance metrics

📌 Tech Stack

- Python (Pandas, Scikit-learn, XGBoost)
- Streamlit
- Git & GitHub

📚 Future Enhancements

- Add SHAP or LIME for explainability
- Store data in SQLite or MongoDB
- Deploy with Docker or Heroku
- Build Power BI dashboards using cleaned_loan_data.csv
- Integrate email alerts for high-risk cases
# Loan Default Prediction

This project predicts the likelihood of loan default using machine learning models. It includes extensive exploratory data analysis (EDA), model training, and a Streamlit UI for prediction. Power BI is used for visual reporting.

---

## 📊 Visualizations & Model Outputs

### 1. 📌 Initial Dataset Overview
![Initial Overview](images/aeff5068-8a5d-46ba-aa45-247d94d1a517.png)
> Displaying the structure and initial statistics of the raw dataset.

---

### 2. 📈 Missing Values Matrix
![Missing Values](images/42b7595d-d3d6-422a-ac72-34b8ff0ee581.png)
> Shows missing values in the dataset using a matrix view. Helps decide imputation strategy.

---

### 3. 📊 Loan Status Distribution
![Loan Status Count](images/dbb624da-de7e-4320-ba32-aa742b722033.png)
> Visualizing the target variable distribution – 'Fully Paid' vs 'Charged Off'.

---

### 4. 🔁 Label Encoding Visualization
![Label Encoded Data](images/5f91daae-5a5d-4a04-b260-ccc1861e1392.png)
> Encoded categorical variables to numerical using LabelEncoder.

---

### 5. 📉 Correlation Heatmap
![Correlation Heatmap](images/22519cf3-4fa1-4de0-804c-360ba7116940.png)
> Identifies relationships between features. Useful to detect multicollinearity.

---

### 6. 🌲 Feature Importance – XGBoost
![XGBoost Importance](images/11eaaa83-3771-4343-b09f-c1b72edceb6f.png)
> Highlights which features contributed most to predictions using XGBoost.

---

### 7. 📉 SHAP Summary Plot
![SHAP Summary](images/44dafe51-59dc-48e3-9ddc-b5e4d31dca1a.png)
> Visual explanation of SHAP values to understand model interpretability.

---

### 8. 🧪 Model Confusion Matrix – Logistic Regression
![Confusion Matrix LR](images/2b912116-89cc-4710-a3e8-ac50912ec33a.png)
> Shows true/false positives/negatives for Logistic Regression.

---

### 9. 🧪 Confusion Matrix – Random Forest
![Confusion Matrix RF](images/70893861-4f1b-4701-9512-1f8436459b68.png)
> Performance visualization for Random Forest model.

---

### 10. 🧪 Confusion Matrix – XGBoost
![Confusion Matrix XGB](images/0a75a0fd-4ea9-4e06-9d73-0a92832b97bf.png)
> Final model’s confusion matrix using XGBoost classifier.

---

## 🚀 Technologies Used

- Python (Pandas, Scikit-learn, XGBoost, SHAP)
- Streamlit (for UI)
- Power BI (for reporting)
- Matplotlib & Seaborn (for plots)

---

## 📂 Project Structure



🙋‍♀️ Author

Lahari Sudhini  
GitHub: https://github.com/laharireddy6453  
LinkedIn: https://www.linkedin.com/in/laharireddy6453

⭐ Star this repo if you found it helpful!
