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

🙋‍♀️ Author

Lahari Sudhini  
GitHub: https://github.com/laharireddy6453  
LinkedIn: https://www.linkedin.com/in/laharireddy6453

⭐ Star this repo if you found it helpful!
