# eda.py - Full ML Pipeline with EDA, Cleaning, Modeling, and SHAP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_csv("loan_data_sample.csv")
print("✅ Dataset loaded. Shape:", df.shape)

# --- Step 1. Visual & Descriptive EDA ---
print("\n📌 Column Names:", df.columns.tolist())
print("\n🔍 Info:")
print(df.info())

print("\n📊 Summary Stats:")
print(df.describe())

print("\n❓ Missing Values:")
print(df.isnull().sum())

# Visuals
plt.figure(figsize=(6, 4))
msno.matrix(df)
plt.title("Missing Value Heatmap")
plt.show()

sns.countplot(x='loan_status', data=df)
plt.title("Loan Status Distribution")
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# More Plots
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    plt.figure(figsize=(5,3))
    sns.countplot(y=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

# Boxplots for numerical features
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

# --- Step 2. Cleaning ---
initial_shape = df.shape
df = df.drop_duplicates()
print(f"\n🧹 Removed {initial_shape[0] - df.shape[0]} duplicate rows")

# Fill missing numeric with median
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing object with 'Unknown'
for col in df.select_dtypes(include='object').columns:
    df[col].fillna('Unknown', inplace=True)

# Fix negative values
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].abs()

print("\n✅ Cleaned data. Remaining nulls:")
print(df.isnull().sum())

# --- Step 3. Encoding ---
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
print("\n🔤 Categorical columns encoded.")

# --- Step 4. Scaling ---
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])
print("\n📏 Feature scaling complete.")

# --- Step 5. Feature-Target Split ---
target_col = 'loan_status'
X = df.drop(columns=[target_col])
y = df[target_col]
print("\n🎯 Target separated. X shape:", X.shape, ", y shape:", y.shape)

# --- Step 6. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 7. Train Models ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n🧠 {name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    plt.figure(figsize=(4,3))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

# --- Step 8. SHAP Explainability ---
explainer = shap.Explainer(models['XGBoost'])
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary_plot.png")
print("\n📊 SHAP summary plot saved as shap_summary_plot.png")

# --- Step 9. Save Models & Data ---
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(models['XGBoost'], f)
with open("rf_model.pkl", "wb") as f:
    pickle.dump(models['Random Forest'], f)
with open("lr_model.pkl", "wb") as f:
    pickle.dump(models['Logistic Regression'], f)

print("\n💾 Models saved as .pkl files.")

# Save cleaned data for Power BI
df_cleaned = X.copy()
df_cleaned['loan_status'] = y
df_cleaned.to_csv("cleaned_loan_data.csv", index=False)
print("📁 Cleaned data saved to cleaned_loan_data.csv")



