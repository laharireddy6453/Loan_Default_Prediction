# Step 1.1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
# Step 1.2: Load the Dataset
df = pd.read_csv("loan_data_sample.csv")  # Make sure the file name matches exactly
print("✅ Dataset loaded successfully!")

# View first 5 rows
print(df.head())
# Step 1.3: Initial Data Exploration

# Shape of the dataset
print("Shape of dataset (rows, columns):", df.shape)

# Data types and non-null counts
print("\nInfo:")
print(df.info())

# Summary statistics for numeric columns
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())
# Step 1.4: Visual EDA
print("\n📌 Columns in the dataset:")
print(df.columns)
# Step 1.4: Visual EDA

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# 1. Missing Value Heatmap
print("\n📊 Missing Value Heatmap:")
msno.matrix(df)
plt.title("Missing Data Heatmap")
plt.show()

# 2. Target Distribution (use correct column: loan_status)
print("\n📊 Loan Status Distribution:")
sns.countplot(x='loan_status', data=df)
plt.title("Loan Status Distribution")
plt.xlabel("Status")
plt.ylabel("Count")
plt.show()

# 3. Correlation Heatmap
print("\n📊 Correlation Matrix:")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation")
plt.show()
# Step 1.5: Data Cleaning

# 1. Remove duplicates
initial_shape = df.shape
df = df.drop_duplicates()
print(f"\n🧹 Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

# 2. Handle missing values

# Fill numeric columns with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with "Unknown"
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna("Unknown", inplace=True)

# Check if all missing values are handled
print("\n✅ Remaining missing values after cleaning:")
print(df.isnull().sum())
# Step 1.6: Encode Categorical Columns

from sklearn.preprocessing import LabelEncoder

# Identify object-type columns
cat_cols = df.select_dtypes(include='object').columns
print(f"\n🔤 Categorical columns to encode: {list(cat_cols)}")

# Apply Label Encoding to each
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("✅ Categorical encoding complete.")
# Step 1.7: Feature Scaling (safe version)

from sklearn.preprocessing import StandardScaler

# Get numeric columns only (exclude target which is non-numeric)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(f"✅ Feature scaling complete for numeric columns: {list(numeric_cols)}")
# Step 1.8: Split Features and Target

# Set target column (label)
target_col = 'loan_status'  # already confirmed

# X = all columns except target
X = df.drop(columns=[target_col])

# y = target column
y = df[target_col]

print("\n✅ Features and target split complete.")
print(f"📐 Shape of X: {X.shape}")
print(f"🎯 Shape of y: {y.shape}")
# Step 2.1: Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Train-test split complete.")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
# Step 2.1: Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Train-test split complete.")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
# Step 2.2: Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and train model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred = lr_model.predict(X_test)

# Evaluate model
print("\n✅ Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Step 2.3: Random Forest Model

from sklearn.ensemble import RandomForestClassifier

# Initialize and train
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Evaluate
print("\n🌲 Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_pred))
# Step 2.4: XGBoost Model

from xgboost import XGBClassifier

# Initialize and train
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict
xgb_pred = xgb_model.predict(X_test)

# Evaluate
print("\n🚀 XGBoost Results:")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("\nClassification Report:\n", classification_report(y_test, xgb_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, xgb_pred))
# Step 2.5A: Feature Importance (XGBoost)

import matplotlib.pyplot as plt

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importances = xgb_model.feature_importances_
features = X.columns
sns.barplot(x=feature_importances, y=features)
plt.title("XGBoost Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
# Step 2.5B: Save Cleaned Data

df_cleaned = X.copy()
df_cleaned['loan_status'] = y

df_cleaned.to_csv("cleaned_loan_data.csv", index=False)
print("✅ Cleaned data saved to 'cleaned_loan_data.csv' for Power BI.")

# ✅ Save trained XGBoost model to JSON
xgb_model.save_model("xgb_model.json")
print("✅ XGBoost model saved to xgb_model.json")




