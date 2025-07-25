# Step 2.1: Train-Test Split

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset from eda.py (optional if you're continuing)
df = pd.read_csv("loan_data_sample.csv")  # Replace with your final cleaned version if saved

# Target column
target_col = 'loan_status'

# Encoding and scaling would already be done in eda.py
# So if reloading raw file here, make sure preprocessing is repeated or use pickle later

# For now, let’s assume you're continuing directly with X, y from eda.py
# If you have them already, paste this below the X, y assignment from eda.py

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Train-test split complete.")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
