"""
modelling.py
Train a basic ML model with MLflow autologging (Basic 2 pts)
"""
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Path setup
data_dir = os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'heart_disease_preprocessed')
train_path = os.path.join(data_dir, 'train_data.csv')
test_path = os.path.join(data_dir, 'test_data.csv')

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Assume target is last column
target_col = train_df.columns[-1]
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# MLflow autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="LogisticRegression_Basic"):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    # MLflow will log model, params, metrics, confusion matrix, etc. automatically

print("\nRun 'mlflow ui' in terminal to view experiment tracking.")
