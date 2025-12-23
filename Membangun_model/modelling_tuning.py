"""
modelling_tuning.py
Manual MLflow logging + hyperparameter tuning (Skilled 3 pts)
"""
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Path setup
data_dir = os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'heart_disease_preprocessed')
train_path = os.path.join(data_dir, 'train_data.csv')
test_path = os.path.join(data_dir, 'test_data.csv')

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = train_df.columns[-1]
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

mlflow.set_tracking_uri("file://" + os.path.abspath(os.path.join(os.path.dirname(__file__), "mlruns")))
mlflow.set_experiment("HeartDiseaseTuning")

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [200, 500, 1000]
}

grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_params = grid.best_params_
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label='Presence')
rec = recall_score(y_test, y_pred, pos_label='Presence')
f1 = f1_score(y_test, y_pred, pos_label='Presence')
cm = confusion_matrix(y_test, y_pred)

with mlflow.start_run(run_name="LogReg_GridSearch"):
    # Manual logging
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")
    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
    mlflow.sklearn.log_model(best_model, "model")
    print(f"Best Params: {best_params}")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

print("\nRun 'mlflow ui' in terminal to view experiment tracking.")
