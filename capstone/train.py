import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import mlflow
import joblib

def main(data_path):
    # Enable automatic logging to Azure ML
    mlflow.autolog(log_models=True)
    mlflow.set_experiment("Capstone Test")

    # Load Data
    df = pd.read_csv(data_path)
    X = df.drop(['DEFAULT'], axis=1)
    y = df['DEFAULT']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model (MLflow automatically logs parameters, metrics, and the model artifact)
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "credit_default.pkl")
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy_score(y_test, y_pred)
    precision_score(y_test, y_pred)
    recall_score(y_test, y_pred)
    f1_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
