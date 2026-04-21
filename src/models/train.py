import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import yaml
import joblib

DATA_DIR = "data/processed"
MODEL_DIR = "models"


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel().astype(int)
    y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel().astype(int)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, params):
    model_params = params["model"]
    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=model_params.get("random_state", 42), 
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds     = model.predict(X_test)
    acc       = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall    = recall_score(y_test, preds, zero_division=0)
    return acc, precision, recall


if __name__ == "__main__":
    params = load_params()
    X_train, X_test, y_train, y_test = load_data()

    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run() as run:
        model = train_model(X_train, y_train, params)

        model_path = os.path.join(MODEL_DIR, "model.joblib")
        joblib.dump(model, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FATAL: joblib failed to create {model_path}")
        print(f"Model saved and verified at: {model_path}")

        acc, precision, recall = evaluate(model, X_test, y_test)

        mlflow.log_params(params["model"])
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="churn_model",
        )