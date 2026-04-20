import os 
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

#path 

DATA_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    return acc, precision, recall

if __name__ == "__main__":
    mlflow.set_experiment("churn prediction")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train)
        acc, precision, recall = evaluate(model, X_test, y_test)

        #parameters logs
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)

        #Metric logs
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        #log model
        mlflow.sklearn.log_model(model, "model")

        print(f"accuracy: {acc}")