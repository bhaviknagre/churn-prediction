import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml

# Paths
RAW_DATA_PATH = "data/raw/telco_data.xlsx"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    df = pd.read_excel(RAW_DATA_PATH)
    df.columns = df.columns.str.strip()
    return df

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)
    
def clean_data(df):
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    return df

def encode_data(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])
    return df

def split_data(df, test_size):
    df.columns = df.columns.str.strip()
    target_col = "Churn Label" 
    
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' not found.")

    churn_related_cols = [col for col in df.columns if "Churn" in col]
    X = df.drop(columns=churn_related_cols)
    y = df[target_col]
    
    print(f"Features used: {X.columns.tolist()}")
    return train_test_split(X, y, test_size=test_size, random_state=42)

def scale_data(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test

def save_data(X_train, X_test, y_train, y_test):
    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = encode_data(df)
    params = load_params()
    X_train, X_test, y_train, y_test = split_data(df, params["split"]["test_size"])
    X_train, X_test = scale_data(X_train, X_test)
    save_data(X_train, X_test, y_train, y_test)
    print("Preprocessing completed successfully")