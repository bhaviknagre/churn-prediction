import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths
RAW_DATA_PATH = "data/raw/telco_data.xlsx"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    return pd.read_excel(RAW_DATA_PATH)

def clean_data(df):
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    return df

def encode_data(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])
    return df

def split_data(df):
    X = df.drop("Churn Label", axis=1)
    y = df["Churn Label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    
    # Identify numeric columns
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    
    # Fit ONLY on training data, transform both
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test

def save_data(X_train, X_test, y_train, y_test):
    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

if __name__ == "__main__":
    # 1. Prepare
    df = load_data()
    df = clean_data(df)
    df = encode_data(df)

    # 2. Split 
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Scale 
    X_train, X_test = scale_data(X_train, X_test)

    # 4. Save
    save_data(X_train, X_test, y_train, y_test)

    print("Preprocessing completed successfully")