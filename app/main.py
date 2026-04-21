from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

MODEL_URI = "models:/churn-model/4"

model = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/")
def home():
    return {"message":  "churn prediction api"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
    
