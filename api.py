
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="E-Commerce Demand Forecasting API")

df = pd.read_csv("data/cleaned_data.csv", parse_dates=["date"])
model = joblib.load("models/demand_model.pkl")
features = joblib.load("models/model_features.pkl")

class PredictionInput(BaseModel):
    store_id: int
    item_id: int


@app.get("/")
def home():
    return {"message": "E-Commerce Demand Forecasting API"}


@app.get("/health")
def health():
    return {"status": "API is running"}


@app.get("/stores")
def stores():
    return {"stores": sorted(df.store.unique().tolist())}


@app.get("/items")
def items():
    return {"items": sorted(df.item.unique().tolist())}


@app.get("/summary")
def summary():
    return {
        "total_sales": int(df.sales.sum()),
        "total_stores": int(df.store.nunique()),
        "total_products": int(df.item.nunique())
    }


@app.get("/metrics")
def metrics():
    data = pd.read_csv("reports/model_metrics.csv")
    return data.to_dict(orient="records")


@app.get("/drift")
def drift():
    data = pd.read_csv("reports/drift_report.csv")
    return data.to_dict(orient="records")


@app.post("/predict")
def predict(data: PredictionInput):

    h = df[
        (df.store == data.store_id) &
        (df.item == data.item_id)
    ].sort_values("date")

    if h.empty:
        raise HTTPException(status_code=404, detail="Store or product not found")

    if len(h) < 28:
        raise HTTPException(status_code=400, detail="At least 28 days of history are required")

    date = h.date.max() + pd.Timedelta(days=1)

    values = {
        "store": data.store_id,
        "item": data.item_id,
        "day": date.day,
        "month": date.month,
        "year": date.year,
        "day_of_week": date.day_of_week,
        "lag_1": h.sales.iloc[-1],
        "lag_7": h.sales.iloc[-7],
        "lag_28": h.sales.iloc[-28],
        "rolling_7": h.sales.tail(7).mean(),
        "rolling_28": h.sales.tail(28).mean()
    }

    X = pd.DataFrame([values])[features]
    result = model.predict(X)[0]

    return {
        "store_id": data.store_id,
        "item_id": data.item_id,
        "forecast_date": str(date.date()),
        "predicted_sales": round(float(result), 2),
        "unit": "units"
    }
