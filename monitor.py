
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/cleaned_data.csv", parse_dates=["date"])
g = df.groupby(["store", "item"])["sales"]

df["day"] = df.date.dt.day
df["month"] = df.date.dt.month
df["year"] = df.date.dt.year
df["day_of_week"] = df.date.dt.day_of_week

for i in [1, 7, 28]:
    df[f"lag_{i}"] = g.shift(i)

for i in [7, 28]:
    df[f"rolling_{i}"] = g.transform(lambda x: x.shift(1).rolling(i).mean())

df = df.dropna()
model = joblib.load("models/demand_model.pkl")
features = joblib.load("models/model_features.pkl")

actual = df.sales
predicted = model.predict(df[features])

print("MAE:", mean_absolute_error(actual, predicted))
print("RMSE:", np.sqrt(mean_squared_error(actual, predicted)))
print("R2:", r2_score(actual, predicted))

pd.DataFrame({"Date": df.date, "Actual": actual, "Predicted": predicted}).to_csv("reports/monitoring_results.csv", index=False)

print("Monitoring completed!")
