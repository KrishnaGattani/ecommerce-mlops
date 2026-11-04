
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

print("DEMAND FORECASTING MODEL TRAINING STARTED")

# Load cleaned dataset
data = pd.read_csv("data/cleaned_data.csv")

data["date"] = pd.to_datetime(data["date"])

# Sort data
data = data.sort_values(["store", "item", "date"])

# Create date features
data["day"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek
data["year"] = data["date"].dt.year

# Create previous sales features
group = data.groupby(["store", "item"])["sales"]

data["lag_1"] = group.shift(1)
data["lag_7"] = group.shift(7)
data["lag_28"] = group.shift(28)

# Create rolling average features
data["rolling_7"] = group.transform(
    lambda x: x.shift(1).rolling(7).mean()
)

data["rolling_28"] = group.transform(
    lambda x: x.shift(1).rolling(28).mean()
)

# Remove rows with missing feature values
data = data.dropna()

# Select input features
features = [
    "store",
    "item",
    "day",
    "month",
    "day_of_week",
    "year",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_7",
    "rolling_28"
]

X = data[features]
y = data["sales"]

# Split data by date
split_date = pd.Timestamp("2017-10-01")

train_data = data[data["date"] < split_date]
test_data = data[data["date"] >= split_date]

X_train = train_data[features]
y_train = train_data["sales"]

X_test = test_data[features]
y_test = test_data["sales"]

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Create model
model = RandomForestRegressor(
    n_estimators=30,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# Train model
print("\nTraining model... Please wait.")

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION RESULTS")
print("------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/demand_model.pkl")

# Save model features
joblib.dump(features, "models/model_features.pkl")

# Save evaluation results
os.makedirs("reports", exist_ok=True)

metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2 Score": r2
}

pd.DataFrame([metrics]).to_csv(
    "reports/model_metrics.csv",
    index=False
)

print("\nModel saved successfully!")
print("Training metrics saved successfully!")
print("\nMODEL TRAINING COMPLETED!")
