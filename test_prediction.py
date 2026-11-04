
import pandas as pd
import joblib

print("MODEL PREDICTION TEST STARTED")

# Load saved model
model = joblib.load("models/demand_model.pkl")

# Load feature names
features = joblib.load("models/model_features.pkl")

# Load cleaned dataset
data = pd.read_csv("data/cleaned_data.csv")

data["date"] = pd.to_datetime(data["date"])

# Select one store and one product
store_data = data[
    (data["store"] == 1) &
    (data["item"] == 1)
].sort_values("date").copy()

# Calculate previous sales
store_data["lag_1"] = store_data["sales"].shift(1)
store_data["lag_7"] = store_data["sales"].shift(7)
store_data["lag_28"] = store_data["sales"].shift(28)

store_data["rolling_7"] = (
    store_data["sales"].shift(1).rolling(7).mean()
)

store_data["rolling_28"] = (
    store_data["sales"].shift(1).rolling(28).mean()
)

# Create date features
store_data["day"] = store_data["date"].dt.day
store_data["month"] = store_data["date"].dt.month
store_data["day_of_week"] = store_data["date"].dt.dayofweek
store_data["year"] = store_data["date"].dt.year

# Remove missing values
store_data = store_data.dropna()

# Select latest record
latest = store_data.iloc[[-1]]

# Prepare input
X = latest[features]

# Predict sales
prediction = model.predict(X)[0]

print("\nStore ID:", 1)
print("Product ID:", 1)
print("Date:", latest["date"].iloc[0].date())

print("\nActual Sales:", latest["sales"].iloc[0])
print("Predicted Sales:", round(prediction, 2))

print("\nMODEL PREDICTION TEST COMPLETED!")
