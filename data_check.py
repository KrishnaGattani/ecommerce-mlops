
import pandas as pd
import os

# Load dataset
data = pd.read_csv("data/train.csv")

print("DATA CLEANING STARTED")

# Convert date into datetime format
data["date"] = pd.to_datetime(data["date"])

# Check duplicate records
duplicates = data.duplicated().sum()
print("\nDuplicate Records:", duplicates)

# Remove duplicate records
data = data.drop_duplicates()

# Sort data by date
data = data.sort_values("date")

# Check dataset information
print("\nDataset Information:")
print(data.info())

# Display date range
print("\nFirst Date:", data["date"].min())
print("Last Date:", data["date"].max())

# Calculate total sales
print("\nTotal Sales:", data["sales"].sum())

# Calculate average sales
print("Average Sales:", data["sales"].mean())

# Calculate highest sales
print("Maximum Sales:", data["sales"].max())

# Calculate lowest sales
print("Minimum Sales:", data["sales"].min())

# Save cleaned dataset
data.to_csv("data/cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("DATA CLEANING COMPLETED!")
