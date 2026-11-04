
import pandas as pd
import plotly.express as px
import os

# Load cleaned dataset
data = pd.read_csv("data/cleaned_data.csv")

# Convert date column
data["date"] = pd.to_datetime(data["date"])

# Create reports folder
os.makedirs("reports", exist_ok=True)

print("EXPLORATORY DATA ANALYSIS STARTED")

# 1. Total sales by product
product_sales = data.groupby("item")["sales"].sum().reset_index()

product_sales = product_sales.sort_values(
    by="sales", ascending=False
)

print("\nTop 10 Products:")
print(product_sales.head(10))

# Create product sales chart
fig1 = px.bar(
    product_sales.head(10),
    x="item",
    y="sales",
    title="Top 10 Best Selling Products",
    labels={"item": "Product ID", "sales": "Total Sales"}
)

fig1.write_html("reports/product_sales.html")

# 2. Total sales by store
store_sales = data.groupby("store")["sales"].sum().reset_index()

print("\nStore Sales:")
print(store_sales)

fig2 = px.bar(
    store_sales,
    x="store",
    y="sales",
    title="Sales by Store",
    labels={"store": "Store ID", "sales": "Total Sales"}
)

fig2.write_html("reports/store_sales.html")

# 3. Monthly sales analysis
data["month"] = data["date"].dt.to_period("M").astype(str)

monthly_sales = data.groupby("month")["sales"].sum().reset_index()

fig3 = px.line(
    monthly_sales,
    x="month",
    y="sales",
    title="Monthly Sales Trend",
    labels={"month": "Month", "sales": "Total Sales"}
)

fig3.write_html("reports/monthly_sales.html")

# 4. Display important statistics
print("\nTotal Products:", data["item"].nunique())
print("Total Stores:", data["store"].nunique())
print("Total Sales:", data["sales"].sum())

print("\nEDA COMPLETED SUCCESSFULLY!")

print("\nCharts saved inside reports folder.")
