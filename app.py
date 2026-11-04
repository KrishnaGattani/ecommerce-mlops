import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="E-Commerce MLOps", layout="wide")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load data
df = pd.read_csv("data/cleaned_data.csv", parse_dates=["date"])
metrics = pd.read_csv("reports/model_metrics.csv")
monitor = pd.read_csv("reports/monitoring_results.csv")


def chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        font_color="white"
    )
    st.plotly_chart(fig, width="stretch")


# Navigation
pages = ["Overview", "Sales Analytics", "Demand Forecasting", "Model Monitoring"]
page = st.sidebar.radio("🛒 E-Commerce MLOps", pages)

st.title("E-Commerce Intelligence Platform")
st.caption("Demand Forecasting | Sales Analytics | Model Monitoring")
st.divider()

# 1. Overview
if page == "Overview":
    c = st.columns(4)
    values = [
        ("Total Sales", f"{df.sales.sum():,}"),
        ("Products", df.item.nunique()),
        ("Stores", df.store.nunique()),
        ("Model R²", round(metrics["R2 Score"][0], 4))
    ]

    for col, (name, value) in zip(c, values):
        col.metric(name, value)

    monthly = df.groupby(df.date.dt.to_period("M")).sales.sum()
    monthly.index = monthly.index.astype(str)
    chart(px.line(monthly, title="Monthly Sales Trend"))

# 2. Sales Analytics
elif page == "Sales Analytics":
    st.subheader("Sales Analytics")

    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", ["All"] + sorted(df.store.unique().tolist()))
    item = c2.selectbox("Product", ["All"] + sorted(df.item.unique().tolist()))

    start = st.date_input("Start Date", df.date.min().date())
    end = st.date_input("End Date", df.date.max().date())

    if start > end:
        st.error("Start date must be before end date.")
    else:
        d = df[(df.date.dt.date >= start) & (df.date.dt.date <= end)]

        if store != "All":
            d = d[d.store == store]
        if item != "All":
            d = d[d.item == item]

        c1, c2 = st.columns(2)
        c1.metric("Total Sales", f"{d.sales.sum():,}")
        c2.metric("Average Sales", round(d.sales.mean(), 2))

        chart(px.line(d.groupby("date", as_index=False).sales.sum(),
                      x="date", y="sales", title="Sales Trend"))
        chart(px.bar(d.groupby("item", as_index=False).sales.sum(),
                     x="item", y="sales", title="Product-wise Sales"))


# 3. Demand Forecasting
elif page == "Demand Forecasting":
    st.subheader("Demand Forecasting")

    store = st.selectbox("Select Store", sorted(df.store.unique()))
    item = st.selectbox("Select Product", sorted(df.item.unique()))

    if st.button("Predict Demand"):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={
                    "store_id": int(store),
                    "item_id": int(item)
                },
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            st.metric(
                "Predicted Demand",
                f"{result['predicted_sales']:.2f} units"
            )

            st.success(f"Forecast Date: {result['forecast_date']}")

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")


# 4. Model Monitoring
elif page == "Model Monitoring":
    st.subheader("Model Monitoring")

    c = st.columns(3)
    for col, name in zip(c, ["MAE", "RMSE", "R2 Score"]):
        col.metric(name, round(metrics[name][0], 4))

    if "status" in monitor.columns:
        errors = monitor.status.value_counts()
        chart(px.pie(values=errors.values, names=errors.index,
                     title="Prediction Error Monitoring"))
        st.metric("High Error Predictions",
                  int((monitor.status == "High Error").sum()))

    chart(px.line(monitor.head(100), y=["Actual", "Predicted"],
                  title="Actual vs Predicted Sales"))
    
    st.subheader("Data Drift Detection")

    drift = pd.read_csv("reports/drift_report.csv")
    st.metric("KS Statistic", round(drift["KS Statistic"][0], 4))
    st.metric("P-value", drift["P-value"][0])

    if drift["Status"][0] == "Drift Detected":
        st.warning("Data Drift Detected!")
    else:
        st.success("No Significant Drift")

