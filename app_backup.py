
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="E-Commerce MLOps", layout="wide")
px.defaults.template = "plotly_dark"

st.markdown("""

<style>
.stApp {background:#0b1120;color:#f8fafc;}
[data-testid="stSidebar"] {background:#111c32;}
[data-testid="stSidebar"] * {color:#f8fafc !important;}
[data-testid="stMetric"] {
    background:#1e293b;
    padding:15px;
    border-radius:12px;
    border:1px solid #334155;
}
[data-testid="stMetricLabel"] {color:#cbd5e1 !important;}
[data-testid="stMetricValue"] {color:#60a5fa !important;}
h1,h2,h3,p,label {color:#f1f5f9;}
</style>

""", unsafe_allow_html=True)

df = pd.read_csv("data/cleaned_data.csv", parse_dates=["date"])
metrics = pd.read_csv("reports/model_metrics.csv")
monitor = pd.read_csv("reports/monitoring_results.csv")

def chart(fig):
    fig.update_layout(paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", font_color="white")
    st.plotly_chart(fig, width="stretch")

page = st.sidebar.radio("E-Commerce MLOps", ["Overview", "Sales Analytics", "Demand Forecasting", "Model Monitoring"])
st.title("E-Commerce Demand Forecasting Platform")

if page == "Overview":
    c = st.columns(4)
    for col, name, value in zip(c, ["Sales", "Products", "Stores", "R²"],
        [f"{df.sales.sum():,}", df.item.nunique(), df.store.nunique(), round(metrics["R2 Score"][0], 4)]):
        col.metric(name, value)
    chart(px.line(df.groupby(df.date.dt.to_period("M")).sales.sum().rename_axis("Month").reset_index().assign(Month=lambda x: x.Month.astype(str)), x="Month", y="sales", title="Monthly Sales"))

elif page == "Sales Analytics":
    chart(px.bar(df.groupby("item", as_index=False).sales.sum().nlargest(10, "sales"), x="item", y="sales", title="Top Products"))
    chart(px.bar(df.groupby("store", as_index=False).sales.sum(), x="store", y="sales", title="Store Sales"))

elif page == "Demand Forecasting":
    model = joblib.load("models/demand_model.pkl")
    features = joblib.load("models/model_features.pkl")
    store = st.selectbox("Store", sorted(df.store.unique()))
    item = st.selectbox("Product", sorted(df.item.unique()))
    h = df[(df.store == store) & (df.item == item)].sort_values("date")

    if st.button("Predict Demand"):
        d = h.date.max() + pd.Timedelta(days=1)
        v = {"store":store, "item":item, "day":d.day, "month":d.month, "year":d.year,
             "day_of_week":d.day_of_week, "lag_1":h.sales.iloc[-1], "lag_7":h.sales.iloc[-7],
             "lag_28":h.sales.iloc[-28], "rolling_7":h.sales.tail(7).mean(),
             "rolling_28":h.sales.tail(28).mean()}
        st.metric("Predicted Demand", f"{model.predict(pd.DataFrame([v])[features])[0]:.2f} units")
        st.success(f"Forecast Date: {d.date()}")

else:
    c = st.columns(3)
    for col, name in zip(c, ["MAE", "RMSE", "R2 Score"]):
        col.metric(name, round(metrics[name][0], 4))

    if "status" in monitor.columns:
        e = monitor.status.value_counts()
        chart(px.pie(values=e.values, names=e.index, title="Error Monitoring"))
        st.metric("High Error Predictions", int((monitor.status == "High Error").sum()))

    chart(px.line(monitor.head(100), y=["Actual", "Predicted"], title="Actual vs Predicted Sales"))
