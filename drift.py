
import pandas as pd
from scipy.stats import ks_2samp

df = pd.read_csv("data/cleaned_data.csv", parse_dates=["date"])

old = df[df.date.dt.year == 2013].sales
new = df[df.date.dt.year == 2017].sales

stat, p = ks_2samp(old, new)
status = "Drift Detected" if p < 0.05 else "No Significant Drift"

print("KS Statistic:", round(stat, 4))
print("P-value:", round(p, 6))
print("Status:", status)

pd.DataFrame({
    "KS Statistic": [stat],
    "P-value": [p],
    "Status": [status]
}).to_csv("reports/drift_report.csv", index=False)
