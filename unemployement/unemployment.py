import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\Home PC\\Downloads\\Unemployment in India.csv")
data.columns = data.columns.str.strip()
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
data = data.dropna()
plt.figure(figsize=(12,6))
plt.plot(data["Date"], data["Estimated Unemployment Rate (%)"])
plt.title("Unemployment Rate Trend in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()
covid_start = pd.to_datetime("2020-03-01")
before_covid = data[data["Date"] < covid_start]
after_covid = data[data["Date"] >= covid_start]
print("Before COVID:", before_covid["Estimated Unemployment Rate (%)"].mean())
print("After COVID:", after_covid["Estimated Unemployment Rate (%)"].mean())
data["Month"] = data["Date"].dt.month
monthly_avg = data.groupby("Month")["Estimated Unemployment Rate (%)"].mean()
monthly_avg.plot(kind="bar")
plt.title("Seasonal Unemployment Trend (Month-wise)")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.show()
