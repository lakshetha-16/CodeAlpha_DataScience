# Car Price Prediction with Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Dataset

data = pd.read_csv(
    r"C:\\Users\\Home PC\\Downloads\\car data.csv")

# 2. Data Exploration

print("First 5 rows:\n", data.head())
print("\nDataset Info:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())

# 3. Data Cleaning

data = data.drop("Car_Name", axis=1)
data = pd.get_dummies(data, drop_first=True)

# 4. Features and Target

X = data.drop("Selling_Price", axis=1)   # input features
y = data["Selling_Price"]                # target variable

# 5. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train the Model

model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predictions

y_pred = model.predict(X_test)

# 8. Model Evaluation

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print("R2 Score:", r2)
print("RMSE:", rmse)

# 9. Visualization

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()
