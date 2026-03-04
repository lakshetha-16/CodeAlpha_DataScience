import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV
data = pd.read_csv("C:\\Users\\Home PC\\Downloads\\Iris.csv")

# Drop the Id column (not needed for training)
data = data.drop("Id", axis=1)

# Inputs (features) and output
X = data.drop("Species", axis=1)
y = data["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
