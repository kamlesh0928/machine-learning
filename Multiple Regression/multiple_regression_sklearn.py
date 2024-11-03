import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# This dataset is provided in this repository.
# Update the path below to your local path where the dataset is located
dataset = pd.read_csv("/content/drive/MyDrive/Datasets/student_performance_dataset.csv")
print(dataset.head())

dataset_encoded = pd.get_dummies(dataset, drop_first = True)
print(dataset_encoded.head())

X = dataset_encoded.iloc[:, 1:].values
y = dataset_encoded.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R2 score: ", r2_score(y_test, y_pred))
