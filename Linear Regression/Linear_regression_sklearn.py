import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv("/content/drive/MyDrive/Advertising_dataset.csv")

X = dataset['TV'].values
y = dataset['Sales'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train.reshape(-1, 1), y_train)

y_pred = model.predict(X_test.reshape(-1, 1))

print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error on test set: {mse:.2f}")
print(f"R-squared on test set: {r2:.2f}")

def plot(X, y, y_pred):

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='Best fit line')
    plt.xlabel('TV Advertising Budget')
    plt.ylabel('Sales')
    plt.title('Simple Linear Regression on Advertising Dataset')
    plt.legend()
    plt.show()

plot(X_test, y_test, y_pred)