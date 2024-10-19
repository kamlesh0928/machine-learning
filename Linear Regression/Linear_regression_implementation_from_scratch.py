import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
# The dataset is provided in this repository
dataset = pd.read_csv("/content/drive/MyDrive/Advertising_dataset.csv")

X = dataset['TV'].values
y = dataset['Sales'].values

# Plot the relationship between TV advertising budget and Sales

plt.figure(figsize=(10, 6))     # Scatter plot to visualize the data points
plt.plot(X, y, 'o')
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("Advertising Dataset")
plt.show()

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

class SimpleLinearRegression:

    def __init__(self):
        self.m = 0    # Sploe
        self.b = 0    # Intercept


    def fit(self, X, y):

        # Calculate the mean of X (TV budget) and y (Sales)
        mean_X = sum(X) / len(X)
        mean_y = sum(y) / len(y)

        # Compute the slope (m) using the formula for linear regression
        numerator = sum((X[i] - mean_X) * (y[i] - mean_y) for i in range(len(X)))
        denominator = sum((X[i] - mean_X) ** 2 for i in range(len(X)))

        self.m = numerator / denominator        # Slope
        self.b = mean_y - (self.m * mean_X)     # Intercept


    # Method to predict the target variable (Sales) for new inputs (TV budgets)
    def predict(self, X):
        return [(self.b + self.m * x) for x in X]


    # Method to compute the Mean Squared Error (MSE) for model evaluation
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


    # Method to calculate R-squared (R²) score for the model
    def r2_score(self, y_true, y_pred):

        # Calculate the mean of the true values (Sales)
        y_mean = np.mean(y_true)

        # Total sum of squares (variance in the actual data)
        ss_total = np.sum((y_true - y_mean) ** 2)

        # Residual sum of squares (variance in the residuals/predictions)
        ss_residual = np.sum((y_true - y_pred) ** 2)

        # R² score formula: proportion of variance explained by the model
        return (1 - (ss_residual / ss_total))


    # Method to plot the data and the best fit line
    def plot(self, X, y, y_pred):

        # Scatter plot of the data points (blue) and best fit line (red)
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Data points')
        plt.plot(X, y_pred, color='red', label='Best fit line')
        plt.xlabel('TV Advertising Budget')
        plt.ylabel('Sales')
        plt.title('Simple Linear Regression on Advertising Dataset')
        plt.legend()
        plt.show()

model = SimpleLinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Intercept: {model.b:.2f}")
print(f"Slope: {model.m:.2f}")

mse = model.mean_squared_error(y_test, y_pred)
r2 = model.r2_score(y_test, y_pred)

print(f"Mean Squared Error on test set: {mse:.2f}")
print(f"R2 score on test set: {r2:.2f}")

model.plot(X_test, y_test, y_pred)
