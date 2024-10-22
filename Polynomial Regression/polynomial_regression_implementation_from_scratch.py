import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate synthetic data
X = np.random.uniform(-3, 3, 1000)  # 1000 random values between -3 and 3

# You may change the following function according to your need.
y = np.sin(X) + 0.5 * np.cos(2 * X) + np.random.normal(0, 0.2, 1000)  # Target variable with noise

# Split the data into training and testing sets (80% train, 20% test )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

class PolynomialRegression:

    def __init__(self, degree):
        self.degree = degree  # Degree of the polynomial
        self.coefficients = None  # To store the coefficients after fitting


    def polynomial_features(self, X):
        # Generate polynomial features up to the specified degree
        X_poly = np.array([(X ** i) for i in range(self.degree + 1)]).T
        return X_poly


    def fit(self, X, y):

        # Fit the model to the training data
        X_poly = self.polynomial_features(X)  # Generate polynomial features

        # Calculate coefficients using the Normal Equation
        X_poly_T_X_poly = np.dot(X_poly.T, X_poly)
        X_poly_T_y = np.dot(X_poly.T, y)
        self.coefficients = np.linalg.inv(X_poly_T_X_poly).dot(X_poly_T_y)


    def predict(self, X):
        # Make predictions using the fitted model
        X_poly = self.polynomial_features(X)  # Generate polynomial features for input

        return np.dot(X_poly, self.coefficients)  # Calculate predictions


    def mean_squared_error(self, y_true, y_pred):
        # Calculate Mean Squared Error (MSE)
        return np.mean((y_true - y_pred) ** 2)


    def r2_score(self, y_true, y_pred):

        # Calculate R² score to evaluate model performance
        ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares

        return (1 - (ss_res / ss_tot))  # R² score calculation


# Set polynomial degree and create model instance
degree = 10
model = PolynomialRegression(degree=degree)
model.fit(X_train, y_train)  # Fit the model to training data

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate and print error metrics
error = model.mean_squared_error(y_test, predictions)  # Mean Squared Error
r2_score = model.r2_score(y_test, predictions)  # R² Score

print("Coefficients:", model.coefficients)  # Display learned coefficients
print("Mean Squared Error:", error)  # Display MSE
print("R² Score:", r2_score)  # Display R² Score

def plot(X_test, y_test):

    # Function to plot the test data and the polynomial fit
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color="blue", label="Test data points")  # Scatter plot of test data

    # Sort the test data for a smooth line plot
    X_sorted = np.sort(X_test)
    predictions_sorted = model.predict(X_sorted)  # Get predictions for sorted values
    plt.plot(X_sorted, predictions_sorted, color="red", label="Polynomial fit", linewidth=2)

    plt.title("Polynomial Regression on Irregular Dataset")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

plot(X_test, y_test)