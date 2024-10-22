import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.random.uniform(-3, 3, 1000)

# You may change the following function according to your need.
y = np.sin(X) + 0.5 * np.cos(2 * X) + np.random.normal(0, 0.2, 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

degree = 10
poly_features = PolynomialFeatures(degree = degree)
X_train_poly = poly_features.fit_transform(X_train.reshape(-1, 1))

model = LinearRegression()
model.fit(X_train_poly, y_train)

X_test_poly = poly_features.transform(X_test.reshape(-1, 1))

predictions = model.predict(X_test_poly)
error = mean_squared_error(y_test, predictions)
r2_score = r2_score(y_test, predictions)

print("Coefficients:", model.coef_)
print("Mean Squared Error:", error)
print("R2 Score:", r2_score)

def plot(X_test, y_test):

    plot_x = np.linspace(-3, 3, 100)
    plot_x_poly = poly_features.transform(plot_x.reshape(-1, 1))
    plot_y = model.predict(plot_x_poly)

    plt.figure(figsize = (10, 6))

    plt.scatter(X_test, y_test, color = "blue", label = "Test data points")
    plt.plot(plot_x, plot_y, color = "red", label = "Polynomial fit", linewidth = 2)

    plt.title("Polynomial Regression on Irregular Dataset")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

plot(X_test, y_test)