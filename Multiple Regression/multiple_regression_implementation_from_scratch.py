import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
# This dataset is provided in this repository.
# Update the path below to your local path where the dataset is located
dataset = pd.read_csv("/content/drive/MyDrive/Datasets/student_performance_dataset.csv")

# Display the first few rows of the dataset
print(dataset.head())

# Encode categorical variables using one-hot encoding, dropping the first category to avoid multicollinearity
dataset_encoded = pd.get_dummies(dataset, drop_first = True)

# Display the first few rows of the encoded dataset
print(dataset_encoded.head())

X = dataset_encoded.iloc[:, 1:].values  # features
y = dataset_encoded.iloc[:, 0].values   # target variables

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Standardize the features by removing the mean and scaling to unit variance
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)     # Fit on training data and transform it
X_test = scaler.transform(X_test)           # Transform the testing data using the same scaler

class MultipleRegression:

    def __init__(self):
        self.coef_ = None       # Coefficients for the features
        self.intercept_ = None  # Intercept of the regression model


    def fit(self, X_train, y_train):

        # Add a column of ones to X_train to account for the intercept
        X_train = np.insert(X_train, 0, 1, axis = 1)

        # Calculate coefficients using the Normal Equation
        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]  # First coefficient is the intercept
        self.coef_ = betas[1:]      # Remaining coefficients are for the features


    def predict(self, X_test):
        # Predict the target variable using the coefficients and intercept
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        return y_pred


    def mse(self, y_true, y_pred):
        # Calculate Mean Squared Error
        return np.mean((y_true - y_pred) ** 2)


    def r2_score(self, y_true, y_pred):
        # Calculate R-squared score
        numerator = np.sum((y_true - y_pred)**2)
        denominator = np.sum((y_true - np.mean(y_true))**2)

        return (1 - (numerator / denominator))


model = MultipleRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the model's intercept and coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Calculate and print performance metrics
mse = model.mse(y_test, y_pred)
r2 = model.r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 score: {r2}")

# Function to plot actual vs predicted values
def plot_predictions(y_test, y_pred):

    plt.figure(figsize=(10, 6))

    # Scatter plot of actual vs predicted values
    plt.scatter(range(len(y_test)), y_test, color = "blue", label = "Actual Point")
    plt.scatter(range(len(y_pred)), y_pred, color = "red", label = "Predicted Point")

    plt.title("Actual vs Predicted Medical Costs")
    plt.xlabel("Test Data Points")
    plt.ylabel("Medical Costs")
    plt.legend()
    plt.show()

plot_predictions(y_test, y_pred)