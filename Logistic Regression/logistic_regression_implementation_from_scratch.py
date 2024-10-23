import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
# This dataset is provided in this repository.
# Update the path below to your local path where the dataset is located
dataset = pd.read_csv('/content/drive/MyDrive/Datasets/creditcard.csv')

print(dataset.head())  # Display the first few rows of the dataset

# Separate features (X) and target variable (y)
X = dataset.drop('Class', axis = 1).values  # Features: all columns except 'Class'
y = dataset['Class'].values  # Target: 'Class' column indicating fraud (1) or not (0)

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

class LogisticRegression:

    def __init__(self, learning_rate = 0.01, num_iterations = 1000):
        # Initialize hyperparameters: learning rate and number of iterations
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None


    def sigmoid(self, z):
        # Sigmoid activation function
        return (1 / (1 + np.exp(-z)))


    def fit(self, X, y):

        # Train the model on the training data
        num_samples, num_features = X.shape  # Get the number of samples and features
        self.weights = np.zeros(num_features)  # Initialize weights to zeros
        self.bias = 0  # Initialize bias to zero

        # Gradient descent loop
        for i in range(self.num_iterations):
            model = np.dot(X, self.weights) + self.bias  # Linear model
            predictions = self.sigmoid(model)  # Apply the sigmoid function

            # Calculate gradients
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))  # Gradient for weights
            db = (1 / num_samples) * np.sum(predictions - y)  # Gradient for bias

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, X):

        # Make predictions on the input data
        model = np.dot(X, self.weights) + self.bias  # Linear model
        predictions = self.sigmoid(model)  # Apply the sigmoid function
        return [1 if p >= 0.5 else 0 for p in predictions]  # Classify based on threshold


    def accuracy(self, y_true, y_pred):
        # Calculate the accuracy of the model
        return np.mean(y_true == y_pred)  # Return the mean of correct predictions
    

model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
accuracy = model.accuracy(y_test, y_pred) * 100  # Calculate accuracy

# Print the accuracy of the model
print(f'Accuracy: {accuracy:.2f}%')