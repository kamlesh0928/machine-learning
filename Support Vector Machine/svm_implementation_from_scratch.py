import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Only keep two classes for binary classification (Setosa and Versicolor)
X = X[y != 2]  # Remove class 2 (Virginica)
y = y[y != 2]  # Remove class 2 (Virginica)


# Convert the target values to binary (-1 for Setosa, 1 for Versicolor)
y = np.where((y == 0), -1, 1)


# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class SVM:

    def __init__(self, learning_rate = 0.001, lambda_param = 0.01, n_iters = 1000):

        # Initialize hyperparameters and weights
        self.lr = learning_rate  # Learning rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters  # Number of iterations for training
        self.w = None  # Weights of the model (initialized to None)
        self.b = None  # Bias term (initialized to None)


    def fit(self, X, y):

        # Train the SVM model using the given training data
        n_samples, n_features = X.shape     # Number of samples and features

        # Initialize the weights (w) and bias (b)
        self.w = np.zeros(n_features)
        self.b = 0

        # Training loop (gradient descent)
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):

                # Check if the sample satisfies the margin condition
                condition = (y[idx] * (np.dot(x_i, self.w) - self.b)) >= 1

                if(condition):
                    # If condition is satisfied (correct classification), apply regularization
                    dw = self.lambda_param * self.w
                    db = 0

                else:
                    # If condition is violated (misclassification), update weights and bias
                    dw = self.lambda_param * self.w - np.dot(x_i, y[idx])
                    db = y[idx]

                # Gradient descent update for weights and bias
                self.w -= (self.lr * dw)
                self.b -= (self.lr * db)


    def predict(self, X):
        # Predict the class labels for the input data X
        linear_output = np.dot(X, self.w) - self.b

        # Return the sign of the linear output (+1 or -1)
        return np.sign(linear_output)


# Initialize and train the SVM model
svm = SVM(learning_rate = 0.001, lambda_param = 0.01, n_iters = 1000)
svm.fit(X_train, y_train)

# Make predictions on the test set
predictions = svm.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Generate a classification report and print it
classification_report = classification_report(y_test, predictions, target_names = ["Setosa", "Versicolor"])
print(f"Classification report: \n {classification_report}")

# Show the confusion matrix
print(f"Confusion Matrix:\n {confusion_matrix(y_test, predictions)}")


# Function to plot the decision boundary for the SVM model
def plot_decision_boundary(X, y, model):

    # Create a meshgrid for plotting the decision boundary
    x_min, x_max = (X[:, 0].min() - 1), (X[:, 0].max() + 1)
    y_min, y_max = (X[:, 1].min() - 1), (X[:, 1].max() + 1)

    xx0, xx1 = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    X_grid = np.c_[xx0.ravel(), xx1.ravel()]    # Create a grid of points to classify

    # Predict the class labels for the points in the grid
    y_pred = model.predict(X_grid)
    y_pred = y_pred.reshape(xx0.shape)  # Reshape the predictions to match the grid shape

    # Plot the decision boundary
    plt.contourf(xx0, xx1, y_pred, alpha = 0.5, cmap = "cividis")
    plt.scatter(X[:, 0], X[:, 1], c = y, edgecolors = 'k', marker = 'o')
    plt.title("SVM")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.show()


# Visualize the decision boundary using the first two features (sepal length and sepal width)
X_train_vis = X_train[:, :2]  # Only use the first two features for visualization
y_train_vis = y_train  # Target labels for visualization

# Train a new SVM model on the 2D data
svm_vis = SVM(learning_rate = 0.001, lambda_param = 0.01, n_iters = 1000)
svm_vis.fit(X_train_vis, y_train_vis)

# Plot the decision boundary for the 2D data
plot_decision_boundary(X_train_vis, y_train_vis, svm_vis)