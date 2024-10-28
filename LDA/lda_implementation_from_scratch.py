import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load the breast cancer dataset
dataset = load_breast_cancer()

# Extract features (X) and target labels (y)
X = dataset.data
y = dataset.target

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Print the shape of the training and testing data
print(f"Original training data shape: {X_train.shape}")
print(f"Original testing data shape: {X_test.shape}")

class LDA:

    def __init__(self):
        # Initialize the weight matrix to None
        self.w_ = None


    def fit(self, X, y):

        # Get unique class labels and number of features
        classes = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(classes)

        # Calculate the overall mean of the data
        overall_mean = np.mean(X, axis = 0)

        # Initialize within-class and between-class scatter matrices
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        # Calculate within-class and between-class scatter matrices
        for c in classes:

            X_c = X[y == c]  # Select the samples for class c
            mean_c = np.mean(X_c, axis=0)  # Mean of class c
            S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))  # Update within-class scatter
            n_c = X_c.shape[0]  # Number of samples in class c
            mean_diff = (mean_c - overall_mean).reshape(n_features, 1)  # Mean difference
            S_B += n_c * (mean_diff).dot(mean_diff.T)  # Update between-class scatter

        # Compute the eigenvalues and eigenvectors of the scatter matrices
        eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

        # Sort the eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigen_values)[::-1]
        self.w_ = eigen_vectors[:, sorted_indices][:, :(n_classes - 1)]  # Select top (n_classes - 1) eigenvectors


    def predict(self, X_train, y_train, X_test):

        predictions = []

        # For each test point, find the nearest training point and assign its label
        for test_point in X_test:
            distances = np.linalg.norm(X_train - test_point, axis = 1)  # Compute distances to all training points
            nearest_index = np.argmin(distances)  # Find the index of the nearest training point
            predictions.append(y_train[nearest_index])  # Append the label of the nearest point

        return np.array(predictions)


    def transform(self, X):
        # Transform the input data using the weight matrix
        return X.dot(self.w_)

# Instantiate and fit the LDA model
lda = LDA()
lda.fit(X_train, y_train)

# Transform the training and testing data
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

# Print the shape of the transformed data
print(f"Transformed training data shape: {X_train_lda.shape}")
print(f"Transformed testing data shape: {X_test_lda.shape}")

# Predict the labels for the test data
y_pred = lda.predict(X_train_lda, y_train, X_test_lda)

# Print confusion matrix and classification report for the model's performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def plot(X_train_lda, y_train):
    
    # Create a scatter plot of the transformed training data
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_lda, np.zeros_like(X_train_lda), c = y_train, cmap = "viridis", alpha = 0.5)
    plt.title("LDA (1D Projection)")  # Title of the plot
    plt.xlabel("Linear Discriminant 1")  # X-axis label
    plt.colorbar(label = "Class (0 = Malignant, 1 = Benign)")  # Colorbar indicating class labels
    plt.grid()  # Enable grid on the plot
    plt.show()  # Display the plot

# Call the plot function to visualize the LDA projection
plot(X_train_lda, y_train)