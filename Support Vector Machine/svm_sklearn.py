import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

predictions = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions) * 100
print(f"Accuracy: {accuracy:.2f}%")

classification_report = classification_report(y_test, predictions, target_names = ["Setosa", "Versicolor"])
print(f"Classification report: \n {classification_report}")

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
svm_vis = SVC()
svm_vis.fit(X_train_vis, y_train_vis)

# Plot the decision boundary for the 2D data
plot_decision_boundary(X_train_vis, y_train_vis, svm_vis)