import numpy as np
import pandas as pd
from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X = iris.data
y = iris.target


print(X)
print(iris.target_names)    # These are target names in which we have to classify our test data, after training the model.
print(y)

# 80% data will be used for training the model and rest 20% of data will used for testing the model's accuracy.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

class KNN:

    def __init__(self, k = 3):
        """
        Initialize the KNN model with the specified number of neighbors i.e. k.
        """
        self.k = k


    def fit(self, X_train, y_train):
        """
        Fiting the KNN model to the training data. It stores the data for future predictions.

        Parameters:
        - X_train: The training data.
        - y_train: The labels for the training data.
        """

        self.X_train = X_train
        self.y_train = y_train


    def euclidean_distance(self, X_train, X_test_point):
        """
        Compute the Euclidean distance between a test point and each point in the training data.

        Parameters:
        - X_train: The training data.
        - X_test_point: A single test point.

        Returns:
        - A DataFrame containing the distances between the test point and each training point.
        """

        distances = []  # List to store distances between the test point and each training point.

        for train_point in X_train:

            # Compute squared difference for each feature
            curr_distance = sum((train_point[col] - X_test_point[col]) ** 2 for col in range(len(train_point)))

            # Take the square root of the sum to get the Euclidean distance
            distances.append(np.sqrt(curr_distance))

        # Convert distances to a DataFrame for easier sorting later
        return pd.DataFrame(data = distances, columns = ["dist"])


    def nearest_neighbors(self, distances):
        """
        Identify the k nearest neighbors based on distance.

        Parameters:
        - distances: A DataFrame containing distances from the test point to each training point.

        Returns:
        - A DataFrame containing the k nearest neighbors.
        """

        # Sort distances in ascending order and pick the first k rows (smallest distances)
        df_nearest = distances.sort_values(by = "dist", axis = 0).head(self.k)

        return df_nearest


    def voting(self, df_nearest, y_train):
        """
        Perform majority voting among the k nearest neighbors to determine the predicted label.

        Parameters:
        - df_nearest: DataFrame containing the k nearest neighbors.
        - y_train: The labels of the training data.

        Returns:
        - The predicted label (the label with the most votes).
        """

        # Get the indices of the nearest neighbors and retrieve their labels from y_train
        nearest_labels = y_train[df_nearest.index]

        # Count the frequency of each label (majority voting)
        vote_counts = {}
        for label in nearest_labels:
            vote_counts[label] = vote_counts.get(label, 0) + 1

        # Return the label with the highest vote count
        return max(vote_counts, key = vote_counts.get)


    def predict(self, X_test):
        """
        Predict the labels for the test data using the KNN algorithm.

        Parameters:
        - X_test: The test dataset.

        Returns:
        - A list of predicted labels for each test point.
        """

        y_pred = []  # List to store predicted labels for each test point.

        # Iterate over each test point and make predictions
        for X_test_point in X_test:

            # Compute distances between this test point and all training points
            distances = self.euclidean_distance(self.X_train, X_test_point)

            # Find the nearest neighbors
            df_nearest_point = self.nearest_neighbors(distances)

            # Perform majority voting to get the predicted label
            y_pred_point = self.voting(df_nearest_point, self.y_train)

            # Append the predicted label to the list
            y_pred.append(y_pred_point)

        return y_pred


    def calc_accuracy(self, y_pred, y_test):
        """
        Calculate the accuracy of the model by comparing predicted labels to actual labels.

        Parameters:
        - y_pred: List of predicted labels.
        - y_test: List of actual labels.

        Returns:
        - Accuracy of the model as a percentage.
        """

        correct_predictions = np.sum(np.array(y_pred) == np.array(y_test))
        total_predictions = len(y_test)
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy

knn = KNN(3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = knn.calc_accuracy(y_pred, y_test)

print(f"Accuracy of KNN Model is which implemented from scratch is {accuracy:.2f}%")
