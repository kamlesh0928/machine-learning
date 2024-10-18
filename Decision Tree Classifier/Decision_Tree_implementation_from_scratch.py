import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# Load dataset
# This dataset is provided in this repository.
dataset = pd.read_csv("/content/drive/MyDrive/bill_authentication.csv")

# Display the first few rows of the dataset
print(dataset.head())

# Split dataset into features (X) and labels (y)
X = dataset.iloc[:, :-1].values  # Features: all columns except the last
y = dataset.iloc[:, -1].values   # Labels: the last column

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)  # 80% train, 20% test

# Define the Node class to represent a node in the decision tree
class Node:

    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        # Parameters used to split data
        self.feature = feature         # Feature index used to split
        self.threshold = threshold     # Threshold value for splitting
        self.left = left               # Left subtree (data <= threshold)
        self.right = right             # Right subtree (data > threshold)
        self.value = value             # Value if it's a leaf node (final label prediction)

    # Check if the node is a leaf node (end of the tree)
    def is_leaf_node(self):
        return self.value is not None  # Leaf node if it contains a value (label)

# Define the DecisionTree class to implement the decision tree classifier
class DecisionTree:

    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = None):

        # Hyperparameters for the decision tree
        self.min_samples_split = min_samples_split  # Minimum number of samples to split a node
        self.max_depth = max_depth                  # Maximum depth of the tree
        self.n_features = n_features                # Number of features to consider for splits
        self.root = None                            # Root node of the decision tree (initialized later)


    # Fit the model to the training data
    def fit(self, X, y):

        # Set the number of features to consider for splits (if not specified, use all features)
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)

        # Build the decision tree by calling the recursive function
        self.root = self.grow_tree(X, y)


    # Recursive function to build the tree
    def grow_tree(self, X, y, depth=0):

        n_samples, n_feats = X.shape  # Number of samples and features
        n_labels = len(np.unique(y))  # Number of unique labels

        # Stopping conditions: max depth reached, only one class, or not enough samples
        if((depth >= self.max_depth) or (n_labels == 1) or (n_samples < self.min_samples_split)):
            leaf_value = self.most_common_label(y)  # Get the most common label

            return Node(value=leaf_value)           # Return a leaf node with the label

        # Randomly select features to consider for the split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best feature and threshold to split the data
        best_feature, best_thresh = self.best_split(X, y, feat_idxs)

        # Split the data based on the best feature and threshold
        left_idxs, right_idxs = self.split(X[:, best_feature], best_thresh)

        # Recursively build the left and right subtrees
        left = self.grow_tree(X[left_idxs, :], y[left_idxs], (depth + 1))
        right = self.grow_tree(X[right_idxs, :], y[right_idxs], (depth + 1))

        # Return the node with the best feature and threshold
        return Node(best_feature, best_thresh, left, right)


    # Function to find the best feature and threshold for splitting
    def best_split(self, X, y, feat_idxs):

        best_gain = -1  # Initialize best information gain
        split_idx, split_threshold = None, None  # Initialize split feature and threshold

        # Iterate through all selected features
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]        # Get the feature values
            thresholds = np.unique(X_column)  # Get unique thresholds

            # Iterate through all unique thresholds
            for thr in thresholds:
                # Calculate information gain for this threshold
                gain = self.information_gain(y, X_column, thr)

                # If gain is better, update the best feature and threshold
                if(gain > best_gain):
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        # Return the best feature index and threshold
        return split_idx, split_threshold


    # Function to calculate information gain from splitting the data
    def information_gain(self, y, X_column, threshold):

        # Calculate entropy before the split (parent entropy)
        parent_entropy = self.entropy(y)

        # Split the data into left and right based on the threshold
        left_idxs, right_idxs = self.split(X_column, threshold)

        # If no split is made (one side is empty), return zero gain
        if(len(left_idxs) == 0 or len(right_idxs) == 0):
            return 0

        # Calculate weighted entropy after the split (children entropy)
        num_samples = len(y)
        num_left, num_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (num_left / num_samples) * entropy_left + (num_right / num_samples) * entropy_right

        # Information gain is the reduction in entropy
        information_gain = parent_entropy - child_entropy

        return information_gain


    # Split the data based on a threshold
    def split(self, X_column, split_thresh):

        left_idxs = np.argwhere(X_column <= split_thresh).flatten()  # Left: values <= threshold
        right_idxs = np.argwhere(X_column > split_thresh).flatten()  # Right: values > threshold

        return left_idxs, right_idxs


    # Function to calculate the entropy of a dataset
    def entropy(self, y):

        hist = np.bincount(y)  # Count occurrences of each label
        ps = hist / len(y)     # Probability of each label

        return -np.sum([p * np.log(p) for p in ps if p > 0])  # Calculate entropy


    # Function to find the most common label in a dataset (for leaf nodes)
    def most_common_label(self, y):

        counter = Counter(y)  # Count occurrences of each label
        value = counter.most_common(1)[0][0]  # Return the most common label

        return value


    # Predict the labels for a dataset
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])  # Traverse the tree for each sample


    # Traverse the tree to make a prediction for a single sample
    def traverse_tree(self, x, node):

        if(node.is_leaf_node()):  # If it's a leaf node, return the node's value (label)
            return node.value

        # Go left if the feature value is <= threshold, else go right
        if(x[node.feature] <= node.threshold):
            return self.traverse_tree(x, node.left)

        return self.traverse_tree(x, node.right)


    # Function to calculate accuracy of predictions
    def accuracy(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)  # Return the proportion of correct predictions

# Initialize the decision tree with max_depth of 10
clf = DecisionTree(max_depth = 10)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = clf.predict(X_test)

# Calculate accuracy of the model
accuracy = clf.accuracy(y_test, y_pred) * 100

# Print the accuracy of the model
print(f"Accuracy of Decision Tree Classifier Model which implemented from scratch is {accuracy:.2f}%")