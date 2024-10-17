import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load the Wine dataset
wine = load_wine()

# Features (X) and target labels (y)
X = wine.data
y = wine.target

print(X)
print(y)

# Split the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

class NaiveBayes:

    def fit(self, X, y):

        # Identify unique classes in the target labels
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        # Calculate mean, variance, and prior probability for each class
        for cls in self.classes:

            X_cls = X[y == cls]  # Select samples belonging to the class
            self.mean[cls] = np.mean(X_cls, axis=0)  # Mean of the features
            self.var[cls] = np.var(X_cls, axis=0)    # Variance of the features
            self.prior[cls] = X_cls.shape[0] / X.shape[0]  # Prior probability


    def predict(self, X):

        predictions = []

        # Iterate over each instance to predict its class
        for x in X:
            class_probabilities = {}

            # Calculate posterior probability for each class
            for cls in self.classes:
                likelihood = np.prod(self.gaussian_likelihood(x, cls))  # Likelihood of the features given the class
                posterior = self.prior[cls] * likelihood  # Posterior probability
                class_probabilities[cls] = posterior

            # Append the class with the highest probability
            predictions.append(max(class_probabilities, key=class_probabilities.get))

        return np.array(predictions)


    def gaussian_likelihood(self, x, cls):
        # Calculate the Gaussian likelihood of the features for the given class
        mean = self.mean[cls]
        var = self.var[cls]

        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

# Create an instance of the NaiveBayes class
nb = NaiveBayes()

# Fit the model to the training data
nb.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = np.mean(y_pred == y_test) * 100
print(f"Accuracy of Naive Bayes Model implemented from scratch is {accuracy:.2f}%")
