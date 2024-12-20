import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generate synthetic dataset with 500 samples, 2 features, and 4 centers (clusters)
X, _ = make_blobs(n_samples = 500, n_features = 2, centers = 4, random_state = 42)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 42)

class KMedoids:

    def __init__(self, n_clusters, max_iter = 100):
        """
        Initialize KMedoids model.

        - n_clusters: Number of clusters
        - max_iter: Maximum number of iterations for the algorithm (default is 100)
        """
        self.n_clusters = n_clusters  # Number of clusters to form
        self.max_iter = max_iter      # Maximum number of iterations for the algorithm
        self.medoids = None           # Array to store the medoids (cluster centers)
        self.labels = None            # Array to store the cluster labels for each sample


    def fit(self, X):
        """
        Fit the K-Medoids algorithm to the dataset.

        - X: Input data to cluster (n_samples x n_features)
        """

        n_samples = X.shape[0]  # Number of samples in the dataset

        # Randomly select initial medoids from the dataset
        random_idx = np.random.choice(n_samples, self.n_clusters, replace = False)
        self.medoids = X[random_idx]

        # Initialize an array to store the previous medoids for convergence check
        prev_medoids = np.zeros_like(self.medoids)

        # Run the algorithm for a maximum number of iterations
        for _ in range(self.max_iter):

            # Calculate the pairwise distances between each point and the medoids
            distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis = 2)

            # Assign each point to the closest medoid (cluster assignment)
            self.labels = np.argmin(distances, axis = 1)

            # Update the medoids for each cluster
            for i in range(self.n_clusters):

                # Extract the points belonging to the current cluster
                cluster_points = X[self.labels == i]

                # Compute the pairwise distances within the cluster
                pairwise_distances = np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis = 2)

                # Calculate the total distance from each point to all other points in the cluster
                distance_sums = np.sum(pairwise_distances, axis = 1)

                # Select the point with the minimum total distance as the new medoid
                new_medoid = cluster_points[np.argmin(distance_sums)]
                self.medoids[i] = new_medoid  # Update the medoid for the cluster

            # Check for convergence (if medoids don't change, exit the loop)
            if np.all(self.medoids == prev_medoids):
                break

            # Save the current medoids for comparison in the next iteration
            prev_medoids = self.medoids.copy()

    def predict(self, X):
        """
        Predict the cluster labels for a given dataset based on the fitted model.

        - Input data to assign to clusters
        - return: Array of cluster labels
        """

        # Calculate the pairwise distances between the points and the medoids
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis = 2)

        # Return the index of the closest medoid for each point
        return np.argmin(distances, axis = 1)

    def fit_predict(self, X):
        """
        Fit the model and return the cluster labels for the dataset in one step.

        - X: Input data to cluster
        - return: Array of cluster labels
        """
        self.fit(X)  # Fit the model to the data
        return self.labels  # Return the cluster labels

    def get_medoids(self):
        """
        Get the current medoids (cluster centers).

        - return: Array of medoids
        """
        return self.medoids

# Create an instance of the KMedoids class with 4 clusters
kmedoids = KMedoids(n_clusters = 4, max_iter = 100)

# Fit the model to the training data
kmedoids.fit(X_train)

# Predict the cluster labels for the training and test data
train_labels = kmedoids.predict(X_train)
test_labels = kmedoids.predict(X_test)

# Plot the clusters and the medoids
plt.figure(figsize = (8, 6))

# Loop over each cluster and plot the points assigned to that cluster
for i in range(kmedoids.n_clusters):
    plt.scatter(X_train[train_labels == i][:, 0], X_train[train_labels == i][:, 1], label = f'Cluster {i+1}')

# Get the medoids and plot them as black stars
medoids = kmedoids.get_medoids()
plt.scatter(medoids[:, 0], medoids[:, 1], c = "black", s = 200, marker = '*', label = "Medoids")

plt.title("K - Medoids Clustering")
plt.legend()
plt.show()
