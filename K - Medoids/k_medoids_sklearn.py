import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn_extra.cluster import KMedoids

# Generate a synthetic dataset with 500 samples, 2 features, and 4 centers (clusters)
X, _ = make_blobs(n_samples = 500, n_features = 2, centers = 4)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 42)

# Create an instance of the KMedoids algorithm with 4 clusters, max 100 iterations, and a fixed random state for reproducibility
kmedoids = KMedoids(n_clusters = 4, max_iter = 100, random_state = 42)

# Fit the KMedoids model to the training data
kmedoids.fit(X_train)

# Predict the cluster labels for the training and test data
train_labels = kmedoids.predict(X_train)
test_labels = kmedoids.predict(X_test)

# Plot the clusters and the medoids
plt.figure(figsize = (8, 6))

# Loop over each cluster and plot the points assigned to that cluster
for i in range(kmedoids.n_clusters):
    plt.scatter(X_train[train_labels == i][:, 0], X_train[train_labels == i][:, 1], label = f"Cluster {i + 1}")

# Get the medoids and plot them as black stars
medoids = kmedoids.cluster_centers_
plt.scatter(medoids[:, 0], medoids[:, 1], c = "black", s = 200, marker = '*', label = "Medoids")

plt.title("K-Medoids Clustering")
plt.legend()
plt.show()
