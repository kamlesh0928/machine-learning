import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Load dataset
# The dataset is provided in this repository
# Update the path below to your local path where the dataset is located
dataset = pd.read_csv("/content/drive/MyDrive/Datasets/mall_customers_datasets.csv")

print(dataset.head())   # Display the first few rows of the dataset

# Extract relevant features for clustering
X = dataset[["Annual Income (k$)", "Spending Score (1-100)"]].values

class KMeans:

    def __init__(self, n_clusters = 2, max_iter = 100):
        self.n_clusters = n_clusters    # Number of clusters to form
        self.max_iter = max_iter    # Maximum number of iterations
        self.centroids = None   # Centroid coordinates


    def fit_predict(self, X):

        # Randomly select initial centroids from the dataset
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        # Iterate to update centroids and assign clusters
        for i in range(self.max_iter):
            cluster_group = self.assign_clusters(X)    # Assign clusters based on centroids
            old_centroids = self.centroids      # Store old centroids for convergence check

            self.centroids = self.move_centroids(X, cluster_group)  # Update centroids based on cluster assignments

            # Check for convergence
            if (old_centroids == self.centroids).all():
                break

        return cluster_group    # Return the final clusters


    def assign_clusters(self,X):

        cluster_group = []  # To hold the cluster assignments
        distances = []  # To hold the distances to centroids

        # For each data point, calculate the distance to each centroid
        for row in X:

            for centroid in self.centroids:
                # Calculate Euclidean distance
                distances.append(np.sqrt(np.dot(row-centroid, row-centroid)))

            min_distance = min(distances)       # Find the minimum distance
            index_pos = distances.index(min_distance)   # Get the index of the closest centroid
            cluster_group.append(index_pos)     # Assign the cluster
            distances.clear()       # Clear distances for the next point

        return np.array(cluster_group)      # Return cluster assignments as a NumPy array


    def move_centroids(self, X, cluster_group):
        new_centroids = []  # List to hold new centroids

        # Get correct cluster types
        cluster_type = np.unique(cluster_group)

        # For each cluster type, calculate the new centroid as the mean of points in the cluster
        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis = 0))

        return np.array(new_centroids)  # Return updated centroids as a NumPy array


# Initialize KMeans with the desired number of clusters
kmeans = KMeans(n_clusters = 5)

# Fit the model to the data and get the cluster assignments
y_means = kmeans.fit_predict(X)

# Get the final centroids
centroids = kmeans.centroids
print("The centroids are:\n", centroids)


def plot_clusters(X, y_means, centroids):

    # Define colors and labels for each cluster
    colors = ["red", "yellow", "blue", "green", "purple"]
    labels = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"]

    plt.figure(figsize = (10, 8))

    # Plot each cluster with a different color
    for i in range(kmeans.n_clusters):
        plt.scatter(X[y_means == i, 0], X[y_means == i, 1], label = labels[i], color = colors[i], marker = 'o', edgecolor = 'k', s = 100)

    # Plot centroids as black 'X' markers
    plt.scatter(centroids[:, 0], centroids[:, 1], color = "black", marker = 'X', s = 200)

    plt.title('K-Means Clustering')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid()
    plt.show()

plot_clusters(X, y_means, centroids)