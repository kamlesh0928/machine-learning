import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs


# Generate synthetic dataset with 3 clusters
data, _ = make_blobs(n_samples = 100, centers = 3, n_features = 2, random_state = 42)


# Visualize the blob dataset
plt.scatter(data[:, 0], data[:, 1], color = "red", marker = 'o', edgecolor = "black", s = 50)
plt.title("Blob Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


class HierarchicalClustering:

    def __init__(self, data, n_clusters = 3, linkage_method = "single"):
        # Initialize the clustering with data, number of clusters, and linkage method
        self.data = data
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.dist_matrix = None     # To store the distance matrix
        self.clusters = None        # To store the resulting clusters
        self.linkage_matrix = None  # To store the linkage matrix for dendrogram


    def calculate_distance_matrix(self):
        # Calculate the distance matrix using the pairwise distance
        self.dist_matrix = squareform(pdist(self.data, metric = "euclidean"))


    def perform_clustering(self):

        # Initialize clusters, each containing a single data point
        clusters = [[i] for i in range(len(self.dist_matrix))]

        # Merge clusters until we reach the desired number of clusters
        while len(clusters) > self.n_clusters:

            min_dist = float("inf")  # Initialize minimum distance to infinity
            to_merge = (None, None)  # Initialize the indices of clusters to merge

            # Find the two closest clusters
            for i in range(len(clusters)):
                for j in range((i + 1), len(clusters)):

                    # Calculate the minimum distance between two clusters
                    dist = np.min([self.dist_matrix[p1][p2] for p1 in clusters[i] for p2 in clusters[j]])

                    # Update minimum distance and clusters to merge if a new minimum is found
                    if(dist < min_dist):
                        min_dist = dist
                        to_merge = (i, j)

            # Merge the two closest clusters
            clusters[to_merge[0]].extend(clusters[to_merge[1]])
            del clusters[to_merge[1]]   # Remove the merged cluster

        # Store the final clusters
        self.clusters = clusters


    def create_dendrogram(self):

        # Create a linkage matrix for the dendrogram
        self.linkage_matrix = linkage(self.data, method = self.linkage_method)

        # Plot the dendrogram
        plt.figure(figsize = (10, 8))
        dendrogram(self.linkage_matrix)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Clusters")
        plt.ylabel("Distance")
        plt.show()


    def get_clusters(self):

        # Ensure clustering has been performed
        if(self.clusters is None):
            self.perform_clustering()

        # Print and return the clusters
        for i, cluster in enumerate(self.clusters):
            print(f"Cluster {i + 1}: {cluster}")

        return self.clusters


# Instantiate the hierarchical clustering with the generated data
hc = HierarchicalClustering(data, n_clusters = 3, linkage_method = "single")
hc.calculate_distance_matrix()  # Calculate distance matrix
hc.create_dendrogram()           # Create and display the dendrogram
clusters = hc.get_clusters()     # Get and print the clusters


# Plot the clustered data points
def plot_clusters(clusters):

    for cluster in clusters:
        cluster_points = data[cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], edgecolor = "black", s = 50)

    plt.title("Hierarchical Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_clusters(clusters)