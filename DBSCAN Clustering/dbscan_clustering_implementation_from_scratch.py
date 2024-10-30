import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Generate the dataset
X, _ = make_moons(n_samples = 1000, noise = 0.1, random_state = 42)

X = StandardScaler().fit_transform(X)

class DBSCAN:

    def __init__(self, eps, min_points):
        self.eps = eps
        self.min_points = min_points
        self.labels = None


    def fit(self, X):
        self.labels = np.full(len(X), -1)   # Initialize all labels to -1 (noise)
        cluster_id = 0

        for point_idx in range(len(X)):
            if(self.labels[point_idx] != -1):   # Already visited
                continue

            neighbors = self.region_query(X, point_idx)

            if(len(neighbors) < self.min_points):
                self.labels[point_idx] = -1     # Mark as noise
            else:
                cluster_id += 1
                self.labels[point_idx] = cluster_id     # Start a new cluster
                self.expand_cluster(X, neighbors, cluster_id)

        return self.labels


    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))


    def region_query(self, X, point_idx):

        neighbors = []

        for idx in range(len(X)):
            if(self.euclidean_distance(X[point_idx], X[idx]) < self.eps):
                neighbors.append(idx)

        return neighbors


    def expand_cluster(self, X, neighbors, cluster_id):

        for neighbor_idx in neighbors:
            if(self.labels[neighbor_idx] == -1):    # Previously marked as noise
                self.labels[neighbor_idx] = cluster_id

            elif(self.labels[neighbor_idx] != 0):   # Already visited
                continue

            self.labels[neighbor_idx] = cluster_id

            # Get new neighbors and expand if needed
            new_neighbors = self.region_query(X, neighbor_idx)

            if(len(new_neighbors) >= self.min_points):
                self.expand_cluster(X, new_neighbors, cluster_id)


dbscan = DBSCAN(eps=0.2, min_points=5)
labels = dbscan.fit(X)


def plot_clusters(X, labels):

    plt.figure(figsize = (10, 6))

    for label in set(labels):
        color = "black" if label == -1 else plt.cm.rainbow(label / len(set(labels)))
        plt.scatter(X[labels == label, 0], X[labels == label, 1], color=color, label=f"Cluster {label}" if label != -1 else "Noise")

    plt.title("DBSCAN Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

plot_clusters(X, labels)