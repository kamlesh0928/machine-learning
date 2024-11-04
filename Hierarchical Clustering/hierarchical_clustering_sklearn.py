import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# Generate synthetic dataset with 3 clusters
data, _ = make_blobs(n_samples = 100, centers = 3, n_features = 2, random_state = 42)


# Visualize the blob dataset
plt.scatter(data[:, 0], data[:, 1], color = "red", marker = 'o', edgecolor = "black", s = 50)
plt.title("Generated Blob Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


clustering = AgglomerativeClustering(n_clusters = 3, linkage = "single")
labels = clustering.fit_predict(data)


# Create dendrogram and visualize it
def create_dendrogram():

    linkage_matrix = linkage(data, method = "single")

    plt.figure(figsize = (10, 8))
    dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Cluster")
    plt.ylabel("Distance")
    plt.show()

create_dendrogram()


# Plot the clustered data points
def plot_clusters():

    plt.scatter(data[:, 0], data[:, 1], c = labels, marker = 'o', edgecolor = "black", s = 50)
    plt.title("Agglomerative Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_clusters()