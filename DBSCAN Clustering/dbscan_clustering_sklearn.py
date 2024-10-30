import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Generate the dataset
X, _ = make_moons(n_samples = 1000, noise = 0.1, random_state = 42)

X = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps = 0.2, min_samples = 5)
labels = dbscan.fit_predict(X)


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