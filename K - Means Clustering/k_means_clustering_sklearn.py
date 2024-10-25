import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# The dataset is provided in this repository
# Update the path below to your local path where the dataset is located
dataset = pd.read_csv("/content/drive/MyDrive/Datasets/mall_customers_datasets.csv")

print(dataset.head())

X = dataset[["Annual Income (k$)", "Spending Score (1-100)"]].values

kmeans = KMeans(n_clusters = 5, random_state = 42)
kmeans.fit(X)

y_means = kmeans.labels_

centroids = kmeans.cluster_centers_
print("The centroids are:\n", centroids)

def plot_clusters(X, y_means, centroids):

    colors = ["red", "yellow", "blue", "green", "purple"]
    labels = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"]

    plt.figure(figsize = (10, 8))

    for i in range(kmeans.n_clusters):
        plt.scatter(X[y_means == i, 0], X[y_means == i, 1], label = labels[i], color = colors[i], marker = 'o', edgecolor = 'k', s = 100)

    plt.scatter(centroids[:, 0], centroids[:, 1], color = "black", marker = 'X', s = 200)

    plt.title('K-Means Clustering')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid()
    plt.show()

plot_clusters(X, y_means, centroids)