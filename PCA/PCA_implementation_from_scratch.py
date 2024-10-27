import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Load the dataset
# Update the path below to your local path where the dataset is located
dataset = pd.read_csv("/content/drive/MyDrive/Digit Recognizer Dataset/train.csv" )

# Split the dataset into features and labels
X_train = dataset.drop(columns = "label").values
y_train = dataset["label"].values

print("Shape of the dataset before applying PCA:", X_train.shape)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components    # Number of principal components to keep
        self.eigenvalues = None             # To store eigenvalues
        self.eigenvectors = None            # To store eigenvectors
        self.mean = None                    # To store the mean of the dataset


    def fit(self, X):

        # Compute the mean of the dataset
        self.mean = np.mean(X, axis = 0)    # Mean of each feature
        X_centered = X - self.mean          # Center the data by subtracting the mean

        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar = False)  # Covariance of the centered data

        # Compute eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(self.eigenvalues)[::-1]         # Indices of eigenvalues in descending order
        self.eigenvalues = self.eigenvalues[sorted_indices]         # Sorted eigenvalues
        self.eigenvectors = self.eigenvectors[:, sorted_indices]    # Sorted eigenvectors

        # Retain only the specified number of components
        self.eigenvectors = self.eigenvectors[:, :self.n_components]


    def transform(self, X):
        X_centered = X - self.mean      # Center the data
        return np.dot(X_centered, self.eigenvectors)    # Project the data onto the principal components


    def fit_transform(self, X):
        self.fit(X)         # Fit the PCA model
        return self.transform(X)        # Transform the data


    def explained_variance(self):

        return ((self.eigenvalues[:self.n_components]) / np.sum(self.eigenvalues))  # Explained variance ratio

# For 3D visualization
n_components = 3
pca_3d = PCA(n_components)      # Create PCA instance for 3 components
X_pca_3d = pca_3d.fit_transform(X_scaled)   # Fit and transform the scaled data

print("Shape of the dataset after applying PCA for 3D:", X_pca_3d.shape)

def plot_3D_pca(X_pca, y_train):

    # Create a DataFrame for the PCA results
    df = pd.DataFrame(X_pca, columns = ["PC1", "PC2", "PC3"])
    df["Label"] = y_train   # Add labels to the DataFrame

    # Create a 3D scatter plot using Plotly
    fig = px.scatter_3d(
        df,
        x = "PC1",
        y = "PC2",
        z = "PC3",
        color="Label",
        title="3D PCA",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2", "PC3": "Principal Component 3"},
        color_continuous_scale = px.colors.sequential.Viridis
    )
    fig.update_traces(marker = dict(size = 3))
    fig.show()

plot_3D_pca(X_pca_3d, y_train)

# For 2D visualization
n_components = 2
pca_2d = PCA(n_components)      # Create PCA instance for 2 components
X_pca_2d = pca_2d.fit_transform(X_scaled)   # Fit and transform the scaled data

print("Shape of the dataset after applying PCA for 2D:", X_pca_2d.shape)

def plot_2D_pca(X_pca, y_train):

    # Create a DataFrame for the PCA results
    df = pd.DataFrame(X_pca, columns = ["PC1", "PC2"])
    df["Label"] = y_train   # Add labels to the DataFrame

    # Create a 2D scatter plot using Plotly
    fig = px.scatter(
        df,
        x = "PC1",
        y = "PC2",
        color = "Label",
        title = "2D PCA",
        labels = {"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        hover_data = {"PC1": True, "PC2": True, "Label": True},
        color_continuous_scale = px.colors.sequential.Viridis
    )
    fig.update_traces(marker = dict(size = 5))
    fig.show()

plot_2D_pca(X_pca_2d, y_train)