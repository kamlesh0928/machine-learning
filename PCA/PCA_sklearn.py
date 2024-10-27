import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# This dataset is provided in this repository.
# Update the path below to your local path where the dataset is located
dataset = pd.read_csv("/content/drive/MyDrive/Digit Recognizer Dataset/train.csv")

X_train = dataset.drop(columns = "label").values
y_train = dataset["label"].values

print("Shape of the dataset before applying PCA:", X_train.shape)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X_train)

# For 3D visualization

n_components = 3
pca_3d = PCA(n_components = n_components)
X_pca_3d = pca_3d.fit_transform(x_scaled)

print("Shape of the dataset after applying PCA for 3D:", X_pca_3d.shape)

def plot_3D_pca(X_pca, y_train):

    df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    df["Label"] = y_train

    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Label",
        title="3D PCA",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2", "PC3": "Principal Component 3"},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_traces(marker=dict(size=3))
    fig.show()

plot_3D_pca(X_pca_3d, y_train)


# For 2D visualization
n_components = 2
pca_2d = PCA(n_components = n_components)
X_pca_2d = pca_2d.fit_transform(x_scaled)

print("Shape of the dataset after applying PCA for 2D:", X_pca_2d.shape)

def plot_2D_pca(X_pca, y_train):

    df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df["Label"] = y_train

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Label",
        title="2D PCA",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        hover_data={"PC1": True, "PC2": True, "Label": True},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_traces(marker=dict(size=5))
    fig.show()

plot_2D_pca(X_pca_2d, y_train)
