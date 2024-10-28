import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report

dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(f"Original training data shape: {X_train.shape}")
print(f"Original testing data shape: {X_test.shape}")

lda = LinearDiscriminantAnalysis()

lda.fit(X_train, y_train)

X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

print(f"Transformed training data shape: {X_train_lda.shape}")
print(f"Transformed testing data shape: {X_test_lda.shape}")

y_pred = lda.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def plot(X_train_lda, y_train):

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_lda, np.zeros_like(X_train_lda), c = y_train, cmap = "viridis", alpha = 0.5)
    plt.title("LDA (1D Projection)")
    plt.xlabel("Linear Discriminant 1")
    plt.colorbar(label = "Class (0 = Malignant, 1 = Benign)")
    plt.grid()
    plt.show()

plot(X_train_lda, y_train)