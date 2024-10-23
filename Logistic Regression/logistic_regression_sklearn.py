import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# This dataset is provided in this repository.
# Update the path below to your local path where the dataset is located
dataset = pd.read_csv('/content/drive/MyDrive/Datasets/creditcard.csv')
print(dataset.head())

X = dataset.drop('Class', axis = 1).values
y = dataset['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LogisticRegression(solver = 'liblinear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print(f'Accuracy: {accuracy:.2f}%')