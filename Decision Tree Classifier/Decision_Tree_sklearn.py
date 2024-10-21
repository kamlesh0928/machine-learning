import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

# This dataset is provided in this repository.
# Update the path below to your local path where the dataset is located
dataset = pd.read_csv("/content/drive/MyDrive/bill_authentication.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

dt = DecisionTreeClassifier(criterion = "entropy", random_state = 42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy of Decision Tree Classifier Model which implemented using sklearn is {accuracy:.2f}%")

report = classification_report(y_test, y_pred)
print("\nClassification report :\n", report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix :\n", conf_matrix)

# The following part will help you to visualize the decision tree how looks.
plt.figure(figsize = (20, 15))
tree.plot_tree(dt, filled = True)
plt.show()
