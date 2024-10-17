import numpy as np
from sklearn.datasets import load_wine

wine = load_wine()

X = wine.data
y = wine.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy of Naive Bayes Model which implemented using sklearn is {accuracy:.2f}%")