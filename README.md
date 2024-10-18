# Machine Learning Algorithms Repository

Welcome to this repository! Here you find practical implementations of popular supervised and unsupervised learning algorithms. All code is provided both from scratch and using the [scikit-learn](https://scikit-learn.org/stable/) library. We hope this repository serves as a valuable resource for your learning journey!


## Table of Contents

- Introduction
- Algorithms
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Decision Tree Classifier
- Contributing
- License

## Introduction

This repository is intended for learners and practitioners of machine learning. By providing implementations of popular algorithms, both from scratch and using a library, we aim to deepen understanding of how these algorithms work and how to apply them effectively.

## Algorithms

### K-Nearest Neighbors (KNN)

The K-Nearest Neighbors algorithm is a simple yet powerful supervised learning algorithm, used for classification and regression purposes. Its work is based on the principle that similar data points are close to each other in feature space.
 
- **Implementation from Scratch:** This is demonstrating how the KNN algorithm works under the hood.
- **Implementation using scikit-learn :** This implementation makes use of the functionality of the sklearn library to come up with a simple and efficient KNN classification.
- **Code:** You can find the code [here](KNN).

### Naive Bayes

Naive Bayes is a probabilistic machine learning algorithm governed by Bayes' Theorem that widely applies on classification tasks. They assume features to be independent given the class label; thus it runs efficiently and fast, especially for big data.

**Types**
- **Gaussian Naive Bayes:** For Continuous data which is assumed to be normally distributed.
- **Multinomial Naive Bayes:** Used over the discrete count data, like text.
- **Bernoulli Naive Bayes:** when we have binary/boolean features.

**Code:** You can find the code [here](Naive%20Bayes).

### Decision Tree Classifier

A Decision Tree Classifier is supervised learning algorithm applied toward classification, splitting the data into subsets based on feature values and thus forming a tree structure. The internal node in the tree presents a feature, while a leaf node presents a class. That is to say, here we want to make data as pure as possible at each stage down the tree.

#### Components of a Decision Tree:
  
  - **Root Node:** It is the topmost node in the tree, which represents the complete dataset. It is the starting point of the decision-making process.
  - **Internal Node:** A node that symbolizes a choice regarding an input feature. Branching off of internal nodes connects them to leaf nodes or other internal nodes.
  - **Parent Node:** The node that divides into one or more child nodes.
  - **Child Node:** The nodes that emerge when a parent node is split.
  - **Leaf Node:** A node without any child nodes that indicates a class label or a numerical value.
  - **Entropy:** Measures impurity or randomness in the dataset. Lower entropy means the data is more pure (belongs to one class).
  - **Gini Impurity:** An alternative to entropy, it measures the probability of incorrect classification.
  - **Information Gain:** The reduction in entropy after splitting based on a feature. Higher information gain means the feature is better for the split.

**Code:** You can find the code [here](Decision%20Tree%20Classifier).

## Contributing
Contributions are welcome! If you want to add more algorithms or improve existing implementations, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. Please see the LICENSE file for more information.
