# Machine Learning Algorithms Repository

Welcome to this repository! Here you find practical implementations of popular supervised and unsupervised learning algorithms. All code is provided both from scratch and using the [scikit-learn](https://scikit-learn.org/stable/) library. We hope this repository serves as a valuable resource for your learning journey!


## Table of Contents

- Introduction
- Algorithms
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
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

## Contributing
Contributions are welcome! If you want to add more algorithms or improve existing implementations, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. Please see the LICENSE file for more information.
