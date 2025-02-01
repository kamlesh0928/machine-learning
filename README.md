# Machine Learning Algorithms Repository

Welcome to this repository! Here you find practical implementations of popular supervised and unsupervised learning algorithms. All code is provided both from scratch and using the [scikit-learn](https://scikit-learn.org/stable/) library. We hope this repository serves as a valuable resource for your learning journey!


## Table of Contents

- [Introduction](#introduction)
- [Algorithms](#algorithms)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Naive Bayes](#naive-bayes)
  - [Decision Tree Classifier](#decision-tree-classifier)
  - [Linear Regression](#linear-regression)
  - [Multiple Regression](#multiple-regression)
  - [Polynomial Regression](#polynomial-regression)
  - [Logistic Regression](#logistic-regression)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [K - Means Clustering](#k---means-clustering)
  - [K - Medoids](#k---medoids)
  - [Hierarchical Clustering](#hierarchical-clustering)
  - [DBSCAN Clustering](#dbscan-clustering)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
  - [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
- [Contributions](#contributions)
- [License](#license)

## Introduction

This repository is intended for learners and practitioners of machine learning. By providing implementations of popular algorithms, both from scratch and using a library, we aim to deepen understanding of how these algorithms work and how to apply them effectively.

## Algorithms

### K-Nearest Neighbors (KNN)

The K-Nearest Neighbors algorithm is a simple yet powerful supervised learning algorithm, used for classification and regression purposes. Its work is based on the principle that similar data points are close to each other in feature space.
 
- **Implementation from Scratch:** This is demonstrating how the KNN algorithm works under the hood.
- **Implementation using scikit-learn :** This implementation makes use of the functionality of the sklearn library to come up with a simple and efficient KNN classification.
- **Code:** You can find the code [here](KNN).

**For more detailed information on K-Nearest Neighbors (KNN) see the [Wikipedia article](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).**

<br>

### Naive Bayes

Naive Bayes is a probabilistic machine learning algorithm governed by Bayes' Theorem that widely applies on classification tasks. They assume features to be independent given the class label; thus it runs efficiently and fast, especially for big data.

**Types**
- **Gaussian Naive Bayes:** For Continuous data which is assumed to be normally distributed.
- **Multinomial Naive Bayes:** Used over the discrete count data, like text.
- **Bernoulli Naive Bayes:** when we have binary/boolean features.

**Code:** You can find the code [here](Naive%20Bayes).

**For more detailed information on Naive Bayes see the [Wikipedia article](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).**

<br>

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

**For more detailed information on Decision Tree Classifier see the [Wikipedia article](https://en.wikipedia.org/wiki/Decision_tree_learning).**

<br>

### Linear Regression

Linear Regression is one of the simplest and most widely used machine learning techniques for modeling relationships between a dependent variable and one independent variable feature. This technique primarily aims to find the linear relationship of the variable and therefore to predict how changes in the independent variable affect the dependent variable.

**Equation of Linear Regression:** y = β0 + β1*x + ϵ

Where,

- y is the dependent variable
- x is the independent variable
- β0 is the intercept
- β1 is the slope
- ϵ is the error term

**Code:** You can find the code [here](Linear%20Regression).

**For more detailed information on Linear Regression see the [Wikipedia article](https://en.wikipedia.org/wiki/Linear_regression).**

<br>

### Multiple Regression

Multiple Regression is a statistical technique in the analysis of understanding one dependent variable and two or more independent variables. Multiple regression is an extension of simple linear regression, where it views the impact of one predictor variable. If it has multiple predictors, multiple regressions help understand the various ways the factors contribute to the outcome as well as the strength with which they have relationships with it.

**Equation of Linear Regression:** Y = β0 + β1*X1 + β2*X2 + ... + βn*Xn + ϵ

Where,

- Y is the dependent variable
- X1, X2, ..., Xn are the independent variable
- β0 is the intercept
- β1, β2, ..., βn are the coefficients for X1, X2, ..., Xn
- ϵ is the error term

**Code:** You can find the code [here](Multiple%20Regression).

**For more detailed information on Multiple Regression see the [Wikipedia article](https://en.wikipedia.org/wiki/Linear_regression).**

<br>

### Polynomial Regression

Polynomial regression is a form of regression analysis, where the relationship between the dependent and one or more independent variables is modeled by fitting the data to a polynomial equation. Unlike linear regressions, where the best fit line is a straight line, polynomial regressions could capture curved relationships.

**Equation of Polynommial regression:** Y = a<sub>0</sub> + a<sub>1</sub> * x + a<sub>2</sub> * x<sup>2</sup> + ... + a<sub>n</sub> * x<sup>n</sup>

Where,

- Y is dependent variable
- x is independent variable
- a<sub>0</sub>, a<sub>1</sub>, a<sub>2</sub>, ..., a<sub>n</sub> are coefficients of x, x<sup>2</sup> + ... + a<sub>n</sub> * x<sup>n</sup> respectively
- n is the degree of the polynomial

**Code:** You can find the code [here](Polynomial%20Regression).

**For more detailed information on Polynomial Regression see the [Wikipedia article](https://en.wikipedia.org/wiki/Polynomial_regression).**

<br>

### Logistic Regression

Logistic regression is used for a binary classification where we use a sigmoid function that takes the input as independent variables and gives a probability value between 0 and 1.

For example, we have two classes Class 0 and Class 1 if the value of logistic function for an input is greater than 0.5 (threshold value) then it belongs to Class 1 otherwise it belongs to Class 0. It's referred to as regression because it is an extension of linear regression but is mainly used for classification problems.

**Code:** You can find the code [here](Logistic%20Regression).

**For more detailed information on Logistic Regression see the [Wikipedia article](https://en.wikipedia.org/wiki/Logistic_regression).**

<br>

### Support Vector Machine (SVM)

Support Vector Machine (SVM) is a supervised algorithm that can be used both for classification and regression. Even though it can apply to regression problems, SVM is particularly best for the application in classification. Mainly, the SVM algorithm identifies the most appropriately separable data points by the optimal hyperplane in the N-dimensional space over the feature space. The algorithmic concept ensures the maximization of difference between closest points of any two classes, also called support vectors.

**Code:** You can find the code [here](Support%20Vector%20Machine).

**For more detailed information on Support Vector Machine see the [Wikipedia article](https://en.wikipedia.org/wiki/Support_vector_machine).**

<br>

### K - Means Clustering

An unsupervised learning technique of machine learning called K-Means clustering groups similar data points based on feature similarity. In short, K-Means partitions a dataset into K groups based on their feature similarity. Each cluster has a centroid, which is just the average of points in it. K-Means is efficient and simple in its execution so is best suited for big datasets, though it suffers from weaknesses; notably sensitivity to the choice of initial centroids and the clusters have to be round and almost the same sizes. It's hard to determine what the ideal number of clusters, K, is; one usually applies the Elbow Method or Silhouette Score technique, among others.

**Code:** You can find the code [here](K%20-%20Means%20Clustering).

**For more detailed information on K - Means Clustering see the [Wikipedia article](https://en.wikipedia.org/wiki/K-means_clusterin).**

<br>

### K - Medoids

The K-Medoids algorithm is a clustering technique that is similar to K-Means but with a key difference: instead of using the mean of points in a cluster as the center (centroid), it uses an actual data point as the center, known as a medoid. The medoid is the point that minimizes the total distance to all other points in the same cluster.

**Note:** The K - Medoids algorithm is provided by the ```sklearn_extra``` library, which is an extension of ```scikit-learn```. To run this code, you'll need to install it first. You can install by running the following command. 

```
pip install scikit-learn-extra
```

**Code:** You can find the code [here](K%20-%20Medoids).

**For more detailed information on K - Medoids see the [Wikipedia article](https://en.wikipedia.org/wiki/K-medoids).**

<br>

### Hierarchical Clustering

Hierarchical clustering uses two approaches to create a hierarchy of clusters, Agglomerative, or by merging the clusters, and Divisive, or by splitting the clusters. Agglomerative starts with single points, whereas divisive starts with one big cluster. Dendrogram can be produced so that one can find how many clusters are the optimal. This is frequently applied in fields such as biology and marketing. In this repository, I have implemented agglomerative hierarchical clustering.

**Code:** You can find the code [here](Hierarchical%20Clustering).

**For more detailed information on Hierarchical Clustering see the [Wikipedia article](https://en.wikipedia.org/wiki/Hierarchical_clustering).**

<br>

### DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm. The group of points is based on their densities. It will highlight the groups of densely packed points and mark the points as noise when it has very low density.

**Parameters:**
-   **Epsilon (ε):** Maximum distance to consider points as neighbors.
-   **MinPts:** Number of minimum points needed for the cluster.

**Types of Points:**
-   **Core point:** This is a point that has at least MinPts points within distance n from itself.
-   **Border point:** This is a point that has at least one Core point at a distance n.
-   **Noise point:** This is a point that is neither a Core nor a Border. And it has less than m points within distance n from itself.

**Code:** You can find the code [here](DBSCAN%20Clustering).

**For more detailed information on DBSCAN Clustering see the [Wikipedia article](https://en.wikipedia.org/wiki/DBSCAN).**

<br>

### Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a dimensionality reduction technique commonly used to simplify large datasets while preserving as much variance as possible. It works by transforming the original features into a new set of uncorrelated variables called principal components, which are ordered by the amount of variance they capture from the data. The first few principal components typically capture the majority of the variance, allowing for data visualization, noise reduction, and faster computation in machine learning tasks. PCA is especially useful in high-dimensional datasets where interpreting and processing data can be challenging.

**Code:** You can find the code [here](PCA).

**For more detailed information on PCA see the [Wikipedia article](https://en.wikipedia.org/wiki/Principal_component_analysis).**

<br>

### Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA) is a supervised machine learning classification and dimensionality reduction method. It seeks a linear combination of features that maximize class separation by maximizing distances between class means and minimizing class variances. LDA assumes that data are normally distributed with a common covariance, so it is optimal for certain conditions. These include applications in pattern recognition, medical diagnosis, finance, where predictive accuracy will be enhanced, and lower dimensions help to visualize complex data.

**Code:** You can find the code [here](LDA).

**For more detailed information on LDA see the [Wikipedia article](https://en.wikipedia.org/wiki/Linear_discriminant_analysis).**

## Contributions
Contributions are welcome! If you want to improve existing implementations, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. Please see the LICENSE file for more information.
