Density Estimators Project
This repository contains implementations of two non-parametric density estimators commonly used in pattern recognition and classification tasks:

Parzen Window PDF Estimator (with Rectangular and Gaussian Kernels)
k-NN Density Estimator
These models are designed to estimate the probability density function (PDF) of a given dataset and use these estimates for classification tasks by applying Bayesian decision theory.


1. Parzen Window PDF Estimator

The Parzen window method is a non-parametric technique for estimating the PDF of a dataset by placing a kernel function (either a rectangular or Gaussian window) over each data point. The estimator sums the contributions of these kernels to approximate the overall density function.

The implementation supports two types of kernels:

Rectangular Window (Hypercube): A uniform kernel that assigns equal weight to points within a fixed window size.
Gaussian Window: A smoother kernel that uses the Gaussian function to weigh points based on their distance from the target.
Both estimators consider prior probabilities to classify new data points by maximizing the posterior probability.

Key Features:

- Customizable window size (h)
- Supports both rectangular and Gaussian kernels
- Handles multiclass classification using Bayesian decision rules



2. k-NN Density Estimator

The k-Nearest Neighbors (k-NN) PDF estimator is another non-parametric method that estimates the PDF by finding the distance to the k-th nearest neighbor of a point in the dataset. The density is computed as the ratio of the number of points within this distance to the volume of a d-dimensional ball centered on the point.

This estimator is also used for classification by calculating the posterior probabilities for each class and predicting the class with the highest posterior.

Key Features:

- Customizable number of neighbors (k)
- Efficient implementation using distance partitioning
- Supports multiclass classification with prior probabilities


3. Naive Bayes Classifier
The Naive Bayes Classifier is a parametric probabilistic model based on Bayes' theorem. It assumes that the features are conditionally independent given the class label, which simplifies the computation of the posterior probabilities.

The classifier works by calculating:

P
(
class
∣
features
)
∝
P
(
features
∣
class
)
×
P
(
class
)
P(class∣features)∝P(features∣class)×P(class)
It predicts the class with the highest posterior probability.

Key Features:

Handles multiclass classification
Supports both discrete and continuous features
Incorporates prior probabilities and likelihoods

4. Naive Bayes Classifier with Risk Implementation
This variant of the Naive Bayes Classifier incorporates decision risk into the classification process. Instead of simply choosing the class with the highest posterior probability, the classifier minimizes the expected risk by taking into account a risk matrix.

The risk matrix defines the cost of misclassifications, allowing the classifier to make cost-sensitive decisions.

Risk-Based Decision Rule:

Decision
=
arg
⁡
min
⁡
i
∑
j
R
(
i
,
j
)
P
(
C
j
∣
features
)
Decision=arg 
i
min
​	
  
j
∑
​	
 R(i,j)P(C 
j
​	
 ∣features)
Where:

R
(
i
,
j
)
R(i,j) is the risk of classifying a sample as class 
i
i when it actually belongs to class 
j
j.
P
(
C
j
∣
features
)
P(C 
j
​	
 ∣features) is the posterior probability for class 
j
j.
Key Features:

Incorporates a risk matrix for cost-sensitive classification
Minimizes expected risk instead of maximizing posterior probability
Supports multiclass classification
