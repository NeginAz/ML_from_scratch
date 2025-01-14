
# ML From Scratch: 
This repository contains custom implementations of several essential machine learning algorithms for pattern recognition, dimensionality reduction, and classification tasks.

These implementations are designed to help with understanding the inner workings of popular algorithms by building them from scratch in Python. The repository covers a range of non-parametric density estimators and linear models that are fundamental to many machine learning applications.

- [Parzen Window PDF Estimator](#1-Parzen-Window-PDF-Estimator)
- [k-NN Density Estimator](#2-k-NN-Density-Estimator)
- [Naive Bayes Classifier](#3-Naive-Bayes-Classifier)
- [Naive Bayes Classifier with Risk Implementation](#4-Naive-Bayes-Classifier-with-Risk-Implementation)
- [Principal Component Analysis (PCA)](#5-Principal_Component_Analysis_(PCA))
- [Linear Discriminant Analysis (LDA)](#6-Linear-Discriminant-Analysis-(LDA))


## 1. Parzen Window PDF Estimator

The Parzen window method is a non-parametric technique for estimating the PDF of a dataset by placing a kernel function (either a rectangular or Gaussian window) over each data point. The estimator sums the contributions of these kernels to approximate the overall density function.

The implementation supports two types of kernels:

Rectangular Window (Hypercube): A uniform kernel that assigns equal weight to points within a fixed window size.
Gaussian Window: A smoother kernel that uses the Gaussian function to weigh points based on their distance from the target.
Both estimators consider prior probabilities to classify new data points by maximizing the posterior probability.

Key Features:

- Customizable window size (h)
- Supports both rectangular and Gaussian kernels
- Handles multiclass classification using Bayesian decision rules



## 2. k-NN Density Estimator

The k-Nearest Neighbors (k-NN) PDF estimator is another non-parametric method that estimates the PDF by finding the distance to the k-th nearest neighbor of a point in the dataset. The density is computed as the ratio of the number of points within this distance to the volume of a d-dimensional ball centered on the point.

This estimator is also used for classification by calculating the posterior probabilities for each class and predicting the class with the highest posterior.

Key Features:

- Customizable number of neighbors (k)
- Efficient implementation using distance partitioning
- Supports multiclass classification with prior probabilities


## 3. Naive Bayes Classifier
The Naive Bayes Classifier is a parametric probabilistic model based on Bayes' theorem. It assumes that the features are conditionally independent given the class label, which simplifies the computation of the posterior probabilities. It predicts the class with the highest posterior probability.

Key Features:

- Handles multiclass classification
- Supports both discrete and continuous features
- Incorporates prior probabilities and likelihoods

## 4. Naive Bayes Classifier with Risk Implementation
This variant of the Naive Bayes Classifier incorporates decision risk into the classification process. Instead of simply choosing the class with the highest posterior probability, the classifier minimizes the expected risk by taking into account a risk matrix.

The risk matrix defines the cost of misclassifications, allowing the classifier to make cost-sensitive decisions.

Key Features:

- Incorporates a risk matrix for cost-sensitive classification
- Minimizes expected risk instead of maximizing posterior probability
- Supports multiclass classification


## 5. Principal Component Analysis (PCA)
The Principal Component Analysis (PCA) is a popular dimensionality reduction technique used to reduce the number of features in a dataset while retaining as much variance as possible.

PCA works by:

Centering the data (subtracting the mean from each feature).
Calculating the covariance matrix.
Finding the eigenvectors and eigenvalues of the covariance matrix.
Projecting the data onto the eigenvectors corresponding to the largest eigenvalues.

Key Features:

- Reduces high-dimensional data to a lower-dimensional subspace.
- Retains the most important features by maximizing the explained variance.
- Helps improve model performance and visualization.


## 6. Linear Discriminant Analysis (LDA)
The Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that seeks to maximize the separation between classes by projecting the data onto a lower-dimensional space.

LDA works by:

Computing the mean vectors for each class.
Calculating the within-class scatter matrix and between-class scatter matrix.
Finding the linear discriminants that maximize the ratio of between-class variance to within-class variance.
Projecting the data onto the linear discriminants.
LDA is often used for classification tasks and works best when:

The classes are linearly separable.
The data distribution is Gaussian.
Key Features:

- Reduces data to a lower-dimensional space.
- Maximizes the separation between classes.
- Helps improve classification accuracy.
- 
## How to Run the Models
#### Clone the repository:

```console
git clone https://github.com/yourusername/ML_From_Scratch
cd ML_From_Scratch
```

#### To install all the required dependencies, run:

```console
pip install -r requirements.txt
```

#### Run the models in your Python environment:
```console
python your_script.py
```
Replace your_script.py with your custom script that imports and uses any of the models.



#### Example usage of the Parzen Window PDF Estimator:

```python
from ParzenPDFEstimator import ParzenPDFEstimator
import pandas as pd

# Load training and testing data
train_data = pd.read_csv('Train_Data.csv', header=None).values
train_labels = pd.read_csv('Train_Labels.csv', header=None).values.ravel()
test_data = pd.read_csv('Test_Data.csv', header=None).values
test_labels = pd.read_csv('Test_Labels.csv', header=None).values.ravel()

# Initialize the Parzen Window Estimator
parzen = ParzenPDFEstimator(window_size=0.1, window_type="gaussian")

# Train the model
parzen.fit(train_data, train_labels)

# Evaluate the model
accuracy = parzen.evaluate(test_data, test_labels)
print(f"Parzen Window Estimator Accuracy: {accuracy * 100:.2f}%")
```

#### Example usage of the k-NN Density Estimator:

```python
from KNNClassifier import KNNClassifier
import pandas as pd

# Load training and testing data
train_data = pd.read_csv('Train_Data.csv', header=None).values
train_labels = pd.read_csv('Train_Labels.csv', header=None).values.ravel()
test_data = pd.read_csv('Test_Data.csv', header=None).values
test_labels = pd.read_csv('Test_Labels.csv', header=None).values.ravel()

# Initialize the k-NN PDF Estimator
knn = KNNClassifier(k=5)

# Train the model
knn.fit(train_data, train_labels)

# Evaluate the model
accuracy = knn.evaluate(test_data, test_labels)
print(f"k-NN Density Estimator Accuracy: {accuracy * 100:.2f}%")
```

#### 3. Naive Bayes Classifier:
Example usage of the Naive Bayes Classifier:

```python
from NaiveBayesClassifier import NaiveBayesClassifier
import pandas as pd

# Load training and testing data
train_data = pd.read_csv('Train_Data.csv', header=None).values
train_labels = pd.read_csv('Train_Labels.csv', header=None).values.ravel()
test_data = pd.read_csv('Test_Data.csv', header=None).values
test_labels = pd.read_csv('Test_Labels.csv', header=None).values.ravel()

# Initialize the Naive Bayes Classifier
naive_bayes = NaiveBayesClassifier()

# Train the model
naive_bayes.fit(train_data, train_labels)

# Evaluate the model
accuracy = naive_bayes.evaluate(test_data, test_labels)
print(f"Naive Bayes Classifier Accuracy: {accuracy * 100:.2f}%")
```

#### 4. Naive Bayes Classifier with Risk Implementation
Example usage of the Naive Bayes Classifier with Risk Implementation:

```python
from NaiveBayesRiskClassifier import NaiveBayesRiskClassifier
import pandas as pd

# Load training and testing data
train_data = pd.read_csv('Train_Data.csv', header=None).values
train_labels = pd.read_csv('Train_Labels.csv', header=None).values.ravel()
test_data = pd.read_csv('Test_Data.csv', header=None).values
test_labels = pd.read_csv('Test_Labels.csv', header=None).values.ravel()

# Define a risk matrix
risk_matrix = [
    [0, 1, 2],  # Cost of predicting class 0
    [1, 0, 1],  # Cost of predicting class 1
    [2, 1, 0],  # Cost of predicting class 2
]

# Initialize the Naive Bayes Classifier with Risk
naive_bayes_risk = NaiveBayesRiskClassifier(risk_matrix=risk_matrix)

# Train the model
naive_bayes_risk.fit(train_data, train_labels)

# Evaluate the model
accuracy = naive_bayes_risk.evaluate(test_data, test_labels)
print(f"Naive Bayes with Risk Classifier Accuracy: {accuracy * 100:.2f}%")
```

#### 5. Principal Component Analysis (PCA):
Example usage of the PCA:

```python
from PCA import PCA
import pandas as pd

# Load training data
train_data = pd.read_csv('Train_Data.csv', header=None).values

# Initialize the PCA with 2 components
pca = PCA(n_components=2)

# Fit and transform the data
train_data_pca = pca.fit_transform(train_data)

print("PCA completed.")
```

#### 6. Linear Discriminant Analysis (LDA)
Example usage of the LDA:

```python
from LDA import LDA
import pandas as pd

# Load training and testing data
train_data = pd.read_csv('Train_Data.csv', header=None).values
train_labels = pd.read_csv('Train_Labels.csv', header=None).values.ravel()
test_data = pd.read_csv('Test_Data.csv', header=None).values
test_labels = pd.read_csv('Test_Labels.csv', header=None).values.ravel()

# Initialize the LDA with 2 components
lda = LDA(n_components=2)

# Train the LDA
lda.fit(train_data, train_labels)

# Transform the data
train_data_lda = lda.transform(train_data)
test_data_lda = lda.transform(test_data)

print("LDA completed.")
```
