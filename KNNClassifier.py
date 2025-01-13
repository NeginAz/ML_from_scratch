#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:25:19 2024

@author: negin
"""
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time
import math 
from scipy.spatial import distance


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.classes = None
        self.class_priors = None
        self.X_train = None
        self.y_train = None


    def fit(self, X, y):
        """
        Store training data for nearest neighbor computation.
        """
        self.X_train = X
        self.y_train = y
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_priors = {cls: count / len(y) for cls, count in zip(self.classes, counts)}

    def _knn_pdf(self, x, target_class):
            """
            Estimate the PDF using the k-NN method for a specific target class.
            """
            # Access class-specific data directly
            class_data = self.X_train[self.y_train == target_class]
    
            # Compute distances between x and all points in class_data
            distances = np.linalg.norm(class_data - x, axis=1)
    
            # Get the distance to the k-th nearest neighbor
            k_nearest_distance = np.partition(distances, self.k - 1)[self.k - 1]
    
            # Compute the volume of the d-dimensional hypersphere
            d = x.shape[0]
            volume = (k_nearest_distance ** d) / (d + 1)
            # Return the estimated density
            return self.k / (self.X_train.shape[1] * (volume + 1e-9))  # Add epsilon for numerical stability

    def predict(self, X):
            """
            Predict class labels based on k-NN PDF estimate and prior probabilities.
            """
            predictions = []
            for x in X:
                pdf_estimates = {}
                for cls in self.classes:
                    # Calculate the PDF for the current class
                    pdf_estimates[cls] = self._knn_pdf(x, cls)
    
                # Multiply by class priors to get the posterior probabilities
                posterior_estimates = {cls: pdf_estimates[cls] * self.class_priors[cls] for cls in self.classes}
    
                # Choose the class with the highest posterior probability
                predictions.append(max(posterior_estimates, key=posterior_estimates.get))
            return np.array(predictions)
        


    def evaluate(self, X, y):
        """
        Evaluate classification accuracy.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy


def min_max_normalization(train_data, test_data):
    min_val = np.min(train_data, axis=0)
    max_val = np.max(train_data, axis=0)
    train_data_normalized = (train_data - min_val) / (max_val - min_val)
    test_data_normalized = (test_data - min_val) / (max_val - min_val)
    return train_data_normalized, test_data_normalized


if __name__ == "__main__":
    # Load dataset
    train_data = pd.read_csv('Train_Data.csv', header=None).values
    train_labels = pd.read_csv('Train_Labels.csv', header=None).values.ravel()
    test_data = pd.read_csv('Test_Data.csv', header=None).values
    test_labels = pd.read_csv('Test_Labels.csv', header=None).values.ravel()

    train_data_normalized, test_data_normalized = min_max_normalization(train_data, test_data)    
    # Parzen Classifier
    print("KNN PDF Estimator:")

    # k-NN PDF Estimation
    print("\nk-NN PDF Estimation:")
    for k in [3, 5, 7, 10]:
        print(f"k = {k}")
        knn_pdf = KNNClassifier(k=k)
        knn_pdf.fit(train_data_normalized, train_labels)
        knn_pdf.evaluate(test_data_normalized, test_labels)


