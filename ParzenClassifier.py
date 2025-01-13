#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:31:56 2024

@author: negin
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class ParzenPDFEstimator:
    def __init__(self, window_size, window_type="gaussian"):
        self.window_size = window_size
        self.window_type = window_type
        self.classes = None
        self.labels = None
        self.data = None

    def fit(self, X, y):
        """
        Store data by class for density estimation.
        """
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_priors = {cls: count / len(y) for cls, count in zip(self.classes, counts)}
        self.data = X
        self.labels = y
        print("Training data and labels stored.")
    # def _parzen_window(self, x, target_class):
    #     """
    #     Estimate the PDF using Parzen window.
    #     """
    #     n_samples = self.data.shape[0]
    #     class_data = self.data[self.labels == target_class]
    #     _ , n_features = class_data.shape
    #     volume  = self.window_size**n_features
        
    #     mean = np.mean(class_data, axis=0)
    #     covariance = np.cov(class_data, rowvar=False)
    #     # Handle singular covariance matrix
    #     # Add a regularization term to ensure positive definiteness
    #     covariance += np.eye(n_features) * 1e-6
        
    #     if self.window_type == "gaussian":
    #         squared_distances = np.sum((class_data - x) ** 2, axis=1)
    #         pdf_values = np.exp(-squared_distances / (2 * self.window_size ** 2))
    #         return np.sum(pdf_values) / (n_samples * volume)
    
    #     elif self.window_type == "rectangular":
    #         inside_window = np.all(np.abs((x - class_data) / self.window_size) <= 0.5, axis=1)
    #         #print("inside window", inside_window)
    #         count_inside = np.sum(inside_window)
    #         #print("class" ,  target_class , "is" , count_inside)
    #         #print(n_samples)
    #         return count_inside/(n_samples*volume)
    #     else:
    #         raise ValueError("Unsupported window type. Use 'gaussian' or 'rectangular'.")
   
    def _parzen_window(self, x, target_class):
        """
        Estimate the PDF using Gaussian or rectangular Parzen window.
        """
        n_samples = self.data.shape[0]
        class_data = self.data[self.labels == target_class]
        _, n_features = class_data.shape
        volume = self.window_size ** n_features  # Keep the volume calculation
    
        if self.window_type == "gaussian":
            # Calculate squared distances between x and all points in class_data
            squared_distances = np.sum((class_data - x) ** 2, axis=1)
    
            # Apply the Gaussian kernel function as per the formula
            pdf_values = (1 / ((2 * np.pi * self.window_size ** 2) ** (n_features / 2))) * np.exp(-squared_distances / (2 * self.window_size ** 2))
    
            # Return the mean PDF value normalized by sample size and volume
            return np.sum(pdf_values) / (n_samples * volume)
    
        elif self.window_type == "rectangular":
            inside_window = np.all(np.abs((x - class_data) / self.window_size) <= 0.5, axis=1)
            count_inside = np.sum(inside_window)
            return count_inside / (n_samples * volume)
    
        else:
            raise ValueError("Unsupported window type. Use 'gaussian' or 'rectangular'.")

   
    
   
    def predict(self, X):
        """
        Predict class labels for X based on maximum Parzen PDF estimate.
        """
        predictions = []
        for x in X:
            pdf_estimates = {cls: self._parzen_window(x, cls) for cls in self.classes}
            posterior_estimates = {cls: pdf_estimates[cls] * self.class_priors[cls] for cls in self.classes}
            predicted_class = max(posterior_estimates, key=posterior_estimates.get)
            #print("Guess is", predicted_class)
            predictions.append(predicted_class)
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

    train_data_normalized , test_data_normalized = min_max_normalization(train_data, test_data)

    # Test Parzen Window Estimator with Different Configurations
    window_sizes = [ 0.1 , 0.3, 0.5,  0.6 , 0.7]
    #window_types = ["gaussian", "rectangular"]
    window_types = ["gaussian"]
    for window_type in window_types:
        print(f"\nTesting Parzen Window Estimation with {window_type.capitalize()} Window:")
        for window_size in window_sizes:
            print(f"  Window Size: {window_size}")
            
            # Initialize the Parzen Window Estimators
            parzen_estimator = ParzenPDFEstimator(window_size=window_size, window_type=window_type)
            
            # Train the model
            parzen_estimator.fit(train_data_normalized, train_labels)
            
            # Evaluate the model
            accuracy = parzen_estimator.evaluate(test_data_normalized, test_labels)
            print(f"    Accuracy: {accuracy * 100:.2f}%")

    print("Testing Completed.")

