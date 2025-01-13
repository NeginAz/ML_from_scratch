

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

class GaussianNaiveBayes:
    def __init__(self, regularization, rejection_risk=0.8, substitution_risk=1.0):
        self.classes = None
        self.priors = None
        self.means = None
        self.cov_matrices = None
        self.regularization = regularization
        self.rejection_risk = rejection_risk  # Risk for rejecting a sample
        self.substitution_risk = substitution_risk  # Risk for misclassification
        self.use_pooled = False  # Flag for pooled covariance

    def fit(self, X, y):
        self.classes = np.unique(y)  # Unique classes
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        # Initialize arrays 
        self.priors = np.zeros((n_classes, 1))
        self.means = np.zeros((n_classes, n_features))
        self.cov_matrices = np.zeros((n_classes, n_features, n_features))
        pooled_covariance = np.zeros((n_features, n_features))  # Initialize pooled covariance
        total_samples = X.shape[0]
        
        for idx, c in enumerate(self.classes):
            class_data = X[y == c]
            n_class_samples = class_data.shape[0]
            self.priors[idx] = n_class_samples / total_samples
            self.means[idx, :] = np.mean(class_data, axis=0)
            covariance = np.cov(class_data, rowvar=False)
            self.cov_matrices[idx, :, :] = covariance + self.regularization * np.eye(n_features)
            pooled_covariance += (n_class_samples - 1) * covariance
        
        self.pooled_covariance = pooled_covariance / (total_samples - n_classes)
        self.pooled_covariance += self.regularization * np.eye(n_features)

    def _compute_posterior(self, X, class_idx):
        mean = self.means[class_idx, :]
        cov_matrix = self.cov_matrices[class_idx, :, :]
        det_cov = np.linalg.det(cov_matrix)
        
        if np.isinf(det_cov) or np.isnan(det_cov) or det_cov == 0:
            cov_matrix = self.pooled_covariance
            det_cov = np.linalg.det(cov_matrix)
        
        inv_cov = np.linalg.inv(cov_matrix)
        n_features = mean.shape[0]
        log_norm_factor = -0.5 * (np.log(det_cov) + n_features * np.log(2 * np.pi))
        diffs = X - mean
        log_exp_factors = -0.5 * np.sum((diffs @ inv_cov) * diffs, axis=1)
        posterior = np.exp(log_norm_factor + log_exp_factors + np.log(self.priors[class_idx]))
        return posterior

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        risks = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            posterior = self._compute_posterior(X, idx)
            risks[:, idx] = self.substitution_risk * (1 - posterior)

        # Minimum risk for each sample
        min_risks = np.min(risks, axis=1)
        predicted_classes = np.argmin(risks, axis=1)

        # Apply rejection logic
        predictions = np.where(min_risks < self.rejection_risk, predicted_classes, -1)
        return predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)

        # Calculate metrics
        correct_predictions = predictions == y
        rejected = predictions == -1
        accuracy = np.mean(correct_predictions[~rejected]) if np.any(~rejected) else 0
        rejection_rate = np.mean(rejected)

        print(f"True Labels:     {y}")
        print(f"Predicted Labels: {predictions}")
        print(f"Accuracy (without rejected): {accuracy * 100:.2f}%")
        print(f"Rejection Rate: {rejection_rate * 100:.2f}%")
        return accuracy, rejection_rate
    
    
def min_max_normalization(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


if __name__ == "__main__":
    train_data = pd.read_csv('Train_Data.csv', header=None).values
    test_data = pd.read_csv('Test_Data.csv', header=None).values
    train_labels = pd.read_csv('Train_Labels.csv', header=None).values.ravel()
    test_labels = pd.read_csv('Test_Labels.csv', header=None).values.ravel()

    train_data_normalized = min_max_normalization(train_data)
    test_data_normalized = min_max_normalization(test_data)

    # Instantiate the Gaussian Naive Bayes with rejection threshold
    gnb = GaussianNaiveBayes(regularization=1e-2, rejection_risk=0.8)
    gnb.fit(train_data_normalized, train_labels)

    # Evaluate with rejection
    accuracy, rejection_rate = gnb.evaluate(test_data_normalized, test_labels)
    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    print(f"Rejection Rate: {rejection_rate * 100:.2f}%")   
