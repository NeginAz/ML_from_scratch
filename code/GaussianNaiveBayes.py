from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

class GaussianNaiveBayes:
    def __init__(self, regularization):
        self.classes = None
        self.priors = None
        self.means = None
        self.cov_matrices = None
        self.regularization = regularization
        self.use_pooled = False # Flag to indicate if pooled covariance should be used

    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes model.
        
        Parameters:
        X (ndarray): 2D array of training data (samples x features).
        y (ndarray): 1D array of labels.
        """
        self.classes = np.unique(y)  # Unique classes
        n_classes = len(self.classes)
        n_features = X.shape[1]
        #Initialize arrays 
        self.priors = np.zeros((n_classes,1))
        self.means = np.zeros((n_classes, n_features))
        self.cov_matrices = np.zeros((n_classes, n_features, n_features))
        pooled_covariance =  np.zeros((n_features, n_features))  # Initialize pooled covariance
        total_samples = X.shape[0]
        for idx, c in enumerate(self.classes):
            class_data = X[y==c]
            n_class_samples = class_data.shape[0]
            # Calculate prior probability P(C)
            self.priors[idx] = n_class_samples/total_samples
            # Calculate mean vector and covariance matrix
            self.means[idx, :] = np.mean(class_data, axis=0)
            covariance = np.cov(class_data, rowvar = False)
            self.cov_matrices[idx, :, :]=  covariance + self.regularization * np.eye(n_features)
            pooled_covariance += (n_class_samples-1)*covariance
        self.pooled_covariance = pooled_covariance/(total_samples - n_classes) 
        self.pooled_covariance += self.regularization*np.eye(n_features)
    def _compute_log_likelihood(self, X, class_idx):
        mean = self.means[class_idx, :]
        
        cov_matrix = self.cov_matrices[class_idx, :, :] 
        det_cov = np.linalg.det(cov_matrix)
        if np.isinf(det_cov) or np.isnan(det_cov) or det_cov == 0:
            print(f"Switching to pooled covariance for class {self.classes[class_idx]}")
            self.use_pooled = True
            cov_matrix = self.pooled_covariance
            det_cov = np.linalg.det(cov_matrix)
        inv_cov = np.linalg.inv(cov_matrix) 
        print(f"the cov is ... {det_cov}")
        n_features = mean.shape[0]
        log_norm_factor = -0.5 * (np.log(det_cov) + n_features* np.log(2*np.pi))
        diffs = X- mean
        log_exp_factors = -0.5 * np.sum((diffs @ inv_cov) * diffs, axis =1)
        return log_norm_factor + log_exp_factors
        
    def predict(self, X):
        """
        Predict the class for each sample in X.
        
        Parameters:
        X (ndarray): 2D array of data to classify (samples x features).
        
        Returns:
        ndarray: 1D array of predicted labels.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))
        for idx in range(n_classes):
            log_likelihood = self._compute_log_likelihood(X, idx)
            log_probs[:, idx] = np.log(self.priors[idx]) + log_likelihood
        return self.classes[np.argmax(log_probs, axis =1)]


    def evaluate(self, X, y):
         """
         Evaluate the model by comparing predictions with true labels.
         
         Parameters:
         X (ndarray): 2D array of test data (samples x features).
         y (ndarray): 1D array of true labels.
         
         Returns:
         float: Accuracy of the model.
         """
         predictions = self.predict(X)
         accuracy = np.mean(predictions == y)

         
         print("True Labels:     ", y)
         print("Predicted Labels:", predictions)
         print(f"Accuracy: {accuracy * 100:.2f}%")
         return accuracy




def min_max_normalization(data):
    """
    Apply Min-Max Normalization to scale data to the range [0, 1].
    
    Parameters:
    data (ndarray): 2D array of data (samples x features).
    """
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
        

if __name__ == "__main__":
    train_data = pd.read_csv('../data/Train_Data.csv', header=None).values
    test_data = pd.read_csv('../data/Test_Data.csv' ,header=None).values
    train_labels = pd.read_csv('../data/Train_Labels.csv', header=None).values.ravel()
    test_labels = pd.read_csv('../data/Test_Labels.csv' ,header=None).values.ravel()    
    train_data_normalized = min_max_normalization(train_data)
    test_data_normalized = min_max_normalization(test_data)

    gnb = GaussianNaiveBayes(regularization=1e-2)
    gnb.fit(train_data_normalized, train_labels)
    accuracy = gnb.evaluate(test_data_normalized, test_labels)
    print("this is the " , accuracy)
    ## Sklearn 
    gnb_sklearn = GaussianNB()
    gnb_sklearn.fit(train_data, train_labels)
    
    # Make predictions
    predictions_sklearn = gnb_sklearn.predict(test_data)
    
    # Evaluate the model
    accuracy_sklearn = accuracy_score(test_labels, predictions_sklearn)
    
    print("True Labels:     ", test_labels)
    print("Predicted Labels:", predictions_sklearn)
    print(f"Accuracy with Scikit-learn: {accuracy_sklearn * 100:.2f}%")
    
    