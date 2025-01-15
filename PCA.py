import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import MNIST_Loader as func
from sklearn.preprocessing import StandardScaler


class PCA: 
    def __init__(self, n_components=None, classifier=None):
        self.n_components = n_components
        self.classifier = classifier if classifier is not None else GaussianNB()
        self.eigenvalues = None
        self.eigenvectors = None
        
    def fit_pca(self, X):
        # Centering the data
        X_meaned = X - np.mean(X, axis=0)

        # Covariance matrix
        covariance_matrix = np.cov(X_meaned, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]

    def transform(self, X):
        # Centering the data
        X_meaned = X - np.mean(X, axis=0)
        if self.n_components is not None:
            selected_vectors = self.eigenvectors[:, :self.n_components]
        else:
            selected_vectors = self.eigenvectors
            
        # Project the data onto principal components
        return np.dot(X_meaned, selected_vectors)
    
    def plot_eigenvalues(self):
        if self.eigenvalues is None:
            raise ValueError("PCA must be fitted before plotting eigenvalues.")

        plt.figure(figsize=(8, 5))
        plt.plot(self.eigenvalues, marker='o')
        plt.title("Eigenvalues of the Covariance Matrix")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.grid()
        plt.show()

    def fit_classifier(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)    
    
    
# Main function
if __name__ == "__main__":
    # Load dataset
    images, labels = func.load_mnist("Fashion-MNIST", kind='train')

    # Reshape and Normalize Data
    X = images.reshape(images.shape[0], -1)  # Flatten the 28x28 images into vectors of length 784
    y = labels

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # PCA Implementation
    num_components = 20
    pca_with_clf = PCA(n_components=num_components)

    # Fit PCA on training data
    pca_with_clf.fit_pca(X_train)

    # Plot Eigenvalues
    pca_with_clf.plot_eigenvalues()

    # Transform data
    X_train_reduced = pca_with_clf.transform(X_train)
    X_test_reduced = pca_with_clf.transform(X_test)

    # Train Classifier on reduced data
    pca_with_clf.fit_classifier(X_train_reduced, y_train)
    y_pred_pca = pca_with_clf.predict(X_test_reduced)

    # Report CCR (Correct Classification Rate) for PCA-transformed data
    ccr_pca = accuracy_score(y_test, y_pred_pca)
    print(f"CCR with PCA: {ccr_pca:.4f}")

    # Train Classifier on original data
    pca_with_clf.fit_classifier(X_train, y_train)
    y_pred_original = pca_with_clf.predict(X_test)

    # Report CCR for original data
    ccr_original = accuracy_score(y_test, y_pred_original)
    print(f"CCR without PCA: {ccr_original:.4f}")

    # Compare CCRs
    print(f"Difference in CCR: {ccr_original - ccr_pca:.4f}")    