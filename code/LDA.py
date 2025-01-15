import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import MNIST_Loader as func
from sklearn.preprocessing import StandardScaler

class LDA: 
    def __init__(self, num_components= None):
        self.num_components = num_components
        self.eigenvectors = None
        self.eigenvalues = None
    def fit(self, X, y):
        overall_mean = np.mean(X, axis = 0)
        n_features = X.shape[1]
        classes = np.unique(y)
        
        #within-class and between-class scatter matrix 
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        for cls in classes:
            X_cls = X[y == cls]
            cls_mean = np.mean(X_cls, axis = 0)
            S_W += (X_cls - cls_mean).T@ (X_cls - cls_mean)
            n_cls = X_cls.shape[0]
            mean_diff = (cls_mean - overall_mean).reshape(-1,1)
            S_B += n_cls * mean_diff @ mean_diff.T
            
        # separability matrix and eigen decomposition
        A = np.linalg.inv(S_W)@ S_B
        eigvals, eigvecs = np.linalg.eigh(A)
        
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        
        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs[:, :self.num_components] if self.num_components else eigvecs

    def transform(self, X):
        return X @ self.eigenvectors
    
    def plot_eigenvalues(self):

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.eigenvalues, marker='o')
        plt.title("Eigenvalues in Descending Order")
        plt.xlabel("Component Index")
        plt.ylabel("Eigenvalue")
        plt.show()
        
    def plot_separability_measure(self):

        cumulative_separability = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues)
        plt.figure()
        plt.plot(cumulative_separability, marker='o')
        plt.title("Separability Measure vs Number of Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Separability Measure")
        plt.show()         
        
if __name__ == "__main__":
    # Load dataset
    images, labels = func.load_mnist("../data/Fashion-MNIST", kind='train')

    # Reshape and Normalize Data
    X = images.reshape(images.shape[0], -1)  # Flatten the 28x28 images into vectors of length 784
    y = labels

    # Standardize features to have zero mean and unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets

    
    lda = LDA(num_components=25)
    lda.fit(X, y)  # X: input data, y: labels
    X_lda = lda.transform(X)  # Project the data into the new subspace
    lda.plot_eigenvalues()
    lda.plot_separability_measure()
    
    X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)
    # Train Naive Bayes classifier
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Compute CCR
    ccr_lda = accuracy_score(y_test, y_pred)
    print("CCR with LDA:", ccr_lda)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ccr_no_lda = accuracy_score(y_test, y_pred)
    print("CCR without LDA:", ccr_no_lda)
    
    print("Improvement in CCR due to LDA:", ccr_lda - ccr_no_lda)

        