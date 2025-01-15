import MNIST_Loader as func
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

class FeatureSelector:
    def __init__(self, classifier = None, scoring = None, accuracy_threshold=0.0001):
        self.classifier = classifier if classifier else GaussianNB()
        self.selected_features = []
        self.accuracies = []
        self.accuracy_threshold = accuracy_threshold
    def fit(self, X, y):
        """
        Perform forward feature selection.

        Parameters:
        - X: Input feature matrix (numpy array).
        - y: Target labels (numpy array).

        Returns:
        - self: The fitted selector.
        """
        remaining_features = list(range(X.shape[1]))
        self.selected_features = []
        self.accuracies = []

        # Step 1: Evaluate individual features
        feature_accuracies = {}
        for feature in remaining_features:
            X_selected = X[:, [feature]]
            self.classifier.fit(X_selected, y)
            accuracy = self.classifier.score(X_selected, y)
            feature_accuracies[feature] = accuracy

        # Step 2: Select the best single feature to start with
        best_feature = max(feature_accuracies, key=feature_accuracies.get)
        best_accuracy = feature_accuracies[best_feature]

        self.selected_features.append(best_feature)
        self.accuracies.append(best_accuracy)
        remaining_features.remove(best_feature)

        # Step 3: Iteratively add features
        while remaining_features:
            improvement = 0
            best_new_feature = None

            for feature in remaining_features:
                current_features = self.selected_features + [feature]
                X_selected = X[:, current_features]

                # Train and evaluate the classifier
                self.classifier.fit(X_selected, y)
                accuracy = self.classifier.score(X_selected, y)

                # Check if this feature improves accuracy
                if accuracy - best_accuracy > improvement:
                    improvement = accuracy - best_accuracy
                    best_new_feature = feature

            # Stop if improvement is below the threshold
            if improvement < self.accuracy_threshold:
                break

            # Add the new best feature
            self.selected_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            best_accuracy += improvement
            self.accuracies.append(best_accuracy)

        return self
    def plot_results(self):
        """
        Plot the classification accuracy as a function of the number of selected features.
        """
        if not self.accuracies:
            raise ValueError("No results to plot. Run the fit method first.")

        plt.plot(range(1, len(self.accuracies) + 1), self.accuracies, marker='o')
        plt.title('Forward Selection: Accuracy vs Number of Features')
        plt.xlabel('Number of Selected Features')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()
    def get_selected_features(self):
        """
        Get the indices of the selected features.

        Returns:
        - List of selected feature indices.
        """
        return self.selected_features

    def get_accuracies(self):
        """
        Get the classification accuracies for each step.

        Returns:
        - List of classification accuracies.
        """
        return self.accuracies    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using the selected features on a test dataset.

        Parameters:
        - X_test: Test feature matrix (numpy array).
        - y_test: Test labels (numpy array).

        Returns:
        - accuracy: Classification accuracy on the test dataset.
        """
        if not self.selected_features:
            raise ValueError("No features selected. Run the fit method first.")

        # Use only the selected features
        X_selected = X_test[:, self.selected_features]
        self.classifier.fit(X_selected, y_test)  # Fit the classifier on the test set
        accuracy = self.classifier.score(X_selected, y_test)  # Evaluate accuracy
        return accuracy    
def min_max_normalization(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the feature selector
    selector = FeatureSelector()
    selector.fit(X_train, y_train)
    
    # Print selected features and accuracies
    print("Selected Features (indices):", selector.get_selected_features())
    print("Accuracies:", selector.get_accuracies())

    # Plot the results
    selector.plot_results()
    
    test_accuracy = selector.evaluate(X_test, y_test)
    print(f"Accuracy: {test_accuracy * 100:.2f}%")