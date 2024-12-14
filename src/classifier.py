import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import cv2
from sklearn.ensemble import RandomForestClassifier

from abc import ABC, abstractmethod

from dataset import TileDataset
from test_dataset import TestDataset

class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class KNNClassifier(Classifier):
    def __init__(self, k=1, distance_metric='euclidean'):
        """
        k: Number of nearest neighbors to consider.
        distance_metric: Metric to use for distance calculation ('euclidean' or 'manhattan').
        """
        self.k = k
        self.distance_metric = distance_metric
        self.model = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class SVMClassifier(Classifier):
    """
    SVM takes to long on the dataset
    """
    def __init__(self, kernel='rbf'):
        from sklearn.svm import SVC
        self.svm = SVC(kernel=kernel)

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict(self, X):
        return self.svm.predict(X)

def test_classifier(model : Classifier, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def get_train_data():
    # load in the training dataset
    x_train = []
    y_train = []
    tile_data = TileDataset("../dataset")
    for i in range(len(tile_data)):
        img, label = tile_data[i] # image is (3, 8, 8)
        x_train.append(img.flatten())
        y_train.append(label)
    return x_train, y_train

def get_test_data():
    test_data = TestDataset()
    x_test = []
    y_test = []
    for i in range(len(test_data)):
        img, label = test_data[i] # image is (16, 16, 3)
        img =  cv2.resize(img.numpy(), (8, 8), interpolation=cv2.INTER_LINEAR)
        img = np.moveaxis(img, -1, 0)  # (3, 8, 8)
        x_test.append(img.flatten())
        y_test.append(label.item())
    return x_test, y_test

# Example Usage
if __name__ == "__main__":
    x_train, y_train = get_train_data()
    x_test, y_test = get_test_data()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)  # Keep 50 principal components
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    model = KNNClassifier(4, 'cosine')
    # model = SVMClassifier()
    accuracy = test_classifier(model, x_train, y_train, x_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")