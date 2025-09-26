import numpy as np
from collections import Counter


class KNN_Classification:
    def __init__(self, k= 3):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        def euclidean_distance(x, y):
            distance = np.sqrt(np.sum((x - y) ** 2))
            return distance

        y_predict = []
        for i in x_test:
            # computer the distance
            distances = [euclidean_distance(i, j) for j in self.x_train]

            # get the closest k
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = [self.y_train[j] for j in k_indices]

            # majority vote
            counter = Counter(k_nearest_labels)
            most_label = counter.most_common(1)[0][0]
            y_predict.append(most_label)
        return y_predict


class KNN_Regression:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        def euclidean_distance(x, y):
            distance = np.sqrt(np.sum((x - y) ** 2))
            return distance

        y_predict = []
        for i in x_test:
            # computer the distance
            distances = [euclidean_distance(i, j) for j in self.x_train]

            # get the closest k
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_values = [self.y_train[j] for j in k_indices]

            # mean target k nearest
            mean_k_nearest = np.mean(k_nearest_values)
            y_predict.append(mean_k_nearest)
        return y_predict




