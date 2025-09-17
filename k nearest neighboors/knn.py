import numpy as np
from collections import Counter


class Knn_classification:
    def __init__(self, k= 3):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        y_predict = []
        for i in x_test:
            # computer the distance
            distances = [self.euclidean_distance(i, j) for j in self.x_train]

            # get the closest k
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = [self.y_train[j] for j in k_indices]

            # majority vote
            counter = Counter(k_nearest_labels)
            most_label = counter.most_common(1)[0][0]
            y_predict.append(most_label)
        return y_predict

    def euclidean_distance(self, x, y):
        distance = np.sqrt(np.sum((x-y)**2))
        return distance



