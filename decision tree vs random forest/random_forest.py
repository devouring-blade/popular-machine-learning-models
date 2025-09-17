from decision_tree import Decision_tree_classification
import numpy as np
from collections import Counter


class Random_forest_classification:
    def __init__(self, n_trees= 10, max_depth= 10, min_samples_split= 2, n_features =None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []


    def fit(self, x, y):
        def _bootstrap_sample(x, y):
            n_samples = x.shape[0]
            idxs = np.random.choice(n_samples, n_samples, replace= True)
            return x[idxs], y[idxs]

        for _ in range(self.n_trees):
            tree = Decision_tree_classification(max_depth= self.max_depth,
                                         min_samples_split= self.min_samples_split,
                                         n_features= self.n_features)
            x_samples, y_samples = _bootstrap_sample(x, y)
            tree.fit(x_samples, y_samples)
            self.trees.append(tree)

    def predict(self, x):
        def _most_label(y):
            counter = Counter(y)
            most_label = counter.most_common(1)[0][0]
            return most_label

        tree_predicts = np.array([tree.predict(x) for tree in self.trees]).T
        predictions = [_most_label(tree_predict) for tree_predict in tree_predicts]
        return predictions


