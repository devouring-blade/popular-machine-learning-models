import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature= None, threshold= None, left= None, right= None, value= None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class Decision_tree_classification:
    def __init__(self, max_depth= 100, min_samples_split= 2, n_features= None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, x, y):
        self.n_features = x.shape[1] if self.n_features is None else min(x.shape[1], self.n_features)

        def _information_gain(column, y, threshold):
            def _entropy(y):
                hist = np.bincount(y)
                ps = hist / len(y)
                return -np.sum([p * np.log2(p) for p in ps if p > 0])

            # parent entropy
            parent_entropy = _entropy(y)

            # child weighted entropy
            left_idxs = np.argwhere(column < threshold).flatten()
            right_idxs = np.argwhere(column >= threshold).flatten()
            if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
            n_l, n_r, n = len(left_idxs), len(right_idxs), len(y)
            e_l, e_r = _entropy(y[left_idxs]), _entropy(y[right_idxs])
            child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

            # calculating information gain
            IG = parent_entropy - child_entropy
            return IG

        def _grow_tree(x, y, depth= 0):
            n_samples, n_features = x.shape
            n_labels = len(np.unique(y))

            # check the stopping criteria
            if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
                counter = Counter(y)
                most_label = counter.most_common(1)[0][0]
                return Node(value= most_label)

            # find best feature and best threshold
            features = np.random.choice(n_features, self.n_features, replace= False)
            best_feature, best_threshold, best_ig = None, None, -1
            for feature in features:
                column = x[:, feature]
                thresholds = np.unique(column)
                for threshold in thresholds:
                    ig = _information_gain(column, y, threshold)
                    if ig > best_ig:
                        best_ig = ig
                        best_feature = feature
                        best_threshold = threshold

            # create child node
            left_idxs = np.argwhere(x[:, best_feature] < best_threshold).flatten()
            right_idxs = np.argwhere(x[:, best_feature] >= best_threshold).flatten()
            left_node = _grow_tree(x[left_idxs], y[left_idxs], depth+1)
            right_node = _grow_tree(x[right_idxs], y[right_idxs], depth+1)
            return Node(feature= best_feature, threshold= best_threshold, left= left_node, right= right_node)

        self.root = _grow_tree(x, y)

    def predict(self, x):
        def traverse_tree(sample, node: Node):
            if node.is_leaf():
                return node.value

            if sample[node.feature] < node.threshold:
                return traverse_tree(sample, node.left)
            else:
                return traverse_tree(sample, node.right)

        predictions = np.array([traverse_tree(sample, self.root) for sample in x])
        return predictions




""""
ID3 (Categorical) → Entropy - Information Gain or Impurity - Gini Gain (classification)

C4.5 → Gain Ratio .

CART (Continuous) → Entropy - Information Gain or Impurity - Gini Gain (classification) or MSE (regression).
"""






