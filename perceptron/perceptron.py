import numpy as np

def activation_func(x):
    return np.where(x >= 0, 1, 0)

class Perceptron:
    def __init__(self, lr= 0.01, n_iters= 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape

        self.weights = np.random.random(n_features)
        self.bias = np.random.random()

        for _ in range(self.n_iters):
            for features, ground_truth in zip(x, y):
                prediction = activation_func(self.weights.T @ features + self.bias)
                self.weights += self.lr * (ground_truth - prediction) * features
                self.bias += self.lr * (ground_truth - prediction)

    def predict(self, x):
        predictions = activation_func(x @ self.weights + self.bias)
        return predictions


