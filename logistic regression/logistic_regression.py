import numpy as np

def sigmoid(linear_predict):
    return 1 / (1 + np.exp(-linear_predict))


class Logistic_regression:

    def __init__(self, lr= 0.001, n_iters= 1000, threshold= 0.5):
        self.lr = lr
        self.n_iters = n_iters
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predict = sigmoid(np.dot(X, self.weights) + self.bias)

            dw = (1/n_samples) * 2 * np.dot(X.T, (y_predict - y))
            db = (1/n_samples) * 2 * np.sum(y_predict - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_predict = sigmoid(np.dot(X, self.weights) + self.bias)
        y_predict = np.array([0 if y < self.threshold else 1 for y in y_predict])
        return y_predict