import numpy as np

class SVM_Classification:
    def __init__(self, lr= 0.001, lambda_param= 0.01, n_iters= 1000):
        self.lr = lr
        self.lambda_param= lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape

        self.weights = np.random.random(n_features)
        self.bias = np.random.random()

        y = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, sample in enumerate(x):
                condition = y[idx] * (sample @ self.weights - self.bias) >= 1
                if condition:
                    self.weights = self.weights - self.lr * (2 * self.lambda_param * self.weights)
                    self.bias = self.bias
                else:
                    self.weights = self.weights - self.lr * (2 * self.lambda_param * self.weights - y[idx] * sample)
                    self.bias = self.bias - self.lr * y[idx]

    def predict(self, x):
        approx = x @ self.weights - self.bias
        return np.sign(approx)

