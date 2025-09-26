import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error

class Linear_regression:
    def __init__(self, lr= 0.01, n_iters= 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(shape= n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predict = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * 2 * np.dot(X.T, (y_predict - y))
            db = (1/n_samples) * 2 * np.sum(y_predict - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_predict = np.dot(X, self.weights) + self.bias
        return y_predict

if __name__ == '__main__':
    data = fetch_california_housing(as_frame= True)
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0, shuffle= True)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = Linear_regression(n_iters= 2000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print("MAE: ", mean_absolute_error(y_true=y_test, y_pred= predictions))
    print("MSE: ", mean_squared_error(y_true=y_test, y_pred=predictions))
    print("R2:  ", r2_score(y_true=y_test, y_pred=predictions))

# 0.5385695598072368
# 0.5390952417922621
# 0.5865691053921712




