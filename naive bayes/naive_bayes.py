import numpy as np


class Naive_bayes:
    def __init__(self):
        self.labels = None
        self.means = None
        self.vars = None
        self.priors = None

    def fit(self, x, y):
        n_samples, n_features = x.shape

        self.labels = np.unique(y)
        self.means = np.zeros((len(self.labels), n_features), dtype= np.float64)
        self.vars = np.zeros((len(self.labels), n_features), dtype= np.float64)
        self.priors = np.zeros(len(self.labels), dtype= np.float64)

        for idx, label in enumerate(self.labels):
            x_label = x[y == label]
            self.means[idx, :] = x_label.mean(axis= 0)
            self.vars[idx, :] = x_label.var(axis= 0)
            self.priors[idx] = len(x_label) / len(x)

    def predict(self, x):
        def pdf_gaussian(idx, sample):
            mean = self.means[idx]
            var = self.vars[idx]
            numerator = np.exp(-((sample - mean)**2) / (2 * var))
            denominator = np.sqrt(2 * np.pi * var)
            return numerator / denominator

        def _predict(sample):
            probs = []
            for idx, label in enumerate(self.labels):
                prior = np.log(self.priors[idx])
                posterior = np.sum(np.log(pdf_gaussian(idx, sample)))
                prob = posterior + prior
                probs.append(prob)
            return self.labels[np.argmax(probs)]

        predictions = [_predict(sample) for sample in x]
        return np.array(predictions)



"""
naive bayes sử dụng các xác suất độc lập để tìm ra nhãn có xác suất cao nhất.
tuy nhiên nó có yếu điểm là coi toàn bộ các features là continuous, hoặc categorical
không có sự pha trộn của cả 2, trong đoạn code này nó coi mọi features là continuous
hướng phát triển: thêm khả năng xử lý riêng cho categorical features đồng thời cùng numerical,
cho phép xứ lý dữ liệu tồn tại cả 2 thành phần đặc trưng này
"""













