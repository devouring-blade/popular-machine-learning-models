from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import  pyplot as plt
from perceptron import Perceptron
import numpy as np
from sklearn.metrics import classification_report


if __name__ == '__main__':
    x, y = datasets.make_blobs(n_samples= 500, n_features= 2, centers= 2,
                               cluster_std= 1, random_state= 0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True,
                                                        test_size= 0.2, random_state= 0,
                                                        stratify= y)

    model = Perceptron()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    print(classification_report(y_true= y_test, y_pred= y_predict))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(x_train[: , 0], x_train[: , 1], marker= "o", c= y_train)

    x0_1 = np.amin(x_train[: , 0])
    x0_2 = np.amax(x_train[: , 0])

    x1_1 = (-model.weights[0] * x0_1 - model.bias) / model.weights[1]
    x1_2 = (-model.weights[0] * x0_2 - model.bias) / model.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    y_min = np.amin(x_train[: , 1])
    y_max = np.amax(x_train[: , 1])
    ax.set_ylim(y_min - 3, y_max + 3)

    plt.show()

#                 precision    recall  f1-score   support
#
#            0       1.00      0.96      0.98        50
#            1       0.96      1.00      0.98        50
#
#     accuracy                           0.98       100
#    macro avg       0.98      0.98      0.98       100
# weighted avg       0.98      0.98      0.98       100
