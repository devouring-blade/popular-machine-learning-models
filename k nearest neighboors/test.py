from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from knn import KNN_Classification, KNN_Regression
from sklearn.metrics import classification_report, r2_score, mean_squared_error, mean_absolute_error


def test_classification():
    iris = load_iris(as_frame=True)
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = KNN_Classification()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    print(classification_report(y_true=y_test, y_pred=y_predict))

def test_regression():
    diabetes = load_diabetes(as_frame=True)
    x = diabetes.data
    y = diabetes.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = KNN_Regression()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    print("MAE: ", mean_absolute_error(y_true= y_test, y_pred=y_predict))
    print("MSE: ", mean_squared_error(y_true= y_test, y_pred=y_predict))
    print("R2:  ", r2_score(y_true= y_test, y_pred=y_predict))

if __name__ == '__main__':
    test_classification()
#       precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00        10
#            1       0.91      1.00      0.95        10
#            2       1.00      0.90      0.95        10
#
#     accuracy                           0.97        30
#    macro avg       0.97      0.97      0.97        30
# weighted avg       0.97      0.97      0.97        30

    test_regression()
# MAE:  51.36704119850187
# MSE:  4489.419475655432
# R2:   0.12451572537935074
