from logistic_regression import Logistic_regression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data = load_breast_cancer(as_frame= True)
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.1, shuffle= True, random_state= 0, stratify= y)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = Logistic_regression()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    print(classification_report(y_true= y_test, y_pred= y_predict))

#                precision    recall  f1-score   support
#
#            0       0.95      0.90      0.93        21
#            1       0.95      0.97      0.96        36
#
#     accuracy                           0.95        57
#    macro avg       0.95      0.94      0.94        57
# weighted avg       0.95      0.95      0.95        57
