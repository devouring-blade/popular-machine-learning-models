import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logistic_regression import logistic_regression
from sklearn.metrics import r2_score, classification_report
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    # data = pd.read_csv("diabetes.csv")
    # x = data.drop(columns= "Outcome")
    # y = data["Outcome"]
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)


    data = datasets.load_breast_cancer()
    x, y = data.data, data.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    print(classification_report(y_test, y_predict))
    print(r2_score(y_test, y_predict))

    for i, j in zip(y_test, y_predict):
        print(f"ground true: {i} --- prediction: {j}")



"""              precision    recall  f1-score   support

           0       1.00      0.95      0.98        43
           1       0.97      1.00      0.99        71

    accuracy                           0.98       114
   macro avg       0.99      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114"""




