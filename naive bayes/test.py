from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from naive_bayes import Naive_Bayes
from sklearn.metrics import classification_report


if __name__ == '__main__':
    breast_cancer = load_breast_cancer(as_frame= True)
    x = breast_cancer.data
    y = breast_cancer.target
    x_train, x_test, y_train, y_test = train_test_split(x , y, test_size= 0.2, shuffle= True, random_state= 0, stratify= y)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = Naive_Bayes()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    print(classification_report(y_true= y_test, y_pred= y_predict))


#                precision    recall  f1-score   support
#
#            0       0.84      0.90      0.87        42
#            1       0.94      0.90      0.92        72
#
#     accuracy                           0.90       114
#    macro avg       0.89      0.90      0.90       114
# weighted avg       0.91      0.90      0.90       114

