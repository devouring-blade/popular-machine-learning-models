from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from decision_tree import Decision_tree
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


data = datasets.load_breast_cancer()
x, y = data.data, data.target

x_train, x_test, y_train, ground_truth = train_test_split(x, y, test_size= 0.2, random_state= 42)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = DecisionTreeClassifier(criterion= "entropy")
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(classification_report(ground_truth, predictions))

# sklearn
"""             precision    recall  f1-score   support

           0       0.91      0.91      0.91        43
           1       0.94      0.94      0.94        71

    accuracy                           0.93       114
   macro avg       0.93      0.93      0.93       114
weighted avg       0.93      0.93      0.93       114"""

# built use entropy
"""              precision    recall  f1-score   support

           0       1.00      0.91      0.95        43
           1       0.95      1.00      0.97        71

    accuracy                           0.96       114
   macro avg       0.97      0.95      0.96       114
weighted avg       0.97      0.96      0.96       114"""


