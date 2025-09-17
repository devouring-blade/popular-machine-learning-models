from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from knn import knn
from sklearn.metrics import classification_report

data = datasets.load_iris()
x, y = data.data, data.target

x_train, x_test, y_train, ground_truth = train_test_split(x, y, test_size= 0.2, random_state= 42)


model = knn()
model.fit(x_train, y_train)
predictions = model.predict(x_test)


print(classification_report(ground_truth, predictions))


plt.scatter(x[: , 2], x[: , 3], c= y, s= 20)
plt.show()



