from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_regression import linear_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression




x, y = datasets.make_regression(n_samples= 100, n_features= 1, noise= 10, random_state= 42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)


model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)

print(mse)
print(mae)

line = model.predict(x)

plt.scatter(x_train, y_train, color= "black", s= 15)
plt.scatter(x_test, y_test, color= "red", s= 15)
plt.plot(x, line)
plt.show()

"""104.2022265318772
8.416659922209387
"""