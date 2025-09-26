from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error
from linear_regression import Linear_regression



if __name__ == '__main__':
    fetch_housing = fetch_california_housing(as_frame= True)
    x, y = fetch_housing.data, fetch_housing.target
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

# MAE:  0.5348531880446492
# MSE:  0.5297564076140712
# R2:   0.5937310357331766

