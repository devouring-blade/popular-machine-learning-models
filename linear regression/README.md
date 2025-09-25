estimation
We assume the dependent variable y varies linearly with the independent variable x. The goal is to find a straight line that fits the data well. imgpng

calculating loss
To calculate the model's loss, we use the mean squared error (MSE) function to compute the average of the squared differences between the model's predictions and the actual values of the data. img1png

update parameters
we take the derivative of the model's loss function to find the direction in which the loss decreases. img2png

To update the weights, we then use the gradient descent algorithm. You need to choose the learning rate (α) carefully because it determines whether the model will converge, and if it does, how long it will take to do so. img3png
