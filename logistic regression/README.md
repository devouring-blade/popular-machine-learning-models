# estimation
Similar to the linear regression model, we assume that the dependent variable (y) varies linearly with the independent variable (x), 
and we try to create a straight line that best fits the data. However, to address a classification problem where the output values are discrete, 
we pass the model’s output through a sigmoid function to map them to probabilities within the range (0, 1).
<img width="959" height="334" alt="{C02AB53C-B735-4939-BA41-27A4F863692E}" src="https://github.com/user-attachments/assets/d205de0d-af74-4448-83da-4c10c5d6e9bf" />

# calculating loss
To calculate the loss for the model, we don’t use MSE like in linear regression, because the output values represent probabilities indicating the confidence that a data point belongs to certain class. Therefore, in this case, we use the cross-entropy loss formula.
At the same time, we also take the derivative of this loss function to determine the direction in which we should adjust the parameters to reduce the loss.
<img width="959" height="278" alt="{5F160029-1902-4ABE-A0EA-29164601125D}" src="https://github.com/user-attachments/assets/6481a009-7d98-4e48-af36-ef9654c45867" />



