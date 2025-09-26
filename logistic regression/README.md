# estimation
Similar to the linear regression model, we assume that the dependent variable (y) varies linearly with the independent variable (x), 
and we try to create a straight line that best fits the data. However, to address a classification problem where the output values are discrete, 
we pass the model’s output through a sigmoid function to map them to probabilities within the range (0, 1).
<img width="959" height="334" alt="{C02AB53C-B735-4939-BA41-27A4F863692E}" src="https://github.com/user-attachments/assets/d205de0d-af74-4448-83da-4c10c5d6e9bf" />

# calculating loss
To calculate the loss for the model, we don’t use MSE like in linear regression, because the output values represent probabilities indicating the confidence that a data point belongs to certain class. Therefore, in this case, we use the cross-entropy loss formula.
<img width="748" height="105" alt="{827935FB-0DC9-4E6B-97E3-A59B79B675A1}" src="https://github.com/user-attachments/assets/2f1c49ef-2471-44fb-9006-2f651c3e32db" />

Note that this formula is specifically for binary classification. If there are more than two classes, a slightly different formula is used.

# update parameters
We will still compute the derivative of the loss function to find the direction in which the loss decreases, in order to optimize the model.
<img width="902" height="130" alt="{4B597363-3BA7-43E6-9BFB-074B419A3F04}" src="https://github.com/user-attachments/assets/4b36dfc8-bf0b-4845-a488-0476187ca9b6" />

Then, we use gradient descent to update the weights and biases.
<img width="970" height="341" alt="{25A0E669-8814-4862-B660-FCB078DB7447}" src="https://github.com/user-attachments/assets/2994a508-c51a-48ad-b6be-e6c24ff3087a" />

# steps
We still follow the same steps as in linear regression, except that we use a different formula. The conditions and procedure remain the same, because, as mentioned, logistic regression is based on linear regression.


