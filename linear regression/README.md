# estimation
We assume the dependent variable y varies linearly with the independent variable x. The goal is to find a straight line that fits the data well. 
<img width="961" height="389" alt="img" src="https://github.com/user-attachments/assets/90046e91-af17-45d9-bdb1-52479f4f0d03" />

# calculating loss
To calculate the model's loss, we use the mean squared error (MSE) function to compute the average of the squared differences between the model's predictions and the actual values of the data.
<img width="952" height="379" alt="img_1" src="https://github.com/user-attachments/assets/5d5812b9-a50e-4674-a3c9-f9cfa995c933" />

# update parameters
we take the derivative of the model's loss function to find the direction in which the loss decreases.
<img width="973" height="385" alt="img_2" src="https://github.com/user-attachments/assets/b370b6d0-2d8b-4b24-9d06-3756c37019a5" />


To update the weights, we then use the gradient descent algorithm. You need to choose the learning rate (α) carefully because it determines whether the model will converge, and if it does, how long it will take to do so.
<img width="962" height="368" alt="img_3" src="https://github.com/user-attachments/assets/59ef67e4-2d7d-4170-a481-86901c89ff35" />

# steps
First, when initializing the model, we create the weights and biases as zeros. Of course, you could assign random values or initialize them using a specific method to make the training converge faster, but since these are simple models, we simplify by initializing them to zero.

each time we fit the data through the model, we calculate the loss, which gives us the derivatives. Then, we update the weights and biases using gradient descent.
<img width="948" height="336" alt="{CC9AD473-3EB6-43D0-8EBE-1513E552E13A}" src="https://github.com/user-attachments/assets/601ce203-3adc-49a6-a168-66276dd53bef" />
