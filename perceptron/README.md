# introduction
the Perceptron is an algorithms for supervised learning of binary classifiers. it can be seem as a single unit of an artificial neural network and it also known as the prototype for neural nets.

it's a simplified model of a biological neuron and simulates the behavior of one cell. 
When an input is fed in, it can be considered as a signal. After going through the computation process, it reaches the activation function, where it decides whether the signal is strong enough to surpass a predefined threshold in order to activate or not.

<img width="631" height="254" alt="{91ED6580-A125-4087-9790-4B44176191A5}" src="https://github.com/user-attachments/assets/9e52e0dc-0cdc-48e5-80f4-5d0581cce5ed" />

# structure
it is a single layer neural network with the unit step function as an activation function.
<img width="1223" height="374" alt="{974A56B4-28D0-4235-8950-EFC399D8923D}" src="https://github.com/user-attachments/assets/0c502922-2f0b-4d7e-8bed-90df27813caa" />

It is similar to logistic regression in that both use a linear function.
<img width="1453" height="188" alt="{217B192A-D579-4D19-B497-3A68AE11FCD8}" src="https://github.com/user-attachments/assets/6e5aebbb-3b1e-490c-b6cb-a1ddcf07306c" />

But its activation function is replaced by a step function.
<img width="1432" height="623" alt="{228CCF66-F31F-444F-BA8E-FDB3B44F9D19}" src="https://github.com/user-attachments/assets/7f156a4c-377c-477c-aa3a-6de525d8d2c4" />


In that case, our prediction result will be: y_pred = g(f(x)) = g(w.x + b).

## update rule
The update rule of the perceptron is not the same as logistic regression because its output is a non-differentiable function, which means we cannot take derivatives and therefore cannot apply gradient descent.
<img width="1415" height="475" alt="{3F82A871-3188-4E24-9EDA-1D9BCF08593A}" src="https://github.com/user-attachments/assets/d5e12213-79b7-4d4a-8d04-d97f11a60072" />


For example, as in rows 0 and 2, if the prediction result and the actual value are the same, we do not need to update the parameters. On the other hand, in row 1, the result is 1, meaning that the prediction is lower than the actual value, so we need to increase the parameters. Similarly, for row 3, the prediction is higher than the actual value, so we need to decrease the parameters.
<img width="1382" height="508" alt="{3896D5C8-C857-4F0F-9633-D477FB9FBE15}" src="https://github.com/user-attachments/assets/1f46d238-4bd4-4547-96fd-6796a7037ee6" />



