# idea
A Decision Tree is a supervised learning model used for classification or regression. Its goal is to predict the value of a target variable based on input features.
- The decision tree splits the data into subsets based on the values of the features.
- Each node in the tree corresponds to a decision (or condition).
- Each branch from a node represents the outcome of that decision.
- The splitting process continues until a stopping condition is met (for example, maximum depth, minimum number of samples, or when all samples in a group have the same label).
- The leaf nodes of the tree contain the predicted value or class label.
  
<img width="623" height="681" alt="{9AB95A10-641E-4FC5-89CF-CBA0C7C13886}" src="https://github.com/user-attachments/assets/05c45237-0730-4de0-a964-057050b73470" />

# problems
The problem is which attribute to choose for splitting, and for that attribute, how to split the data (for numerical features, we use a threshold; for categorical features, we split according to the categories). All of this is based on the entropy formula and information gain.

<img width="1230" height="276" alt="{F5671ED6-A5DD-4F97-8A59-A9D1D1CF9020}" src="https://github.com/user-attachments/assets/5904be97-dbe0-44f1-ad38-f66be9cf15eb" />

