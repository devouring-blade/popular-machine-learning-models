# idea
for Support Vector Machine Linear: use a linear model and try to find a decision boundary (hyperplane) that best separates data. the best hyperplane is the one that yields the largest separation/margin between both classes. so we choose the hyperplane so that the distance from it to the nearest datapoint on each size is maximined.
“The nearest data points are the support vectors. All of these data points, when projected onto the hyperplane 
𝑤⋅𝑥 + 𝑏, will equal 1 or -1, while the data points lying outside the margin will be greater than 1 or less than -1.
<img width="1375" height="572" alt="{214C0377-8CD9-4C63-8434-FD80B1F4B8EB}" src="https://github.com/user-attachments/assets/edcaabbc-8500-46ff-9947-fdfccd84b270" />

# loss function and regulization
“We use Hinge Loss because, in a binary classification problem with two labels 1 and -1, as explained above, if the label is -1, the prediction must lie on or beyond the negative margin (𝑤.𝑥 + 𝑏 = −1); conversely, if the label is 1, the prediction must lie on or beyond the positive margin (𝑤.𝑥 + 𝑏 = 1). In this case, we always have 𝑦_true.𝑦_pred ≥ 1. If the model predicts correctly, then 𝑦_true.𝑦_pred ≥ 1, which makes 1 − 𝑦_true.𝑦_pred < 0, and thus the loss is 0. On the other hand, if the model predicts a label that lies within the two margins—or worse, predicts the wrong label—then 𝑦_true.𝑦_pred < 1, which makes 1 −𝑦_true.𝑦_pred > 0. That is how the Hinge Loss function works.
<img width="1381" height="554" alt="{325B2D79-29D0-4200-B132-5A61617CE47A}" src="https://github.com/user-attachments/assets/fbac7ca0-4ced-41b2-a18a-a49adeef643a" />
