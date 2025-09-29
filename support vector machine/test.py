import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from svm import SVM_Classification
from sklearn.metrics import classification_report


# --- Load dataset ---
data = load_breast_cancer()
X = data.data
y = data.target
y = 2*y - 1  # {-1,1}

# --- train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Fit Linear SVM ---
clf = SVM_Classification()
clf.fit(X_train_scaled, y_train)

# ---  ---
test_preds = clf.predict(X_test_scaled)
print(classification_report(y_true= y_test, y_pred= test_preds))


#                precision    recall  f1-score   support
# 
#           -1       0.98      0.94      0.96        64
#            1       0.96      0.99      0.98       107
# 
#     accuracy                           0.97       171
#    macro avg       0.97      0.96      0.97       171
# weighted avg       0.97      0.97      0.97       171


