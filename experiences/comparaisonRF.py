import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
sys.path.append('.')
from scripts.RF import *


# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict using scikit-learn's RandomForestClassifier
sklearn_rf = RandomForestClassifier(n_estimators=25, random_state=42)
sklearn_rf.fit(X_train, y_train)
sklearn_predictions = sklearn_rf.predict(X_test)

# Train and predict using your custom Random Forest
custom_rf = RF(n_estimators=25)  
custom_rf.fit(X_train, y_train)  
custom_predictions = custom_rf.predict(X_test)  

# Compare the predictions
accuracy_sklearn = accuracy_score(y_test, sklearn_predictions)
accuracy_custom = accuracy_score(y_test, custom_predictions)


# Print the accuracies
print(f"Accuracy of scikit-learn's RandomForestClassifier: {accuracy_sklearn:.4f}")
print(f"Accuracy of your custom Random Forest: {accuracy_custom:.4f}")
