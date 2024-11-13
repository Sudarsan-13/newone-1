# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train and evaluate each classifier individually
for clf, name in zip([rf_clf, ada_clf, gb_clf], ['Random Forest', 'AdaBoost', 'Gradient Boosting']):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.2f}')

# Ensemble classifiers
ensemble_clf = VotingClassifier(estimators=[('rf', rf_clf), ('ada', ada_clf), ('gb', gb_clf)], voting='soft')

# Train ensemble classifier
ensemble_clf.fit(X_train, y_train)

# Predict using the ensemble classifier
y_pred = ensemble_clf.predict(X_test)

# Calculate accuracy of ensemble classifier
ensemble_accuracy = accuracy_score(y_test, y_pred)
print('Ensemble Accuracy:', ensemble_accuracy)
