# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Creating a simple DataFrame
data = {
    'Feature1': [2, 3, 10, 6, 8, 11, 5, 9],
    'Feature2': [3, 6, 7, 8, 3, 9, 4, 10],
    'Label': [0, 0, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Splitting features and labels
X = df[['Feature1', 'Feature2']]  # Features
y = df['Label']  # Labels

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Training Features:\n", X_train)
print("Training Labels:\n", y_train)
print("Testing Features:\n", X_test)
print("Testing Labels:\n", y_test)

# Initialize the Support Vector Classifier (SVC)
svm = SVC(kernel='linear')

# Train the model
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the SVM model: {accuracy * 100:.2f}%")

# Display actual vs predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

# Testing with new data

# Step 1: Creating a new data frame
data = {'Feature1': [2, 13, 21], 'Feature2': [3, 16, 27]}

# Create DataFrame
new_df = pd.DataFrame(data)

# Print the new DataFrame
print("New Data for Prediction:\n", new_df)

# Step 2: Predict with our model
y_pred_new = svm.predict(new_df)
print("Predictions for New Data:\n", y_pred_new)
