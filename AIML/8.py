import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Define the dataset directly
data = {
    'Pregnancies': [0, 1, 2, 3, 4, 5],
    'Glucose': [6, 85, 8, 1, 0, 50],
    'Blood Pressure': [148, 66, 183, 137, 66, 72],
    'Skin Thickness': [72, 29, 64, 40, 23, 30],
    'Insulin': [35, 0, 0, 35, 94, 15],
    'BMI': [33.6, 26.6, 28.1, 43.1, 23.3, 27.4],
    'Diabetes Pedigree Function': [0.627, 0.351, 0.167, 2.288, 0.672, 1.234],
    'Age': [50, 31, 21, 33, 21, 40],
    'Outcome': [1, 0, 1, 1, 0, 0]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the dataset into features (X) and target (y)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The last column (Outcome)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a list of classification algorithms
algorithms = [
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=3),  # Reduced n_neighbors to 3
    DecisionTreeClassifier(),
    GaussianNB(),
    RandomForestClassifier(),
    AdaBoostClassifier(algorithm='SAMME'),
    GradientBoostingClassifier()
]

# Train and evaluate each algorithm
for algorithm in algorithms:
    model = algorithm.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{algorithm.__class__.__name__} accuracy: {accuracy:.2f}")
