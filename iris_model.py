import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Dataset
# The Iris dataset is so famous, it comes built-in with scikit-learn!
print("Loading the Iris dataset...")
iris = load_iris()

# Convert it to a Pandas DataFrame so it's easy to work with
X = pd.DataFrame(iris.data, columns=iris.feature_names) # The features (sepal/petal length and width)
y = iris.target # The target variable (the species of the flower)

# 2. Split the Data
# Split into training data (80%) and testing data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build and Train the Model
# We are using K-Nearest Neighbors (KNN), a classic algorithm for this specific dataset
print("Training the K-Nearest Neighbors model...")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. Make Predictions and Evaluate
print("Evaluating model...")
predictions = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\n--- Results ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# To make the report readable, we map the target numbers (0, 1, 2) back to flower names
print("Detailed Classification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))