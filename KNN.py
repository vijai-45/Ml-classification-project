import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('./Cancer.csv')

# Preprocess the data: remove unnecessary columns ('id' and 'Unnamed: 32')
X = data.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)

# Encode 'diagnosis' column (M = 1, B = 0)
y = data['diagnosis'].replace({'M': 1, 'B': 0}).astype(int)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a KNN classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = knn_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example new input (a new tumor with certain features) - ensure it has the same number of features
new_input = np.array([[15.0, 20.0, 100.0, 800.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 
                       0.2, 0.4, 5.0, 50.0, 0.05, 0.1, 0.15, 0.1, 0.2, 0.15, 
                       16.0, 25.0, 120.0, 900.0, 0.1, 0.25, 0.35, 0.15, 0.25, 0.3]])  # Replace with real values

# Standardize the new input just like the training data
new_input_scaled = scaler.transform(new_input)

# Predict the class of the new input
predicted_class = knn_clf.predict(new_input_scaled)

# Output the predicted class (0 = Benign, 1 = Malignant)
print(f"Predicted class: {'Malignant' if predicted_class[0] == 1 else 'Benign'}")
