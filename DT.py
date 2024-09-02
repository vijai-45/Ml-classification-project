import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your Cancer dataset
data = pd.read_csv('./Cancer.csv')

# Handle missing values in numeric columns by replacing zeros with NaN and filling with the mean
columns_to_replace = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']  # Update as necessary
data[columns_to_replace] = data[columns_to_replace].replace(0, pd.NA)

# Fill missing values only for numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Features and target variable
X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)  # Exclude non-feature columns
y = data['diagnosis']  # Target (Outcome) column

# Convert categorical target variable to numeric if necessary
y = y.map({'M': 1, 'B': 0})  # Map Malignant to 1 and Benign to 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualization of Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Benign', 'Malignant'])
plt.title("Decision Tree")
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
