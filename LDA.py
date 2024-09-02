import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('./Cancer.csv')

# Preprocess the data
X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
y = data['diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply LDA
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

# Print the first 5 rows of LDA-transformed data
print("First 5 rows of LDA-transformed data:")
print(X_lda[:5])

# Plot the LDA results
plt.figure(figsize=(8, 6))
plt.scatter(X_lda[:, 0], [0] * len(X_lda), c=(y == 'M').astype(int), cmap='coolwarm', edgecolor='k', s=40)
plt.xlabel('Linear Discriminant 1')
plt.title('LDA of Cancer Dataset')
plt.colorbar(label='Diagnosis (0 = Benign, 1 = Malignant)')
plt.show()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model on the LDA-transformed data
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("\nLogistic Regression Performance on LDA Data:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
