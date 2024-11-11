# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('emails.csv')
df.dropna(inplace=True)  # Handle missing values by removing rows with NaN

# Drop unnecessary columns
df.drop(['Email No.'], axis=1, inplace=True)

# Split the dataset into features and target variable
X = df.drop(['Prediction'], axis=1)
y = df['Prediction']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Determine the optimal number of neighbors (k) for KNN
k_values = range(1, 30)
cross_val_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cross_val_scores.append(scores.mean())

# Plot cross-validation accuracy to find the best k
plt.plot(k_values, cross_val_scores)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Optimal Number of Neighbors for KNN')
plt.show()

# Choose the best k
optimal_k = k_values[cross_val_scores.index(max(cross_val_scores))]
print(f"The optimal number of neighbors is {optimal_k}")

# K-Nearest Neighbors Classifier with optimal k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)  # Train the KNN model
y_pred_knn = knn.predict(X_test)  # Predict on test set

# KNN Accuracy and Confusion Matrix
print("K-NN Accuracy:", metrics.accuracy_score(y_test, y_pred_knn))
print("K-NN Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_knn))

# Support Vector Machine Classifier (Linear Kernel)
svm = SVC(C=1, kernel='linear')  # Initialize SVM with linear kernel
svm.fit(X_train, y_train)  # Train the SVM model
y_pred_svm = svm.predict(X_test)  # Predict on test set

# SVM Accuracy and Confusion Matrix
print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred_svm))
print("SVM Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_svm))
