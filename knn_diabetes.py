# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import plotly.express as px  # For data visualization
from sklearn.preprocessing import MinMaxScaler  # For scaling features
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier
from sklearn import metrics  # For evaluation metrics
from mlxtend.plotting import plot_confusion_matrix  # For confusion matrix visualization
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For visualizations
import warnings  # To ignore warnings
warnings.filterwarnings("ignore")

# Loading and exploring the dataset
df = pd.read_csv("diabetes.csv")
print("Dataset preview:")
print(df.head())

# Displaying dataset information to check for null values and data types
print("\nDataset info:")
print(df.info())

# Statistical summary of the dataset
print("\nDataset summary statistics:")
print(df.describe().T)

# Checking the balance of the target variable
print("\nTarget variable distribution:")
print(df["Outcome"].value_counts())
sns.countplot(data=df, x="Outcome")
plt.show()

# Splitting features and target variable
X = df.drop("Outcome", axis=1)  # Features
y = df["Outcome"]  # Target variable

# Scaling the features using Min-Max Scaler for better model performance
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Implementing K-Nearest Neighbors with optimal k selection
k_values = range(1, 51)  # Range of k values to evaluate
accuracy_values = []

# Looping over k values to find the best k with the highest accuracy
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)

# Plotting accuracy values to choose optimal k
px.line(x=list(k_values), y=accuracy_values, labels={'x': "k", 'y': "Accuracy"}).show()

# Choosing the best k based on highest accuracy
optimal_k = k_values[accuracy_values.index(max(accuracy_values))]
print(f"Optimal k: {optimal_k}")

# Building the KNN model with the optimal k
knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
knn_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluating the model
accuracy = metrics.accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Error rate: {error_rate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Generating the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_matrix)
plt.show()

# Displaying classification report for detailed metrics
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred))
