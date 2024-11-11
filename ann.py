import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix

# Load dataset
dataset = pd.read_csv('/content/Churn_Modelling.csv', index_col='RowNumber')

# Selecting feature columns and target
X_columns = dataset.columns.tolist()[2:12]  # Select features from 'CreditScore' to 'EstimatedSalary'
Y_columns = dataset.columns.tolist()[-1:]  # Select the target 'Exited'

X = dataset[X_columns].values
Y = dataset[Y_columns].values

# ColumnTransformer to one-hot encode categorical columns and standardize numerical columns
column_transformer = ColumnTransformer(
    transformers=[
        ("Gender OneHot Encoder", OneHotEncoder(drop='first'), [2]),  # OneHotEncode Gender
        ("Geography OneHot Encoder", OneHotEncoder(drop='first'), [1])  # OneHotEncode Geography
    ],
    remainder='passthrough'  # Leave the rest of the columns as they are (no transformation)
)

# Apply transformations to the features
X = column_transformer.fit_transform(X)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Initialize the neural network model
classifier = Sequential()

# Add the input layer and the first hidden layer
classifier.add(Dense(6, activation='relu', input_shape=(X_train.shape[1],)))
classifier.add(Dropout(0.1))  # Dropout for regularization

# Add a second hidden layer
classifier.add(Dense(6, activation='relu'))
classifier.add(Dropout(0.1))  # Dropout for regularization

# Add the output layer (binary classification)
classifier.add(Dense(1, activation='sigmoid'))

# Compile the model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = classifier.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.1, verbose=2)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Apply threshold of 0.5 to get binary predictions
y_pred = (y_pred > 0.5).astype(int)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Output the confusion matrix
print("Confusion Matrix:")
print(cm)

# Accuracy of the model
accuracy = (cm[0][0] + cm[1][1]) * 100 / len(y_test)
print(f"Accuracy: {accuracy}%")
