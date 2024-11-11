import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, asin

warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv("uber.csv")
df = df.drop(["Unnamed: 0", "key"], axis=1)
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df = df.dropna()

# Define distance transformation function
def distance_transform(longitude1, latitude1, longitude2, latitude2):
    distance = []
    for pos in range(len(longitude1)):
        long1, lati1, long2, lati2 = map(radians, [longitude1[pos], latitude1[pos], longitude2[pos], latitude2[pos]])
        dist_long = long2 - long1
        dist_lati = lati2 - lati1
        a = sin(dist_lati/2)**2 + cos(lati1) * cos(lati2) * sin(dist_long/2)**2
        c = 2 * asin(sqrt(a)) * 6371  # Radius of Earth in kilometers
        distance.append(c)
    return distance

# Apply distance transformation and create new columns
df["distance_km"] = distance_transform(
    df["pickup_longitude"].to_numpy(), 
    df["pickup_latitude"].to_numpy(), 
    df["dropoff_longitude"].to_numpy(), 
    df["dropoff_latitude"].to_numpy()
)
df = df.assign(
    pickup_hr=df.pickup_datetime.dt.hour,
    day=df.pickup_datetime.dt.day,
    month=df.pickup_datetime.dt.month,
    year=df.pickup_datetime.dt.year,
    day_of_week=df.pickup_datetime.dt.dayofweek,
    day_name=df.pickup_datetime.dt.day_name()
)

# Identify and remove outliers
def find_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    outliers = df[((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]
    return outliers

# Drop unwanted rows based on conditions
df.drop(df[df['distance_km'] == 0].index, inplace=True)
df.drop(df[df['distance_km'] > 60].index, inplace=True)
df.drop(df[df['fare_amount'] > 100].index, inplace=True)
df.drop(df[df['fare_amount'] < 0].index, inplace=True)
df.drop(df[df['passenger_count'] > 6].index, inplace=True)

# Display correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True)
plt.show()

# Train-test split and model training
x = df[["year", "distance_km"]]
y = df["fare_amount"]
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

# Predictions and regression plot
y_pred = model.predict(x_test)
sns.regplot(x=y_test, y=y_pred, color="red", line_kws={"color": "blue"})
plt.show()
