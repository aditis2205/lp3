import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
df = pd.read_csv("sales_data_sample.csv", encoding="latin")
df.head()
df.info()
df = df[['ORDERLINENUMBER', 'SALES']]
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df.values)
wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, init='k-means++')
    model.fit_predict(scaled_values)
    wcss.append(model.inertia_)
plt.plot(range(1, 11), wcss, 'ro-')
plt.show()
model = KMeans(n_clusters=7, init='k-means++')
clusters = model.fit_predict(scaled_values)
clusters

df['cluster'] = clusters
df
model.inertia_

plt.scatter(df['ORDERLINENUMBER'], df['SALES'], c=df['cluster'])
plt.show()
