import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Dataset: https://www.kaggle.com/hellbuoy/online-retail-customer-clustering
df = pd.read_csv('OnlineRetail.csv', sep=',', encoding='ISO-8859-1', header=0)

# Drop rows with missing values and change CustomerID type to str
df.dropna(inplace=True)
df['CustomerID'] = df['CustomerID'].astype(str)

# Create 3 new columns
df['Amount'] = df['Quantity'] * df['UnitPrice']
df_1 = df.groupby('CustomerID')['Amount'].sum()
df_1 = df_1.reset_index()

df_2 = df.groupby('CustomerID')['InvoiceNo'].count()
df_2 = df_2.reset_index()
df_2.columns = ['CustomerID', 'Frequency']

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
max_date = max(df['InvoiceDate'])
df['Diff'] = (max_date - df['InvoiceDate']).dt.days
df_3 = df.groupby('CustomerID')['Diff'].min()
df_3 = df_3.reset_index()

# Merge the 3 DataFrame
new_df = pd.merge(df_1, df_2, on='CustomerID', how='inner')
new_df = pd.merge(new_df, df_3, on='CustomerID', how='inner')
new_df.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

# Remove Outliers
Q1 = new_df['Amount'].quantile(0.05)
Q3 = new_df['Amount'].quantile(0.95)
IQR = Q3 - Q1
new_df = new_df[(new_df['Amount'] >= Q1 - 1.5 * IQR) & (new_df['Amount'] <= Q3 + 1.5 * IQR)]

Q1 = new_df['Frequency'].quantile(0.05)
Q3 = new_df['Frequency'].quantile(0.95)
IQR = Q3 - Q1
new_df = new_df[(new_df['Frequency'] >= Q1 - 1.5 * IQR) & (new_df['Frequency'] <= Q3 + 1.5 * IQR)]

Q1 = new_df['Recency'].quantile(0.05)
Q3 = new_df['Recency'].quantile(0.95)
IQR = Q3 - Q1
new_df = new_df[(new_df['Recency'] >= Q1 - 1.5 * IQR) & (new_df['Recency'] <= Q3 + 1.5 * IQR)]

# Standard Scaler
from sklearn.preprocessing import StandardScaler

x = new_df[['Amount', 'Frequency', 'Recency']]
scaler = StandardScaler()
new_df_scaled = scaler.fit_transform(x)
new_df_scaled = pd.DataFrame(new_df_scaled)
new_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

# Find optimal n_clusters of KMeans with Elbow Method
from sklearn.cluster import KMeans

inertias = []
for i in range(1,11):
    km = KMeans(n_clusters=i, random_state=21)
    km.fit(new_df_scaled)
    inertias.append(km.inertia_)

km = KMeans(n_clusters=3, random_state=21)
km.fit(new_df_scaled)
new_df['Cluster'] = km.labels_

# Find optimal n_clusters of Hierarchical Clustering with Dendogram
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

mergings = linkage(new_df_scaled, method='average', metric='euclidean')
cluster_id = cut_tree(mergings, n_clusters=3).reshape(-1,)
new_df1 = new_df.copy()
new_df1['Cluster'] = cluster_id

# Last but not least, plot the clusters with scatterplot
