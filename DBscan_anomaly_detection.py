import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

'''
*** DBscan clustering ***

inspired by
https://medium.com/@dilip.voleti/dbscan-algorithm-for-fraud-detection-outlier-detection-in-a-data-set-60a10ad06ea8
'''

# import data
df = pd.read_csv("iris.csv")
print(df.info())


# knn to find optimal Epsilon
knn = NearestNeighbors(n_neighbors=2).fit(df[["sepal_length", "sepal_width"]])
distances, indx = knn.kneighbors(df[["sepal_length", "sepal_width"]])

# find max distance between and a point and its nearest neighbor
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure()
plt.plot(distances)
plt.title('Knn distance graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.savefig('KNN-Epsilon.png')

# Epsilon
eps = distances.max()

# DBscan clustering
dbscan = DBSCAN(eps=eps, min_samples=10).fit(df[['sepal_length', 'sepal_width']])
# add cluster labels to df
df['label'] = dbscan.labels_
# visualize
plt.figure()
plt.scatter(df["sepal_length"], df["sepal_width"], c = dbscan.labels_)
plt.title('sepal length & width clusters',fontsize=20)
plt.xlabel('length',fontsize=14)
plt.ylabel('width', fontsize=14)
plt.savefig('DBscan-scatter.png')

# outliers
outliers = df[df['label'] == -1]
print(outliers)
