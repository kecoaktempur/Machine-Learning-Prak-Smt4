# import package
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# import datasets
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
print(cancer.keys())

df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df.head(5))

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df)
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_data=scaler.transform(df)
print(scaled_data)

pca=PCA(n_components=2)
pca.fit(scaled_data)

x_pca=pca.transform(scaled_data)
print(scaled_data.shape)

print(x_pca.shape)

scaled_data

print(x_pca)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.show()

