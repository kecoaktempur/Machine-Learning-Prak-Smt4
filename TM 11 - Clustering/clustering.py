import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# Read Dataset
iris = pd.read_csv("D:\Coolyeah\Mata Kuliah\SMT 4\Machine Learning Praktikum\TugasClustering_Kelompok 2\iris.csv")
x = iris.iloc[:, [0, 1, 2, 3]].values

iris.info()
iris[0:10]

iris_outcome = pd.crosstab(index=iris["Species"],  # Make a crosstab
                           columns="count")      # Name the count column

iris_outcome

iris_setosa=iris.loc[iris["Species"]=="Iris-setosa"]
iris_virginica=iris.loc[iris["Species"]=="Iris-virginica"]
iris_versicolor=iris.loc[iris["Species"]=="Iris-versicolor"]

# Distribution Plots

# Plot Each Flower to Histogram
sns.FacetGrid(iris,hue="Species",height=3).map(sns.distplot,"petal_length").add_legend()
sns.FacetGrid(iris,hue="Species",height=3).map(sns.distplot,"petal_width").add_legend()
sns.FacetGrid(iris,hue="Species",height=3).map(sns.distplot,"sepal_length").add_legend()
sns.FacetGrid(iris,hue="Species",height=3).map(sns.distplot,"sepal_width").add_legend()
plt.show()

# Box Plot
sns.boxplot(x="Species",y="petal_length",data=iris)
plt.show()

# Violin Plot
sns.violinplot(x="Species",y="petal_length",data=iris)
plt.show()

# Scatter Plot
sns.set_style("whitegrid")
sns.pairplot(iris,hue="Species",height=3);
plt.show()

# K-Means

# Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Elbow Method to Determine The Optimal Number of Clusters for K-Means Clustering
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

# Clustering
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
    
# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')

plt.legend()

# 3d scatterplot using matplotlib

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.show()