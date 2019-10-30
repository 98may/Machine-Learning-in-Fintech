import numpy as np
import scipy.io as sio
from plot import plot
from todo import kmeans
from todo import spectral
from todo import knn_graph

cluster_data = sio.loadmat('/Users/may/Desktop/实验3-聚类与分类/clustering/cluster_data.mat')
X = cluster_data['X']

#idx = kmeans(X, 2)
#plot(X, idx, "Clustering-kmeans")

W = knn_graph(X, 20, 0.5)  # W = knn_graph(X, 15, 1.45) ：recommend parameters
idx = spectral(W, 2)
plot(X, idx, "Clustering-Spectral-knn_graph(X, 20, 0.5)")
