import numpy as np
from sklearn.cluster import MeanShift# as ms
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt


centers=[[1,1],[5,5],[3,10]]
X,_=make_blobs(n_samples=500,centers=centers,cluster_std=0.3)

#plt.scatter(X[:,0],X[:,1])
#plt.show()
ms=MeanShift()
ms.fit(X)
labels=ms.labels_
cluster_clusters=ms.cluster_centers_


n_clusters_=len(np.unique(labels))
print('Number of cluster esimated:',n_clusters_)
colors=10*['r.','g.','b.','c.','y.','m.']

print(colors)
print(colors)

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
    
plt.scatter(cluster_clusters[:,0],cluster_clusters[:,1],marker='x',s=150,linewidths=5,zorder=10)
    
plt.show()