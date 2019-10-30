import numpy as np
from scipy.spatial.distance import cdist

def kmeans(X, k):
    '''
    K-Means clustering algorithm

    Input:  x: data point features, N-by-P maxtirx
            k: the number of clusters

    OUTPUT:  idx: cluster label, N-by-1 vector
    '''

    N, P = X.shape
    idx = np.zeros((N, 1))
    # Your Code Here
    # Answer begin

    #把所有数据点划分成k个子集（如何保证非空？）
    idx = np.mat(np.random.randint(1,k+1,size=(N,1)))
    #计算每个子集的中心点
    count = np.zeros((k, 1))

    centers = np.zeros((k, P))

    for i in range(N):
            # print("whyfaillll?")
            centers[idx[i]-1] += X[i]
            # centers[idx[i][0]-1][j] += X[i][j]# index 1 is out of bounds for axis 0 with size 1
            count[idx[i]-1] +=1

    for i in range(k):
            centers[i] = centers[i]/count[i]

    #将所有数据点重新划分
    tmp_dist = cdist(X, centers, 'euclidean')
    for i in range(N):
        for j in range(k):
            if(tmp_dist[i][j] < tmp_dist[i][idx[i]-1]):
                idx[i] = j+1

    recenterTimes=0
    flag=1
    while(1):
        if(flag==0):
            break;
        recenterTimes+=1

         #计算每个子集的中心点
        count = np.zeros((k, 1))

        centers = np.zeros((k, P))

        for i in range(N):
            # print("i,j is",i,j)
            # print("idx[i]",idx[i])
            # print("whyfaillll?")
            centers[idx[i]-1] += X[i]
            # centers[idx[i][0]-1][j] += X[i][j]# index 1 is out of bounds for axis 0 with size 1
            count[idx[i]-1] +=1

        for i in range(k):
                centers[i] = centers[i]/count[i]

        #将所有数据点重新划分
        tmp_dist = cdist(X, centers, 'euclidean')
        flag=0
        for i in range(N):
            for j in range(k):
                if(tmp_dist[i][j] < tmp_dist[i][idx[i]-1]):
                    idx[i] = j+1
                    flag=1
        # print("recenter times,flag",recenter,flag)

    ttmp=np.ones((N, 1))
    idx = idx - ttmp
    print("recenter times",recenterTimes)

    idx = np.asarray(idx)
    idx = idx.squeeze()

    # Answer end
    return idx

def spectral(W, k):
    '''
    Spectral clustering algorithm

    Input:  W: Adjacency matrix, N-by-N matrix
            k: number of clusters

    Output:  idx: data point cluster labels, N-by-1 vector
    '''
    N = W.shape[0]
    idx = np.zeros((N, 1))
    # Your code here
    # Answer begin
    D = np.diag(np.sum(W,axis=1))
    invD = np.linalg.inv(D) 

    tmpMatrix = np.dot(invD, D-W)

    eigValue, eigVector = np.linalg.eig(tmpMatrix)
    
    dim = len(eigValue)
    dictEigValue=dict(zip(eigValue,range(0,dim)))
    kEig=np.sort(eigValue)[0:k]
    ix=[dictEigValue[k] for k in kEig]
    X = eigVector[:,ix]
# return the k smallest eigenvalues' corresponding eigenvector matrix.

    print("N =",N)
    print("k =",k)
    print("X.shape =",X.shape)

    # X: data point features, N-by-k maxtirx
    # Answer end
    X = X.astype(float)
    idx = kmeans(X, k)
    return idx

def knn_graph(X, k, threshold):
    '''
    Construct W using KNN graph

    Input:  X:data point features, N-by-P maxtirx.
            k: number of nearest neighbour.
            threshold: distance threshold.

    Output:  W - adjacency matrix, N-by-N matrix.
    '''
    N = X.shape[0]
    W = np.zeros((N, N))
    aj = cdist(X, X, 'euclidean')
    for i in range(N):
        index = np.argsort(aj[i])[:(k+1)]  # aj[i,i] = 0
        W[i, index] = 1
    W[aj >= threshold] = 0
    return W
