

# <center>Fintech_Report3

<center>毛阿妍3170102656
<center>2019/8/22


[TOC]

## 1:实验⽬的 

1. 通过实验了解聚类与分类的应用场景;
2. 熟练掌握至少两种聚类算法的原理，并了解不同算法的优劣势;
3. 熟练掌握至少一种分类算法的原理。



## 2:实验步骤 

### 2.1:聚类算法实现

> 1. clustering 文件夹下的 todo.py 文件部分:
>
>    - 在kmeans函数下实现k-means或k-medoids算法 
>
>    -  在spectral函数下实现谱聚类算法 
>
> 2. clustering 文件夹下的 clustering.py 文件部分
>
>    - 调试knn_graph的参数，优化谱聚类的效果
>    - 运行clustering.py文件，保存两种算法的聚类结果，比较分析其不同 

#### k-means算法

##### k-means思路

1.把所有数据点划分成k个子集

2.

##### k-means源代码

```python
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

    recenter=0
    flag=1
    while(1):
        if(flag==0):
            break;
        recenter+=1

         #计算每个子集的中心点
        count = np.zeros((k, 1))

        centers = np.zeros((k, P))

        for i in range(N):
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

    idx = np.asarray(idx)
    idx = idx.squeeze()

    # Answer end
    return idx
```

#### 谱聚类算法Spectral Clustering

##### 谱聚类思路

> 第一步：数据准备，生成图的邻接矩阵；
>
> 第二步：归一化普拉斯矩阵；
>
> 第三步：生成最小的k个特征值和对应的特征向量；
>
> 第四步：将特征向量kmeans聚类(少量的特征向量)；
>
> 一般来说，谱聚类主要的注意点为相似矩阵的生成方式，切图的方式以及最后的聚类方法。
> 具体介绍一下：
> 最常用的相似矩阵的生成方式是基于高斯核距离的全连接方式，最常用的切图方式是Ncut。而到最后常用的聚类方法为K-Means。下面以Ncut总结谱聚类算法流程。
>
> 输入：样本集D=(x1,x2,…,xn)，相似矩阵的生成方式, 降维后的维度k1, 聚类方法，聚类后的维度k2
> 输出： 簇划分C(c1,c2,…ck2)
>
> 1）根据输入的相似矩阵的生成方式构建样本的相似矩阵S
>
> 2）根据相似矩阵S构建邻接矩阵W，构建度矩阵D
>
> 3）计算出拉普拉斯矩阵L
>
> 4）构建标准化后的拉普拉斯矩阵D−1/2LD−1/2
> 　　　　
> 5）计算D−1/2LD−1/2最小的k1个特征值所各自对应的特征向量f
>
> 6）将各自对应的特征向量f组成的矩阵按行标准化，最终组成n×k1维的特征矩阵F
>
> 7）对F中的每一行作为一个k1维的样本，共n个样本，用输入的聚类方法进行聚类，聚类维数为k2。
>
> 8）得到簇划分C(c1,c2,…ck2 ).
>  ———————————————— 
> 版权声明：本文为CSDN博主「水煮洋洋洋」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/weixin_43526820/article/details/89638596

```
# W = knn_graph(X, 15, 1.45)  # recommend parameters

N = 4078
k = 2
X.shape = (4078, 2)

recenter times 2
```

##### 谱聚类源代码

```python
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
```

##### 调试knn_graph的参数，优化谱聚类的效果

详见”3:实验结果“

### 2.2:分类算法实现

> 1. classification 文件夹下 todo.py 文件部分:
>
>    - 在func函数下实现LR、SVM和感知机(Perceptron)三种分类算法的其中一种 
>
> 2. 在 classification 文件夹下 classification.py 文件部分:
>
>    - 选择划分训练集和测试集的比例
>
>    - 编写计算训练和测试错误率的代码 
>    - 保存分类结果，报告训练和测试错误率

#### 2.2.1:感知机(Perceptron)算法

```python
     #训练算法的核心，即随机梯度下降：
    def func(X, y):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned perceptron parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.zeros((P+1, 1))

    # Your code here
    # Answer begin
   
    riTimes = 500
    for ri in range(riTimes):
        for ni in range(N):
            #let b == w[0][0]
            t = w[0][0]
            for i in range(P):
                t += X[i][ni]*w[i+1][0]
            if((t*y[0][ni])<=0):
                w[0][0] += y[0][ni]
                for j in range(P):
                    w[j+1][0] += X[j][ni]*y[0][ni]

    # Answer end
    return w
```



> ref：<https://blog.csdn.net/xueyingxue001/article/details/50672890>
>

#### 2.2.2:选择划分训练集和测试集的比例、编写计算训练和测试错误率的代码

划分训练集0.7和测试集0.3

```python
for i in range(no_iter):
    X, y, w_f = gen_data(no_data)
    X_train, X_test = X[:, :no_train], X[:, no_train:]
    y_train, y_test = y[:, :no_train], y[:, no_train:]
    w_g = func(X_train, y_train)
    # Compute training, testing error
    # Your code here
    # Answer begin
    
    train_err = 0
    test_err = 0
    for j in range(no_train):
        if(( X[0][j] * w_g[1][0] + X[1][j] * w_g[2][0] + w_g[0][0]) * y[0][j] <= 0):
            train_err = 1
            break
    for j in range(no_test):
        if(( X[0][no_train+j] * w_g[1][0] + X[1][no_train+j] * w_g[2][0] + w_g[0][0]) * y[0][no_train+j] <= 0):
            test_err = 1
            break

    # Answer end
    cumulative_train_err += train_err
    cumulative_test_err += test_err

train_err = cumulative_train_err / no_iter
test_err = cumulative_test_err / no_iter
```



## 3:实验结果

### 3.1:聚类算法

#### 3.1.1:k-means

![Figure_1](/Users/may/Desktop/assets/Figure_1.png)

#### 3.1.2:谱聚类Spectral Clustering

W = knn_graph(X, 15, 1.45)  # recommend parameters,recenter times : 2

![Figure_2](/Users/may/Desktop/assets/Figure_2.png)

#### 3.1.3:调试knn_graph的参数，优化谱聚类的效果

如图所示，推荐的参数W=knn_graph(X, 15, 1.45)效果不错，微调参数以后的效果也相近；变动太大则效果不佳。

![IMG_3249](/Users/may/Desktop/assets/IMG_3249.JPG)

#### 3.1.4:分析k-means和Spectral Clustering的其不同

+ K-means:

  - 简单快速 (思想很简单易懂）
  - 􏰡􏱂􏰞􏰀􏰗􏰌􏰮􏰢聚类结果在向量空间为球状（凸集）
  - 聚类结果易受到起始点影响
  - 聚类结果易受到噪声（脏数据）影响

+ Spectral Clustering

  - 相对来说更复杂缓慢

  - 能在任意形状的样本空间上得到较好的聚类效果（如示例中的同心圆情况）

  - 利用了图论的思想和其他的聚类算法（例如k-means），优化了聚类效果


### 3.2:分类算法

#### 感知机

我写的是任意维度情况下都可用的代码，增加训练次数会优化分类效果，但次数太多还导致运行缓慢，大约500次的效果已经不错。

（黑框里的数字为训练次数）

![IMG_3279](/Users/may/Desktop/Fintech/Fintech_report/Fintech_Report3_毛阿妍3170102656/assets/IMG_3279.JPG)

![IMG_3288](/Users/may/Desktop/Fintech/Fintech_report/Fintech_Report3_毛阿妍3170102656/assets/IMG_3288.JPG)

#### 训练集与测试集的比例

训练集与测试集的比例在7:3左右时效果不错，不过合理范围内的分类效果都区别不大

![IMG_3282](/Users/may/Desktop/Fintech/Fintech_report/Fintech_Report3_毛阿妍3170102656/assets/IMG_3282.JPG)