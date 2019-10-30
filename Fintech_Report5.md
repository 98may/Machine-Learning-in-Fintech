# <center>Fintech_Report5


<center>2019/8/26

[TOC]

## 1:实验⽬的 

1.学习投资组合基本流程

2.了解投资组合算法框架

3.并实现经典投资组合策略



## 2:实验步骤

> 程序中的策略代码文件和配置文件strategy.\__init__.py

### 2.1:学习并编译示例代码

> 下载示例代码后，安装示例代码所需的 python 包(scipy, pandas, numpy, matplotlib, cvxopt)， 我使用了pycharm进行安装。运行示例代码，分析示例代码中 EW 策略的表现，理解示例代码中的评价指标。 

因为别的代码都已经是设置好的，直接run main.py就能得到结果，详见3:实验结果。



夏普比率计算公式：＝[E(Rp)－Rf]/σp，其中E(Rp)：投资组合预期报酬率，Rf：无风险利率，σp：投资组合的[标准差]。计算投资组合每承受一单位总风险，会产生多少的超额报酬。



### 2.2:实现马科维茨投资组合策略（MV）

>使用 cvxopt 库中的二次规划函数，根据理论课中讲述的算法流程，实现马科维茨投资组合策略(MV)。 

马科维茨投资组合策略(MV)核心思想：

![屏幕快照 2019-08-31 下午5.19.34](/Users/may/Desktop/assets/屏幕快照 2019-08-31 下午5.19.34.png)

具体实现中在其他基本设置里根据报错提示进行了相应修改，例如

```python
	Best = {"name": "Best", "function": "Best", "data_type": "density"}
    EW = {"name": "EW", "function": "EW", "data_type": "density"}
    mayMV = {"name": "mayMV", "function": "mayMV", "data_type": "density"}

    methods = [Best, EW, mayMV]
    methods_name = ["Best", "EW", "mayMV"]
```

实现马科维茨投资组合策略(MV)的核心代码如下：

```python
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
from cvxopt import solvers, matrix

'''
context参数说明
R：k - span_t + 1:k一段时间的return
'''
span_t = 120

def mayMV_weight_compute(n, context):

    #key code begin
    #context["R"] : span_t-by-N matrix
    meanR = np.mean(context["R"].T, axis = 1)
    # print("meanR.shape =", meanR.shape)
    # print("context["R"].shape =", context["R"].shape)
    Q = np.cov(context["R"].T)
    # print("Q.shape =", Q.shape)

    P = 2 * matrix(Q)
    q = -1 * matrix(meanR)

    G = -matrix(np.eye(n))
    h = matrix(0.0, (n, 1))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    sol = solvers.qp(P, q, G, h, A, b)
    w = np.array(sol['x']).T
    w = w[0]

    #key code end
    return w


if __name__ == "__main__":
    print("this is mayMV Portfolio")
```

> ref：CVXOPT中求解二次规划的问题https://uqer.io/v3/community/share/55c9a55df9f06c91f818c675
>
> ![屏幕快照 2019-08-31 下午5.17.51](/Users/may/Desktop/assets/屏幕快照 2019-08-31 下午5.17.51.png)



## 3:实验结果

如图所示，总收益远大于EW，但是也不如Best。

夏普比率、不稳定性、最大回撤与EW相仿。

![屏幕快照 2019-08-31 下午5.11.21](/Users/may/Desktop/assets/屏幕快照 2019-08-31 下午5.11.21.png)

如下图，MV的收益在交易后期（轮数多）时收益远高于EW且快速上涨。

![屏幕快照 2019-08-31 下午5.23.37](/Users/may/Desktop/assets/屏幕快照 2019-08-31 下午5.23.37.png)

两者的Regret相近。

![屏幕快照 2019-08-31 下午5.23.15](/Users/may/Desktop/assets/屏幕快照 2019-08-31 下午5.23.15.png)