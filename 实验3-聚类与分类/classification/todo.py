import numpy as np

# You may find below useful for Support Vector Machine
# More details in
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# from scipy.optimize import minimize

def func(X, y):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned perceptron parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.zeros((P+1, 1))

#   Second Edition
    riTimes = 1
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
    
    '''First Edition
    defaultx = np.ones(N) # -by-N
    defaultx = np.mat(defaultx) #1-by-N
    defaultx = defaultx.T #N-by-1
    newX = np.c_[X.T,defaultx] #N-by-(P+1) matrix
    y = np.array(y)
    newX = np.array(newX)
    w = np.array(w)

    for ri in range(500):
        for i in range(N):
            #predictY := F(wx+b)
            #let t represents wx+b==wx+b*1== w*newX
            t = newX[i].dot(w)
            if(t*y[0][i]<=0):
                for j in range(P+1):
                    w[j][0] += newX[i][j]*y[0][i]
    '''

    # Answer end
    return w
