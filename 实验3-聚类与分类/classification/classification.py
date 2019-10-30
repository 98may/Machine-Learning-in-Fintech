import numpy as np
from gen_data import gen_data
from plot import plot
from todo import func

no_iter = 1000  # number of iteration
no_train = 50# Your code here  # number of training data
no_test = 50# Yourcode here  # number of testing data
no_data = 100  # number of all data
assert(no_train + no_test == no_data)

cumulative_train_err = 0
cumulative_test_err = 0

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

plot(X, y, w_f, w_g, "Classification")
print("Training error: %s" % train_err)
print("Testing error: %s" % test_err)

