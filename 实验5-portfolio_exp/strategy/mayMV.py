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

    # w = np.zeros(n)

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

    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mat_data_path = PARENT_DIR + '/data/ff25_input.mat'
    # mat_data_path = PARENT_DIR + '/data/ff48_input.mat'
    ff25 = Stocks(path=mat_data_path)
    m = ff25.Nmonths
    n = ff25.Nportfolios
    R = ff25.portfolios
    portfolio = Portfolio(stock=ff25)
    for k in range(span_t - 1, m, 1):
        wk = mayMV_weight_compute(n, context)
        portfolio.rebalance(target_weights=wk)
    print(portfolio.eval(portfolio.cumulative_wealth))
    print(portfolio.eval(portfolio.sharpe_ratio))
    print(portfolio.eval(portfolio.volatility))
    print(portfolio.eval(portfolio.max_drawdown))
    print(portfolio.eval(portfolio.turnover))