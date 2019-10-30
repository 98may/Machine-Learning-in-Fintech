# EW Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio


span_t = 120


def EW_weight_compute(n, context=None):
    w = np.ones(n)
    w = w / n

    return w


if __name__ == "__main__":
    print("this is EW Portfolio")
    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mat_data_path = PARENT_DIR + '/data/ff25_input.mat'
    # mat_data_path = PARENT_DIR + '/data/ff48_input.mat'
    ff25 = Stocks(path=mat_data_path)
    m = ff25.Nmonths
    n = ff25.Nportfolios
    R = ff25.portfolios
    portfolio = Portfolio(stock=ff25)
    for k in range(span_t - 1, m, 1):
        wk = EW_weight_compute(n)
        portfolio.rebalance(target_weights=wk)
    print(portfolio.eval(portfolio.cumulative_wealth))
    print(portfolio.eval(portfolio.sharpe_ratio))
    print(portfolio.eval(portfolio.volatility))
    print(portfolio.eval(portfolio.max_drawdown))
    print(portfolio.eval(portfolio.turnover))
    # plot_cumulative_wealth()
