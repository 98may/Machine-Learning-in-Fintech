import datetime

from strategy.Best import Best_weight_compute
from strategy.EW import EW_weight_compute
from strategy.mayMV import mayMV_weight_compute
'''
context参数说明

Pk：k时刻的价格
Rk：k时刻的price relative vector
P：k - span_t + 1:k一段时间的价格
R：k - span_t + 1:k一段时间的return
frequency: rebanlance的周期
'''
'''
method说明

"EW": equal weighted
"VW": value weighted
"BH": buy and hold
"MV": minumum variance
"MVWC":Minumum-variance Portfolio with the Constraint
"MC":Mean-CVaR
"MCWC":Mean-CVaR Portfolio with the constraint
"TP_VW": Blending VW MV
"TP_EM": Blending EW MV
"OBP": Orthogonal Bandit Portfolio
"PBTS"
"PCTS" : portfolio choices via Thompson sampling
'''


def methods_config():
    """
    :param name: 算法名称
    :param function: 算法所在的函数名称
    :param duplicate: 实验重复次数
    :param k: PBTS特有参数
    :param stock_size: 一共有几只股票
    :param portfolio_size: 每个组合有几只股票，若0则不限制
    :param update_type: 更新类型，不同算法意义不同
    """
    Best = {"name": "Best", "function": "Best", "data_type": "density"}
    EW = {"name": "EW", "function": "EW", "data_type": "density"}
    mayMV = {"name": "mayMV", "function": "mayMV", "data_type": "density"}

    methods = [Best, EW, mayMV]
    methods_name = ["Best", "EW", "mayMV"]

    return methods, methods_name


def datasets_config():
    # !!!根据特征工程，init_t一定一定要大于12个单位
    ff25_csv = {"name": "ff25_csv", "filename": "portfolio25.csv", "span_t": 120, "init_t": 20, "frequency": "month"}

    datasets = [ff25_csv]
    dataset_name = ["ff25_csv"]
    return datasets, dataset_name


def runPortfolio(stocks, portfolio, method, dataset):
    # get stock data
    m = stocks.Nmonths
    n = stocks.Nportfolios
    R = stocks.portfolios
    P = stocks.portfolios_price

    MF = stocks.market_feature

    SF = stocks.stock_feature

    weight_compute = eval(method["function"] + "_weight_compute")
    context = {"frequency": portfolio.frequency, "return_list": []}

    for k in range(dataset["span_t"] - 1 + dataset["init_t"], m, 1):
        context["Pk"] = P[k]
        context["Rk"] = R[k]
        context["MF"] = MF[k]
        context["SF"] = SF[k * n:(k + 1) * n, :]
        context["next_Rk"] = None
        if k < m - 1:
            context["next_Rk"] = R[k + 1]
        context["P"] = P[k - dataset["span_t"] + 1: k + 1]
        context["R"] = R[k - dataset["span_t"] + 1: k + 1]

        wk = weight_compute(n, context)

        portfolio.rebalance(target_weights=wk)

        context["return_list"].append(portfolio.return_list[-1])


if __name__ == "__main__":
    print("this is config and run script, start please go to main.py")
