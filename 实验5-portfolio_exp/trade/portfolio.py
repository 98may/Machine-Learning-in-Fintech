#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    File Name:      stocks.py
    Created Time:   11/15/18 3:56 PM
    Author:         zmy
"""
import numpy as np


class Portfolio:
    def __init__(self, stock, total_price=1000000, time_init=120, frequency="month"):
        """
        :param stock: 股票的全局价格
        :param total_price: 组合初始资金
        """
        self.Nportfolios = stock.Nportfolios
        self.Nmonths = stock.Nmonths
        self.portfolios_price = stock.portfolios_price
        self.portfolios = stock.portfolios
        self.weight = []
        self.total_price = total_price
        self.return_list = []
        self.net_return_list = []
        self.time_init = time_init
        self.price_init = total_price
        self.frequency = frequency

    # todo:输入权重，输出return_list
    # option可以控制要让权重调整的最小值为1
    def rebalance(self, target_weights, option="Normal"):
        # 计算return，不使用新的，计算的是到这个时间的return
        time = len(self.weight)

        if time == 0:
            new_return = 1
            self.return_list.append(new_return)
            self.net_return_list.append(new_return)
        else:
            old_weight = self.weight[-1]
            new_return = 0

            for k in range(len(old_weight)):
                # 针对random10
                new_return += (self.portfolios[time + self.time_init - 2][k] - 1) * old_weight[k]

            self.net_return_list.append(new_return + 1.0)
            new_return = self.return_list[-1] * (new_return + 1.0)
            self.return_list.append(new_return)
        # 存weight vectors
        new_weights = np.zeros(self.Nportfolios)
        if option == "Normal":
            new_weights = target_weights
        else:
            for k in range(self.Nportfolios):
                postion = self.total_price * target_weights[k]
                new_weights[k] = int(postion / self.portfolios_price[time + self.time_init][k])
            self.total_price = self.price_init * new_return
        self.weight.append(new_weights)
        return True

    # todo:评价指标的计算，调用其他的函数
    # Sharpe ratios
    # cumulative wealth
    # turnovers
    # volatility
    # max drawdown
    def eval(self, evaluation_function):
        # print("now start computing:", evaluation_function.__name__)
        return evaluation_function()

    def max_drawdown(self):
        """最大回撤率"""
        i = np.argmax((np.maximum.accumulate(self.return_list) - self.return_list) / np.maximum.accumulate(
            self.return_list))  # 结束位置
        if i == 0:
            return 0
        j = np.argmax(self.return_list[:i])  # 开始位置
        return (self.return_list[j] - self.return_list[i]) / (self.return_list[j])

    def sharpe_ratio(self):
        """夏普比率"""
        net_return = []
        for k in range(len(self.return_list) - 1):
            net_return.append(float(self.return_list[k + 1] / self.return_list[k]))
        average_return = np.mean(net_return)
        return_stdev = np.std(net_return)
        non_risk_rate = 0  # 无风险利率为0.02
        sharpe_ratio = (average_return - non_risk_rate) / return_stdev
        ASR = np.sqrt(12) * sharpe_ratio
        return sharpe_ratio

    def cumulative_wealth(self):
        """累计return"""
        return self.return_list[-1]

    def turnover(self):
        total_turnover = 0.0
        for k in range(len(self.weight)):
            if k != 0:
                self_weight = np.multiply(self.weight[k-1], self.portfolios[k + self.time_init - 1])
                total_weight = np.dot(self.weight[k-1], self.portfolios[k + self.time_init - 1])
                weight_ = self_weight / total_weight
                total_turnover += sum(abs(self.weight[k] - weight_))

        return total_turnover / len(self.weight)

    def volatility(self):
        net_return = [ ]
        for k in range(len(self.return_list) - 1):
            net_return.append(float(self.return_list[k + 1] / self.return_list[k]))
        return_stdev = np.std(net_return)
        if self.frequency == "month":
            volatility = np.sqrt(12) * return_stdev  # for month
        elif self.frequency == "week":
            volatility = np.sqrt(52) * return_stdev  # for week
        elif self.frequency == "day":
            volatility = np.sqrt(365) * return_stdev  # for week
        else:
            volatility = return_stdev
        return volatility


