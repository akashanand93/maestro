# this module, given the prediction from the network, will decide whether to buy or sell
# this will also calculate the profit and loss

import numpy as np

class Decider:

    def __init__(self, cutoff_deviation, seq_len, pred_len, funds=100000, min_cutoff=2, cutoff_len=None, log=False):

        self.log = log
        self.funds = funds
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.cutoff_deviation = cutoff_deviation
        self.min_cutoff = (funds*min_cutoff)/100
        self.commision = min(20, funds*0.03/100) + (funds*(0.025 + 0.00325)*1.18)/100
        if not cutoff_len:
            self.cutoff_len = pred_len

    def decide(self, pred, true, last_price):

        num_stock = self.funds // last_price

        # calculate the diff between current price and max price of pred in percentage
        max_diff = (np.max(pred) - last_price)
        min_diff = (np.min(pred) - last_price)

        action = "sell"
        target_price = np.min(pred)
        if abs(max_diff) > abs(min_diff):
            action = "buy"
            target_price = (np.max(pred))

        expected_profit = abs(target_price - last_price) * num_stock

        if expected_profit < max(self.commision, self.min_cutoff):
            return ["hold", 0, 0, 0]

        # find out the first index where the true price is true_price + max_diff, withing cutoff_len
        index = -1
        for i in range(self.cutoff_len):
            if true[i] >= target_price and action == "buy":
                index = i
                break
            elif true[i] <= target_price and action == "sell":
                index = i
                break

        if action == "buy":
            profit = (true[index] - last_price) * num_stock
        else:
            profit = (last_price - true[index]) * num_stock

        profit -= self.commision

        if self.log:
            # print all used variable
            print("current price: {}".format(last_price))
            print("max_diff: {}".format(max_diff))
            print("min_diff: {}".format(min_diff))
            print("action: {}".format(action))
            print("target_price: {}".format(target_price))
            print("num_stock: {}".format(num_stock))
            print("index: {}".format(index))
            print("profit: {}".format(profit))
            print("cuttoff price: {}".format(true[index]))

        fund_utilized = num_stock * last_price
        return [action, round(profit), fund_utilized, self.commision]








