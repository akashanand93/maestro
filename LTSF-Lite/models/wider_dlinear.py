import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size1, kernel_size2):
        super(SeriesDecomp, self).__init__()
        self.moving_avg1 = MovingAvg(kernel_size1, stride=1)
        self.moving_avg2 = MovingAvg(kernel_size2, stride=1)

    def forward(self, x):
        moving_mean1 = self.moving_avg1(x)
        moving_mean2 = self.moving_avg2(x)
        res = x - moving_mean2 - moving_mean1
        return res, moving_mean1, moving_mean2

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecomp(5, 25)

        self.Linear_Seasonal1 = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Seasonal2 = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init1, seasonal_init2, trend_init = self.decompsition(x)
        seasonal_init1, seasonal_init2, trend_init = seasonal_init1.permute(0,2,1), seasonal_init2.permute(0,2,1), trend_init.permute(0,2,1)

        seasonal_output1 = self.Linear_Seasonal1(seasonal_init1)
        seasonal_output2 = self.Linear_Seasonal2(seasonal_init2)
        trend_output = self.Linear_Trend(trend_init)

        x = trend_output + seasonal_output1 + seasonal_output2
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
