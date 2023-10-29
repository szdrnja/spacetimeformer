"""
Reversible Instance Normalization from
https://github.com/ts-kim/RevIN
"""

import torch
import torch.nn as nn

from typing import Tuple


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
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


class SeriesDecomposition(nn.Module):
    """
        A given time series is thought to consist of three systematic
        components including level, trend, seasonality, and one non-systematic
        component called noise.
            Level: The average value in the series.
            Trend: The increasing or decreasing value in the series.
            Seasonality: The repeating short-term cycle in the series.
            Noise: The random variation in the series.
        https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html

    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Adds `x` to the moving average and then removes the
            Trend (T) from `x` to get Seasonality (S) and Noise (e).
            Assumes X = T + S + e

            @param x: input

            @returns (res, moving_mean)
                res: the residue left after removing T
                moving_mean: the current value of T
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class RevIN(nn.Module):
    """
        Reversible Instance Normalization for Accurate Time-Series Forecasting
        against Distribution Shift
        ref: https://openreview.net/pdf?id=cGDAkQo1C0p
        Statistical properties such as mean and variance often change over time
        in time series, i.e., time-series data suffer from a distribution shift
        problem. This change in temporal distribution is one of the main
        challenges that prevent accurate timeseries forecasting.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, update_stats=True):
        assert x.ndim == 3
        if mode == "norm":
            if update_stats:
                self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
