import numpy as np

from . import oddity
from .collections import TimeSeries

from typing import *


class Oddity:
    """
    Oddity anomaly detector

    Parameters
    ----------
    params: Dictionary of the parameters and settings.
    """

    def __init__(self, params: dict=None):
        self.params = {
            'trend':
                {'kernel': 'rbf',
                 'l': 10,
                 'sigma_y': 2.0},
            'seasonal':
                {'kernel': 'periodic',
                 'sigma_y': 0.25}
        }

        if params is not None:
            self.params = params

        self.trend_mu, self.trend_cov = None, None
        self.seasonal_mu, self.seasonal_cov = None, None

    def fit(self, data: Union[List, np.ndarray, oddity.TimeSeries]):
        """
        Fits the Oddity detector on the data.

        Parameters
        ----------
        data: time series data.
            Data can be either List, numpy ndarray, rust oddity.TimeSeries or TimeSeries

        """

        data = TimeSeries(data)

        self.trend_mu, self.trend_cov = oddity.gprfit(data.series,
                                                      self.params['trend']['sigma_y'],
                                                      self.params['trend'])

        self.seasonal_mu, self.seasonal_cov = oddity.gprfit(
            oddity.timeseries((np.array(data.data) - np.array(self.trend_mu).reshape(-1, 1))),
            self.params['seasonal']['sigma_y'],
            self.params['seasonal'])


    @property
    def mu(self) -> np.ndarray:
        """
        The posterior mean for every point in the time series

        Returns
        -------
        numpy ndarray
            The posterior mean for every point in the time series

        """

        return np.array(self.seasonal_mu) + np.array(self.trend_mu)

    @property
    def cov(self) -> np.ndarray:
        """
        Seasonality covariance matrix

        Returns
        -------
        numpy ndarray
            Seasonality covariance matrix
        """

        return np.array(self.seasonal_cov)
