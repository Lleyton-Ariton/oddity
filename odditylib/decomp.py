import numpy as np

from . import oddity
from .collections import TimeSeries

from typing import *


def decompose(data: Union[List, np.ndarray, oddity.TimeSeries, TimeSeries], period: int=None) -> Tuple[TimeSeries,
                                                                                                       TimeSeries,
                                                                                                       TimeSeries]:
    """
    Wrapper around rust oddity.decompose

    Parameters
    ----------
    data: Time series data.
        Can be either List, numpy ndarray, rust oddity.TimeSeries or TimeSeries
    period: Seasonality of the data
        If None (default), the periodicity will be attempted to be inferred.
        Otherwise, period must be an integer of the period of the data.

    Returns
    -------
    tuple
        Tuple of 3 TimeSeries: trend, seasonality and residual, respectively.
    """

    if not isinstance(data, TimeSeries):
        data = TimeSeries(data)

    if period is None:
        return oddity.decompose(data.series)

    return oddity.decompose(data.series, {'period': period})
