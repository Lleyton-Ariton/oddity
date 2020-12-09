import numpy as np

from . import oddity
from typing import *


class TimeSeries:
    """
    Time series wrapper around rust oddity.TimeSeries.

    Parameters
    ----------
    data: Time series data.
        Can be a List, numpy ndarray, rust oddity.TimeSeries or TimeSeries.
        Leave None (default) to create an empty TimeSeries
    """

    @staticmethod
    def _dimension_check(data: Union[List, np.ndarray], dimension: int=2) -> bool:
        """Checks the number of dimensions of data

        Parameters
        ----------
        data: Either a list or numpy ndarray.
        dimension: Integer of the number of dimensions data should have

        Returns
        ------
        bool
            Whether or not data has dimension number of dimensions

        """

        if isinstance(data, list):
            try:
                for _ in range(dimension):
                    data = data[0]
                return True

            except IndexError:
                return False
        elif isinstance(data, np.ndarray):
            return len(data.shape) == dimension

        return False

    def __init__(self, data: Union[List, np.ndarray, oddity.TimeSeries, "TimeSeries"]=None):
        if data is None:
            self.series = oddity.TimeSeries.empty()
        else:
            if not self._dimension_check(data, 2):
                raise RuntimeError('Time series data must be 2D')

            options_ = {
                list: oddity.timeseries(data),
                oddity.TimeSeries: data,
            }

            # Exceptions
            if isinstance(data, TimeSeries):
                self.series = data.series

            if isinstance(data, np.ndarray):
                self.series = oddity.timeseries(data.tolist())
            else:
                self.series = options_[type(data)]

    @property
    def mean(self) -> float:
        """
        Mean of the time series

        Returns
        -------
        float
            The mean of the time series
        """

        return self.series.mean()

    @property
    def std(self) -> float:
        """
        Standard deviation of the time series

        Returns
        -------
        float
            The standard deviation of the time series
        """

        return self.series.std()

    @property
    def data(self) -> List:
        """
        Gets the data stored in the TimeSeries as a Python list

        Returns
        -------
        list
            The time series
        """

        return self.series.data

    def tolist(self) -> List:
        """
        Gets the data stored in the TimeSeries as a Python list

        Returns
        -------
        list
            The time series
        """

        return self.data

    def append(self, value: Union[int, float]):
        """
        Appends a value at the end of the time series.
        Works identically to Python's builtin 'append' method on lists.

        Parameters
        ----------
        value: Value that will be appended.
            Can be either integer of float
        """

        self.series.push(value)

    def extend(self, values: List[Union[int, float]]):
        """
        Appends every value from an iterator to the end of the time series.
        Works identically to Python's builtin 'extend' method on lists.

        Parameters
        ----------
        values: Iterator of values.
            Items in the iterator can be either integer of float
        """

        [self.series.push(value) for value in values]

    def outliers(self) -> List[Tuple[int, float]]:
        """
        Returns points that are more than 2.5 standard deviations from the mean of the time series

        Returns
        -------
        list[Tuple(int, float)]
            A list of tuples of the outlier points,
            with the index of the point in the time step and it's value, respectively

        """

        return oddity.timeseries_outliers(self.series)

    def __len__(self) -> int:
        return self.series.len()

    def __str__(self):
        return self.data.__str__()

    def __iter__(self):
        for timestep in self.data:
            yield timestep

    def __getitem__(self, item) -> List[float]:
        return self.series.data[item]
