# Oddity: Time Series Anomaly Detection

Oddity is a time series anomaly detection tool for Python, implemented in Rust. Oddity is capable of learning trend, global seasonality and even local seasonality from time series data, and works best in these situations.

Being written in Rust, Oddity is incredibly fast and can generally fit to even a few thousand time steps in minimal time.

Oddity also provides a few other tools along with anomaly detection, such as: 

- STL decomposition
- gaussian process fitting
- gaussian distribution fitting
- Periodicity inference

More functionality along with general optimizations will be added in the future.
