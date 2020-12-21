<br></br>

<p align="center">
  <img src="./imgs/odditylogo.png" alt="Logo"/, width="500", height="160">
</p>

<br></br>

---

<br></br>

# Oddity: Time Series Anomaly Detection

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-light.svg)](https://opensource.org/licenses/)
[![GitHub Release](https://img.shields.io/github/release/Lleyton-Ariton/oddity.svg?style=flat)]()
[![PyPi Version](https://img.shields.io/pypi/v/oddity.svg)](https://pypi.python.org/pypi/oddity/)
[![made-with-rust](https://img.shields.io/badge/Made%20with-Rust-1f425f.svg)](https://www.rust-lang.org/)


Oddity is a time series anomaly detection tool for Python, implemented in Rust. Oddity is capable of learning trend, global seasonality and even local seasonality from time series data, and works best in these situations.

<p align="center">
  <img src="./imgs/oddity-demo-banner.gif" alt="Oddity Demo: flagging severe anomalies"/>
</p>

Being written in Rust, Oddity is incredibly fast and can generally fit to even a few thousand time steps in minimal time.

Oddity also provides a few other tools along with anomaly detection, such as: 

- STL decomposition
- gaussian process fitting
- gaussian distribution fitting
- Periodicity inference

More functionality along with general optimizations will be added in the future.

Currently Oddity is intended to be used on static datasets, however online learning can potentially be implemented by using a rolling/sliding window. With enough hacking, it can potentially also be used for forecasting.

## Oddity Demo

Web app demo of the Oddity engine detecting anomalies in some data sets. The web app was deployed on a google cloud kubernetes cluster open to the public, but will not be forever available due to ressource reasons. 

- A local version of the web app can still be run by follwing the instructions on: [https://github.com/Lleyton-Ariton/oddity-demo](https://github.com/Lleyton-Ariton/oddity-demo)

<p align="center">
  <img src="./imgs/oddity-demo.gif" alt="Oddity Demo: flagging severe anomalies"/>
</p>

## Important Links

The following are some important links for more information:

> PyPi: [https://pypi.org/project/oddity/](https://pypi.org/project/oddity/)

> Oddity Engine (Rust): [https://github.com/Lleyton-Ariton/oddity-engine](https://github.com/Lleyton-Ariton/oddity-engine)

> Oddity Demo: [https://github.com/Lleyton-Ariton/oddity-demo](https://github.com/Lleyton-Ariton/oddity-demo)
