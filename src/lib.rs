mod algos;
mod collections;
mod gaussian;

use collections::timeseries::TimeSeries;

use pyo3::prelude::*;
use pyo3::prelude::PyModule;
use pyo3::{wrap_pyfunction, PyResult, Python};
use pyo3::types::PyDict;

use crate::algos::stl::{stl, time_series_outliers};
use crate::algos::gaussian::GaussianProcessRegression;
use crate::gaussian::kernels::Kernel;

#[pyfunction]
pub fn timeseries(data: Vec<Vec<f32>>) -> TimeSeries {
    TimeSeries::new(data)
}

#[pyfunction]
pub fn decompose(data: TimeSeries, py_kwargs: Option<&PyDict>) -> (TimeSeries, TimeSeries, TimeSeries) {
    if py_kwargs.is_some() {
        match py_kwargs.unwrap().get_item("period") {
            Some(item) => stl(data, &item.extract::<usize>().unwrap()),
            None => stl(data, &Option::<usize>::None)
        }
    }

    else {
        stl(data, &Option::<usize>::None)
    }
}

#[pyfunction]
pub fn timeseries_outliers(data: TimeSeries) -> Vec<(usize, f32)> {
    time_series_outliers(data)
}

#[pyfunction]
pub fn gprfit(data: TimeSeries, sigma_y: f32, py_kwargs: Option<&PyDict>) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut gp: GaussianProcessRegression = {
        if py_kwargs.is_some() {
            let l: f32 = match py_kwargs.unwrap().get_item("l") {
                Some(item) => item.extract::<f32>().unwrap(),
                None => 1.0
            };

            let sigma: f32 = match py_kwargs.unwrap().get_item("sigma") {
                Some(item) => item.extract::<f32>().unwrap(),
                None => 1.0
            };

            let p = match py_kwargs.unwrap().get_item("period") {
                Some(item) => item.extract::<f32>().unwrap(),
                None => 12.0
            };

            GaussianProcessRegression::new(
                match py_kwargs.unwrap().get_item("kernel") {
                    Some(item) => match item.extract::<&str>().unwrap() {
                        "rbf" => Kernel::RBF {l, sigma},
                        "periodic" => Kernel::Periodic {l, sigma, p},
                        "locally periodic" => Kernel::LocallyPeriodic {l, sigma, p},
                        _ => Kernel::RBF{l, sigma}
                    }
                    None => Kernel::RBF{l, sigma}
                })
        }

        else {
            GaussianProcessRegression::default()
        }
    };

    gp.fit(&data, sigma_y);

    (gp.mu, gp.cov)
}

#[pymodule]
fn oddity(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TimeSeries>()?;

    m.add_function(wrap_pyfunction!(timeseries, m)?)?;
    m.add_function(wrap_pyfunction!(decompose, m)?)?;
    m.add_function(wrap_pyfunction!(timeseries_outliers, m)?)?;

    m.add_function(wrap_pyfunction!(gprfit, m)?)?;

    Ok(())
}