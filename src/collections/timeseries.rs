use pyo3::prelude::*;
use std::ops::{Sub, Add, Mul, Div};

#[pyclass]
#[derive(Debug, Clone)]
pub struct TimeSeries {
    #[pyo3(get)]
    pub data: Vec<Vec<f32>>
}

#[pymethods]
impl TimeSeries {
    #[new]
    pub fn new(data: Vec<Vec<f32>>) -> TimeSeries {
        TimeSeries{
            data
        }
    }

    #[staticmethod]
    pub fn empty() -> TimeSeries {
        TimeSeries{
            data: Vec::new()
        }
    }

    pub fn as_flat_vec(&self) -> Vec<f32> {
        self.data.iter().map(|x|x[0]).collect()
    }

    pub fn push(&mut self, value: f32) {
        self.data.push(vec![value])
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn mean(&self) -> f32 {
        self.data.iter().map(|x|x[0]).sum::<f32>() / self.len() as f32
    }

    pub fn std(&self) -> f32 {
        let mean = self.mean();
        (self.data
            .iter()
            .map(|x|(x[0] - mean).powi(2))
            .sum::<f32>() / self.data.len() as f32
        ).sqrt()
    }
}

impl Sub for TimeSeries {
    type Output = TimeSeries;

    fn sub(self, rhs: Self) -> Self::Output {
        TimeSeries::new(self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|x| vec![x.0[0] - x.1[0]])
            .collect::<Vec<Vec<f32>>>())
    }
}


impl Add for TimeSeries {
    type Output = TimeSeries;

    fn add(self, rhs: Self) -> Self::Output {
        TimeSeries::new(self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|x| vec![x.0[0] + x.1[0]])
            .collect::<Vec<Vec<f32>>>())
    }
}


impl Mul for TimeSeries {
    type Output = TimeSeries;

    fn mul(self, rhs: Self) -> Self::Output {
        TimeSeries::new(self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|x| vec![x.0[0] * x.1[0]])
            .collect::<Vec<Vec<f32>>>())
    }
}

impl Div for TimeSeries {
    type Output = TimeSeries;

    fn div(self, rhs: Self) -> Self::Output {
        TimeSeries::new(self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|x| vec![x.0[0] / x.1[0]])
            .collect::<Vec<Vec<f32>>>())
    }
}
