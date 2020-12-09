#![allow(dead_code)]

use std::f32::consts::{PI, E};
use crate::collections::timeseries::TimeSeries;
use crate::collections::matrix::Matrix;

#[derive(Clone)]
pub struct GaussianDistribution {
    pub mean: f32,
    pub variance: f32
}

impl GaussianDistribution {
    pub fn new() -> GaussianDistribution {
        GaussianDistribution{
            mean: 0.0,
            variance: 0.0
        }
    }

    pub fn with(mean: f32, variance: f32) -> GaussianDistribution {
        GaussianDistribution{
            mean,
            variance
        }
    }

    pub fn vec_fit(&mut self, x_data: Vec<f32>) {
        self.mean = x_data
            .iter()
            .map(|x|x)
            .sum::<f32>();

        self.variance = x_data
            .iter()
            .map(|x|(x - self.mean).powi(2))
            .sum::<f32>()
            .sqrt();
    }

    pub fn fit(&mut self, x_data: &TimeSeries) {
        self.vec_fit(x_data.data.iter().map(|x|x[0]).collect())
    }

    pub fn predict(&self, x: f32) -> f32 {
        let base = 1.0/(self.variance * (2.0 * PI).sqrt());
        let power = -0.5 * ((x - self.mean)/self.variance).powi(2);

        base * E.powf(power)
    }

    pub fn is_anomaly(&self, x: f32, epsilon: f32) -> u32 {
        match self.predict(x) < epsilon {
            true => 1,
            false => 0
        }
    }
}

#[derive(Clone)]
pub struct MultiDistGaussian {
    pub distributions: Vec<GaussianDistribution>
}

impl MultiDistGaussian {

    pub fn new() -> MultiDistGaussian {
        MultiDistGaussian{
            distributions: Vec::new()
        }
    }

    pub fn from_dists(distributions: Vec<GaussianDistribution>) -> MultiDistGaussian {
        MultiDistGaussian{
            distributions
        }
    }

    pub fn fit(&mut self, x_data: Vec<Vec<f32>>) {
        let n_features = x_data[0].len();
        let features = Matrix{
            iterators: x_data.into_iter().map(|x|x.into_iter()).collect()
        };

        if self.distributions.len() != n_features {
            self.distributions = vec![GaussianDistribution::new(); n_features]
        }
        for (dist, feature) in self.distributions
            .iter_mut()
            .zip(features) {

            dist.vec_fit(feature)
        }
    }

    pub fn predict(&self, x: Vec<f32>) -> f32 {
        x.iter()
            .zip(self.distributions.iter())
            .map(|x|x.1.predict(*x.0))
            .product()
    }

    pub fn is_anomaly(&self, x: Vec<f32>, epsilon: f32) -> u32 {
        match self.predict(x) < epsilon {
            true => 1,
            false => 0
        }
    }
}