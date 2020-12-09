use crate::gaussian::kernels::Kernel;
use crate::gaussian::posterior::Posterior;

use crate::collections::timeseries::TimeSeries;
use crate::collections::matrix::ndarray_to_vec;

#[derive(Clone)]
pub struct GaussianProcessRegression {
    pub kernel: Kernel,
    pub mu: Vec<f32>,
    pub cov: Vec<Vec<f32>>
}

impl GaussianProcessRegression {
    pub fn new(kernel: Kernel) -> GaussianProcessRegression {
        GaussianProcessRegression {
            kernel,
            mu: Vec::new(),
            cov: Vec::new(),
        }
    }

    pub fn compute(&mut self, x_s: &Vec<Vec<f32>>, x_train: &Vec<Vec<f32>>,
                   y_train: &TimeSeries, sigma_y: f32) {
        let mut posterior = Posterior::new(sigma_y);
        posterior.compute(x_s, x_train, y_train.as_flat_vec(), self.kernel, sigma_y);

        let shape = posterior.cov_s.shape();

        self.mu = posterior.mu().into_raw_vec();
        self.cov = ndarray_to_vec(posterior.cov(), (shape[0], shape[1]));
    }

    pub fn fit(&mut self, y_train: &TimeSeries, sigma_y: f32) {
        let x_train: Vec<Vec<f32>> = (0..y_train.len())
            .map(|x|vec![x as f32])
            .collect::<Vec<Vec<f32>>>();

        let mut posterior = Posterior::new(sigma_y);
        posterior.compute(&x_train, &x_train, y_train.as_flat_vec(), self.kernel, sigma_y);

        let shape = posterior.cov_s.shape();

        self.mu = posterior.mu().into_raw_vec();
        self.cov = ndarray_to_vec(posterior.cov(), (shape[0], shape[1]));
    }
}

impl Default for GaussianProcessRegression {
    fn default() -> GaussianProcessRegression {
        GaussianProcessRegression::new(Kernel::RBF {l: 1.0, sigma: 1.0})
    }
}