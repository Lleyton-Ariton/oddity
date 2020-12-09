#![allow(dead_code)]

use nalgebra::DMatrix;

use crate::gaussian::kernels::*;
use crate::collections::matrix::inverse_matrix;

use ndarray::prelude::*;

#[derive(Clone)]
pub struct Posterior {
    pub mu_s: Array<f32, Dim<[usize; 1]>>,
    pub cov_s: Array<f32, Dim<[usize; 2]>>,
    pub sigma_y: f32
}

impl Posterior {
    pub fn new(sigma_y: f32) -> Posterior {
        Posterior {
            mu_s: Array::zeros((1,)),
            cov_s: Array::zeros((1, 1)),

            sigma_y
        }
    }

    fn compute_kernel(&self, x: &Vec<Vec<f32>>, x_prime: &Vec<Vec<f32>>, kernel: Kernel) -> Vec<Vec<f32>> {
        match kernel {
            Kernel::RBF {l, sigma} => Kernels::rbf(x, x_prime, l, sigma),
            Kernel::Periodic {l, sigma, p} => Kernels::periodic(x, x_prime, l, sigma, p),
            Kernel::LocallyPeriodic {l, sigma, p} => Kernels::locally_periodic(x, x_prime, l, sigma, p),
        }
    }

    fn compute_kernel_flattened(&self, x: &Vec<Vec<f32>>, x_prime: &Vec<Vec<f32>>, kernel: Kernel) -> Vec<f32> {
        self.compute_kernel(x, x_prime, kernel).into_iter().flatten().collect::<Vec<f32>>()
    }

    fn compute_kernel_dmatrix(&self, x: &Vec<Vec<f32>>, x_prime: &Vec<Vec<f32>>, kernel: Kernel) -> DMatrix<f32> {
        DMatrix::from_vec(x.len(), x_prime.len(),
                          self.compute_kernel(x, x_prime, kernel).into_iter().flatten().collect())
    }

    pub fn compute(&mut self, x_s: &Vec<Vec<f32>>, x_train: &Vec<Vec<f32>>,
                   y_train: Vec<f32>, kernel: Kernel, sigma_y: f32) {
        let x_s_len = x_s.len();
        let x_train_len = x_train.len();

        let k = Array::from_shape_vec((x_train_len, x_train_len),
                                          self.compute_kernel_flattened(x_train,
                                                                        x_train,
                                                                        kernel)
        ).unwrap() + sigma_y.powi(2) * Array::eye(x_train_len);

        let k_s = Array::from_shape_vec((x_train_len, x_s_len),
                                            self.compute_kernel_flattened(x_train,
                                                                          x_s,
                                                                          kernel)).unwrap();
        let k_ss = Array::from_shape_vec((x_s_len, x_s_len),
                                             self.compute_kernel_flattened(x_s,
                                                                           x_s,
                                                                           kernel)
        ).unwrap() + 1e-8f32 * Array::eye(x_s_len);

        let k_inv = inverse_matrix(&k);

        self.mu_s = k_s.clone().reversed_axes().dot(&k_inv).dot(&Array::from(y_train));
        self.cov_s = k_ss - k_s.clone().reversed_axes().dot(&k_inv).dot(&k_s);
    }

    pub fn mu(&self) -> Array<f32, Dim<[usize; 1]>> {
        self.mu_s.clone()
    }

    pub fn cov(&self) -> Array<f32, Dim<[usize; 2]>> {
        self.cov_s.clone()
    }
}