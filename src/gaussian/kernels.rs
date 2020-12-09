#![allow(dead_code)]

use std::f32::consts::{PI, E};

#[derive(Clone)]
pub struct Kernels;

fn exp(value: f32) -> f32 {
    E.powf(value)
}

fn standalone_rbf(x: f32, x_prime: f32, l: f32, sigma: f32) -> f32 {
    sigma.powi(2) * exp(-1.0 * ((x - x_prime).powi(2)/(2.0 * l.powi(2))))
}

fn standalone_periodic(x: f32, x_prime: f32, l: f32, sigma: f32, p: f32) -> f32 {
    sigma.powi(2) * exp(-1.0 * (2.0 * (((PI * (x - x_prime).abs())/p).sin()).powi(2)) / l.powi(2))

}

impl Kernels {
    pub fn rbf(x: &Vec<Vec<f32>>, x_prime: &Vec<Vec<f32>>,
               l: f32, sigma: f32) -> Vec<Vec<f32>> {
        x.iter()
            .map(|m|x_prime.iter().map(|n|
                standalone_rbf(m[0], n[0], l, sigma)
            ).collect())
            .collect::<Vec<Vec<f32>>>()
    }

    pub fn periodic(x: &Vec<Vec<f32>>, x_prime: &Vec<Vec<f32>>,
                    l: f32, sigma: f32, p: f32) -> Vec<Vec<f32>> {
        x.iter()
            .map(|m|x_prime.iter().map(|n|
                standalone_periodic(m[0], n[0], l, sigma, p)
            ).collect())
            .collect::<Vec<Vec<f32>>>()
    }

    pub fn locally_periodic(x: &Vec<Vec<f32>>, x_prime: &Vec<Vec<f32>>,
                            l: f32, sigma: f32, p: f32) -> Vec<Vec<f32>> {
        x.iter()
            .map(|m|x_prime.iter().map(|n|
                standalone_periodic(m[0], n[0], l, sigma, p) * exp(-1.0 * ((m[0] - n[0]).powi(2)/(2.0 * l.powi(2))))
            ).collect()).collect::<Vec<Vec<f32>>>()
    }
}

#[derive(Clone, Copy)]
pub enum Kernel {
    RBF {l: f32, sigma: f32},
    Periodic {l: f32, sigma: f32, p: f32},
    LocallyPeriodic {l: f32, sigma: f32, p: f32}
}