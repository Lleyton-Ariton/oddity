#![allow(dead_code)]

use nalgebra::DMatrix;
use ndarray::prelude::*;

#[derive(Clone)]
pub struct Matrix<I> where I: Iterator {
    pub iterators: Vec<I>
}

impl<I, T> Iterator for Matrix<I>
    where I: Iterator<Item = T> {
    type Item = Vec<T>;
    fn next(&mut self) -> Option<Self::Item> {
        let output: Option<Vec<T>> = self.iterators.iter_mut().map(|iter| iter.next()).collect();
        output
    }
}

pub fn dmatrix_to_vec(dmtarix: DMatrix<f32>, shape: (usize, usize)) -> Vec<Vec<f32>> {
    let mut rust_vec: Vec<Vec<f32>> = Vec::with_capacity(shape.1);
    let mut row: Vec<f32> = Vec::with_capacity(shape.1);

    for row_index in 0..shape.0 {
        row = Vec::with_capacity(shape.1);

        row.extend(dmtarix.row(row_index).iter());
        rust_vec.push(row);
    }

    rust_vec
}

pub fn ndarray_to_vec(dmtarix: Array<f32, Dim<[usize; 2]>>, shape: (usize, usize)) -> Vec<Vec<f32>> {
    let mut rust_vec: Vec<Vec<f32>> = Vec::with_capacity(shape.1);
    let mut row: Vec<f32> = Vec::with_capacity(shape.1);

    for row_index in 0..shape.0 {
        row = Vec::with_capacity(shape.1);

        row.extend(dmtarix.row(row_index).iter());
        rust_vec.push(row);
    }

    rust_vec
}

pub fn inverse_matrix(x: &Array<f32, Dim<[usize; 2]>>) -> Array<f32, Dim<[usize; 2]>> {
    let shape = x.shape();

    let axes = x.clone().reversed_axes();
    let mut nalg_matrix = DMatrix::from_iterator(axes.cols(),
                                                 axes.rows(), axes.iter().cloned());
    nalg_matrix.try_inverse_mut();

    Array2::from_shape_vec((shape[0], shape[1]), {
        let mut vec = Vec::with_capacity(shape[0]);
        vec.extend(nalg_matrix.iter());

        vec
    }).unwrap().reversed_axes()
}