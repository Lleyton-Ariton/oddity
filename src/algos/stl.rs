use std::any::Any;

extern crate num;
use num::complex::Complex;
use std::f64::consts::PI;

use std::collections::VecDeque;
use crate::collections::matrix::Matrix;
use crate::collections::timeseries::TimeSeries;

const I: Complex<f64> = Complex { re: 0.0, im: 1.0 };

fn complex_abs(number: Complex<f64>) -> f64 {
    (number.re.powi(2) + number.im.powi(2)).sqrt()
}

pub fn fft(x_data: &TimeSeries) -> TimeSeries {

    let input = &x_data.data
        .iter()
        .map(|x|Complex::from(x[0] as f64))
        .collect::<Vec<Complex<f64>>>()[..];

    fn fft_inner(buf_a: &mut [Complex<f64>], buf_b: &mut [Complex<f64>], n: usize, step: usize) {

        if step >= n {
            return;
        }

        fft_inner(buf_b, buf_a, n, step * 2);
        fft_inner(&mut buf_b[step..], &mut buf_a[step..], n, step * 2);

        let (left, right) = buf_a.split_at_mut(n / 2);

        for i in (0..n).step_by(step * 2) {
            let t = (-I * PI * (i as f64) / (n as f64)).exp() * buf_b[i + step];
            left[i / 2] = buf_b[i] + t;
            right[i / 2] = buf_b[i] - t;
        }
    }

    let n_orig = input.len();
    let n = n_orig.next_power_of_two();

    let mut buf_a = input.to_vec();
    buf_a.append(&mut vec![Complex { re: 0.0, im: 0.0 }; n - n_orig]);

    let mut buf_b = buf_a.clone();
    fft_inner(&mut buf_a, &mut buf_b, n, 1);

    TimeSeries::new(buf_a
        .into_iter()
        .map(|x|vec![complex_abs(x) as f32])
        .collect::<Vec<Vec<f32>>>()
    )
}

pub fn sma(x_data: &TimeSeries, period: usize) -> TimeSeries {
    let mut moving_average: Vec<Vec<f32>>= Vec::new();
    let mut period_deque: VecDeque<&Vec<f32>> = VecDeque::with_capacity(period);
    let mut i = 0;

    if period > x_data.len() {
        panic!("period can not be greater than length of the time series!")
    }

    loop {
        if i >= x_data.len() {
            return TimeSeries::new(moving_average);
        }

        period_deque.push_back(&x_data.data[..][i]);

        if period_deque.len() == period {
            moving_average.push(
                vec![period_deque
                    .iter()
                    .map(|x|x[0])
                    .sum::<f32>() / period as f32
                ]);
            period_deque.pop_front();
        }

        i += 1;
    }
}

pub fn stl(x_data: TimeSeries, period: &dyn Any) -> (TimeSeries, TimeSeries, TimeSeries) {
    let trend = sma(&x_data, (x_data.len() as f32 * 0.2) as usize);
    let detrended = x_data.clone() - trend;

    let estimated_period = {
        if period.is::<usize>() {
            *period.downcast_ref::<usize>().unwrap()
        } else {
            (fft(&detrended).data[0][0] / x_data.len() as f32).round() as usize
        }
    };

    if estimated_period < 1 {
        panic!("period can not be less than 1!")
    }

    let matrix = Matrix {
        iterators: detrended.data
            .chunks(estimated_period)
            .filter(|x|x.len() == estimated_period)
            .into_iter()
            .map(|v| v.into_iter())
            .collect()
    };

    let mut seasonality: Vec<f32> = Vec::new();
    for column in matrix {
        seasonality.push(column
            .iter()
            .map(|x|x[0])
            .sum::<f32>() / column.len() as f32
        );
    }

    seasonality = seasonality
        .repeat(x_data.len() / estimated_period as usize)[..detrended.len()]
        .to_vec();

    let series_seasonality = TimeSeries::new(seasonality
        .iter()
        .map(|x|vec![*x])
        .collect::<Vec<Vec<f32>>>());


    let residual = detrended.clone() - series_seasonality.clone();
    (detrended, series_seasonality, residual)
}

pub fn time_series_outliers(x_data: TimeSeries) -> Vec<(usize, f32)> {
    let mean = x_data.mean();
    let std = x_data.std();

    let upper_bound = mean + 3.0 * std;
    let lower_bound = mean - 3.0 * std;

    x_data.data
        .iter()
        .enumerate()
        .map(|x|(x.0, x.1[0]))
        .filter(|x|x.1 >= upper_bound || x.1 <= lower_bound)
        .collect::<Vec<(usize, f32)>>()
}
