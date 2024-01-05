#![allow(dead_code)]
use crate::utils::Matrix;
use rand::{thread_rng, rngs::ThreadRng, Rng};
type Layer = Vec<FeatureMap>;
pub struct ConvolutionalNetwork {
    layers : Vec<Layer>,
}

impl ConvolutionalNetwork {
    pub fn new(layers : Vec<Layer>) -> ConvolutionalNetwork {
        let mut layer: ConvolutionalNetwork = ConvolutionalNetwork {
            layers
        };
        layer.initialize_weights(&mut thread_rng());
        layer
    }

    pub fn run(&self, input : &Matrix<f64>) -> Matrix<f64> {
        let mut to_compute = input.clone();
        for layer in self.layers.iter() {
            let mut result = vec![];
            for filter in layer.iter() {
                let mut filter_result = filter.convolve(&to_compute);
                result.append(&mut filter_result);
            }
            to_compute = result.clone();
        }
        to_compute.clone()
    }

    fn initialize_weights(&mut self, rng: &mut ThreadRng) {
        for layer in self.layers.iter_mut() {
            for filter in layer.iter_mut() {
                for row in filter.kernel.iter_mut() {
                   row.iter_mut().for_each(|weight| *weight = rng.gen_range(-1.0..1.0));
                }
            }
        }    
    }
}

pub struct FeatureMap {
    kernel : Matrix<f64>,
    bias : f64,
}

impl FeatureMap {
    pub fn new(kernel : Matrix<f64>, bias : f64) -> FeatureMap {
        FeatureMap {
            kernel,
            bias
        }
    }
    pub fn convolve(&self, input : &Matrix<f64>) -> Matrix<f64> {
        let mut result = vec![];

        //assuming kernel with no padding and stride of 1 and one dimensional input
        for i in 0..input.len() - self.kernel.len() + 1 {
            let mut row = vec![];
            for j in 0..input[0].len() - self.kernel[0].len() + 1 {
                let sum = self.calculate_pixel(input, i, j);
                row.push(sum - self.bias);
            }
            result.push(row);
        }
        result
    }

    fn calculate_pixel(&self, input : &Matrix<f64>, i : usize, j : usize) -> f64 {
        let mut sum = 0.0;
        for k in 0..self.kernel.len() {
            for l in 0..self.kernel[0].len() {
                sum += input[i + k][j + l] * self.kernel[k][l];
            }
        }
        sum
    }
}

pub struct PoolingFilter {
    width: usize,
    height: usize
}

impl PoolingFilter {
    pub fn max_pooling(&self, input: &Matrix<f64>) {
        let image_width = input.len();
        let image_height = input[0].len();
        for _ in 0..image_width / self.width {
            for _ in 0..image_height / self.height {
                
            }
        }
    }
}