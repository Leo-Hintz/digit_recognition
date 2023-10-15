extern crate rand;
use rand::{Rng, rngs::ThreadRng};

use crate::utils;
type Layer = Vec<Neuron>;
type Matrix<T> = Vec<Vec<T>>;
static mut LEARNING_RATE: f64 = 0.01;
const MOMENTUM: f64 = 0.8;
const BATCH_SIZE : usize = 20;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Network {
        Network { layers }
    }

    pub fn initialize_weights(&mut self, rng: &mut ThreadRng) {
        for layer in self.layers.iter_mut() {
            for neuron in layer.iter_mut() {
                for weight in neuron.weights.iter_mut() {
                    *weight = rng.gen_range(-1.0..1.0);
                }
                neuron.bias = 0.0;
            }
        }
    }
    
    pub fn train(&mut self, inputs: &Matrix<f64>, expected: &Matrix<f64>, iterations: usize) {

        let training_size = inputs.len();
        for i in 0..iterations {
            println!("iteration: {}", i);
            for i in 0..training_size / BATCH_SIZE {
                let mut batch_inputs = vec![];
                let mut batch_expected = vec![];
                for j in 0..BATCH_SIZE {
                    batch_inputs.push(inputs[i * BATCH_SIZE + j].clone());
                    batch_expected.push(expected[i * BATCH_SIZE + j].clone());
                }
                //run network and calculate cumulative output for all neurons
                for single_input in batch_inputs.iter() {
                    self.run(single_input);
                }
                let batch_expected_transposed = utils::transpose(batch_expected.clone());
                self.propagate_backward(&batch_inputs, &batch_expected_transposed);
                //reset outputs
                self.reset_network();   
            }
            let mut outputs = vec![];
            for single_input in inputs.iter() {
                outputs.push(self.run(single_input));
            }
        }
    }
    
    pub fn reset_network(&mut self) {
        for layer in self.layers.iter_mut() {
            for neuron in layer.iter_mut() {
                neuron.outputs.clear();
            }
        }
    }   

    pub fn run(&mut self, input_data: &Vec<f64>) -> Vec<f64> {
        let mut to_compute = input_data.clone();
        for layer in self.layers.iter_mut() {
            let mut layer_output = vec![];            
            for neuron in layer.iter_mut() {
                layer_output.push(neuron.calculate_output(&to_compute));
            }
            to_compute = layer_output.clone();
        }
        to_compute.clone()
    }

   fn propagate_backward(&mut self, inputs : &Matrix<f64>, expected_output_vectors: &Matrix<f64>) {
        
        //calculate derivatives for output layer (probably correct because mse is actaully smaller after training)
        let training_size = inputs.len();
        let mut previous_layer_derivatives = vec![];
        for (neuron, expected_outputs) in self.layers.last().unwrap().iter().zip(expected_output_vectors.iter()) {
            let mut neuron_mse_derivatives = vec![];
            for (expected, actual) in expected_outputs.iter().zip(neuron.outputs.iter()) {
                neuron_mse_derivatives.push(utils::mse_derivative(*expected, *actual));
            }
            previous_layer_derivatives.push(neuron_mse_derivatives);
        }
        //perform backpropagation
        for i in (0..self.layers.len()).rev() {

            //calculate weight derivatives for current layer (identical for every neuron in layer bc. all take same inputs)
            let prev_layer_outputs = if i > 0 {
                utils::transpose(self.layers[i - 1]
                    .iter()
                    .map(|neuron|{ neuron.outputs.clone() })
                    .collect())
            } else {
                inputs.clone()
            };

            //Initialize input gradients for current layer (used in next iteration)
            let mut current_input_gradients = vec![vec![0.0; training_size]; self.layers[i][0].weights.len()]; 
            for j in 0..self.layers[i].len() {
                let mut neuron_weight_gradients = vec![0.0; self.layers[i][j].weights.len()];
                let mut bias_gradient = 0.0;

                for k in 0..training_size {
                    //calculate derivative for sigmoid function
                    //Source: https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
                    let sigmoid_derivative = self.layers[i][j].outputs[k] * (1.0 - self.layers[i][j].outputs[k]);

                    //calculate weight, input and bias gradients
                    for (weight_gradient, prev_layer_output) in neuron_weight_gradients.iter_mut().zip(prev_layer_outputs[k].iter()) {
                        *weight_gradient += prev_layer_output * sigmoid_derivative * previous_layer_derivatives[j][k];
                    }
                    bias_gradient -= sigmoid_derivative * previous_layer_derivatives[j][k];
                }

                //update weights and biases
                let prev_bias_gradient = self.layers[i][j].bias_gradient;
                let prev_weight_gradients = self.layers[i][j].weight_gradients.clone();

                self.layers[i][j].previous_weights = self.layers[i][j].weights.clone();
                self.layers[i][j].previous_bias = self.layers[i][j].bias;

                self.layers[i][j].weight_gradients = neuron_weight_gradients.clone();
                self.layers[i][j].bias_gradient = bias_gradient;

                for k in 0..neuron_weight_gradients.len() {
                    self.layers[i][j].previous_weights[k] = self.layers[i][j].weights[k];
                    unsafe {
                        self.layers[i][j].weights[k] -= LEARNING_RATE * (neuron_weight_gradients[k] + prev_weight_gradients[k] * MOMENTUM);
                    }
                }
                unsafe {
                    self.layers[i][j].previous_bias = self.layers[i][j].bias;
                    self.layers[i][j].bias -= LEARNING_RATE * (bias_gradient + prev_bias_gradient * MOMENTUM);
                }
                for k in 0..training_size {
                    let sigmoid_derivative = self.layers[i][j].outputs[k] * (1.0 - self.layers[i][j].outputs[k]);
                    for (input_gradient, neuron_weight) in current_input_gradients.iter_mut().zip(self.layers[i][j].weights.iter()) {
                        input_gradient[k] += neuron_weight * sigmoid_derivative * previous_layer_derivatives[j][k];
                    }
                }
            }
            //update previous derivatives
            previous_layer_derivatives = current_input_gradients;
        }
    }

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }
}   

 
#[derive(Clone)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    outputs: Vec<f64>,
    weight_gradients: Vec<f64>,
    bias_gradient: f64,
    previous_weights: Vec<f64>,
    previous_bias: f64,
}

impl Neuron {
    pub fn new(input_count: usize) -> Neuron {
        Neuron {
            weights: vec![1.0; input_count],
            bias: 0.0,
            outputs: vec![],
            weight_gradients: vec![0.0; input_count],
            bias_gradient: 0.0,
            previous_weights: vec![1.0; input_count],
            previous_bias: 0.0,
        }
    }

    pub fn calculate_output(&mut self, inputs: &Vec<f64>) -> f64 {
        let mut output = 0.0;
        for (weight, input) in self.weights.iter().zip(inputs.iter()) {
            output += weight * input;
        }
        output -= self.bias;
        let result = utils::sigmoid(output);
        self.outputs.push(result);
        result
    }

    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }
}
