mod network;
mod utils;
mod data_importing;

use network::Network;
use network::Neuron;

//Todo: expand for color images
//Todo: add convolution
    // convolution layer
    // pooling layer
    // expand backpropagation for convolution layer
fn main() {

    let args = std::env::args();
    let mut args = args.skip(1);
    let iteration_count = args.next().unwrap().parse::<usize>().or_else(|_| Err("Invalid number of iterations")).unwrap();
    // Initialize weights and biases
    let mut network = Network::new(vec![
        vec![Neuron::new(784); 16],
        vec![Neuron::new(16); 16],
        vec![Neuron::new(16); 10]
    ]);
    
    //read all data
    let (training_images, training_labels, testing_images, testing_labels) = data_importing::read_mnist_data();

    //test network before training
    let mut outputs = vec![];
    for image in testing_images.iter() {
        outputs.push(network.run(image));
    }
    println!("before training");
    println!("{}% accuracy", utils::calculate_accuracy(testing_labels.clone(), outputs.clone()) * 100.0);
    println!("mean squared error is: {}", utils::mse(&testing_labels, &outputs));
    network.reset_network();
    
    //train network
    network.train(&training_images, &training_labels, iteration_count);
    
    
    //test network after training
    let mut outputs = vec![];
    for image in testing_images.iter() {
        outputs.push(network.run(image));
    }
    println!("after training");
    println!("{}% accuracy", utils::calculate_accuracy(testing_labels.clone(), outputs.clone()) * 100.0);
    println!("average mean squared error is: {}", utils::mse(&testing_labels, &outputs));
}