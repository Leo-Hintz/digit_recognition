mod network;
mod utils;
mod parsing;

use std::cmp::Ordering;

use network::Network;
use network::Neuron;

use rand::thread_rng;

fn main() {
    // Initialize weights and biases
    let mut network = Network::new(vec![
        vec![Neuron::new(784); 16],
        vec![Neuron::new(16); 16],
        vec![Neuron::new(16); 10]
    ]);
    network.initialize_weights(&mut thread_rng());

    //read all data
    let (training_images, training_labels, testing_images, testing_labels) = parsing::read_data();

    //test network
    let mut outputs = vec![];
    for image in testing_images.iter() {
        outputs.push(network.run(image));
    }

    println!("{}", utils::mse(&testing_labels, &outputs));
    println!("train network");
    network.reset_network();
    
    //train network
    network.train(&training_images, &training_labels, 30);
    
    
    //test network
    let mut outputs = vec![];
    for image in testing_images.iter() {
        outputs.push(network.run(image));
    }
    let mut classified_correctly = 0;
    let mut classified_count = 0;
    
    for (expected, actual) in testing_labels.iter().zip(outputs.iter()) {
        if expected.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index) 
        == 
        actual.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index) {
            classified_correctly += 1;
        }

        classified_count += 1;
    }
    println!("classified correctly: {} out of {}", classified_correctly, classified_count);
    println!("{}", utils::mse(&testing_labels, &outputs));
}
