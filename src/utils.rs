use std::cmp::Ordering;
pub type Matrix<T> = Vec<Vec<T>>;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn mse(expected_outs: &Vec<Vec<f64>>, actual_outs: &Vec<Vec<f64>>) -> f64 {
    let mut result = 0.0;
    for (expected, actual) in expected_outs.iter().zip(actual_outs.iter()) {
        result += error(expected, actual);
    }
    result / expected_outs.len() as f64
}

pub fn transpose<T>(matrix: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!matrix.is_empty());
    let len = matrix[0].len();
    let mut iters: Vec<_> = matrix.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

pub fn mse_derivative(expected: f64, actual: f64) -> f64 {
    2.0 * (actual - expected)
}

pub fn error(expected_outs: &Vec<f64>, actual_outs: &Vec<f64>) -> f64 {
    let mut error = 0.0;
    for (expected, actual) in expected_outs.iter().zip(actual_outs.iter()) {
        error += (actual - expected).powi(2);
    }
    error
}

pub fn calculate_accuracy(testing_labels: Vec<Vec<f64>>, outputs: Vec<Vec<f64>>) -> f64 {
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
    classified_correctly as f64 / classified_count as f64
}
