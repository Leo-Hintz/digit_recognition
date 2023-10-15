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
