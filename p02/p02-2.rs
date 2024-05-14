use std::fs::File;
use std::io::{BufRead, BufReader};

fn mean(values: Vec<f64>) -> f64 {
    return values.iter().sum::<f64>() / values.len() as f64;
}

fn covariance(x: Vec<f64>, mean_x: f64, y: Vec<f64>, mean_y: f64) -> f64 {
    let mut covar = 0.0;
    for i in 0..x.len() {
        covar += (x[i] - mean_x) * (y[i] - mean_y);
    }
    return covar;
}

fn variance(list: Vec<f64>, mean: f64) -> f64 {
    return list.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
}
fn coefficient(covar: f64, var: f64, mean_x: f64, mean_y: f64) -> (f64, f64) {
    let b1 = covar / var;
    let b0 = mean_y - (b1 * mean_x);
    return (b1, b0);
}
fn load_data(dataset: &str) -> (Vec<f64>, Vec<f64>) {
    let mut init = 0;
    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    let file = File::open(dataset).expect("file not found");

    let content = BufReader::new(file);

    for line in content.lines() {
        if init == 0 {
            init += 1;
            continue;
        }
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_terminator(",").collect();
        let row0 = parts[0]
            .chars()
            .filter(|&c| c != '\\' && c != '"')
            .collect::<String>()
            .parse::<f64>()
            .unwrap();

        let row1 = parts[1]
            .chars()
            .filter(|&c| c != '\\' && c != '"')
            .collect::<String>()
            .parse::<f64>()
            .unwrap();

        x.push(row0);
        y.push(row1);
    }
    return (x, y);
}
fn split_dataset(x: Vec<f64>, y: Vec<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x_train: Vec<f64> = Vec::new();
    let mut x_test: Vec<f64> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    let mut y_test: Vec<f64> = Vec::new();

    let training_size = (x.len() as f64 * 0.8).round() as usize;

    x_train = x[0..training_size].to_vec();
    x_test = x[training_size..].to_vec();
    y_train = y[0..training_size].to_vec();
    y_test = y[training_size..].to_vec();

    return (x_train, x_test, y_train, y_test);
}
fn predict(b0: f64, b1: f64, test_x: Vec<f64>) -> Vec<f64> {
    let mut predicted_y: Vec<f64> = Vec::new();
    for i in test_x {
        predicted_y.push(b0 + b1 * i);
    }
    return predicted_y;
}
fn rmse(predicted_y: Vec<f64>, test_y: Vec<f64>) -> f64 {
    let mut error = 0.0;
    for i in 0..predicted_y.len() {
        error = (predicted_y[i] - test_y[i]).powi(2);
        //        let sum_error = predicted_y[i] - test_y[i];
        //      error += sum_error.powi(2);
    }
    let mean_error = error / test_y.len() as f64;
    return mean_error.sqrt();
}

fn main() {
    let data = load_data("./data/dataset.csv");
    let (x, y) = data;

    let mean_x = mean(x.clone());
    let mean_y = mean(y.clone());
    let covariance = covariance(x.clone(), mean_x, y.clone(), mean_y);
    let var = variance(x.clone(), mean_x);

    let (x_train, x_test, y_train, y_test) = split_dataset(x, y);
    let (b1, b0) = coefficient(covariance, var, mean_x, mean_y);

    println!("Coefficients");
    println!("B0: {}", b0);
    println!("B1: {}", b1);

    let predicted_y = predict(b0, b1, x_test);

    let root_mean = rmse(predicted_y, y_test);

    println!("Linear Regression Model without framework");
    println!("Root Mean Squared Error: {}", root_mean);

    let new_y = b0 + b1 * 70.0;

    println!("Prediction for x=70: {}", new_y);
}
