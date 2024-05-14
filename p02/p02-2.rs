use std::fs::File;
use std::io::{BufRead, BufReader};

type Float64 = f64;

fn mean(values: &Vec<Float64>) -> Float64 {
    return values.iter().sum::<Float64>() / values.len() as Float64;
}

fn covariance(x: &Vec<Float64>, mean_x: Float64, y: &Vec<Float64>, mean_y: Float64) -> Float64 {
    let mut covar = 0.0;
    for i in 0..x.len() {
        covar += (x[i] - mean_x) * (y[i] - mean_y);
    }
    return covar;
}

fn variance(list: &Vec<Float64>, mean: Float64) -> Float64 {
    return list.iter().map(|x| (x - mean).powi(2)).sum::<Float64>();
}
fn coefficient(
    covar: Float64,
    var: Float64,
    mean_x: Float64,
    mean_y: Float64,
) -> (Float64, Float64) {
    let b1 = covar / var;
    let b0 = mean_y - (b1 * mean_x);
    return (b1, b0);
}
fn load_data(dataset: &str) -> (Vec<Float64>, Vec<Float64>) {
    let mut init = 0;
    let mut x: Vec<Float64> = Vec::new();
    let mut y: Vec<Float64> = Vec::new();

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
            .parse::<Float64>()
            .unwrap();

        let row1 = parts[1]
            .chars()
            .filter(|&c| c != '\\' && c != '"')
            .collect::<String>()
            .parse::<Float64>()
            .unwrap();

        x.push(row0);
        y.push(row1);
    }
    return (x, y);
}
fn split_dataset(
    x: &Vec<Float64>,
    y: &Vec<Float64>,
) -> (Vec<Float64>, Vec<Float64>, Vec<Float64>, Vec<Float64>) {
    let mut x_train: Vec<Float64> = Vec::new();
    let mut x_test: Vec<Float64> = Vec::new();
    let mut y_train: Vec<Float64> = Vec::new();
    let mut y_test: Vec<Float64> = Vec::new();

    let training_size = (x.len() as Float64 * 0.8).round() as usize;

    x_train = x[0..training_size].to_vec();
    x_test = x[training_size..].to_vec();
    y_train = y[0..training_size].to_vec();
    y_test = y[training_size..].to_vec();

    return (x_train, x_test, y_train, y_test);
}
fn predict(b0: Float64, b1: Float64, test_x: &Vec<Float64>) -> Vec<Float64> {
    let mut predicted_y: Vec<Float64> = Vec::new();
    for i in test_x {
        predicted_y.push(b0 + b1 * i);
    }
    return predicted_y;
}
fn rmse(predicted_y: &Vec<Float64>, test_y: &Vec<Float64>) -> Float64 {
    let mut error = 0.0;
    for i in 0..predicted_y.len() {
        error = (predicted_y[i] - test_y[i]).powi(2);
    }
    let mean_error = error / test_y.len() as Float64;
    return mean_error.sqrt();
}

fn main() {
    let data = load_data("./data/dataset.csv");
    let (x, y) = data;

    let mean_x = mean(&x);
    let mean_y = mean(&y);
    let covariance = covariance(&x, mean_x, &y, mean_y);
    let var = variance(&x, mean_x);

    let (x_train, x_test, y_train, y_test) = split_dataset(&x, &y);
    let (b1, b0) = coefficient(covariance, var, mean_x, mean_y);

    println!("Coefficients");
    println!("B0: {}", b0);
    println!("B1: {}", b1);

    let predicted_y = predict(b0, b1, &x_test);

    let root_mean = rmse(&predicted_y, &y_test);

    println!("Linear Regression Model without framework");
    println!("Root Mean Squared Error: {}", root_mean);

    let new_y = b0 + b1 * 70.0;

    println!("Prediction for x=70: {}", new_y);
}
