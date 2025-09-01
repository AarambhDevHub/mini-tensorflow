use mini_tensorflow::{Adam, SGD, Tensor};

struct NeuralNetwork {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        NeuralNetwork {
            w1: Tensor::random(vec![input_size, hidden_size]),
            b1: Tensor::zeros(vec![hidden_size]),
            w2: Tensor::random(vec![hidden_size, output_size]),
            b2: Tensor::zeros(vec![output_size]),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // First layer: x @ w1 + b1 (with broadcasting)
        let z1 = x.matmul(&self.w1).add_broadcast(&self.b1);
        let a1 = z1.relu();

        // Second layer: a1 @ w2 + b2 (with broadcasting)
        let z2 = a1.matmul(&self.w2).add_broadcast(&self.b2);
        z2.sigmoid()
    }

    fn loss(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        // Simple mean squared error
        let diff = predictions.sub(targets);
        let squared_diff = diff.mul(&diff);
        squared_diff.mean()
    }
}

fn main() {
    println!("Neural Network Example");

    // Create a simple 2-3-1 network
    let network = NeuralNetwork::new(2, 3, 1);

    // Sample input data
    let x = Tensor::new(vec![0.5, -0.2, 0.3, 0.8, -0.1, 0.6], vec![3, 2]); // 3 samples, 2 features
    let y = Tensor::new(vec![1.0, 0.0, 1.0], vec![3, 1]); // 3 targets

    println!("Input shape: {:?}", x.shape());
    println!("Target shape: {:?}", y.shape());

    // Forward pass
    let predictions = network.forward(&x);
    println!("Predictions: {}", predictions);

    // Compute loss
    let loss = network.loss(&predictions, &y);
    println!("Loss: {:.6}", loss);

    // Demonstrate different activation functions
    println!("\nActivation Functions Demo:");
    let test_input = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    println!("Input: {}", test_input);
    println!("ReLU: {}", test_input.relu());
    println!("Sigmoid: {}", test_input.sigmoid());
    println!("Tanh: {}", test_input.tanh());
    println!("Softmax: {}", test_input.softmax());

    // Show optimizer initialization
    let sgd = SGD::new(0.01);
    let adam = Adam::new(0.001);
    println!("\nOptimizers initialized:");
    println!("SGD learning rate: {}", sgd.learning_rate);
    println!("Adam learning rate: {}", adam.learning_rate);
}
