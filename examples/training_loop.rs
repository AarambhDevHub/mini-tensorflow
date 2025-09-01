use mini_tensorflow::{Adam, Optimizer, SGD, Tensor};

struct LinearModel {
    weight: Tensor,
    bias: Tensor,
}

impl LinearModel {
    fn new(input_dim: usize) -> Self {
        LinearModel {
            weight: Tensor::random(vec![input_dim, 1]),
            bias: Tensor::zeros(vec![1]),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // Use add_broadcast to handle bias addition properly
        x.matmul(&self.weight).add_broadcast(&self.bias)
    }

    fn mse_loss(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        let diff = predictions.sub(targets);
        let squared = diff.mul(&diff);
        squared.mean()
    }

    // Compute numerical gradients (finite difference approximation)
    fn compute_gradients(&self, x: &Tensor, y: &Tensor, h: f64) -> (Tensor, Tensor) {
        let current_loss = self.mse_loss(&self.forward(x), y);

        // Gradient for weight
        let mut weight_grad_data = Vec::new();
        for i in 0..self.weight.size() {
            let mut weight_plus = self.weight.clone();
            weight_plus.data[i] += h;
            let model_plus = LinearModel {
                weight: weight_plus,
                bias: self.bias.clone(),
            };
            let loss_plus = model_plus.mse_loss(&model_plus.forward(x), y);

            let gradient = (loss_plus - current_loss) / h;
            weight_grad_data.push(gradient);
        }
        let weight_grad = Tensor::new(weight_grad_data, self.weight.shape.clone());

        // Gradient for bias
        let mut bias_grad_data = Vec::new();
        for i in 0..self.bias.size() {
            let mut bias_plus = self.bias.clone();
            bias_plus.data[i] += h;
            let model_plus = LinearModel {
                weight: self.weight.clone(),
                bias: bias_plus,
            };
            let loss_plus = model_plus.mse_loss(&model_plus.forward(x), y);

            let gradient = (loss_plus - current_loss) / h;
            bias_grad_data.push(gradient);
        }
        let bias_grad = Tensor::new(bias_grad_data, self.bias.shape.clone());

        (weight_grad, bias_grad)
    }
}

fn main() {
    println!("Training Loop Example");

    // Generate synthetic data: y = 2*x + 1 + noise
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_data = vec![3.1, 4.9, 7.2, 9.1, 10.8, 13.2, 15.0, 16.9]; // 2*x + 1 + small noise

    let x_train = Tensor::new(x_data, vec![8, 1]);
    let y_train = Tensor::new(y_data, vec![8, 1]);

    println!("Training data:");
    println!("X: {}", x_train);
    println!("Y: {}", y_train);

    // Create model and optimizer
    let mut model = LinearModel::new(1);
    let mut sgd_optimizer = SGD::new(0.01);

    println!("\nInitial parameters:");
    println!("Weight: {}", model.weight);
    println!("Bias: {}", model.bias);

    // Training with SGD
    println!("\n=== Training with SGD ===");

    for epoch in 0..20 {
        // Forward pass
        let predictions = model.forward(&x_train);
        let loss = model.mse_loss(&predictions, &y_train);

        if epoch % 5 == 0 || epoch < 3 {
            println!("Epoch {}: Loss = {:.6}", epoch + 1, loss);
        }

        // Compute gradients using numerical differentiation
        let (weight_grad, bias_grad) = model.compute_gradients(&x_train, &y_train, 1e-5);

        // Update parameters using optimizer
        sgd_optimizer.step(
            &mut [&mut model.weight, &mut model.bias],
            &[&weight_grad, &bias_grad],
        );
    }

    println!(
        "SGD Final Loss: {:.6}",
        model.mse_loss(&model.forward(&x_train), &y_train)
    );
    println!("SGD Final Weight: {}", model.weight);
    println!("SGD Final Bias: {}", model.bias);

    // Reset model for Adam comparison
    let mut model_adam = LinearModel::new(1);
    let mut adam_optimizer = Adam::new(0.1);

    println!("\n=== Training with Adam ===");

    for epoch in 0..20 {
        // Forward pass
        let predictions = model_adam.forward(&x_train);
        let loss = model_adam.mse_loss(&predictions, &y_train);

        if epoch % 5 == 0 || epoch < 3 {
            println!("Epoch {}: Loss = {:.6}", epoch + 1, loss);
        }

        // Compute gradients using numerical differentiation
        let (weight_grad, bias_grad) = model_adam.compute_gradients(&x_train, &y_train, 1e-5);

        // Update parameters using Adam optimizer
        adam_optimizer.step(
            &mut [&mut model_adam.weight, &mut model_adam.bias],
            &[&weight_grad, &bias_grad],
        );
    }

    println!(
        "Adam Final Loss: {:.6}",
        model_adam.mse_loss(&model_adam.forward(&x_train), &y_train)
    );
    println!("Adam Final Weight: {}", model_adam.weight);
    println!("Adam Final Bias: {}", model_adam.bias);

    println!("\n=== Training Results Summary ===");
    println!("Target relationship: y = 2*x + 1");
    println!("Expected: weight ≈ 2.0, bias ≈ 1.0");

    // Test predictions on new data
    println!("\n=== Testing on new data ===");
    let test_x = Tensor::new(vec![9.0, 10.0], vec![2, 1]);
    let sgd_predictions = model.forward(&test_x);
    let adam_predictions = model_adam.forward(&test_x);

    println!("Test input: {}", test_x);
    println!("SGD predictions: {}", sgd_predictions);
    println!("Adam predictions: {}", adam_predictions);
    println!("Expected: ~[19.0, 21.0]");

    // Demonstrate different learning rates
    println!("\n=== Learning Rate Comparison ===");

    let learning_rates = [0.001, 0.01, 0.1];

    for &lr in &learning_rates {
        let mut model_lr = LinearModel::new(1);
        let mut optimizer_lr = SGD::new(lr);

        // Quick training for 10 epochs
        for _epoch in 0..10 {
            let (weight_grad, bias_grad) = model_lr.compute_gradients(&x_train, &y_train, 1e-5);
            optimizer_lr.step(
                &mut [&mut model_lr.weight, &mut model_lr.bias],
                &[&weight_grad, &bias_grad],
            );
        }

        let final_loss = model_lr.mse_loss(&model_lr.forward(&x_train), &y_train);
        println!(
            "Learning Rate {}: Final Loss = {:.6}, Weight = {:.4}, Bias = {:.4}",
            lr, final_loss, model_lr.weight.data[0], model_lr.bias.data[0]
        );
    }
}
