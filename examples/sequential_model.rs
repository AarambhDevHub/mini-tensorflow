use mini_tensorflow::{Adam, Dense, Optimizer, ReLU, SGD, Saveable, Sequential, Sigmoid, Tensor};

fn main() {
    println!("Sequential Model Example");

    // Build a multi-layer perceptron
    let mut model = Sequential::new()
        .add(Dense::new(784, 256)) // Input layer: 784 -> 256
        .add(ReLU::new())
        .add(Dense::new(256, 128)) // Hidden layer: 256 -> 128
        .add(ReLU::new())
        .add(Dense::new(128, 64)) // Hidden layer: 128 -> 64
        .add(ReLU::new())
        .add(Dense::new(64, 10)) // Output layer: 64 -> 10
        .add(Sigmoid::new());

    model.summary();

    // Create sample data (flattened 28x28 images)
    let batch_size = 8;
    let input = Tensor::random(vec![batch_size, 784]);
    let targets = Tensor::random(vec![batch_size, 10]);

    println!("Input shape: {:?}", input.shape);

    // Forward pass
    let predictions = model.forward(input.clone());
    println!("Predictions shape: {:?}", predictions.shape);

    // Calculate loss (MSE)
    let diff = predictions.sub(&targets);
    let loss = diff.mul(&diff).mean();
    println!("Initial loss: {:.6}", loss);

    // Training simulation
    let _sgd = SGD::new(0.001);
    let mut adam = Adam::new(0.0001);

    println!("\nTraining simulation:");

    for epoch in 0..10 {
        // Forward pass
        let predictions = model.forward(input.clone());

        // Calculate loss
        let diff = predictions.sub(&targets);
        let loss = diff.mul(&diff).mean();

        if epoch % 2 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch + 1, loss);
        }

        // Simplified parameter update (in practice, you'd use gradients)
        let mut params = model.parameters_mut();
        let dummy_grads: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::new(vec![0.001; p.size()], p.shape.clone()))
            .collect();

        let grad_refs: Vec<&Tensor> = dummy_grads.iter().collect();
        adam.step(&mut params, &grad_refs);
    }

    println!("Training completed!");

    // Save the model
    model
        .save("sequential_model.json")
        .expect("Failed to save model");
    println!("Model saved as sequential_model.json");
}
