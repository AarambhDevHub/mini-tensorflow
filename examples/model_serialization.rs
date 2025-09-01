use mini_tensorflow::{ModelState, SGD, Saveable, Tensor};

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
        let z1 = x.matmul(&self.w1).add_broadcast(&self.b1);
        let a1 = z1.relu();
        let z2 = a1.matmul(&self.w2).add_broadcast(&self.b2);
        z2.sigmoid()
    }

    fn loss(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        let diff = predictions.sub(targets);
        let squared_diff = diff.mul(&diff);
        squared_diff.mean()
    }
}

impl Saveable for NeuralNetwork {
    fn save_state(&self) -> ModelState {
        let mut state = ModelState::new();

        // Save parameters
        state.add_parameter("w1".to_string(), &self.w1);
        state.add_parameter("b1".to_string(), &self.b1);
        state.add_parameter("w2".to_string(), &self.w2);
        state.add_parameter("b2".to_string(), &self.b2);

        // Save metadata
        state.add_metadata("model_type".to_string(), "NeuralNetwork".to_string());
        state.add_metadata("input_size".to_string(), self.w1.shape[0].to_string());
        state.add_metadata("hidden_size".to_string(), self.w1.shape[1].to_string());
        state.add_metadata("output_size".to_string(), self.w2.shape[1].to_string());

        state
    }

    fn load_state(&mut self, state: &ModelState) -> Result<(), String> {
        self.w1 = state.get_parameter("w1").ok_or("Missing w1 parameter")?;
        self.b1 = state.get_parameter("b1").ok_or("Missing b1 parameter")?;
        self.w2 = state.get_parameter("w2").ok_or("Missing w2 parameter")?;
        self.b2 = state.get_parameter("b2").ok_or("Missing b2 parameter")?;

        Ok(())
    }
}

fn main() {
    println!("Model Serialization Example");

    // Create and train a model
    let network = NeuralNetwork::new(2, 3, 1);
    let _optimizer = SGD::new(0.01);

    // Sample training data
    let x = Tensor::new(vec![0.5, -0.2, 0.3, 0.8], vec![2, 2]);
    let y = Tensor::new(vec![1.0, 0.0], vec![2, 1]);

    println!("Original model parameters:");
    println!("W1: {}", network.w1);
    println!("B1: {}", network.b1);

    // Train for a few epochs
    for epoch in 0..5 {
        let predictions = network.forward(&x);
        let loss = network.loss(&predictions, &y);
        println!("Epoch {}: Loss = {:.6}", epoch + 1, loss);
    }

    // Save the model
    println!("\n=== Saving Model ===");

    // Save as JSON (human-readable)
    network
        .save("model.json")
        .expect("Failed to save model as JSON");
    println!("✅ Model saved as JSON: model.json");

    // Save as binary (compact)
    network
        .save("model.bin")
        .expect("Failed to save model as binary");
    println!("✅ Model saved as binary: model.bin");

    // Save individual tensor
    network
        .w1
        .save("weight1.json")
        .expect("Failed to save tensor");
    println!("✅ Weight tensor saved: weight1.json");

    // Create a new model and load the saved state
    println!("\n=== Loading Model ===");
    let mut new_network = NeuralNetwork::new(2, 3, 1);

    println!("New model before loading:");
    println!("W1: {}", new_network.w1);

    // Load from saved file
    new_network
        .load("model.json")
        .expect("Failed to load model");

    println!("New model after loading:");
    println!("W1: {}", new_network.w1);

    // Test that the loaded model produces the same results
    let original_predictions = network.forward(&x);
    let loaded_predictions = new_network.forward(&x);

    println!("\n=== Verification ===");
    println!("Original predictions: {}", original_predictions);
    println!("Loaded predictions: {}", loaded_predictions);

    // Check if they're the same (within floating point precision)
    let diff = original_predictions.sub(&loaded_predictions);
    let max_diff = diff.data.iter().map(|x| x.abs()).fold(0.0, f64::max);

    if max_diff < 1e-10 {
        println!(
            "✅ Model serialization successful! (max difference: {:.2e})",
            max_diff
        );
    } else {
        println!(
            "❌ Model serialization failed! (max difference: {:.2e})",
            max_diff
        );
    }

    // File size comparison
    let json_size = std::fs::metadata("model.json").unwrap().len();
    let bin_size = std::fs::metadata("model.bin").unwrap().len();

    println!("\n=== File Sizes ===");
    println!("JSON format: {} bytes", json_size);
    println!("Binary format: {} bytes", bin_size);
    println!(
        "Space savings: {:.1}%",
        (1.0 - bin_size as f64 / json_size as f64) * 100.0
    );
}
