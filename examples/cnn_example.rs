use mini_tensorflow::{Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential, Softmax, Tensor};

fn main() {
    println!("CNN Example");

    // Create a simple CNN for image classification
    let model = Sequential::new()
        .add(Conv2D::new(1, 32, 3)) // 1 input channel, 32 filters, 3x3 kernel
        .add(ReLU::new())
        .add(MaxPool2D::new(2)) // 2x2 max pooling
        .add(Conv2D::new(32, 64, 3)) // 32->64 channels
        .add(ReLU::new())
        .add(MaxPool2D::new(2))
        .add(Flatten::new()) // Flatten for dense layers
        .add(Dense::new(1600, 128)) // Adjust based on input size
        .add(ReLU::new())
        .add(Dense::new(128, 10)) // 10 classes
        .add(Softmax::new());

    model.summary();

    // Create a dummy input image: batch_size=1, channels=1, height=28, width=28
    let input = Tensor::random(vec![1, 1, 28, 28]);
    println!("Input shape: {:?}", input.shape);

    // Forward pass
    let output = model.forward(input);
    println!("Output shape: {:?}", output.shape);
    println!("Output probabilities: {}", output);

    // Verify it's a probability distribution
    let sum: f64 = output.data.iter().sum();
    println!("Sum of probabilities: {:.6}", sum);
}
