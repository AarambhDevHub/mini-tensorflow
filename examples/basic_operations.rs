use mini_tensorflow::Tensor;

fn main() {
    println!("Basic Operations Example");

    // Create some tensors
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0], vec![2, 3]);

    println!("Tensor A (2x3):\n{}", a);
    println!("Tensor B (2x3):\n{}", b);

    // Element-wise operations
    println!("\nElement-wise operations:");
    println!("A + B = {}", a.add(&b));
    println!("A - B = {}", a.sub(&b));
    println!("A * B = {}", a.mul(&b));
    println!("A / B = {}", a.div(&b));

    // Matrix operations
    let c = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    println!("\nTensor C (3x2):\n{}", c);
    println!("A @ C (matrix multiplication):\n{}", a.matmul(&c));

    // Scalar operations
    println!("\nScalar operations:");
    println!("A + 10 = {}", a.add_scalar(10.0));
    println!("A * 2 = {}", a.mul_scalar(2.0));

    // Reshape and transpose
    println!("\nReshape and transpose:");
    let reshaped = a.reshape(vec![3, 2]);
    println!("A reshaped to (3x2):\n{}", reshaped);
    println!("Transposed:\n{}", reshaped.transpose());

    // Aggregation operations
    println!("\nAggregation:");
    println!("Sum of A: {}", a.sum());
    println!("Mean of A: {}", a.mean());
}
