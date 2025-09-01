use crate::serialization::{ModelState, Saveable};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

// Base trait for all layers
pub trait Layer: Send + Sync {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn zero_grad(&mut self);
    fn name(&self) -> String;
}

// Dense (Fully Connected) Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dense {
    pub weight: Tensor,
    pub bias: Tensor,
    pub input_size: usize,
    pub output_size: usize,
    weight_grad: Option<Tensor>,
    bias_grad: Option<Tensor>,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Xavier initialization
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();
        let weight_data: Vec<f64> = (0..input_size * output_size)
            .map(|_| (rand::random::<f64>() - 0.5) * 2.0 * limit)
            .collect();

        Dense {
            weight: Tensor::new(weight_data, vec![input_size, output_size]),
            bias: Tensor::zeros(vec![output_size]),
            input_size,
            output_size,
            weight_grad: None,
            bias_grad: None,
        }
    }

    pub fn with_weights(weight: Tensor, bias: Tensor) -> Self {
        let input_size = weight.shape[0];
        let output_size = weight.shape[1];

        Dense {
            weight,
            bias,
            input_size,
            output_size,
            weight_grad: None,
            bias_grad: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.matmul(&self.weight).add_broadcast(&self.bias)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // This is a simplified backward pass
        // In a real implementation, you'd store the input from forward pass
        grad_output.clone()
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn zero_grad(&mut self) {
        self.weight_grad = None;
        self.bias_grad = None;
    }

    fn name(&self) -> String {
        format!("Dense({}, {})", self.input_size, self.output_size)
    }
}

// Activation Layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Layer for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        grad_output.clone() // Simplified
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
    fn zero_grad(&mut self) {}
    fn name(&self) -> String {
        "ReLU".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Layer for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.sigmoid()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        grad_output.clone() // Simplified
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
    fn zero_grad(&mut self) {}
    fn name(&self) -> String {
        "Sigmoid".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl Layer for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.tanh()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        grad_output.clone() // Simplified
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
    fn zero_grad(&mut self) {}
    fn name(&self) -> String {
        "Tanh".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Softmax
    }
}

impl Layer for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.softmax()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        grad_output.clone() // Simplified
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
    fn zero_grad(&mut self) {}
    fn name(&self) -> String {
        "Softmax".to_string()
    }
}

// Sequential Model Container
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("num_layers", &self.layers.len())
            .field(
                "layer_names",
                &self.layers.iter().map(|l| l.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    pub fn add<L: Layer + 'static>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn forward(&self, mut input: Tensor) -> Tensor {
        for layer in &self.layers {
            input = layer.forward(&input);
        }
        input
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params
    }

    pub fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    pub fn summary(&self) {
        println!("Model Summary:");
        println!("{:-<50}", "");
        for (i, layer) in self.layers.iter().enumerate() {
            println!("Layer {}: {}", i + 1, layer.name());
        }
        println!("{:-<50}", "");
        let total_params: usize = self.parameters().iter().map(|t| t.size()).sum();
        println!("Total parameters: {}", total_params);
    }
}

impl Saveable for Sequential {
    fn save_state(&self) -> ModelState {
        let mut state = ModelState::new();

        for (i, layer) in self.layers.iter().enumerate() {
            for (j, param) in layer.parameters().iter().enumerate() {
                state.add_parameter(format!("layer_{}_param_{}", i, j), param);
            }
        }

        state.add_metadata("model_type".to_string(), "Sequential".to_string());
        state.add_metadata("num_layers".to_string(), self.layers.len().to_string());

        state
    }

    fn load_state(&mut self, state: &ModelState) -> Result<(), String> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let mut params = layer.parameters_mut();
            for (j, param) in params.iter_mut().enumerate() {
                let param_name = format!("layer_{}_param_{}", i, j);
                if let Some(loaded_param) = state.get_parameter(&param_name) {
                    **param = loaded_param;
                } else {
                    return Err(format!("Missing parameter: {}", param_name));
                }
            }
        }
        Ok(())
    }
}
