use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape");

        Tensor {
            data,
            shape,
            requires_grad: false,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor::new(vec![0.0; size], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor::new(vec![1.0; size], shape)
    }

    pub fn random(shape: Vec<usize>) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Tensor::new(data, shape)
    }

    pub fn from_scalar(value: f64) -> Self {
        Tensor::new(vec![value], vec![1])
    }

    pub fn requires_grad(&mut self, requires_grad: bool) -> &mut Self {
        self.requires_grad = requires_grad;
        self
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            self.size(),
            new_size,
            "Cannot reshape tensor to incompatible size"
        );

        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            requires_grad: self.requires_grad,
        }
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2, "Transpose only supported for 2D tensors");

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut transposed_data = vec![0.0; self.size()];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Tensor::new(transposed_data, vec![cols, rows])
    }

    pub fn matmul(&self, other: &Tensor) -> Self {
        assert_eq!(self.ndim(), 2, "First tensor must be 2D");
        assert_eq!(other.ndim(), 2, "Second tensor must be 2D");
        assert_eq!(
            self.shape[1], other.shape[0],
            "Incompatible dimensions for matrix multiplication"
        );

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    result[i * n + j] += self.data[i * k + l] * other.data[l * n + j];
                }
            }
        }

        Tensor::new(result, vec![m, n])
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        self.sum() / self.size() as f64
    }

    pub fn relu(&self) -> Self {
        let data = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn sigmoid(&self) -> Self {
        let data = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn tanh(&self) -> Self {
        let data = self.data.iter().map(|&x| x.tanh()).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn softmax(&self) -> Self {
        let max_val = self
            .data
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let exp_data: Vec<f64> = self.data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f64 = exp_data.iter().sum();
        let softmax_data: Vec<f64> = exp_data.iter().map(|&x| x / sum_exp).collect();

        Tensor::new(softmax_data, self.shape.clone())
    }

    // Element-wise operations
    pub fn add(&self, other: &Tensor) -> Self {
        self.element_wise_op(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Self {
        self.element_wise_op(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor) -> Self {
        self.element_wise_op(other, |a, b| a * b)
    }

    pub fn div(&self, other: &Tensor) -> Self {
        self.element_wise_op(other, |a, b| a / b)
    }

    pub fn add_scalar(&self, scalar: f64) -> Self {
        let data = self.data.iter().map(|&x| x + scalar).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn mul_scalar(&self, scalar: f64) -> Self {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::new(data, self.shape.clone())
    }

    fn element_wise_op<F>(&self, other: &Tensor, op: F) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        assert_eq!(self.shape, other.shape, "Tensors must have the same shape");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| op(a, b))
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    fn get_flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        let mut flat_index = 0;
        let mut stride = 1;

        for i in (0..indices.len()).rev() {
            assert!(indices[i] < self.shape[i]);
            flat_index += indices[i] * stride;
            stride *= self.shape[i];
        }

        flat_index
    }

    pub fn get(&self, indices: &[usize]) -> f64 {
        let flat_index = self.get_flat_index(indices);
        self.data[flat_index]
    }

    pub fn set(&mut self, indices: &[usize], value: f64) {
        let flat_index = self.get_flat_index(indices);
        self.data[flat_index] = value;
    }

    pub fn add_broadcast(&self, other: &Tensor) -> Self {
        // Handle bias addition: [batch, features] + [features] -> [batch, features]
        if self.ndim() == 2 && other.ndim() == 1 && self.shape[1] == other.shape[0] {
            let mut result_data = Vec::with_capacity(self.size());
            let batch_size = self.shape[0];
            let feature_size = self.shape[1];

            for i in 0..batch_size {
                for j in 0..feature_size {
                    let self_idx = i * feature_size + j;
                    result_data.push(self.data[self_idx] + other.data[j]);
                }
            }

            return Tensor::new(result_data, self.shape.clone());
        }

        // Handle scalar broadcast: [any_shape] + [1] -> [any_shape]
        if other.shape == vec![1] {
            return self.add_scalar(other.data[0]);
        }

        // Fall back to regular addition for same shapes
        if self.shape == other.shape {
            return self.add(other);
        }

        panic!(
            "Incompatible shapes for broadcasting: {:?} and {:?}",
            self.shape, other.shape
        );
    }

    pub fn save<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let serializable = crate::serialization::SerializableTensor::from(self);
        let json = serde_json::to_string_pretty(&serializable)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let serializable: crate::serialization::SerializableTensor =
            serde_json::from_str(&contents)?;
        Ok(Tensor::from(serializable))
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape: {:?}, data: [", self.shape)?;
        for (i, &val) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", val)?;
            if i >= 10 {
                // Limit output for large tensors
                write!(f, ", ...")?;
                break;
            }
        }
        write!(f, "])")
    }
}
