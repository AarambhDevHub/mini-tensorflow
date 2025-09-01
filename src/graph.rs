use crate::tensor::Tensor;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Sum,
    Mean,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub operation: Option<Operation>,
    pub inputs: Vec<usize>,
    pub value: Option<Tensor>,
    pub gradient: Option<Tensor>,
}

impl Node {
    pub fn new(id: usize, value: Tensor) -> Self {
        Node {
            id,
            operation: None,
            inputs: vec![],
            value: Some(value),
            gradient: None,
        }
    }

    pub fn new_op(id: usize, operation: Operation, inputs: Vec<usize>) -> Self {
        Node {
            id,
            operation: Some(operation),
            inputs,
            value: None,
            gradient: None,
        }
    }
}

#[derive(Debug)]
pub struct Graph {
    nodes: HashMap<usize, Node>,
    next_id: usize,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn add_variable(&mut self, tensor: Tensor) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let node = Node::new(id, tensor);
        self.nodes.insert(id, node);
        id
    }

    pub fn add_operation(&mut self, operation: Operation, inputs: Vec<usize>) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let node = Node::new_op(id, operation, inputs);
        self.nodes.insert(id, node);
        id
    }

    pub fn forward(&mut self, node_id: usize) -> Result<(), String> {
        let node = self.nodes.get(&node_id).ok_or("Node not found")?;

        if node.value.is_some() {
            return Ok(());
        }

        let operation = node.operation.clone().ok_or("No operation found")?;
        let input_ids = node.inputs.clone();

        // Ensure all inputs are computed
        for input_id in &input_ids {
            self.forward(*input_id)?;
        }

        // Get input values
        let input_tensors: Vec<Tensor> = input_ids
            .iter()
            .map(|&id| self.nodes[&id].value.clone().unwrap())
            .collect();

        // Compute output
        let result = match operation {
            Operation::Add => {
                if input_tensors.len() != 2 {
                    return Err("Add requires exactly 2 inputs".to_string());
                }
                input_tensors[0].add(&input_tensors[1])
            }
            Operation::Sub => {
                if input_tensors.len() != 2 {
                    return Err("Sub requires exactly 2 inputs".to_string());
                }
                input_tensors[0].sub(&input_tensors[1])
            }
            Operation::Mul => {
                if input_tensors.len() != 2 {
                    return Err("Mul requires exactly 2 inputs".to_string());
                }
                input_tensors[0].mul(&input_tensors[1])
            }
            Operation::Div => {
                if input_tensors.len() != 2 {
                    return Err("Div requires exactly 2 inputs".to_string());
                }
                input_tensors[0].div(&input_tensors[1])
            }
            Operation::MatMul => {
                if input_tensors.len() != 2 {
                    return Err("MatMul requires exactly 2 inputs".to_string());
                }
                input_tensors[0].matmul(&input_tensors[1])
            }
            Operation::ReLU => {
                if input_tensors.len() != 1 {
                    return Err("ReLU requires exactly 1 input".to_string());
                }
                input_tensors[0].relu()
            }
            Operation::Sigmoid => {
                if input_tensors.len() != 1 {
                    return Err("Sigmoid requires exactly 1 input".to_string());
                }
                input_tensors[0].sigmoid()
            }
            Operation::Tanh => {
                if input_tensors.len() != 1 {
                    return Err("Tanh requires exactly 1 input".to_string());
                }
                input_tensors[0].tanh()
            }
            Operation::Softmax => {
                if input_tensors.len() != 1 {
                    return Err("Softmax requires exactly 1 input".to_string());
                }
                input_tensors[0].softmax()
            }
            Operation::Sum => {
                if input_tensors.len() != 1 {
                    return Err("Sum requires exactly 1 input".to_string());
                }
                Tensor::from_scalar(input_tensors[0].sum())
            }
            Operation::Mean => {
                if input_tensors.len() != 1 {
                    return Err("Mean requires exactly 1 input".to_string());
                }
                Tensor::from_scalar(input_tensors[0].mean())
            }
        };

        // Store result
        self.nodes.get_mut(&node_id).unwrap().value = Some(result);
        Ok(())
    }

    pub fn get_value(&self, node_id: usize) -> Option<&Tensor> {
        self.nodes.get(&node_id)?.value.as_ref()
    }

    pub fn backward(&mut self, output_id: usize) -> Result<(), String> {
        // Initialize gradient of output
        let output_tensor = self
            .nodes
            .get(&output_id)
            .ok_or("Output node not found")?
            .value
            .clone()
            .ok_or("Output has no value")?;

        let grad_shape = output_tensor.shape.clone();
        let ones_grad = Tensor::ones(grad_shape);
        self.nodes.get_mut(&output_id).unwrap().gradient = Some(ones_grad);

        // Topological sort and backward pass would go here
        // This is a simplified version
        Ok(())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
