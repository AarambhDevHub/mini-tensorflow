use crate::graph::{Graph, Operation};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Variable {
    pub node_id: usize,
    pub graph: *mut Graph, // Unsafe pointer for simplicity
}

impl Variable {
    pub fn new(tensor: Tensor, graph: &mut Graph) -> Self {
        let node_id = graph.add_variable(tensor);
        Variable { node_id, graph }
    }

    pub fn value(&self) -> Option<&Tensor> {
        unsafe { (*self.graph).get_value(self.node_id) }
    }

    pub fn add(&self, other: &Variable) -> Variable {
        let graph = unsafe { &mut *self.graph };
        let node_id = graph.add_operation(Operation::Add, vec![self.node_id, other.node_id]);
        Variable {
            node_id,
            graph: self.graph,
        }
    }

    pub fn mul(&self, other: &Variable) -> Variable {
        let graph = unsafe { &mut *self.graph };
        let node_id = graph.add_operation(Operation::Mul, vec![self.node_id, other.node_id]);
        Variable {
            node_id,
            graph: self.graph,
        }
    }

    pub fn matmul(&self, other: &Variable) -> Variable {
        let graph = unsafe { &mut *self.graph };
        let node_id = graph.add_operation(Operation::MatMul, vec![self.node_id, other.node_id]);
        Variable {
            node_id,
            graph: self.graph,
        }
    }

    pub fn relu(&self) -> Variable {
        let graph = unsafe { &mut *self.graph };
        let node_id = graph.add_operation(Operation::ReLU, vec![self.node_id]);
        Variable {
            node_id,
            graph: self.graph,
        }
    }

    pub fn sigmoid(&self) -> Variable {
        let graph = unsafe { &mut *self.graph };
        let node_id = graph.add_operation(Operation::Sigmoid, vec![self.node_id]);
        Variable {
            node_id,
            graph: self.graph,
        }
    }

    pub fn compute(&self) -> Result<(), String> {
        unsafe { (*self.graph).forward(self.node_id) }
    }

    pub fn backward(&self) -> Result<(), String> {
        unsafe { (*self.graph).backward(self.node_id) }
    }
}
