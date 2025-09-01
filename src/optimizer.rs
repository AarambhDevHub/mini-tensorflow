use crate::tensor::Tensor;
use std::collections::HashMap;

pub trait Optimizer {
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &[&Tensor]);
    fn zero_grad(&mut self);
}

pub struct SGD {
    pub learning_rate: f64,
    pub momentum: f64,
    pub velocity: HashMap<usize, Tensor>,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        SGD {
            learning_rate,
            momentum: 0.0,
            velocity: HashMap::new(),
        }
    }

    pub fn with_momentum(learning_rate: f64, momentum: f64) -> Self {
        SGD {
            learning_rate,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &[&Tensor]) {
        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            if self.momentum > 0.0 {
                let velocity = self
                    .velocity
                    .entry(i)
                    .or_insert_with(|| Tensor::zeros(param.shape.clone()));

                // v = momentum * v - learning_rate * grad
                *velocity = velocity
                    .mul_scalar(self.momentum)
                    .sub(&grad.mul_scalar(self.learning_rate));

                // param = param + v
                **param = param.add(velocity);
            } else {
                // Simple SGD: param = param - learning_rate * grad
                **param = param.sub(&grad.mul_scalar(self.learning_rate));
            }
        }
    }

    fn zero_grad(&mut self) {
        // In a real implementation, this would zero out gradients
    }
}

pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub t: usize,                  // time step
    pub m: HashMap<usize, Tensor>, // first moment
    pub v: HashMap<usize, Tensor>, // second moment
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn with_params(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &[&Tensor]) {
        self.t += 1;

        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let m = self
                .m
                .entry(i)
                .or_insert_with(|| Tensor::zeros(param.shape.clone()));
            let v = self
                .v
                .entry(i)
                .or_insert_with(|| Tensor::zeros(param.shape.clone()));

            // m = beta1 * m + (1 - beta1) * grad
            *m = m
                .mul_scalar(self.beta1)
                .add(&grad.mul_scalar(1.0 - self.beta1));

            // v = beta2 * v + (1 - beta2) * grad^2
            let grad_squared = grad.mul(grad);
            *v = v
                .mul_scalar(self.beta2)
                .add(&grad_squared.mul_scalar(1.0 - self.beta2));

            // Bias correction
            let m_hat = m.mul_scalar(1.0 / (1.0 - self.beta1.powi(self.t as i32)));
            let v_hat = v.mul_scalar(1.0 / (1.0 - self.beta2.powi(self.t as i32)));

            // param = param - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
            let v_hat_sqrt = Tensor::new(
                v_hat
                    .data
                    .iter()
                    .map(|&x| x.sqrt() + self.epsilon)
                    .collect(),
                v_hat.shape.clone(),
            );
            let update = m_hat.div(&v_hat_sqrt).mul_scalar(self.learning_rate);
            **param = param.sub(&update);
        }
    }

    fn zero_grad(&mut self) {
        // In a real implementation, this would zero out gradients
    }
}
