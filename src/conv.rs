use crate::layers::Layer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2D {
    pub weight: Tensor, // [out_channels, in_channels, kernel_h, kernel_w]
    pub bias: Tensor,   // [out_channels]
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_params(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1),
            (0, 0),
        )
    }

    pub fn with_params(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        // He initialization for ReLU networks
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let std_dev = (2.0 / fan_in as f64).sqrt();

        let weight_size = out_channels * in_channels * kernel_size.0 * kernel_size.1;
        let weight_data: Vec<f64> = (0..weight_size)
            .map(|_| rand::random::<f64>() * std_dev * 2.0 - std_dev)
            .collect();

        Conv2D {
            weight: Tensor::new(
                weight_data,
                vec![out_channels, in_channels, kernel_size.0, kernel_size.1],
            ),
            bias: Tensor::zeros(vec![out_channels]),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    pub fn conv2d(&self, input: &Tensor) -> Tensor {
        // Input shape: [batch, in_channels, height, width]
        // Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        // Output shape: [batch, out_channels, out_height, out_width]

        assert_eq!(
            input.ndim(),
            4,
            "Input must be 4D: [batch, channels, height, width]"
        );
        assert_eq!(input.shape[1], self.in_channels, "Input channels mismatch");

        let batch_size = input.shape[0];
        let in_h = input.shape[2];
        let in_w = input.shape[3];
        let (kernel_h, kernel_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;

        // Calculate output dimensions
        let out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

        let mut output = Tensor::zeros(vec![batch_size, self.out_channels, out_h, out_w]);

        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = self.bias.data[oc]; // Add bias

                        for ic in 0..self.in_channels {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let ih = oh * stride_h + kh;
                                    let iw = ow * stride_w + kw;

                                    // Check bounds (simple padding with zeros)
                                    if ih >= pad_h
                                        && ih < in_h + pad_h
                                        && iw >= pad_w
                                        && iw < in_w + pad_w
                                    {
                                        let input_idx = b * (self.in_channels * in_h * in_w)
                                            + ic * (in_h * in_w)
                                            + (ih - pad_h) * in_w
                                            + (iw - pad_w);

                                        let weight_idx = oc
                                            * (self.in_channels * kernel_h * kernel_w)
                                            + ic * (kernel_h * kernel_w)
                                            + kh * kernel_w
                                            + kw;

                                        sum += input.data[input_idx] * self.weight.data[weight_idx];
                                    }
                                }
                            }
                        }

                        let output_idx = b * (self.out_channels * out_h * out_w)
                            + oc * (out_h * out_w)
                            + oh * out_w
                            + ow;

                        output.data[output_idx] = sum;
                    }
                }
            }
        }

        output
    }
}

impl Layer for Conv2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.conv2d(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // Simplified backward pass
        grad_output.clone()
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn zero_grad(&mut self) {}

    fn name(&self) -> String {
        format!(
            "Conv2D({}, {}, kernel={}x{})",
            self.in_channels, self.out_channels, self.kernel_size.0, self.kernel_size.1
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool2D {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2D {
    pub fn new(kernel_size: usize) -> Self {
        MaxPool2D {
            kernel_size: (kernel_size, kernel_size),
            stride: (kernel_size, kernel_size),
            padding: (0, 0),
        }
    }

    pub fn with_params(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        MaxPool2D {
            kernel_size,
            stride,
            padding,
        }
    }

    pub fn max_pool2d(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.ndim(),
            4,
            "Input must be 4D: [batch, channels, height, width]"
        );

        let batch_size = input.shape[0];
        let channels = input.shape[1];
        let in_h = input.shape[2];
        let in_w = input.shape[3];
        let (kernel_h, kernel_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;

        let out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

        let mut output = Tensor::zeros(vec![batch_size, channels, out_h, out_w]);

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f64::NEG_INFINITY;

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                if ih >= pad_h
                                    && ih < in_h + pad_h
                                    && iw >= pad_w
                                    && iw < in_w + pad_w
                                {
                                    let input_idx = b * (channels * in_h * in_w)
                                        + c * (in_h * in_w)
                                        + (ih - pad_h) * in_w
                                        + (iw - pad_w);

                                    max_val = max_val.max(input.data[input_idx]);
                                }
                            }
                        }

                        let output_idx =
                            b * (channels * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow;

                        output.data[output_idx] = max_val;
                    }
                }
            }
        }

        output
    }
}

impl Layer for MaxPool2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.max_pool2d(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        grad_output.clone()
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
    fn zero_grad(&mut self) {}

    fn name(&self) -> String {
        format!(
            "MaxPool2D(kernel={}x{}, stride={}x{})",
            self.kernel_size.0, self.kernel_size.1, self.stride.0, self.stride.1
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flatten;

impl Flatten {
    pub fn new() -> Self {
        Flatten
    }
}

impl Layer for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        if input.ndim() <= 2 {
            return input.clone();
        }

        let batch_size = input.shape[0];
        let flattened_size = input.size() / batch_size;

        input.reshape(vec![batch_size, flattened_size])
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        grad_output.clone()
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
    fn zero_grad(&mut self) {}
    fn name(&self) -> String {
        "Flatten".to_string()
    }
}
