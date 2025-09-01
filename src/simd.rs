use crate::tensor::Tensor;

#[cfg(target_arch = "x86_64")]
use wide::f64x4;

pub trait SIMDOps {
    fn simd_add(&self, other: &Tensor) -> Tensor;
    fn simd_mul(&self, other: &Tensor) -> Tensor;
    fn simd_sum(&self) -> f64;
    fn simd_relu(&self) -> Tensor;
}

impl SIMDOps for Tensor {
    #[cfg(target_arch = "x86_64")]
    fn simd_add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Tensors must have same shape");

        let mut result = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;

        // Process 4 elements at a time using SIMD
        for i in 0..chunks {
            let a = f64x4::new([
                self.data[i * 4],
                self.data[i * 4 + 1],
                self.data[i * 4 + 2],
                self.data[i * 4 + 3],
            ]);
            let b = f64x4::new([
                other.data[i * 4],
                other.data[i * 4 + 1],
                other.data[i * 4 + 2],
                other.data[i * 4 + 3],
            ]);

            let sum = a + b;
            let array = sum.to_array();
            result.extend_from_slice(&array);
        }

        // Handle remaining elements
        for i in (chunks * 4)..self.data.len() {
            result.push(self.data[i] + other.data[i]);
        }

        Tensor::new(result, self.shape.clone())
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_add(&self, other: &Tensor) -> Tensor {
        // Fallback to regular addition on non-x86_64 platforms
        self.add(other)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Tensors must have same shape");

        let mut result = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;

        for i in 0..chunks {
            let a = f64x4::new([
                self.data[i * 4],
                self.data[i * 4 + 1],
                self.data[i * 4 + 2],
                self.data[i * 4 + 3],
            ]);
            let b = f64x4::new([
                other.data[i * 4],
                other.data[i * 4 + 1],
                other.data[i * 4 + 2],
                other.data[i * 4 + 3],
            ]);

            let product = a * b;
            let array = product.to_array();
            result.extend_from_slice(&array);
        }

        for i in (chunks * 4)..self.data.len() {
            result.push(self.data[i] * other.data[i]);
        }

        Tensor::new(result, self.shape.clone())
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_mul(&self, other: &Tensor) -> Tensor {
        self.mul(other)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_sum(&self) -> f64 {
        let chunks = self.data.len() / 4;
        let mut sum_vec = f64x4::splat(0.0);

        for i in 0..chunks {
            let chunk = f64x4::new([
                self.data[i * 4],
                self.data[i * 4 + 1],
                self.data[i * 4 + 2],
                self.data[i * 4 + 3],
            ]);
            sum_vec += chunk;
        }

        let sum_array = sum_vec.to_array();
        let mut total = sum_array.iter().sum::<f64>();

        // Add remaining elements
        for i in (chunks * 4)..self.data.len() {
            total += self.data[i];
        }

        total
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_sum(&self) -> f64 {
        self.sum()
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_relu(&self) -> Tensor {
        let mut result = Vec::with_capacity(self.data.len());
        let chunks = self.data.len() / 4;
        let zero = f64x4::splat(0.0);

        for i in 0..chunks {
            let chunk = f64x4::new([
                self.data[i * 4],
                self.data[i * 4 + 1],
                self.data[i * 4 + 2],
                self.data[i * 4 + 3],
            ]);

            let relu_result = chunk.max(zero);
            let array = relu_result.to_array();
            result.extend_from_slice(&array);
        }

        for i in (chunks * 4)..self.data.len() {
            result.push(self.data[i].max(0.0));
        }

        Tensor::new(result, self.shape.clone())
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_relu(&self) -> Tensor {
        self.relu()
    }
}

// Parallel operations using Rayon
pub trait ParallelOps {
    fn parallel_add(&self, other: &Tensor) -> Tensor;
    fn parallel_mul(&self, other: &Tensor) -> Tensor;
    fn parallel_matmul(&self, other: &Tensor) -> Tensor;
}

impl ParallelOps for Tensor {
    fn parallel_add(&self, other: &Tensor) -> Tensor {
        use rayon::prelude::*;

        assert_eq!(self.shape, other.shape);

        let result_data: Vec<f64> = self
            .data
            .par_iter()
            .zip(&other.data)
            .map(|(&a, &b)| a + b)
            .collect();

        Tensor::new(result_data, self.shape.clone())
    }

    fn parallel_mul(&self, other: &Tensor) -> Tensor {
        use rayon::prelude::*;

        assert_eq!(self.shape, other.shape);

        let result_data: Vec<f64> = self
            .data
            .par_iter()
            .zip(&other.data)
            .map(|(&a, &b)| a * b)
            .collect();

        Tensor::new(result_data, self.shape.clone())
    }

    fn parallel_matmul(&self, other: &Tensor) -> Tensor {
        use rayon::prelude::*;

        assert_eq!(self.ndim(), 2);
        assert_eq!(other.ndim(), 2);
        assert_eq!(self.shape[1], other.shape[0]);

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let result_data: Vec<f64> = (0..m * n)
            .into_par_iter()
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;

                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                sum
            })
            .collect();

        Tensor::new(result_data, vec![m, n])
    }
}
