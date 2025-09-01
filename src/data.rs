use crate::tensor::Tensor;
use csv::Reader;
use rand::seq::SliceRandom;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Dataset {
    pub data: Vec<Tensor>,
    pub labels: Vec<Tensor>,
    pub feature_names: Option<Vec<String>>,
}

impl Dataset {
    pub fn new(data: Vec<Tensor>, labels: Vec<Tensor>) -> Self {
        assert_eq!(
            data.len(),
            labels.len(),
            "Data and labels must have same length"
        );

        Dataset {
            data,
            labels,
            feature_names: None,
        }
    }

    pub fn from_arrays(
        data_array: &[Vec<f64>],
        labels_array: &[f64],
        input_shape: Vec<usize>,
    ) -> Self {
        let data: Vec<Tensor> = data_array
            .iter()
            .map(|row| Tensor::new(row.clone(), input_shape.clone()))
            .collect();

        let labels: Vec<Tensor> = labels_array
            .iter()
            .map(|&label| Tensor::from_scalar(label))
            .collect();

        Dataset::new(data, labels)
    }

    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        features: Vec<usize>,
        target: usize,
        has_header: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = Reader::from_reader(file);

        let mut data_rows: Vec<Vec<f64>> = Vec::new();
        let mut label_vec: Vec<f64> = Vec::new();
        let mut feature_names: Option<Vec<String>> = None;

        if has_header {
            if let Ok(headers) = reader.headers() {
                feature_names = Some(
                    features
                        .iter()
                        .map(|&i| headers.get(i).unwrap_or("unknown").to_string())
                        .collect(),
                );
            }
        }

        for result in reader.records() {
            let record = result?;

            // Extract features
            let mut row_data = Vec::new();
            for &feature_idx in &features {
                if let Some(value_str) = record.get(feature_idx) {
                    let value: f64 = value_str.parse()?;
                    row_data.push(value);
                }
            }

            // Extract target
            if let Some(target_str) = record.get(target) {
                let target_value: f64 = target_str.parse()?;
                label_vec.push(target_value);
            }

            data_rows.push(row_data);
        }

        let input_shape = vec![features.len()];
        let mut dataset = Dataset::from_arrays(&data_rows, &label_vec, input_shape);
        dataset.feature_names = feature_names;

        Ok(dataset)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.shuffle(&mut rng);

        let shuffled_data: Vec<Tensor> = indices.iter().map(|&i| self.data[i].clone()).collect();

        let shuffled_labels: Vec<Tensor> =
            indices.iter().map(|&i| self.labels[i].clone()).collect();

        self.data = shuffled_data;
        self.labels = shuffled_labels;
    }

    pub fn split(&self, train_ratio: f64) -> (Dataset, Dataset) {
        assert!(
            train_ratio > 0.0 && train_ratio < 1.0,
            "Train ratio must be between 0 and 1"
        );

        let split_idx = (self.len() as f64 * train_ratio) as usize;

        let train_data = self.data[..split_idx].to_vec();
        let train_labels = self.labels[..split_idx].to_vec();
        let test_data = self.data[split_idx..].to_vec();
        let test_labels = self.labels[split_idx..].to_vec();

        let train_dataset = Dataset::new(train_data, train_labels);
        let test_dataset = Dataset::new(test_data, test_labels);

        (train_dataset, test_dataset)
    }

    pub fn normalize_features(&mut self) {
        if self.data.is_empty() {
            return;
        }

        let num_features = self.data[0].size();
        let mut means = vec![0.0; num_features];
        let mut stds = vec![0.0; num_features];

        // Calculate means
        for sample in &self.data {
            for (i, &value) in sample.data.iter().enumerate() {
                means[i] += value;
            }
        }

        for mean in &mut means {
            *mean /= self.len() as f64;
        }

        // Calculate standard deviations
        for sample in &self.data {
            for (i, &value) in sample.data.iter().enumerate() {
                let diff = value - means[i];
                stds[i] += diff * diff;
            }
        }

        for std in &mut stds {
            *std = (*std / self.len() as f64).sqrt();
            if *std == 0.0 {
                *std = 1.0;
            } // Avoid division by zero
        }

        // Normalize data
        for sample in &mut self.data {
            for (i, value) in sample.data.iter_mut().enumerate() {
                *value = (*value - means[i]) / stds[i];
            }
        }
    }

    pub fn get_batch(&self, indices: &[usize]) -> (Vec<Tensor>, Vec<Tensor>) {
        let batch_data: Vec<Tensor> = indices.iter().map(|&i| self.data[i].clone()).collect();

        let batch_labels: Vec<Tensor> = indices.iter().map(|&i| self.labels[i].clone()).collect();

        (batch_data, batch_labels)
    }

    pub fn summary(&self) {
        println!("Dataset Summary:");
        println!("  Samples: {}", self.len());
        if !self.data.is_empty() {
            println!("  Feature shape: {:?}", self.data[0].shape);
            println!("  Label shape: {:?}", self.labels[0].shape);
        }

        if let Some(ref names) = self.feature_names {
            println!("  Features: {:?}", names);
        }

        // Basic statistics
        if !self.labels.is_empty() {
            let label_values: Vec<f64> = self.labels.iter().map(|t| t.data[0]).collect();

            let min_label = label_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_label = label_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mean_label: f64 = label_values.iter().sum::<f64>() / label_values.len() as f64;

            println!("  Label range: [{:.3}, {:.3}]", min_label, max_label);
            println!("  Label mean: {:.3}", mean_label);
        }
    }
}

pub struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: usize) -> Self {
        DataLoader {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
        }
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn iter(&mut self) -> BatchIterator {
        if self.shuffle {
            self.dataset.shuffle();
        }

        BatchIterator::new(&self.dataset, self.batch_size, self.drop_last)
    }
}

pub struct BatchIterator<'a> {
    dataset: &'a Dataset,
    batch_size: usize,
    current_idx: usize,
    drop_last: bool,
}

impl<'a> BatchIterator<'a> {
    fn new(dataset: &'a Dataset, batch_size: usize, drop_last: bool) -> Self {
        BatchIterator {
            dataset,
            batch_size,
            current_idx: 0,
            drop_last,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Vec<Tensor>, Vec<Tensor>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let batch_size = end_idx - self.current_idx;

        // Skip incomplete batches if drop_last is true
        if self.drop_last && batch_size < self.batch_size {
            return None;
        }

        let indices: Vec<usize> = (self.current_idx..end_idx).collect();
        let (batch_data, batch_labels) = self.dataset.get_batch(&indices);

        self.current_idx = end_idx;

        Some((batch_data, batch_labels))
    }
}

// Utility function to create synthetic datasets for testing
pub fn make_classification_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> Dataset {
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..n_samples {
        let sample: Vec<f64> = (0..n_features)
            .map(|_| rand::random::<f64>() * 2.0 - 1.0)
            .collect();

        let label = rand::random::<usize>() % n_classes;

        data.push(Tensor::new(sample, vec![n_features]));
        labels.push(Tensor::from_scalar(label as f64));
    }

    Dataset::new(data, labels)
}

pub fn make_regression_dataset(n_samples: usize, n_features: usize, noise: f64) -> Dataset {
    let mut data = Vec::new();
    let mut labels = Vec::new();

    // True weights for synthetic linear relationship
    let true_weights: Vec<f64> = (0..n_features)
        .map(|_| rand::random::<f64>() * 2.0 - 1.0)
        .collect();

    for _ in 0..n_samples {
        let sample: Vec<f64> = (0..n_features)
            .map(|_| rand::random::<f64>() * 2.0 - 1.0)
            .collect();

        // Linear combination + noise
        let label: f64 = sample
            .iter()
            .zip(&true_weights)
            .map(|(&x, &w)| x * w)
            .sum::<f64>()
            + (rand::random::<f64>() - 0.5) * noise;

        data.push(Tensor::new(sample, vec![n_features]));
        labels.push(Tensor::from_scalar(label));
    }

    Dataset::new(data, labels)
}
