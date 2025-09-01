use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableTensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
}

impl From<&Tensor> for SerializableTensor {
    fn from(tensor: &Tensor) -> Self {
        SerializableTensor {
            data: tensor.data.clone(),
            shape: tensor.shape.clone(),
            requires_grad: tensor.requires_grad,
        }
    }
}

impl From<SerializableTensor> for Tensor {
    fn from(serializable: SerializableTensor) -> Self {
        Tensor {
            data: serializable.data,
            shape: serializable.shape,
            requires_grad: serializable.requires_grad,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelState {
    pub parameters: HashMap<String, SerializableTensor>,
    pub metadata: HashMap<String, String>,
}

impl ModelState {
    pub fn new() -> Self {
        ModelState {
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_parameter(&mut self, name: String, tensor: &Tensor) {
        self.parameters
            .insert(name, SerializableTensor::from(tensor));
    }

    pub fn get_parameter(&self, name: &str) -> Option<Tensor> {
        self.parameters.get(name).map(|st| Tensor::from(st.clone()))
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    // Save to JSON format (human-readable)
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    // Load from JSON format
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let model_state: ModelState = serde_json::from_str(&contents)?;
        Ok(model_state)
    }

    // Save to binary format (smaller file size)
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    // Load from binary format
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let model_state: ModelState = bincode::deserialize(&buffer)?;
        Ok(model_state)
    }
}

// Trait for models that can be serialized
pub trait Saveable {
    fn save_state(&self) -> ModelState;
    fn load_state(&mut self, state: &ModelState) -> Result<(), String>;

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let state = self.save_state();
        if path.as_ref().extension().and_then(|s| s.to_str()) == Some("json") {
            state.save_json(path)
        } else {
            state.save_binary(path)
        }
    }

    fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let state = if path.as_ref().extension().and_then(|s| s.to_str()) == Some("json") {
            ModelState::load_json(path)?
        } else {
            ModelState::load_binary(path)?
        };
        self.load_state(&state).map_err(|e| e.into())
    }
}
