//! Tensor operations for hybrid quantum-classical models.
//!
//! Provides a handle-based tensor store with standard DNN operations.
//! Backend: ndarray (CPU). API designed to swap to libtorch for GPU.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use ndarray::{Array1, Array2, Axis};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("tensor {0} not found")]
    NotFound(u32),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
}

pub type TensorId = u32;

/// A stored tensor (1D or 2D for now).
#[derive(Clone, Debug)]
enum TensorData {
    Vec1(Array1<f64>),
    Mat2(Array2<f64>),
}

/// Handle-based tensor store. WASM guests interact with tensors via u32 handles.
pub struct TensorStore {
    tensors: HashMap<TensorId, TensorData>,
    next_id: AtomicU32,
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorStore {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            next_id: AtomicU32::new(1),
        }
    }

    fn alloc_id(&self) -> TensorId {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    // ---- Creation ----

    /// Create a 1D tensor from a slice.
    pub fn from_vec(&mut self, data: &[f64]) -> TensorId {
        let id = self.alloc_id();
        self.tensors
            .insert(id, TensorData::Vec1(Array1::from_vec(data.to_vec())));
        id
    }

    /// Create a 2D tensor from flat data + shape.
    pub fn from_data_2d(&mut self, data: &[f64], rows: usize, cols: usize) -> TensorId {
        let id = self.alloc_id();
        let mat = Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap();
        self.tensors.insert(id, TensorData::Mat2(mat));
        id
    }

    /// Create a 2D tensor filled with zeros.
    pub fn zeros_2d(&mut self, rows: usize, cols: usize) -> TensorId {
        let id = self.alloc_id();
        self.tensors
            .insert(id, TensorData::Mat2(Array2::zeros((rows, cols))));
        id
    }

    // ---- Data access ----

    /// Read tensor data as flat f64 slice.
    pub fn to_vec(&self, id: TensorId) -> Result<Vec<f64>, TensorError> {
        match self.tensors.get(&id) {
            Some(TensorData::Vec1(v)) => Ok(v.to_vec()),
            Some(TensorData::Mat2(m)) => Ok(m.iter().copied().collect()),
            None => Err(TensorError::NotFound(id)),
        }
    }

    /// Get shape as (rows, cols). 1D tensors return (len, 1).
    pub fn shape(&self, id: TensorId) -> Result<(usize, usize), TensorError> {
        match self.tensors.get(&id) {
            Some(TensorData::Vec1(v)) => Ok((v.len(), 1)),
            Some(TensorData::Mat2(m)) => Ok((m.nrows(), m.ncols())),
            None => Err(TensorError::NotFound(id)),
        }
    }

    /// Free a tensor.
    pub fn free(&mut self, id: TensorId) {
        self.tensors.remove(&id);
    }

    // ---- Operations ----

    /// Matrix multiply: C = A @ B
    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TensorError> {
        let a_mat = self.get_mat2(a)?;
        let b_mat = self.get_mat2(b)?;
        let c = a_mat.dot(&b_mat);
        let id = self.alloc_id();
        self.tensors.insert(id, TensorData::Mat2(c));
        Ok(id)
    }

    /// Matrix-vector multiply: y = A @ x
    pub fn matvec(&mut self, a: TensorId, x: TensorId) -> Result<TensorId, TensorError> {
        let a_mat = self.get_mat2(a)?;
        let x_vec = self.get_vec1(x)?;
        let y = a_mat.dot(&x_vec);
        let id = self.alloc_id();
        self.tensors.insert(id, TensorData::Vec1(y));
        Ok(id)
    }

    /// Element-wise ReLU: max(0, x)
    pub fn relu(&mut self, t: TensorId) -> Result<TensorId, TensorError> {
        let id = self.alloc_id();
        match self.tensors.get(&t) {
            Some(TensorData::Vec1(v)) => {
                self.tensors
                    .insert(id, TensorData::Vec1(v.mapv(|x| x.max(0.0))));
            }
            Some(TensorData::Mat2(m)) => {
                self.tensors
                    .insert(id, TensorData::Mat2(m.mapv(|x| x.max(0.0))));
            }
            None => return Err(TensorError::NotFound(t)),
        }
        Ok(id)
    }

    /// Element-wise tanh
    pub fn tanh(&mut self, t: TensorId) -> Result<TensorId, TensorError> {
        let id = self.alloc_id();
        match self.tensors.get(&t) {
            Some(TensorData::Vec1(v)) => {
                self.tensors
                    .insert(id, TensorData::Vec1(v.mapv(|x| x.tanh())));
            }
            Some(TensorData::Mat2(m)) => {
                self.tensors
                    .insert(id, TensorData::Mat2(m.mapv(|x| x.tanh())));
            }
            None => return Err(TensorError::NotFound(t)),
        }
        Ok(id)
    }

    /// Softmax over a 1D vector.
    pub fn softmax(&mut self, t: TensorId) -> Result<TensorId, TensorError> {
        let v = self.get_vec1(t)?;
        let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Array1<f64> = v.mapv(|x| (x - max).exp());
        let sum: f64 = exp.sum();
        let result = exp / sum;
        let id = self.alloc_id();
        self.tensors.insert(id, TensorData::Vec1(result));
        Ok(id)
    }

    /// Linear layer: y = W @ x + b
    pub fn linear(
        &mut self,
        weight: TensorId,
        input: TensorId,
        bias: TensorId,
    ) -> Result<TensorId, TensorError> {
        let w = self.get_mat2(weight)?;
        let x = self.get_vec1(input)?;
        let b = self.get_vec1(bias)?;
        let y = w.dot(&x) + &b;
        let id = self.alloc_id();
        self.tensors.insert(id, TensorData::Vec1(y));
        Ok(id)
    }

    /// Add two tensors element-wise.
    pub fn add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TensorError> {
        let id = self.alloc_id();
        match (self.tensors.get(&a), self.tensors.get(&b)) {
            (Some(TensorData::Vec1(va)), Some(TensorData::Vec1(vb))) => {
                self.tensors.insert(id, TensorData::Vec1(va + vb));
            }
            (Some(TensorData::Mat2(ma)), Some(TensorData::Mat2(mb))) => {
                self.tensors.insert(id, TensorData::Mat2(ma + mb));
            }
            (None, _) => return Err(TensorError::NotFound(a)),
            (_, None) => return Err(TensorError::NotFound(b)),
            _ => {
                return Err(TensorError::ShapeMismatch(
                    "mismatched tensor types for add".into(),
                ))
            }
        }
        Ok(id)
    }

    /// Scalar multiply.
    pub fn scale(&mut self, t: TensorId, scalar: f64) -> Result<TensorId, TensorError> {
        let id = self.alloc_id();
        match self.tensors.get(&t) {
            Some(TensorData::Vec1(v)) => {
                self.tensors
                    .insert(id, TensorData::Vec1(v.mapv(|x| x * scalar)));
            }
            Some(TensorData::Mat2(m)) => {
                self.tensors
                    .insert(id, TensorData::Mat2(m.mapv(|x| x * scalar)));
            }
            None => return Err(TensorError::NotFound(t)),
        }
        Ok(id)
    }

    /// Sum all elements.
    pub fn sum(&self, t: TensorId) -> Result<f64, TensorError> {
        match self.tensors.get(&t) {
            Some(TensorData::Vec1(v)) => Ok(v.sum()),
            Some(TensorData::Mat2(m)) => Ok(m.sum()),
            None => Err(TensorError::NotFound(t)),
        }
    }

    // ---- Internal helpers ----

    fn get_mat2(&self, id: TensorId) -> Result<Array2<f64>, TensorError> {
        match self.tensors.get(&id) {
            Some(TensorData::Mat2(m)) => Ok(m.clone()),
            Some(TensorData::Vec1(v)) => {
                // Promote to column matrix
                Ok(v.clone().insert_axis(Axis(1)))
            }
            None => Err(TensorError::NotFound(id)),
        }
    }

    fn get_vec1(&self, id: TensorId) -> Result<Array1<f64>, TensorError> {
        match self.tensors.get(&id) {
            Some(TensorData::Vec1(v)) => Ok(v.clone()),
            Some(TensorData::Mat2(m)) => {
                // Flatten
                Ok(Array1::from_iter(m.iter().copied()))
            }
            None => Err(TensorError::NotFound(id)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_read() {
        let mut store = TensorStore::new();
        let id = store.from_vec(&[1.0, 2.0, 3.0]);
        let data = store.to_vec(id).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert_eq!(store.shape(id).unwrap(), (3, 1));
    }

    #[test]
    fn test_matmul() {
        let mut store = TensorStore::new();
        // 2x3 @ 3x2 = 2x2
        let a = store.from_data_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = store.from_data_2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = store.matmul(a, b).unwrap();
        let data = store.to_vec(c).unwrap();
        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        assert_eq!(data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_linear() {
        let mut store = TensorStore::new();
        let w = store.from_data_2d(&[1.0, 0.0, 0.0, 1.0], 2, 2); // identity
        let x = store.from_vec(&[3.0, 4.0]);
        let b = store.from_vec(&[1.0, 2.0]);
        let y = store.linear(w, x, b).unwrap();
        let data = store.to_vec(y).unwrap();
        assert_eq!(data, vec![4.0, 6.0]); // [3+1, 4+2]
    }

    #[test]
    fn test_relu() {
        let mut store = TensorStore::new();
        let t = store.from_vec(&[-1.0, 0.0, 1.0, -0.5, 2.0]);
        let r = store.relu(t).unwrap();
        let data = store.to_vec(r).unwrap();
        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_softmax() {
        let mut store = TensorStore::new();
        let t = store.from_vec(&[1.0, 2.0, 3.0]);
        let s = store.softmax(t).unwrap();
        let data = store.to_vec(s).unwrap();
        let total: f64 = data.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
        // softmax should be monotonically increasing
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);
    }

    #[test]
    fn test_tanh() {
        let mut store = TensorStore::new();
        let t = store.from_vec(&[0.0, 1.0, -1.0]);
        let r = store.tanh(t).unwrap();
        let data = store.to_vec(r).unwrap();
        assert!((data[0] - 0.0).abs() < 1e-10);
        assert!((data[1] - 1.0_f64.tanh()).abs() < 1e-10);
        assert!((data[2] - (-1.0_f64).tanh()).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_linear_relu_linear() {
        let mut store = TensorStore::new();
        // Simple 2-layer network: linear(2->3) -> relu -> linear(3->1)
        let w1 = store.from_data_2d(&[0.5, -0.3, 0.2, 0.1, -0.4, 0.7], 3, 2);
        let b1 = store.from_vec(&[0.1, 0.0, -0.1]);
        let w2 = store.from_data_2d(&[1.0, 1.0, 1.0], 1, 3);
        let b2 = store.from_vec(&[0.0]);

        let x = store.from_vec(&[1.0, 2.0]);

        // Forward pass
        let h = store.linear(w1, x, b1).unwrap();
        let h_relu = store.relu(h).unwrap();
        let y = store.linear(w2, h_relu, b2).unwrap();

        let output = store.to_vec(y).unwrap();
        assert_eq!(output.len(), 1);
        // w1 @ [1,2] = [0.5*1+(-0.3)*2, 0.2*1+0.1*2, -0.4*1+0.7*2] = [-0.1, 0.4, 1.0]
        // + b1 = [0.0, 0.4, 0.9]
        // relu = [0.0, 0.4, 0.9]
        // w2 @ relu + b2 = [0+0.4+0.9] = [1.3]
        assert!((output[0] - 1.3).abs() < 1e-10, "output = {:?}", output);
    }
}
