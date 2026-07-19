#![deny(unsafe_code)]

pub mod dtype;
pub mod error;
pub mod shape;
pub mod tensor;
pub mod types;

pub use dtype::DType;
pub use error::{Error, Result};
pub use shape::Shape;
pub use tensor::TensorId;
pub use types::*;

