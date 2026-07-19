use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("missing required tensor: {0}")]
    MissingTensor(String),

    #[error("tensor has incorrect shape: expected {expected:?}, got {got:?}")]
    TensorShape { expected: Vec<usize>, got: Vec<usize> },

    #[error("model requires {required} but only {available} available")]
    InsufficientMemory { required: String, available: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("format error: {0}")]
    Format(String),

    #[error("invalid operation: {0}")]
    InvalidOperation(String),

    #[error("context overflow: max {max} tokens, requested {requested}")]
    ContextOverflow { max: usize, requested: usize },

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

