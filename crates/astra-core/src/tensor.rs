use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(String);

impl TensorId {
    pub fn new(name: impl Into<String>) -> Self {
        TensorId(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TensorId {
    fn from(s: &str) -> Self {
        TensorId(s.to_string())
    }
}

impl std::fmt::Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

