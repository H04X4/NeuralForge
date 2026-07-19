use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn num_elements(&self) -> Option<usize> {
        self.0.iter().try_fold(1usize, |a, b| a.checked_mul(*b))
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty() || self.0 == [1]
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape(dims)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.0.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" × ")
        )
    }
}


