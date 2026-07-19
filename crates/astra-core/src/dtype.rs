use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
    Q4K,
    Q4KM,
    Q5K,
    Q6K,
    Q8K,
    Unknown(String),
}

impl DType {
    pub fn size_in_bytes(&self) -> Option<usize> {
        match self {
            DType::F32 => Some(4),
            DType::F16 | DType::BF16 => Some(2),
            DType::I32 => Some(4),
            DType::I64 => Some(8),
            DType::U8 => Some(1),
            DType::Q4K => Some(1),
            DType::Q4KM => Some(1),
            DType::Q5K => Some(1),
            DType::Q6K => Some(1),
            DType::Q8K => Some(1),
            DType::Unknown(_) => None,
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::Q4K => write!(f, "q4k"),
            DType::Q4KM => write!(f, "q4km"),
            DType::Q5K => write!(f, "q5k"),
            DType::Q6K => write!(f, "q6k"),
            DType::Q8K => write!(f, "q8k"),
            DType::Unknown(s) => write!(f, "unknown({})", s),
        }
    }
}


