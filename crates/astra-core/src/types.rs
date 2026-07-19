use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(u32),
    Metal,
    Rocm(u32),
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "CPU"),
            Device::Cuda(i) => write!(f, "CUDA({})", i),
            Device::Metal => write!(f, "Metal"),
            Device::Rocm(i) => write!(f, "ROCm({})", i),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelKind {
    Dense,
    MoE,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Vision,
    Audio,
    Embedding,
    ImageGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityPolicy {
    Exact,
    Quality,
    Balanced,
    Fast,
    Tiny,
}

impl QualityPolicy {
    pub fn as_str(&self) -> &'static str {
        match self {
            QualityPolicy::Exact => "exact",
            QualityPolicy::Quality => "quality",
            QualityPolicy::Balanced => "balanced",
            QualityPolicy::Fast => "fast",
            QualityPolicy::Tiny => "tiny",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub model_name: String,
    pub model_kind: ModelKind,
    pub modalities: Vec<Modality>,
    pub weight_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub peak_ram_bytes: u64,
    pub peak_vram_bytes: u64,
    pub disk_bytes: u64,
    pub quality: QualityPolicy,
    pub estimated_tok_s: f64,
    pub bottleneck: String,
    pub feasible: bool,
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    Rocm,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Cpu => write!(f, "CPU"),
            BackendKind::Cuda => write!(f, "CUDA"),
            BackendKind::Metal => write!(f, "Metal"),
            BackendKind::Vulkan => write!(f, "Vulkan"),
            BackendKind::Rocm => write!(f, "ROCm"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInventory {
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    pub physical_cpus: usize,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub storage_free_bytes: u64,
    pub backends: Vec<BackendKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub name: String,
    pub kind: ModelKind,
    pub architecture: String,
    pub modalities: Vec<Modality>,
    pub tensors: Vec<TensorEntry>,
    pub metadata: std::collections::HashMap<String, String>,
    pub source_url: Option<String>,
    pub source_revision: Option<String>,
    pub license: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub id: crate::tensor::TensorId,
    pub dtype: crate::dtype::DType,
    pub shape: crate::shape::Shape,
    pub offset: u64,
    pub size_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model not found: {0}")]
    NotFound(String),

    #[error("download failed: {0}")]
    Download(String),

    #[error("checksum mismatch")]
    ChecksumMismatch,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

