use astra_core::{Error, Result};
use astra_formats::gguf::GgufFile;

#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl QwenConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .architecture()
            .ok_or_else(|| Error::Other("no architecture in GGUF metadata".into()))?;
        let valid = ["qwen3", "qwen2", "llama"];
        if !valid.contains(&arch) {
            return Err(Error::UnsupportedArchitecture(arch.to_string()));
        }

        let prefix = arch;

        let hidden_size = gguf
            .metadata
            .get(&format!("{prefix}.embedding_length"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::Other("missing embedding_length".into()))? as usize;

        let n_layers = gguf
            .metadata
            .get(&format!("{prefix}.block_count"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::Other("missing block_count".into()))? as usize;

        let n_heads = gguf
            .metadata
            .get(&format!("{prefix}.attention.head_count"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::Other("missing head_count".into()))? as usize;

        let n_kv_heads = gguf
            .metadata
            .get(&format!("{prefix}.attention.head_count_kv"))
            .and_then(|v| v.as_u64())
            .map_or(n_heads, |v| v as usize);

        let ffn_dim = gguf
            .metadata
            .get(&format!("{prefix}.feed_forward_length"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::Other("missing feed_forward_length".into()))? as usize;

        let vocab_size = gguf
            .metadata
            .get(&format!("{prefix}.vocab_size"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::Other("missing vocab_size".into()))? as usize;

        let max_seq_len = gguf
            .metadata
            .get(&format!("{prefix}.context_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(8192) as usize;

        let rope_theta = gguf
            .metadata
            .get(&format!("{prefix}.rope.freq_base"))
            .and_then(|v| v.as_f64().map(|x| x as f32))
            .unwrap_or(1000000.0);

        let rms_norm_eps = gguf
            .metadata
            .get(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f64().map(|x| x as f32))
            .unwrap_or(1e-6);

        let head_dim = hidden_size / n_heads;

        Ok(QwenConfig {
            hidden_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
        })
    }
}

