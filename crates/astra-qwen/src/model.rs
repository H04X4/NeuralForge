use crate::config::QwenConfig;
use astra_core::Result;
use astra_formats::gguf::GgufFile;
use astra_kernels::{matmul, rms_norm, rope_qwen, silu, softmax_inplace};

pub struct QwenWeights {
    pub token_embd: Vec<f32>,

    pub attn_norm: Vec<Vec<f32>>,
    pub attn_q: Vec<Vec<f32>>,
    pub attn_k: Vec<Vec<f32>>,
    pub attn_v: Vec<Vec<f32>>,
    pub attn_output: Vec<Vec<f32>>,

    pub ffn_norm: Vec<Vec<f32>>,
    pub ffn_gate: Vec<Vec<f32>>,
    pub ffn_up: Vec<Vec<f32>>,
    pub ffn_down: Vec<Vec<f32>>,

    pub output_norm: Vec<f32>,
    pub output: Vec<f32>,

    pub cfg: QwenConfig,
}

fn load_layer(gguf: &GgufFile, i: usize, name: &str) -> Result<Vec<f32>> {
    gguf.read_tensor_f32(&format!("blk.{i}.{name}"))
}

impl QwenWeights {
    pub fn load(gguf: &GgufFile) -> Result<Self> {
        let cfg = QwenConfig::from_gguf(gguf)?;
        let n = cfg.n_layers;

        let token_embd = gguf.read_tensor_f32("token_embd.weight")?;

        let mut attn_norm = Vec::with_capacity(n);
        let mut attn_q = Vec::with_capacity(n);
        let mut attn_k = Vec::with_capacity(n);
        let mut attn_v = Vec::with_capacity(n);
        let mut attn_output = Vec::with_capacity(n);
        let mut ffn_norm = Vec::with_capacity(n);
        let mut ffn_gate = Vec::with_capacity(n);
        let mut ffn_up = Vec::with_capacity(n);
        let mut ffn_down = Vec::with_capacity(n);

        for i in 0..n {
            attn_norm.push(load_layer(gguf, i, "attn_norm.weight")?);
            attn_q.push(load_layer(gguf, i, "attn_q.weight")?);
            attn_k.push(load_layer(gguf, i, "attn_k.weight")?);
            attn_v.push(load_layer(gguf, i, "attn_v.weight")?);
            attn_output.push(load_layer(gguf, i, "attn_output.weight")?);
            ffn_norm.push(load_layer(gguf, i, "ffn_norm.weight")?);
            ffn_gate.push(load_layer(gguf, i, "ffn_gate.weight")?);
            ffn_up.push(load_layer(gguf, i, "ffn_up.weight")?);
            ffn_down.push(load_layer(gguf, i, "ffn_down.weight")?);
        }

        let output_norm = gguf.read_tensor_f32("output_norm.weight")?;
        let output = gguf.read_tensor_f32("output.weight")?;

        Ok(QwenWeights {
            token_embd,
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            output_norm,
            output,
            cfg,
        })
    }
}

pub struct KVCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub max_seq_len: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl KVCache {
    pub fn new(cfg: &QwenConfig, max_seq_len: usize) -> Self {
        let n_layers = cfg.n_layers;
        let kv_size = max_seq_len * cfg.n_kv_heads * cfg.head_dim;
        KVCache {
            k: vec![vec![0.0f32; kv_size]; n_layers],
            v: vec![vec![0.0f32; kv_size]; n_layers],
            max_seq_len,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
        }
    }

    pub fn insert(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let kv_len = self.n_kv_heads * self.head_dim;
        let base = pos * kv_len;
        let k_slot = &mut self.k[layer][base..base + kv_len];
        k_slot.copy_from_slice(k);
        let v_slot = &mut self.v[layer][base..base + kv_len];
        v_slot.copy_from_slice(v);
    }

    pub fn k(&self, layer: usize, seq_len: usize) -> &[f32] {
        &self.k[layer][..seq_len * self.n_kv_heads * self.head_dim]
    }

    pub fn v(&self, layer: usize, seq_len: usize) -> &[f32] {
        &self.v[layer][..seq_len * self.n_kv_heads * self.head_dim]
    }
}

pub struct QwenModel;

impl QwenModel {
    pub fn forward(weights: &QwenWeights, token: u32, pos: usize, cache: &mut KVCache) -> Result<Vec<f32>> {
        let cfg = &weights.cfg;
        let h = cfg.hidden_size;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let group_size = n_heads / n_kv;

        let mut hidden = vec![0.0f32; h];

        let token = token as usize;
        let embd_row = &weights.token_embd[token * h..(token + 1) * h];
        hidden.copy_from_slice(embd_row);

        for layer in 0..cfg.n_layers {
            let mut residual = hidden.clone();

            rms_norm(&mut hidden, &weights.attn_norm[layer], cfg.rms_norm_eps);

            let mut q = vec![0.0f32; n_heads * head_dim];
            matmul(&mut q, &hidden, &weights.attn_q[layer], 1, n_heads * head_dim, h)?;

            let mut k = vec![0.0f32; n_kv * head_dim];
            matmul(&mut k, &hidden, &weights.attn_k[layer], 1, n_kv * head_dim, h)?;

            let mut v = vec![0.0f32; n_kv * head_dim];
            matmul(&mut v, &hidden, &weights.attn_v[layer], 1, n_kv * head_dim, h)?;

            rope_qwen(&mut q, pos, head_dim, cfg.rope_theta);
            rope_qwen(&mut k, pos, head_dim, cfg.rope_theta);

            cache.insert(layer, pos, &k, &v);

            let seq_len = pos + 1;
            let k_cache = cache.k(layer, seq_len);
            let v_cache = cache.v(layer, seq_len);

            let mut attn_out = vec![0.0f32; n_heads * head_dim];

            let scale = 1.0 / (head_dim as f32).sqrt();

            for g in 0..n_kv {
                for h_idx in 0..group_size {
                    let q_head = g * group_size + h_idx;
                    let q_off = q_head * head_dim;

                    let mut scores = vec![0.0f32; seq_len];
                    for (t, score) in scores.iter_mut().enumerate() {
                        let k_off = t * n_kv * head_dim + g * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_off + d] * k_cache[k_off + d];
                        }
                        *score = dot * scale;
                    }

                    softmax_inplace(&mut scores);

                    for d in 0..head_dim {
                        let mut val = 0.0f32;
                        for (t, score) in scores.iter().enumerate() {
                            let v_off = t * n_kv * head_dim + g * head_dim;
                            val += score * v_cache[v_off + d];
                        }
                        attn_out[q_off + d] = val;
                    }
                }
            }

            let mut attn_proj = vec![0.0f32; h];
            matmul(
                &mut attn_proj,
                &attn_out,
                &weights.attn_output[layer],
                1,
                h,
                n_heads * head_dim,
            )?;

            for i in 0..h {
                hidden[i] = residual[i] + attn_proj[i];
            }

            residual = hidden.clone();

            rms_norm(&mut hidden, &weights.ffn_norm[layer], cfg.rms_norm_eps);

            let ffn_dim = cfg.ffn_dim;
            let mut gate = vec![0.0f32; ffn_dim];
            matmul(&mut gate, &hidden, &weights.ffn_gate[layer], 1, ffn_dim, h)?;

            let mut up = vec![0.0f32; ffn_dim];
            matmul(&mut up, &hidden, &weights.ffn_up[layer], 1, ffn_dim, h)?;

            for i in 0..ffn_dim {
                gate[i] = silu(gate[i]);
                gate[i] *= up[i];
            }

            let mut ffn_out = vec![0.0f32; h];
            matmul(&mut ffn_out, &gate, &weights.ffn_down[layer], 1, h, ffn_dim)?;

            for i in 0..h {
                hidden[i] = residual[i] + ffn_out[i];
            }
        }

        rms_norm(&mut hidden, &weights.output_norm, cfg.rms_norm_eps);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        matmul(&mut logits, &hidden, &weights.output, 1, cfg.vocab_size, h)?;

        Ok(logits)
    }
}


