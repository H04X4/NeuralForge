# Architecture

NeuralForge is a modular Rust workspace with 11 crates. Each crate has a single responsibility, and the dependency graph is a directed acyclic graph rooted at `astra-core`.

## Crate Dependency Graph

```
astra-core  ←──────────────────────────────────────────────────────────┐
  ├── astra-formats  ←─────────────────────────────────────────────┐   │
  │     ├── astra-planner     astra-tokenizers                     │   │
  │     ├── astra-qwen ─── astra-kernels                           │   │
  │     │     └── astra-runtime ───┬── astra-api                   │   │
  │     └──────────────────────────┤                               │   │
  └────────────────────────────────┤                               │   │
                                   │                               │   │
  astra-cli ─── depends on ALL other crates                        │   │
  astra-hardware ── depends only on astra-core                     │   │
  astra-models ── depends on astra-core + reqwest                  │   │
                                                                   │   │
  astra-core provides: DType, Shape, TensorId, Error, Result,      │   │
                      HardwareInventory, ExecutionPlan,             │   │
                      ModelManifest, ModelError                     │   │
```

## Crate Responsibilities

### astra-core (foundation)

Shared types used by every other crate. No dependencies beyond serde.

**Key types:**
- `DType` — data type enum (F32, F16, BF16, I32, Q4_K, Q8_0, etc.)
- `Shape` — tensor shape with overflow-safe `num_elements()`
- `TensorId` — newtype wrapper around String for tensor identification
- `Device` — compute device (Cpu, Cuda, Metal, Rocm)
- `HardwareInventory` — OS, arch, CPU count, RAM, storage, detected backends
- `ExecutionPlan` — feasibility estimate with weight size, KV cache, peak RAM, tokens/sec
- `ModelKind` — Dense or Mixture-of-Experts
- `Modality` — Text, Vision, Audio, Embedding, ImageGeneration
- `QualityPolicy` — Exact, Quality, Balanced, Fast, Tiny
- `Error` — unified error enum with thiserror

### astra-formats (model readers)

Two file format readers: GGUF (v3) and Safetensors.

**GGUF reader** (`GgufReader`):
- Validates magic bytes (`0x46554747` = "GGUF")
- Reads version, metadata KV pairs (string, integer, float, bool, arrays)
- Reads tensor index (name, dimensions, dtype, offset)
- Provides `read_tensor(name)` for raw bytes and `read_tensor_f32(name)` with F16→F32 conversion
- Safety limits: max 100K tensors, 1 MB string length, 100K array length
- 32-byte aligned data start offset

**Safetensors reader** (`SafetensorsReader`):
- Reads 8-byte header size, then JSON header
- Parses tensor metadata (dtype, shape, data_offsets)
- Safety limit: 100 MB max header size

### astra-hardware (system discovery)

Platform-specific hardware detection using conditional compilation:

| OS | RAM detection | GPU detection |
|----|---------------|---------------|
| macOS | `sysctl hw.memsize` + Mach `host_statistics64` for available RAM | Metal framework presence |
| Linux | (stub — returns 8 GB default) | (not yet) |
| Windows | (stub — returns 8 GB default) | (not yet) |

Also detects logical/physical CPU count and free disk space via `statfs`.

### astra-planner (estimation engine)

Given a model file (GGUF/Safetensors) or raw parameters, estimates:
- Weight size (from file metadata or raw bytes)
- KV cache size (~576 bytes per token per layer for Qwen3 MLA; ~2×4×hidden_size×n_layers for MHA)
- Peak RAM (weights + KV cache + ~64 MB working set)
- Tokens/second estimate (heuristic based on model size / RAM ratio)

### astra-kernels (math primitives)

CPU kernels for transformer inference. No BLAS dependency — pure scalar Rust.

| Module | Functions | Complexity |
|--------|-----------|------------|
| `matmul` | `matmul`, `matmul_tiled`, `matmul_q8_0`, `matmul_q4_0`, `quantize_q8_0` | O(m·n·k) scalar, tiled (32×32×256) |
| `attention` | `causal_attention` | O(n_heads · seq_len² · head_dim) |
| `norm` | `rms_norm` | O(n) |
| `op` | `silu`, `swiglu`, `rope`, `rope_qwen`, `softmax_inplace` | O(n) |

**Quantization support:**

`matmul_q8_0` — weight matrix stored as Q8_0 blocks (32 elements + f16 scale per block), dequantized on-the-fly during multiplication. `matmul_q4_0` — similar but 4-bit with symmetric quantization.

### astra-tokenizers (BPE tokenizer)

tiktoken-compatible byte-level BPE tokenizer:
- Reads standard `.tiktoken` rank files (base64-encoded token bytes → rank)
- GPT-2 pre-tokenization pattern (split on regex)
- BPE merge via greedy rank-based pair selection
- Handles OOV bytes by falling through to single-byte tokens

### astra-qwen (Qwen3 model)

The model implementation:

**`QwenConfig`** — parsed from GGUF metadata:
- `embedding_length`, `block_count`, `head_count`, `head_dim`
- `feed_forward_length`, `rope_theta`, `rms_norm_eps`
- `max_seq_len`, `n_kv_heads` (GQA support)

**`QwenWeights`** — loads all per-layer tensors from GGUF:
- `token_embd` — token embedding table
- Per layer: `attn_norm`, `attn_q/k/v/output`, `ffn_norm`, `ffn_gate/up/down`
- `output_norm`, `output` (LM head, tied with embeddings)

**`KVCache`** — incremental key-value cache:
- Per-layer storage for K and V with `insert()`, `k()`, `v()` methods
- Grows with sequence length, indexed by position

**`QwenModel::forward()`** — single-token forward pass:

```
Input token + position
    │
    ├── Token embedding lookup (token_embd[token])
    │
    └── For each layer:
        │
        ├── x = RMSNorm(x, attn_norm)
        ├── Q = attn_q(x), K = attn_k(x), V = attn_v(x)
        ├── Apply RoPE to Q and K (rope_qwen)
        ├── cache.insert(pos, K, V)
        ├── K_full = cache.k()[:pos+1], V_full = cache.v()[:pos+1]
        ├── scores = Q @ K_full^T / sqrt(head_dim)
        ├── causal softmax(scores)
        ├── out = scores @ V_full
        ├── x = x + attn_output(out)
        │
        ├── residual = x
        ├── x = RMSNorm(x, ffn_norm)
        ├── gate = ffn_gate(x), up = ffn_up(x), down = ffn_down(x)
        ├── x = residual + down(swiglu(gate, up))
        │
    └── End layers
    
    ├── x = RMSNorm(x, output_norm)
    └── logits = x @ output^T (LM head)
```

### astra-runtime (inference engine)

High-level session management:

**`Session::load(model, tokenizer)`** — loads GGUF, weights, config, and tokenizer in one call.

**`Session::generate(prompt, max_tokens, temperature)`** — full generation loop:
1. Tokenize prompt
2. Prefill: forward all prompt tokens sequentially
3. Decode: autoregressive loop with KV cache reuse
4. Sample: greedy (argmax) or temperature-based multinomial

**Sampling functions:**
- `greedy_sample(logits)` — returns argmax token
- `temperature_sample(logits, temperature)` — softmax with temperature scaling, then multinomial draw

### astra-api (HTTP server)

Axum-based HTTP server exposing three OpenAI-compatible endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (Qwen3 chat format) |
| `/v1/completions` | POST | Raw text completion |

**Chat format:** Converts OpenAI message format to Qwen3's `<|system|>`, `<|user|>`, `<|assistant|>` format.

**Auth:** Optional Bearer token authentication via `Authorization: Bearer <key>`.

**Limits:** Context length enforced, graceful error responses for overflow.

### astra-models (model lifecycle)

Model cache management:
- `default_model_dir()` — returns `~/.cache/neural-forge/models/` (XDG-compatible)
- `pull(repo_id, cache, progress)` — async download from HuggingFace Hub with progress callback
- `list_models(cache)` — scans cache for GGUF files
- `resolve_model(name, cache)` — checks raw path, then cache directory

## Data Flow

```
neural-forge pull
  → astra_models::pull(repo_id)
    → HuggingFace API: /api/models/{repo_id}/tree/main
    → Find first .gguf file
    → Download to ~/.cache/neural-forge/models/{org}__{name}/
    → Progress bar via callback

neural-forge run model.gguf --tokenizer tok.tiktoken "prompt"
  → astra_runtime::Session::load(model, tokenizer)
    → GgufReader::open(model)
    → QwenWeights::load(&gguf)
    → KVCache::new(&cfg, max_seq_len)
    → TiktokenTokenizer::from_file(tokenizer)
  → Session::generate(prompt, max_tokens, temperature)
    → tok.encode(prompt) → token IDs
    → For each prompt token: QwenModel::forward() → prefill KV cache
    → For each decode step:
        → QwenModel::forward() → logits
        → sample(logits) → next token
        → tok.decode(ids) → text

neural-forge serve model.gguf --tokenizer tok.tiktoken
  → axum HTTP server
  → /v1/chat/completions: build prompt → Session::generate → JSON response
```

## Memory Layout

```
┌─────────────────────────┐
│   Token Embeddings      │  vocab_size × embedding_length × dtype_size
├─────────────────────────┤
│   Per Layer (×32):      │
│   ┌───────────────────┐ │
│   │ Attention Q       │ │  embedding_length × head_count × head_dim
│   │ Attention K       │ │  embedding_length × n_kv_heads × head_dim
│   │ Attention V       │ │  embedding_length × n_kv_heads × head_dim
│   │ Attention Output  │ │  head_count × head_dim × embedding_length
│   │ FFN Gate          │ │  embedding_length × feed_forward_length
│   │ FFN Up            │ │  embedding_length × feed_forward_length
│   │ FFN Down          │ │  feed_forward_length × embedding_length
│   │ Attn Norm (RMS)   │ │  embedding_length
│   │ FFN Norm (RMS)    │ │  embedding_length
│   └───────────────────┘ │
├─────────────────────────┤
│   Output Norm (RMS)     │  embedding_length
│   Output (LM head)      │  vocab_size × embedding_length (tied)
├─────────────────────────┤
│   KV Cache              │  2 × n_layers × n_kv_heads × head_dim × seq_len × 4 bytes
└─────────────────────────┘
```
