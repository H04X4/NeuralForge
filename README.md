# NeuralForge

**forge intelligence — universal local AI runtime for Qwen3 models**

NeuralForge is a high-performance CPU-first inference engine for open-source Qwen3 models. It runs entirely on your machine — no cloud, no telemetry, no GPU required. Download a model from HuggingFace and run it in seconds.

```
neural-forge pull Qwen/Qwen3-1.5B-GGUF
neural-forge run Qwen3-1.5B-GGUF/qwen3-1.5b-q4_k_m.gguf --tokenizer qwen3.tiktoken "write a poem about Rust"
```

## Features

- **CPU-first** — runs on any x86_64 or ARM64 machine with 8 GB RAM or more
- **Qwen3-native** — built for Qwen3 dense and MoE architectures from day one
- **GGUF + Safetensors** — reads both formats, with GGUF quantized inference (Q4_0, Q8_0)
- **OpenAI-compatible API** — drop-in replacement for local development
- **Hardware-aware planner** — estimates memory and throughput before loading
- **No Python** — pure Rust, single static binary
- **Apache 2.0** — free for any use

## Quick Start

### 1. Download a Model

```bash
neural-forge pull Qwen/Qwen3-1.5B-GGUF
```

Lists available GGUF files from HuggingFace and downloads with a progress bar. Cached in `~/.cache/neural-forge/models/`.

### 2. Run a Prompt

```bash
neural-forge run ./models/qwen3-1.5b-q4_k_m.gguf \
  --tokenizer ./qwen3.tiktoken \
  "explain quantum computing in one paragraph"
```

### 3. Chat Interactively

```bash
neural-forge chat ./models/qwen3-1.5b-q4_k_m.gguf --tokenizer ./qwen3.tiktoken
```

### 4. Start an API Server

```bash
neural-forge serve ./models/qwen3-1.5b-q4_k_m.gguf \
  --tokenizer ./qwen3.tiktoken \
  --host 0.0.0.0 --port 8080
```

Then use any OpenAI client:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-1.5b",
    "messages": [{"role": "user", "content": "hello!"}]
  }'
```

## Requirements

| Requirement | Minimum |
|-------------|---------|
| **RAM** | 8 GB (16 GB recommended for 7B+ models) |
| **OS** | macOS 12+, Linux, Windows 10+ |
| **Arch** | x86_64, ARM64 (Apple Silicon) |
| **Disk** | 2 GB free per model |

## CLI Reference

| Command | Description |
|---------|-------------|
| `doctor` | Inspect system hardware and available backends |
| `plan` | Estimate memory and speed for a model before downloading |
| `pull` | Download a model from HuggingFace Hub |
| `list` | Show locally cached models |
| `run` | Run a single prompt and print the completion |
| `chat` | Interactive multi-turn chat session |
| `serve` | Start an OpenAI-compatible HTTP API server |
| `inspect` | Show metadata and tensor layout of a GGUF/Safetensors file |
| `tokenize` | Tokenize text and show token IDs |
| `build` | Build the engine (compile binaries) |

### Common Flags

- `--json` — machine-readable JSON output (doctor, plan, list, inspect, tokenize)
- `--temperature <float>` — sampling temperature (0.0 = greedy, 0.7 = default for chat)
- `--max-tokens <int>` — maximum tokens to generate
- `--tokenizer <path>` — path to tiktoken `.tiktoken` rank file (required for run/chat/serve)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     neural-forge (CLI)                   │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌───────────────┐ │
│  │ doctor│ │ plan │ │ pull │ │ list │ │ run/chat/serve│ │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └───────┬───────┘ │
└─────┼────────┼────────┼────────┼──────────────┼─────────┘
      │        │        │        │              │
┌─────▼────────▼────────▼────────▼──────────────▼─────────┐
│                    Workspace Crates                       │
│                                                          │
│  astra-core    astra-formats    astra-hardware            │
│  astra-planner astra-kernels    astra-tokenizers          │
│  astra-qwen    astra-runtime    astra-api                 │
│  astra-models                                            │
└──────────────────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full crate dependency graph and data flow.

## Project Structure

```
crates/
├── astra-core/        # Shared types: DType, Shape, TensorId, HardwareInventory, ExecutionPlan
├── astra-cli/         # CLI binary and command implementations
├── astra-formats/     # GGUF and Safetensors file format readers
├── astra-hardware/    # Hardware detection (CPU, RAM, GPU backends)
├── astra-planner/     # Memory and performance estimation
├── astra-kernels/     # CPU math kernels: matmul, attention, RoPE, RMS norm, softmax, SwiGLU
├── astra-tokenizers/  # tiktoken BPE tokenizer
├── astra-qwen/        # Qwen3 model definition: config, weights, forward pass, KV cache
├── astra-runtime/     # Session management, generation loop, sampling strategies
├── astra-api/         # OpenAI-compatible HTTP API server (axum)
└── astra-models/      # Model lifecycle: download from HuggingFace, local cache management
```

## How It Works

### Inference Pipeline

```
Input Text
    │
    ▼
┌─────────────────┐
│  Tokenizer       │  Qwen3 tiktoken BPE → token IDs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  QwenModel      │  Prefill: process all prompt tokens in sequence
│  .forward()     │     - Token embedding lookup
│                 │     - For each layer:
│                 │         RMSNorm → QKV projection → RoPE
│                 │         GQA causal attention (KV cache update)
│                 │         Output projection → residual
│                 │         SwiGLU FFN → residual
│                 │     - Final RMSNorm → LM head → logits
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Sampler        │  Greedy (argmax) or temperature-based sampling
└────────┬────────┘
         │
         ▼
┌─────────────────┐  Decode: one token at a time, reusing KV cache
│  Autoregressive │
│  Decode Loop    │  Repeat until max_tokens or EOS
└────────┬────────┘
         │
         ▼
      Output Text
```

### KV Cache

The key-value cache stores per-layer K and V projections from previous tokens. At each decode step, only the new token's K/V are computed and appended. This avoids recomputing the full attention for every token — the cache grows linearly with sequence length.

### Quantization

NeuralForge supports GGUF-quantized models via on-the-fly dequantization in the matmul kernel:

| Quant | Storage per weight | Speed vs F32 |
|-------|-------------------|--------------|
| F32   | 4 bytes           | 1×           |
| F16   | 2 bytes           | ≈1.5×        |
| Q8_0  | 1 byte + scale    | ≈2-3×        |
| Q4_0  | 0.5 byte + scale  | ≈3-4×        |
| Q4_K_M| 0.5 byte + scale  | ≈3-4×        |

## License

Apache 2.0. See [LICENSE](LICENSE) or http://apache.org/licenses/LICENSE-2.0.
