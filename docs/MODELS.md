# Model Support

## Supported Formats

| Format | Status | Features |
|--------|--------|----------|
| GGUF v3 | Full | Metadata, tensors, Q4_0/Q8_0 quantized inference |
| Safetensors | Read-only | Metadata extraction, planning only |

## Supported Architectures

| Architecture | Support Level | Notes |
|-------------|---------------|-------|
| Qwen3 (dense) | Full | 0.5B through 7B tested |
| Qwen3 (MoE) | Partial | Weight loading works, inference tuning in progress |
| Qwen3-VL | Planned | Vision modality support |
| Qwen3 Embeddings | Planned | Reranker and embedding endpoints |
| Qwen3 Audio | Planned | ASR/TTS support |

## Hardware Requirements

| Model Size | Min RAM | Recommended RAM | Speed (tok/s, CPU) |
|-----------|---------|----------------|-------------------|
| 0.5B Q4 | 4 GB | 8 GB | 25-50 |
| 1.5B Q4 | 6 GB | 8 GB | 15-25 |
| 1.5B F16 | 8 GB | 16 GB | 8-15 |
| 7B Q4 | 8 GB | 16 GB | 3-8 |
| 7B F16 | 16 GB | 32 GB | 1-3 |
| 32B Q4 | 16 GB | 32 GB | 0.5-1.5 |
| 72B Q4 | 32 GB | 64 GB | 0.1-0.5 |

## How to Get Tokenizer Files

NeuralForge uses tiktoken `.tiktoken` rank files. You can extract them from HuggingFace model repos:

```bash
# For Qwen3 models, download from HuggingFace:
# https://huggingface.co/Qwen/Qwen3-1.5B/blob/main/qwen3.tiktoken
curl -LO https://huggingface.co/Qwen/Qwen3-1.5B/raw/main/qwen3.tiktoken
```

The tokenizer file path is passed via `--tokenizer` to `run`, `chat`, and `serve` commands.

## GGUF Quantization Types

| GGML Type ID | Name | Bytes per element | Description |
|-------------|------|-------------------|-------------|
| 0 | F32 | 4 | Full precision |
| 1 | F16 | 2 | Half precision |
| 2 | Q4_0 | 0.5 + overhead | 4-bit symmetric, 32-element blocks |
| 3 | Q4_1 | 0.5 + overhead | 4-bit asymmetric |
| 6 | Q5_0 | 0.625 + overhead | 5-bit symmetric |
| 7 | Q5_1 | 0.625 + overhead | 5-bit asymmetric |
| 8 | Q8_0 | 1 + overhead | 8-bit symmetric, 32-element blocks |
| 10 | Q6_K | ~0.75 | K-quant 6-bit |
| 12 | Q4_K | ~0.5 | K-quant 4-bit (best quality/size) |
| 13 | Q5_K | ~0.625 | K-quant 5-bit |
| 14 | Q6_K | ~0.75 | K-quant 6-bit |
| 15 | Q8_K | ~1 | K-quant 8-bit |

During inference, NeuralForge reads tensor dtype from the GGUF file and uses the appropriate kernel:
- F32: `matmul()` — scalar F32 multiply
- F16: `read_tensor_f32()` converts to F32 on load, then uses F32 kernel
- Q8_0: `matmul_q8_0()` — on-the-fly dequantization with F32 activations
- Q4_0: `matmul_q4_0()` — on-the-fly dequantization with F32 activations

## Cache Directory

Models are cached locally at:

| Platform | Path |
|----------|------|
| macOS/Linux | `~/.cache/neural-forge/models/` |
| Windows | `%LOCALAPPDATA%/neural-forge/models/` |

Each model repo gets its own subdirectory (e.g., `Qwen__Qwen3-1.5B-GGUF/`). Use `neural-forge list` to see cached models and `neural-forge pull <repo>` to download new ones.
