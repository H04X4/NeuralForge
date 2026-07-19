# Development

## Building from Source

```bash
# Clone and build
git clone https://github.com/NeuralForge/neural-forge.git
cd neural-forge
cargo build --release

# The binary is at target/release/neural-forge
```

### Build Dependencies

- **Rust** 1.85+ (edition 2024)
- **macOS**: Xcode Command Line Tools (`xcode-select --install`)
- **Linux**: `build-essential`, `libssl-dev`, `pkg-config`
- **Windows**: Visual Studio Build Tools (for C++ linker)

## Workspace Structure

```
Cargo.toml              # Workspace root (11 member crates)
crates/
├── astra-core/         # Shared types — no internal deps
├── astra-cli/          # Binary crate — depends on all others
├── astra-formats/      # GGUF + Safetensors readers
├── astra-hardware/     # System discovery
├── astra-planner/      # Memory estimation
├── astra-kernels/      # CPU math kernels
├── astra-tokenizers/   # tiktoken BPE
├── astra-qwen/         # Qwen3 model
├── astra-runtime/      # Session + generation
├── astra-models/       # Model lifecycle
└── astra-api/          # HTTP API server
```

## Adding a New Model Architecture

1. **Create a new crate** (e.g., `astra-llama`) with:
   - Config struct (parsed from GGUF metadata)
   - Weights struct (tensor loading)
   - KVCache (if different from Qwen3's GQA cache)
   - Model struct with `forward()` method

2. **Register in the runtime** — add a variant to `ModelType` enum and dispatch in `Session::load`

3. **Expose in CLI** — the CLI auto-discovers via `Session::load`

## Coding Conventions

- Rust edition 2024
- `#![deny(unsafe_code)]` in all crates (the kernels crate is the only exception for performance)
- `camelCase` for JSON fields, `snake_case` for Rust
- All errors through `astra_core::Error` enum
- CLI output through `term.rs` styling helpers (never raw println! colors)

## Performance Guidelines

- Keep hot loops in `astra-kernels` simple — the compiler auto-vectorizes well
- Use tiling for matmul (32×32×256) to improve cache locality
- Profile with `cargo flamegraph` before optimizing
- Quantization (Q8_0, Q4_0) gives 2-4× speedup on memory-bandwidth-bound workloads
