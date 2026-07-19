use std::path::Path;

use astra_core::{ExecutionPlan, Modality, ModelKind, QualityPolicy};

pub fn plan_from_file(
    path: &Path,
    context: usize,
    quality: QualityPolicy,
    ram_bytes: u64,
) -> Result<ExecutionPlan, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let (weight_bytes, model_kind, mods) = match ext {
        "gguf" => {
            let f = astra_formats::GgufReader::open(path).map_err(|e| format!("cannot read GGUF: {e}"))?;
            let data_offset_guess = f.file_size.min(f.file_size / 10 * 8);
            let weight = f.file_size.saturating_sub(data_offset_guess);
            let kind = detect_model_kind(&f.metadata);
            (weight, kind, vec![Modality::Text])
        }
        "safetensors" | "sft" => {
            let f =
                astra_formats::SafetensorsReader::open(path).map_err(|e| format!("cannot read Safetensors: {e}"))?;
            let weight = f.file_size.saturating_sub(8 + f.header_size);
            (weight, ModelKind::Dense, vec![Modality::Text])
        }
        _ => {
            let flen = std::fs::metadata(path)
                .map_err(|e| format!("cannot stat file: {e}"))?
                .len();
            (flen, ModelKind::Dense, vec![Modality::Text])
        }
    };

    Ok(compute_plan(
        &path.to_string_lossy(),
        weight_bytes,
        context,
        ram_bytes,
        quality,
        model_kind,
        mods,
    ))
}

pub fn plan(
    model_name: &str,
    weight_bytes: u64,
    context: usize,
    ram_bytes: u64,
    quality: QualityPolicy,
) -> ExecutionPlan {
    compute_plan(
        model_name,
        weight_bytes,
        context,
        ram_bytes,
        quality,
        ModelKind::Dense,
        vec![Modality::Text],
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_plan(
    model_name: &str,
    weight_bytes: u64,
    context: usize,
    ram_bytes: u64,
    quality: QualityPolicy,
    model_kind: ModelKind,
    modalities: Vec<Modality>,
) -> ExecutionPlan {
    let ctx = context as u64;
    let kv_per_token = 576u64; // per layer; multiply by layers below
    let kv_total = kv_per_token * ctx * 32; // assume 32 layers

    let working_set = 64u64 * 1024 * 1024;
    let peak = weight_bytes + kv_total + working_set;
    let feasible = peak <= ram_bytes;

    let bottleneck = if weight_bytes > ram_bytes {
        "insufficient RAM for weights".into()
    } else if peak > ram_bytes {
        "insufficient RAM for full context".into()
    } else if ram_bytes < 8_000_000_000 {
        "low system memory".into()
    } else {
        "CPU (no GPU backend)".into()
    };

    let tok_s_estimate = if !feasible {
        0.0
    } else if weight_bytes > 10_000_000_000 {
        (ram_bytes as f64 / weight_bytes as f64) * 5.0
    } else if weight_bytes > 3_000_000_000 {
        8.0
    } else {
        25.0
    };

    let note = if !feasible {
        format!(
            "needs ~{:.1} GB RAM, only {:.1} GB available",
            peak as f64 / 1e9,
            ram_bytes as f64 / 1e9
        )
    } else {
        String::new()
    };

    ExecutionPlan {
        model_name: model_name.to_string(),
        model_kind,
        modalities,
        weight_bytes,
        kv_bytes_per_token: kv_total / ctx.max(1),
        peak_ram_bytes: peak,
        peak_vram_bytes: 0,
        disk_bytes: weight_bytes,
        quality,
        estimated_tok_s: tok_s_estimate,
        bottleneck,
        feasible,
        note,
    }
}

fn detect_model_kind(metadata: &std::collections::HashMap<String, astra_formats::GgufValue>) -> ModelKind {
    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if arch.contains("moe") || arch.contains("MoE") {
        return ModelKind::MoE;
    }
    if metadata.contains_key("llama.expert_count") || metadata.contains_key("qwen3.expert_count") {
        return ModelKind::MoE;
    }
    ModelKind::Dense
}


