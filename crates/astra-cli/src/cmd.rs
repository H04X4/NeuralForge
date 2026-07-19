use crate::term::{Style, kv, print_json};
use astra_core::HardwareInventory;
use astra_formats::{GgufFile, GgufReader, GgufValue, SafetensorsReader};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

pub fn cmd_doctor(json: bool) -> anyhow::Result<()> {
    let inventory = astra_hardware::discover()?;
    if json {
        print_json(&inventory);
        return Ok(());
    }
    println!();
    println!("{}", Style::heading("hardware inventory"));
    println!();
    print_hardware(&inventory);
    println!();
    Ok(())
}

pub fn cmd_plan(model: &str, context: usize, policy: &str, json: bool) -> anyhow::Result<()> {
    let quality = parse_policy(policy);
    let inventory = astra_hardware::discover()?;

    let plan = if std::path::Path::new(model).exists() {
        match astra_planner::plan_from_file(
            std::path::Path::new(model),
            context,
            quality,
            inventory.available_ram_bytes,
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  {} {}", Style::err(), e);
                return Ok(());
            }
        }
    } else {
        astra_planner::plan(model, 0, context, inventory.available_ram_bytes, quality)
    };

    if json {
        print_json(&plan);
        return Ok(());
    }

    println!();
    println!("{}", Style::heading(&format!("plan {}", plan.model_name)));
    println!("  {} {:22} {}", Style::dim(), Style::label("context"), context);
    println!(
        "  {} {:22} {}",
        Style::dim(),
        Style::label("policy"),
        Style::value(policy)
    );
    println!();
    print_plan(&plan);
    println!();
    Ok(())
}

pub fn cmd_run(
    model: &str,
    prompt: &[String],
    tokenizer: &str,
    max_tokens: usize,
    temperature: f64,
    _seed: Option<u64>,
) -> anyhow::Result<()> {
    let full_prompt = prompt.join(" ");

    let session =
        astra_runtime::Session::load(model, tokenizer).map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

    let input_ids = session.tok.encode(&full_prompt);
    let n_prompt = input_ids.len();
    let mut cache = astra_qwen::KVCache::new(&session.cfg, session.cfg.max_seq_len);

    println!();
    let model_name = std::path::Path::new(model).file_stem().unwrap().to_string_lossy();
    println!("{}", Style::heading(&format!("run {}", model_name)));
    println!(
        "  {} {} {} tok, temp {}",
        Style::dim(),
        Style::label("prompt"),
        n_prompt,
        temperature
    );
    println!();

    for (pos, &token) in input_ids.iter().enumerate() {
        astra_qwen::QwenModel::forward(&session.weights, token, pos, &mut cache)?;
    }

    let mut output_ids = input_ids.clone();
    let mut is_first = true;

    for pos in (n_prompt..).take(max_tokens) {
        let logits = if is_first {
            is_first = false;
            astra_qwen::QwenModel::forward(&session.weights, output_ids[n_prompt - 1], n_prompt - 1, &mut cache)?
        } else {
            let prev = output_ids[pos - 1];
            astra_qwen::QwenModel::forward(&session.weights, prev, pos - 1, &mut cache)?
        };

        let next_token = if temperature <= 0.0 {
            astra_runtime::greedy_sample(&logits)
        } else {
            astra_runtime::temperature_sample(&logits, temperature)
        };

        output_ids.push(next_token);
    }

    let generated = &output_ids[n_prompt..];
    let text = session.tok.decode(generated);
    println!("{}", text);
    println!();
    println!(
        "  {} {:>22} {}",
        Style::dim(),
        Style::label("generated"),
        Style::value(&format!("{} tok", generated.len()))
    );
    println!();

    Ok(())
}

pub fn cmd_chat(model: &str, tokenizer: &str, max_tokens: usize, temperature: f64) -> anyhow::Result<()> {
    let session =
        astra_runtime::Session::load(model, tokenizer).map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

    let name = std::path::Path::new(model).file_stem().unwrap().to_string_lossy();
    println!();
    println!("{}", Style::heading(&format!("chat {}", name)));
    println!("  {} {}", Style::dim(), Style::sub("type your message · /q to quit"),);
    println!();

    let mut conversation = String::new();

    loop {
        print!("  {} ", Style::prompt_label());
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim().to_string();

        if input == "/q" || input == ":q" || input == "/exit" {
            break;
        }
        if input.is_empty() {
            continue;
        }

        conversation.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", input));

        let mut cache = astra_qwen::KVCache::new(&session.cfg, session.cfg.max_seq_len);
        let input_ids = session.tok.encode(&conversation);
        let n_prompt = input_ids.len();

        if n_prompt > session.cfg.max_seq_len {
            println!("  {} context limit reached, resetting", Style::warn());
            conversation.clear();
            continue;
        }

        for (pos, &token) in input_ids.iter().enumerate() {
            astra_qwen::QwenModel::forward(&session.weights, token, pos, &mut cache)?;
        }

        let mut output_ids = input_ids.clone();
        let mut is_first = true;

        print!("  {} ", Style::model_label());
        std::io::stdout().flush()?;

        for pos in (n_prompt..).take(max_tokens) {
            let logits = if is_first {
                is_first = false;
                astra_qwen::QwenModel::forward(&session.weights, output_ids[n_prompt - 1], n_prompt - 1, &mut cache)?
            } else {
                let prev = output_ids[pos - 1];
                astra_qwen::QwenModel::forward(&session.weights, prev, pos - 1, &mut cache)?
            };

            let next_token = if temperature <= 0.0 {
                astra_runtime::greedy_sample(&logits)
            } else {
                astra_runtime::temperature_sample(&logits, temperature)
            };

            output_ids.push(next_token);
        }

        let generated = &output_ids[n_prompt..];
        let text = session.tok.decode(generated);
        for word in text.split_inclusive(' ') {
            print!("{}", word);
            std::io::stdout().flush()?;
        }
        println!();
        println!();
        conversation.push_str(&text);
        conversation.push('\n');
    }

    println!();
    Ok(())
}

pub fn cmd_serve(model: &str, tokenizer: &str, host: &str, port: u16, api_key: Option<&str>) -> anyhow::Result<()> {
    let name = std::path::Path::new(model).file_stem().unwrap().to_string_lossy();
    println!();
    println!("{}", Style::heading("serve"));
    println!("  {} {}", Style::value(&name), Style::sub("ready"));
    println!(
        "  {} listening on {}",
        Style::dim(),
        Style::value(&format!("{}:{}", host, port))
    );
    if api_key.is_some() {
        println!("  {} {}", Style::ok(), Style::sub("api key auth"));
    }
    println!();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(astra_api::serve(
        model,
        tokenizer,
        host,
        port,
        api_key.map(|s| s.to_string()),
    ))?;
    Ok(())
}

pub fn cmd_pull(model: &str, cache_dir: Option<&str>) -> anyhow::Result<()> {
    let cache = cache_dir
        .map(PathBuf::from)
        .unwrap_or_else(astra_models::default_model_dir);

    println!();
    println!("{}", Style::heading(&format!("pull {}", model)));

    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("=> "),
    );

    let pb_clone = pb.clone();
    let model_owned = model.to_string();
    let cache_clone = cache.clone();

    let rt = tokio::runtime::Runtime::new()?;
    let pulled = rt.block_on(astra_models::pull(&model_owned, &cache_clone, move |current, total| {
        if total > 0 {
            pb_clone.set_length(total);
        }
        pb_clone.set_position(current);
    }))?;

    pb.finish_and_clear();

    println!("  {}", Style::ok());
    println!("{}", kv("file", pulled.path.display()));
    println!("{}", kv("size", format_bytes(pulled.file_size)));
    println!();
    Ok(())
}

pub fn cmd_list(json: bool) -> anyhow::Result<()> {
    let cache = astra_models::default_model_dir();
    let models = astra_models::list_models(&cache)?;

    if json {
        let list: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "repo_id": m.repo_id,
                    "path": m.path,
                    "file_size": m.file_size,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&list)?);
        return Ok(());
    }

    println!();
    if models.is_empty() {
        println!("{}", Style::heading("cached models"));
        println!("  {} {}", Style::dim(), Style::sub("no models cached"));
        println!("  {} neural-forge pull <model>", Style::dim());
    } else {
        println!("{}", Style::heading("cached models"));
        for m in &models {
            println!("  {}  {}", Style::ok(), Style::value(&m.repo_id));
            println!("    {}", kv("size", format_bytes(m.file_size)));
        }
    }
    println!();
    Ok(())
}

pub fn cmd_inspect(path: &str, json: bool) -> anyhow::Result<()> {
    if json {
        let result = inspect_file_json(path)?;
        print_json(&result);
        return Ok(());
    }
    match inspect_file(path)? {
        InspectResult::Gguf(f) => print_gguf_inspect(&f),
        InspectResult::Safetensors(f) => print_safetensors_inspect(&f),
    };
    Ok(())
}

fn parse_policy(s: &str) -> astra_core::QualityPolicy {
    match s {
        "exact" => astra_core::QualityPolicy::Exact,
        "quality" => astra_core::QualityPolicy::Quality,
        "balanced" => astra_core::QualityPolicy::Balanced,
        "fast" => astra_core::QualityPolicy::Fast,
        "tiny" => astra_core::QualityPolicy::Tiny,
        _ => astra_core::QualityPolicy::Quality,
    }
}

fn inspect_file_json(path: &str) -> Result<serde_json::Value, anyhow::Error> {
    let fmt = detect_format(path)?;
    match fmt.as_str() {
        "gguf" => {
            let f = GgufReader::open(path)?;
            Ok(serde_json::to_value(&f)?)
        }
        "safetensors" => {
            let f = SafetensorsReader::open(path)?;
            Ok(serde_json::to_value(&f)?)
        }
        _ => Err(anyhow::anyhow!("unsupported format: {fmt}")),
    }
}

pub fn cmd_tokenize(model: &str, text: &[String], json: bool) -> anyhow::Result<()> {
    let full_text = text.join(" ");
    let tok = astra_tokenizers::TiktokenTokenizer::from_file(model).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = tok.encode(&full_text);

    if json {
        let result = serde_json::json!({
            "tokens": ids,
            "count": ids.len(),
        });
        print_json(&result);
        return Ok(());
    }

    println!();
    println!("{}", Style::heading("tokenize"));
    println!("{}", kv("vocab", tok.vocab_size()));
    println!("{}", kv("tokens", ids.len()));
    println!();
    println!("  {:?}", ids);
    println!();
    Ok(())
}

pub fn cmd_build() -> anyhow::Result<()> {
    println!();
    println!("{}", Style::heading("build"));
    println!("  {} cargo build --release", Style::dim());
    println!();
    Ok(())
}

enum InspectResult {
    Gguf(GgufFile),
    Safetensors(astra_formats::SafetensorsFile),
}

fn detect_format(path: &str) -> Result<String, anyhow::Error> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;

    if magic == [0x47, 0x47, 0x55, 0x46] {
        return Ok("gguf".into());
    }

    if path.ends_with(".safetensors") || path.ends_with(".sft") {
        return Ok("safetensors".into());
    }

    Err(anyhow::anyhow!(
        "unknown format (magic: {:02x} {:02x} {:02x} {:02x})",
        magic[0],
        magic[1],
        magic[2],
        magic[3]
    ))
}

fn inspect_file(path: &str) -> Result<InspectResult, anyhow::Error> {
    let fmt = detect_format(path)?;
    match fmt.as_str() {
        "gguf" => Ok(InspectResult::Gguf(GgufReader::open(path)?)),
        "safetensors" => Ok(InspectResult::Safetensors(SafetensorsReader::open(path)?)),
        _ => Err(anyhow::anyhow!("unsupported format: {}", fmt)),
    }
}

fn format_gguf_value(val: &GgufValue) -> String {
    match val {
        GgufValue::Uint8(v) => v.to_string(),
        GgufValue::Int8(v) => v.to_string(),
        GgufValue::Uint16(v) => v.to_string(),
        GgufValue::Int16(v) => v.to_string(),
        GgufValue::Uint32(v) => v.to_string(),
        GgufValue::Int32(v) => v.to_string(),
        GgufValue::Float32(v) => format!("{v}"),
        GgufValue::Bool(v) => v.to_string(),
        GgufValue::String(v) => v.clone(),
        GgufValue::Array(v) => format!("[{}]", v.iter().map(format_gguf_value).collect::<Vec<_>>().join(", ")),
        GgufValue::Uint64(v) => v.to_string(),
        GgufValue::Int64(v) => v.to_string(),
        GgufValue::Float64(v) => format!("{v}"),
    }
}

fn print_gguf_inspect(file: &GgufFile) {
    println!();
    println!("{}", Style::heading(&format!("inspect {}", file.path)));
    println!();

    let sec = |s: &str| println!("  {}", s.truecolor(120, 140, 160).bold());

    sec("format");
    println!("{}", kv("type", "GGUF"));
    println!("{}", kv("version", format!("v{}", file.version)));
    println!("{}", kv("file size", format_bytes(file.file_size)));

    if let Some(arch) = file.architecture() {
        println!();
        sec("architecture");
        println!("{}", kv("name", arch));
        if let Some(ctx) = file.context_length() {
            println!("{}", kv("context", ctx));
        }
        if let Some(emb) = file.embedding_length() {
            println!("{}", kv("embedding", emb));
        }
        if let Some(blocks) = file.block_count() {
            println!("{}", kv("blocks", blocks));
        }
    }

    if !file.metadata.is_empty() {
        println!();
        sec(&format!("metadata ({} keys)", file.metadata.len()));
        let sorted: BTreeMap<&String, &GgufValue> = file.metadata.iter().collect();
        for (key, val) in &sorted {
            println!(
                "  {} {:30} {}",
                Style::dim(),
                Style::label(key),
                Style::value(&format_gguf_value(val))
            );
        }
    }

    println!();
    sec(&format!("tensors ({})", file.tensors.len()));
    for t in file.tensors.iter().take(30) {
        let dims = t.dimensions.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
        println!(
            "  {} {:35} [{}]  dtype={}  offset={}",
            Style::dim(),
            t.name,
            dims,
            t.dtype,
            t.offset
        );
    }
    if file.tensors.len() > 30 {
        println!("  {} ... and {} more", Style::dim(), file.tensors.len() - 30);
    }
    println!();
}

fn print_safetensors_inspect(file: &astra_formats::SafetensorsFile) {
    println!();
    println!("{}", Style::heading(&format!("inspect {}", file.path)));
    println!();

    let sec = |s: &str| println!("  {}", s.truecolor(120, 140, 160).bold());

    sec("format");
    println!("{}", kv("type", "Safetensors"));
    println!("{}", kv("file size", format_bytes(file.file_size)));
    println!("{}", kv("tensors", file.tensor_count()));

    println!();
    sec(&format!("tensors ({})", file.tensors.len()));
    for (name, entry) in file.tensors.iter().take(30) {
        let dims = entry.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
        println!("  {} {:35} [{}]  {}", Style::dim(), name, dims, entry.dtype);
    }
    if file.tensors.len() > 30 {
        println!("  {} ... and {} more", Style::dim(), file.tensors.len() - 30);
    }
    println!();
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    format!("{:.1} {}", size, UNITS[unit_idx])
}

fn print_hardware(inv: &HardwareInventory) {
    println!("{}", kv("os", &inv.os));
    println!("{}", kv("arch", &inv.arch));
    println!("{}", kv("logical cpus", inv.logical_cpus.to_string()));
    println!("{}", kv("physical cpus", inv.physical_cpus.to_string()));
    println!("{}", kv("total ram", format_bytes(inv.total_ram_bytes)));
    println!("{}", kv("available ram", format_bytes(inv.available_ram_bytes)));
    println!("{}", kv("free disk", format_bytes(inv.storage_free_bytes)));

    if !inv.backends.is_empty() {
        println!();
        println!("  {}", "backends".truecolor(120, 140, 160).bold());
        for b in &inv.backends {
            println!("  {} {}", Style::ok(), Style::dim_label(&b.to_string()));
        }
    }
}

fn print_plan(plan: &astra_core::ExecutionPlan) {
    println!("{}", kv("weights", format_bytes(plan.weight_bytes)));
    println!("{}", kv("kv per token", format_bytes(plan.kv_bytes_per_token)));
    println!("{}", kv("peak ram", format_bytes(plan.peak_ram_bytes)));
    println!("{}", kv("peak vram", format_bytes(plan.peak_vram_bytes)));
    println!("{}", kv("disk", format_bytes(plan.disk_bytes)));
    println!("{}", kv("policy", plan.quality.as_str()));
    println!("{}", kv("estimated tok/s", format!("{:.1}", plan.estimated_tok_s)));
    println!("{}", kv("bottleneck", &plan.bottleneck));
    println!();

    if plan.feasible {
        println!("  {} feasible", Style::ok());
    } else {
        println!("  {} not feasible", Style::err());
    }
    if !plan.note.is_empty() {
        println!("  {} {}", Style::dim(), plan.note);
    }
}

