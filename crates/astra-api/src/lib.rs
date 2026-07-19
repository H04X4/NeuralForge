use axum::{
    Router,
    extract::State,
    http::HeaderMap,
    response::Json,
    routing::{get, post},
};
use std::sync::Arc;

use astra_qwen::{KVCache, QwenConfig, QwenWeights};
use astra_runtime::{greedy_sample, temperature_sample};

type ApiResult<T> = std::result::Result<T, (axum::http::StatusCode, Json<serde_json::Value>)>;

#[derive(Clone)]
struct AppState {
    weights: Arc<QwenWeights>,
    tok: Arc<astra_tokenizers::TiktokenTokenizer>,
    cfg: QwenConfig,
    api_key: Option<String>,
    model_name: String,
}

pub async fn serve(
    model: &str,
    tokenizer: &str,
    host: &str,
    port: u16,
    api_key: Option<String>,
) -> astra_core::Result<()> {
    let gguf = astra_formats::gguf::GgufReader::open(model)?;
    let weights = Arc::new(QwenWeights::load(&gguf)?);
    let cfg = weights.cfg.clone();
    let tok = Arc::new(astra_tokenizers::TiktokenTokenizer::from_file(tokenizer).map_err(astra_core::Error::Other)?);
    let model_name = std::path::Path::new(model)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    let state = AppState {
        weights,
        tok,
        cfg,
        api_key,
        model_name,
    };

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    tracing::info!("starting API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| astra_core::Error::Other(format!("bind failed: {e}")))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| astra_core::Error::Other(format!("server error: {e}")))?;

    Ok(())
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> ApiResult<()> {
    if let Some(ref expected) = state.api_key {
        let provided = headers
            .get("Authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        if provided != expected {
            return Err(err(axum::http::StatusCode::UNAUTHORIZED, "unauthorized"));
        }
    }
    Ok(())
}

async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> ApiResult<Json<serde_json::Value>> {
    check_auth(&state, &headers)?;
    let body = serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "neural-forge",
        }]
    });
    Ok(Json(body))
}

#[derive(serde::Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    stream: Option<bool>,
}

#[derive(serde::Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    stream: Option<bool>,
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    check_auth(&state, &headers)?;

    if req.stream.unwrap_or(false) {
        return Err(err(
            axum::http::StatusCode::NOT_IMPLEMENTED,
            "streaming not yet supported",
        ));
    }

    let max_tokens = req.max_tokens.unwrap_or(512).min(state.cfg.max_seq_len);
    let temperature = req.temperature.unwrap_or(0.7);
    let prompt = build_chat_prompt(&req.messages);
    let result = generate_text(&state, &prompt, max_tokens, temperature).await?;

    let body = serde_json::json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        "model": req.model.as_deref().unwrap_or(&state.model_name),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result.text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.token_count,
            "total_tokens": result.prompt_tokens + result.token_count,
        }
    });
    Ok(Json(body))
}

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    check_auth(&state, &headers)?;

    let max_tokens = req.max_tokens.unwrap_or(512).min(state.cfg.max_seq_len);
    let temperature = req.temperature.unwrap_or(0.7);
    let result = generate_text(&state, &req.prompt, max_tokens, temperature).await?;

    let body = serde_json::json!({
        "id": format!("cmpl-{}", uuid::Uuid::new_v4()),
        "object": "text_completion",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        "model": req.model.as_deref().unwrap_or(&state.model_name),
        "choices": [{
            "index": 0,
            "text": result.text,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.token_count,
            "total_tokens": result.prompt_tokens + result.token_count,
        }
    });
    Ok(Json(body))
}

async fn generate_text(
    state: &AppState,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
) -> ApiResult<astra_runtime::GeneratedText> {
    let input_ids = state.tok.encode(prompt);
    let n_prompt = input_ids.len();

    if n_prompt + max_tokens > state.cfg.max_seq_len {
        return Err(err(
            axum::http::StatusCode::BAD_REQUEST,
            &format!(
                "prompt too long: {} tokens, max context is {}",
                n_prompt, state.cfg.max_seq_len
            ),
        ));
    }

    let mut cache = KVCache::new(&state.cfg, state.cfg.max_seq_len);

    let mut last_logits = None;
    for (pos, &token) in input_ids.iter().enumerate() {
        let logits = astra_qwen::QwenModel::forward(&state.weights, token, pos, &mut cache)
            .map_err(|e| err(axum::http::StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
        if pos + 1 == n_prompt {
            last_logits = Some(logits);
        }
    }

    let mut output_ids = input_ids.clone();

    for pos in (n_prompt..).take(max_tokens) {
        let logits = if pos == n_prompt {
            last_logits.take().unwrap()
        } else {
            let prev = output_ids[pos - 1];
            astra_qwen::QwenModel::forward(&state.weights, prev, pos - 1, &mut cache)
                .map_err(|e| err(axum::http::StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?
        };

        let next_token = if temperature <= 0.0 {
            greedy_sample(&logits)
        } else {
            temperature_sample(&logits, temperature)
        };

        output_ids.push(next_token);
    }

    let generated = &output_ids[n_prompt..];
    let text = state.tok.decode(generated);

    Ok(astra_runtime::GeneratedText {
        text,
        tokens: generated.to_vec(),
        token_count: generated.len(),
        prompt_tokens: n_prompt,
    })
}

fn build_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut result = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => result.push_str(&format!("<|system|>\n{}\n", msg.content)),
            "user" => result.push_str(&format!("<|user|>\n{}\n", msg.content)),
            "assistant" => result.push_str(&format!("<|assistant|>\n{}\n", msg.content)),
            _ => result.push_str(&format!("{}\n", msg.content)),
        }
    }
    result.push_str("<|assistant|>\n");
    result
}

fn err(status: axum::http::StatusCode, message: &str) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "error",
            }
        })),
    )
}


