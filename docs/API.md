# API Reference

NeuralForge provides an OpenAI-compatible HTTP API for local model inference. Start the server:

```bash
neural-forge serve ./model.gguf --tokenizer ./tok.tiktoken --host 127.0.0.1 --port 8080
```

Optional API key auth:

```bash
neural-forge serve ./model.gguf --tokenizer ./tok.tiktoken --api-key sk-my-secret
```

## Endpoints

### `GET /v1/models`

List available models.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "model-name",
      "object": "model",
      "created": 0,
      "owned_by": "neural-forge"
    }
  ]
}
```

### `POST /v1/chat/completions`

Chat completion using Qwen3's native chat format.

**Request:**

```json
{
  "model": "optional-model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the meaning of life?"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**

```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  }
}
```

The chat prompt is converted to Qwen3's internal format:
```
<|system|>
You are a helpful assistant.
<|user|>
What is the meaning of life?
<|assistant|>
```

### `POST /v1/completions`

Raw text completion (non-chat).

**Request:**

```json
{
  "model": "optional-model-name",
  "prompt": "Once upon a time,",
  "max_tokens": 256,
  "temperature": 0.8,
  "stream": false
}
```

**Response:**

```json
{
  "id": "cmpl-<uuid>",
  "object": "text_completion",
  "created": 1234567890,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "text": "...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 256,
    "total_tokens": 261
  }
}
```

## Authentication

If `--api-key` is provided on startup, every request must include:

```
Authorization: Bearer <api-key>
```

Requests missing or with wrong key receive `401 Unauthorized`.

## Error Responses

```json
{
  "error": {
    "message": "description of the error",
    "type": "error"
  }
}
```

| Status | Cause |
|--------|-------|
| 400 | Prompt too long (exceeds max context) |
| 401 | Missing or invalid API key |
| 500 | Model inference error |
| 501 | Streaming not yet implemented |

## Limitations

- **No streaming** (`stream: true` returns 501) — planned for a future release
- **No stop sequences** — generation stops at `max_tokens` only
- **Single model** — one model per server process
- **No batching** — requests are processed serially
