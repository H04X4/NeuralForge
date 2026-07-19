use astra_core::Result;
use astra_formats::gguf::GgufReader;
use astra_qwen::{KVCache, QwenConfig, QwenModel, QwenWeights};
use astra_tokenizers::TiktokenTokenizer;
use rand::Rng;

pub struct Session {
    pub model_path: String,
    pub weights: QwenWeights,
    pub cache: KVCache,
    pub tok: TiktokenTokenizer,
    pub cfg: QwenConfig,
}

impl Session {
    pub fn load(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let gguf = GgufReader::open(model_path)?;
        let weights = QwenWeights::load(&gguf)?;
        let cfg = weights.cfg.clone();
        let max_seq_len = cfg.max_seq_len;
        let cache = KVCache::new(&cfg, max_seq_len);
        let tok = TiktokenTokenizer::from_file(tokenizer_path).map_err(|e| astra_core::Error::Other(e.to_string()))?;
        Ok(Session {
            model_path: model_path.to_string(),
            weights,
            cache,
            tok,
            cfg,
        })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<GeneratedText> {
        let input_ids = self.tok.encode(prompt);
        let n_prompt = input_ids.len();

        if n_prompt + max_tokens > self.cfg.max_seq_len {
            return Err(astra_core::Error::ContextOverflow {
                max: self.cfg.max_seq_len,
                requested: n_prompt + max_tokens,
            });
        }
        let mut last_logits = None;
        for (pos, &token) in input_ids.iter().enumerate() {
            let logits = QwenModel::forward(&self.weights, token, pos, &mut self.cache)?;
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
                QwenModel::forward(&self.weights, prev, pos - 1, &mut self.cache)?
            };

            let next_token = if temperature <= 0.0 {
                greedy_sample(&logits)
            } else {
                temperature_sample(&logits, temperature)
            };

            output_ids.push(next_token);
        }

        let generated = &output_ids[n_prompt..];
        let text = self.tok.decode(generated);

        Ok(GeneratedText {
            text,
            tokens: generated.to_vec(),
            token_count: generated.len(),
            prompt_tokens: n_prompt,
        })
    }
}

pub struct GeneratedText {
    pub text: String,
    pub tokens: Vec<u32>,
    pub token_count: usize,
    pub prompt_tokens: usize,
}

pub fn greedy_sample(logits: &[f32]) -> u32 {
    let mut best = 0;
    let mut max_val = logits[0];
    for (i, &v) in logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            best = i;
        }
    }
    best as u32
}

pub fn temperature_sample(logits: &[f32], temperature: f64) -> u32 {
    let n = logits.len();
    let temp = temperature.max(1e-8);
    let mut rng = rand::thread_rng();

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    let mut probs = vec![0.0f64; n];
    for (i, &v) in logits.iter().enumerate() {
        let p = ((v - max_logit) as f64 / temp).exp();
        probs[i] = p;
        sum += p;
    }

    if sum == 0.0 || !sum.is_finite() {
        return greedy_sample(logits);
    }

    let inv_sum = 1.0 / sum;
    let r: f64 = rng.r#gen();
    let mut cum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cum += p * inv_sum;
        if r < cum {
            return i as u32;
        }
    }
    (n - 1) as u32
}


