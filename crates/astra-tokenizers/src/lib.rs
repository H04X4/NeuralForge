use std::collections::HashMap;
use std::path::Path;

pub struct TiktokenTokenizer {
    encoder: HashMap<Vec<u8>, usize>,
    decoder: Vec<Vec<u8>>,
    pattern: regex::Regex,
}

impl TiktokenTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let data = std::fs::read_to_string(path.as_ref()).map_err(|e| format!("cannot read tiktoken file: {e}"))?;
        Self::from_data(&data)
    }

    pub fn from_data(data: &str) -> Result<Self, String> {
        let mut encoder = HashMap::new();
        let mut max_id = 0usize;

        for line in data.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.splitn(2, ' ');
            let b64 = parts.next().ok_or_else(|| "missing base64 token".to_string())?;
            let rank_str = parts.next().ok_or_else(|| "missing rank".to_string())?;
            let rank: usize = rank_str
                .parse()
                .map_err(|e| format!("invalid rank '{rank_str}': {e}"))?;
            let bytes = decode_base64(b64)?;
            if let Some(&prev) = encoder.get(&bytes) {
                if prev != rank {
                    return Err(format!("duplicate token with different ranks: {prev} vs {rank}"));
                }
            }
            encoder.insert(bytes, rank);
            if rank > max_id {
                max_id = rank;
            }
        }

        let mut decoder: Vec<Vec<u8>> = vec![Vec::new(); max_id + 1];
        for (bytes, &id) in &encoder {
            if id < decoder.len() {
                decoder[id] = bytes.clone();
            }
        }

        let pattern = regex::Regex::new(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
            .map_err(|e| format!("invalid regex: {e}"))?;

        Ok(Self {
            encoder,
            decoder,
            pattern,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for cap in self.pattern.find_iter(text) {
            self.encode_piece(cap.as_str().as_bytes(), &mut ids);
        }
        ids
    }

    fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        let mut tokens: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();
        loop {
            let mut best_idx = None;
            let mut best_rank = usize::MAX;
            for i in 0..tokens.len().saturating_sub(1) {
                let merged = [&tokens[i][..], &tokens[i + 1][..]].concat();
                if let Some(&rank) = self.encoder.get(&merged) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = Some(i);
                    }
                }
            }
            match best_idx {
                None => break,
                Some(i) => {
                    let merged = [&tokens[i][..], &tokens[i + 1][..]].concat();
                    tokens.splice(i..=i + 1, [merged]);
                }
            }
        }
        for t in &tokens {
            if let Some(&id) = self.encoder.get(t) {
                out.push(id as u32);
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(seg) = self.decoder.get(id as usize) {
                bytes.extend_from_slice(seg);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn encoder(&self) -> &HashMap<Vec<u8>, usize> {
        &self.encoder
    }

    pub fn vocab_size(&self) -> usize {
        self.decoder.len()
    }
}

fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let input = input.trim_end_matches('=');
    let bytes = input.as_bytes();
    let mut result = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u8;

    for &c in bytes {
        let val = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(format!("invalid base64 char: {c}")),
        };
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(result)
}



