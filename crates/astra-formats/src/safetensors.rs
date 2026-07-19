use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use astra_core::{Error, Result};
use serde::{Deserialize, Serialize};

const MAX_HEADER_SIZE: u64 = 104_857_600;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorEntry {
    pub dtype: String,
    pub shape: Vec<u64>,
    #[serde(rename = "data_offsets")]
    pub data_offsets: [u64; 2],
}

#[derive(Debug, Clone, Serialize)]
pub struct SafetensorsFile {
    pub path: String,
    pub file_size: u64,
    pub header_size: u64,
    pub tensors: Vec<(String, SafetensorEntry)>,
}

impl SafetensorsFile {
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

#[derive(Default)]
pub struct SafetensorsReader;

impl SafetensorsReader {
    pub fn new() -> Self {
        SafetensorsReader
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<SafetensorsFile> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(Error::Io)?;
        let file_size = file.metadata().map_err(Error::Io)?.len();

        let mut header_size_buf = [0u8; 8];
        file.read_exact(&mut header_size_buf).map_err(Error::Io)?;
        let header_size = u64::from_le_bytes(header_size_buf);

        if header_size == 0 {
            return Err(Error::Format("safetensors header size is zero".into()));
        }
        if header_size > MAX_HEADER_SIZE {
            return Err(Error::Format(format!(
                "safetensors header size {header_size} exceeds safety limit {MAX_HEADER_SIZE}"
            )));
        }
        if 8 + header_size > file_size {
            return Err(Error::Format("safetensors header exceeds file bounds".into()));
        }

        let mut json_buf = vec![0u8; header_size as usize];
        file.read_exact(&mut json_buf).map_err(Error::Io)?;

        let tensors_raw: HashMap<String, SafetensorEntry> = serde_json::from_slice(&json_buf)
            .map_err(|e| Error::Format(format!("invalid safetensors JSON header: {e}")))?;

        let mut tensors: Vec<(String, SafetensorEntry)> = tensors_raw.into_iter().collect();
        tensors.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(SafetensorsFile {
            path: path.to_string_lossy().into_owned(),
            file_size,
            header_size,
            tensors,
        })
    }
}


