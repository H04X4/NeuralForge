use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use astra_core::{Error, Result};
use serde::Serialize;

const GGUF_MAGIC: u32 = 0x46554747;
const MAX_STRING_LEN: u64 = 1_048_576;
const MAX_ARRAY_LEN: u64 = 100_000;
const MAX_TENSORS: u64 = 100_000;

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::Uint64(v) => Some(*v),
            GgufValue::Uint32(v) => Some(*v as u64),
            GgufValue::Int32(v) => Some(*v as u64),
            GgufValue::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::Float32(v) => Some(*v as f64),
            GgufValue::Float64(v) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dimensions: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

impl GgufTensorInfo {
    pub fn n_elems(&self) -> usize {
        self.dimensions.iter().product::<u64>() as usize
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufFile {
    pub path: String,
    pub file_size: u64,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_start: u64,
}

impl GgufFile {
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    pub fn name(&self) -> Option<&str> {
        self.metadata.get("general.name").and_then(|v| v.as_str())
    }

    pub fn file_type(&self) -> Option<u64> {
        self.metadata.get("general.file_type").and_then(|v| v.as_u64())
    }

    pub fn block_count(&self) -> Option<u64> {
        let prefix = self.architecture().unwrap_or("unknown");
        self.metadata
            .get(&format!("{prefix}.block_count"))
            .and_then(|v| v.as_u64())
    }

    pub fn context_length(&self) -> Option<u64> {
        let prefix = self.architecture().unwrap_or("unknown");
        self.metadata
            .get(&format!("{prefix}.context_length"))
            .and_then(|v| v.as_u64())
    }

    pub fn embedding_length(&self) -> Option<u64> {
        let prefix = self.architecture().unwrap_or("unknown");
        self.metadata
            .get(&format!("{prefix}.embedding_length"))
            .and_then(|v| v.as_u64())
    }

    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::Other(format!("tensor '{name}' not found")))?;
        let mut file = File::open(&self.path).map_err(Error::Io)?;
        let elem_size = ggml_type_size(info.dtype);
        let n_elems = info.dimensions.iter().product::<u64>() as usize;
        let n_bytes = n_elems * elem_size;
        let offset = self.data_start + info.offset;
        file.seek(SeekFrom::Start(offset)).map_err(Error::Io)?;
        let mut buf = vec![0u8; n_bytes];
        file.read_exact(&mut buf).map_err(Error::Io)?;
        Ok(buf)
    }

    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::Other(format!("tensor '{name}' not found")))?;
        let raw = self.read_tensor(name)?;
        match info.dtype {
            0 => Ok(bytemuck::cast_slice(&raw).to_vec()),
            1 => {
                let n = info.n_elems();
                let f16s: Vec<half::f16> = raw
                    .chunks_exact(2)
                    .take(n)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                    .collect();
                Ok(f16s.iter().map(|x| f32::from(*x)).collect())
            }
            other => Err(Error::Other(format!(
                "unsupported GGML dtype {} for tensor '{name}'",
                other
            ))),
        }
    }

    pub fn list_tensor_names(&self) -> Vec<String> {
        self.tensors.iter().map(|t| t.name.clone()).collect()
    }
}

fn ggml_type_size(dtype: u32) -> usize {
    match dtype {
        0 => 4,
        1 => 2,
        _ => 1,
    }
}

#[derive(Default)]
pub struct GgufReader;

impl GgufReader {
    pub fn new() -> Self {
        GgufReader
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<GgufFile> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(Error::Io)?;
        let file_size = file.metadata().map_err(Error::Io)?.len();

        let magic = read_u32_le(&mut file)?;
        if magic != GGUF_MAGIC {
            return Err(Error::Format(format!("not a GGUF file (magic: {magic:#010x})")));
        }

        let version = read_u32_le(&mut file)?;
        let tensor_count = read_u64_le(&mut file)?;
        let metadata_kv_count = read_u64_le(&mut file)?;

        if tensor_count > MAX_TENSORS {
            return Err(Error::Format(format!(
                "tensor count {tensor_count} exceeds safety limit {MAX_TENSORS}"
            )));
        }
        if metadata_kv_count > MAX_ARRAY_LEN {
            return Err(Error::Format(format!(
                "metadata KV count {metadata_kv_count} exceeds safety limit {MAX_ARRAY_LEN}"
            )));
        }

        let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut file, file_size)?;
            let value = read_gguf_value(&mut file, file_size)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut file, file_size)?;
            let n_dims = read_u32_le(&mut file)?;
            let pos = file.stream_position().map_err(Error::Io)?;
            let remaining = (file_size.saturating_sub(pos)) / 8;
            if n_dims as u64 > remaining {
                return Err(Error::Format("tensor dimensions exceed file bounds".into()));
            }
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d = read_u64_le(&mut file)?;
                dimensions.push(d);
            }
            let dtype = read_u32_le(&mut file)?;
            let offset = read_u64_le(&mut file)?;
            tensors.push(GgufTensorInfo {
                name,
                n_dims,
                dimensions,
                dtype,
                offset,
            });
        }

        let pos = file.stream_position().map_err(Error::Io)?;
        let data_start = (pos + 31) & !31;
        let padding = data_start.saturating_sub(pos);
        if padding > 0 && pos + padding <= file_size {
            let mut pad = vec![0u8; padding as usize];
            read_exact(&mut file, &mut pad)?;
        } else {
            let _ = padding;
        }

        Ok(GgufFile {
            path: path.to_string_lossy().into_owned(),
            file_size,
            version,
            tensor_count,
            metadata,
            tensors,
            data_start,
        })
    }
}

fn read_exact<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<()> {
    reader.read_exact(buf).map_err(Error::Io)
}

fn read_u32_le<R: Read>(reader: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    read_exact(reader, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64_le<R: Read>(reader: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    read_exact(reader, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_gguf_string<R: Read + Seek>(reader: &mut R, file_size: u64) -> Result<String> {
    let len = read_u64_le(reader)?;
    if len > MAX_STRING_LEN {
        return Err(Error::Format(format!(
            "GGUF string length {len} exceeds safety limit {MAX_STRING_LEN}"
        )));
    }
    let pos = reader.stream_position().map_err(Error::Io)?;
    if pos + len > file_size {
        return Err(Error::Format("GGUF string exceeds file bounds".into()));
    }
    let mut buf = vec![0u8; len as usize];
    read_exact(reader, &mut buf)?;
    String::from_utf8(buf).map_err(|_| Error::Format("GGUF string is not valid UTF-8".into()))
}

fn read_gguf_value<R: Read + Seek>(reader: &mut R, file_size: u64) -> Result<GgufValue> {
    let type_tag = read_u32_le(reader)?;
    match type_tag {
        0 => Ok(GgufValue::Uint8(read_u32_le(reader)? as u8)),
        1 => Ok(GgufValue::Int8(read_u32_le(reader)? as i8)),
        2 => Ok(GgufValue::Uint16(read_u32_le(reader)? as u16)),
        3 => Ok(GgufValue::Int16(read_u32_le(reader)? as i16)),
        4 => Ok(GgufValue::Uint32(read_u32_le(reader)?)),
        5 => Ok(GgufValue::Int32(read_u32_le(reader)? as i32)),
        6 => {
            let mut b = [0u8; 4];
            read_exact(reader, &mut b)?;
            Ok(GgufValue::Float32(f32::from_le_bytes(b)))
        }
        7 => Ok(GgufValue::Bool(read_u32_le(reader)? != 0)),
        8 => Ok(GgufValue::String(read_gguf_string(reader, file_size)?)),
        9 | 13 => {
            let elem_tag = read_u32_le(reader)?;
            let count = read_u64_le(reader)?;
            if count > MAX_ARRAY_LEN {
                return Err(Error::Format(format!(
                    "GGUF array length {count} exceeds safety limit {MAX_ARRAY_LEN}"
                )));
            }
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_gguf_value_typed(reader, file_size, elem_tag)?);
            }
            Ok(GgufValue::Array(items))
        }
        10 => Ok(GgufValue::Uint64(read_u64_le(reader)?)),
        11 => Ok(GgufValue::Int64(read_u64_le(reader)? as i64)),
        12 => {
            let mut b = [0u8; 8];
            read_exact(reader, &mut b)?;
            Ok(GgufValue::Float64(f64::from_le_bytes(b)))
        }
        t => Err(Error::Format(format!("unknown GGUF value type tag: {t}"))),
    }
}

fn read_gguf_value_typed<R: Read + Seek>(reader: &mut R, file_size: u64, type_tag: u32) -> Result<GgufValue> {
    match type_tag {
        0 => {
            let mut b = [0u8; 1];
            read_exact(reader, &mut b)?;
            Ok(GgufValue::Uint8(b[0]))
        }
        1 => {
            let mut b = [0u8; 1];
            read_exact(reader, &mut b)?;
            Ok(GgufValue::Int8(b[0] as i8))
        }
        2 => {
            let mut b = [0u8; 2];
            read_exact(reader, &mut b)?;
            Ok(GgufValue::Uint16(u16::from_le_bytes(b)))
        }
        3 => {
            let mut b = [0u8; 2];
            read_exact(reader, &mut b)?;
            Ok(GgufValue::Int16(i16::from_le_bytes(b)))
        }
        4 => Ok(GgufValue::Uint32(read_u32_le(reader)?)),
        5 => Ok(GgufValue::Int32(read_u32_le(reader)? as i32)),
        6 => {
            let mut b = [0u8; 4];
            read_exact(reader, &mut b)?;
            Ok(GgufValue::Float32(f32::from_le_bytes(b)))
        }
        7 => {
            let mut b = [0u8; 1];
            read_exact(reader, &mut b)?;
            Ok(GgufValue::Bool(b[0] != 0))
        }
        8 => Ok(GgufValue::String(read_gguf_string(reader, file_size)?)),
        9 | 13 => {
            let elem_tag = read_u32_le(reader)?;
            let count = read_u64_le(reader)?;
            if count > MAX_ARRAY_LEN {
                return Err(Error::Format(format!(
                    "GGUF nested array length {count} exceeds safety limit"
                )));
            }
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_gguf_value_typed(reader, file_size, elem_tag)?);
            }
            Ok(GgufValue::Array(items))
        }
        10 => Ok(GgufValue::Uint64(read_u64_le(reader)?)),
        11 => Ok(GgufValue::Int64(read_u64_le(reader)? as i64)),
        12 => {
            let mut b = [0u8; 8];
            read_exact(reader, &mut b)?;
            Ok(GgufValue::Float64(f64::from_le_bytes(b)))
        }
        t => Err(Error::Format(format!("unknown GGUF typed value tag: {t}"))),
    }
}


