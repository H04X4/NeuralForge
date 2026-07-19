use astra_core::{Error, Result};
use std::path::{Path, PathBuf};

pub struct PulledModel {
    pub path: PathBuf,
    pub repo_id: String,
    pub file_size: u64,
}

pub struct CachedModel {
    pub repo_id: String,
    pub path: PathBuf,
    pub file_size: u64,
}

pub fn default_model_dir() -> PathBuf {
    let base = if cfg!(target_os = "windows") {
        std::env::var("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".cache"))
    } else {
        std::env::var("XDG_CACHE_HOME").map(PathBuf::from).unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".cache")
        })
    };
    base.join("neural-forge").join("models")
}

pub fn list_models(cache_dir: &Path) -> Result<Vec<CachedModel>> {
    let mut models = Vec::new();
    if !cache_dir.exists() {
        return Ok(models);
    }
    for entry in std::fs::read_dir(cache_dir).map_err(Error::Io)? {
        let entry = entry.map_err(Error::Io)?;
        let dir_path = entry.path();
        if !dir_path.is_dir() {
            continue;
        }
        let dir_name = dir_path.file_name().unwrap().to_string_lossy().to_string();
        for file_entry in std::fs::read_dir(&dir_path).map_err(Error::Io)? {
            let file_entry = file_entry.map_err(Error::Io)?;
            let path = file_entry.path();
            if path.extension().is_some_and(|e| e == "gguf") {
                let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let repo_id = dir_name.replace("__", "/");
                models.push(CachedModel {
                    repo_id,
                    path,
                    file_size,
                });
            }
        }
    }
    models.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
    Ok(models)
}

pub fn resolve_model(model: &str, cache_dir: &Path) -> Option<PathBuf> {
    if Path::new(model).exists() {
        return Some(PathBuf::from(model));
    }
    let dir_name = model.replace('/', "__");
    let model_dir = cache_dir.join(&dir_name);
    if model_dir.is_dir() {
        for entry in std::fs::read_dir(&model_dir).ok()? {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "gguf") {
                return Some(path);
            }
        }
    }
    None
}

pub async fn pull<F>(repo_id: &str, cache_dir: &Path, on_progress: F) -> Result<PulledModel>
where
    F: Fn(u64, u64) + Send + 'static,
{
    let dir_name = repo_id.replace('/', "__");
    let model_dir = cache_dir.join(&dir_name);
    std::fs::create_dir_all(&model_dir).map_err(Error::Io)?;

    let client = reqwest::Client::new();

    let files = fetch_gguf_files(&client, repo_id).await?;
    if files.is_empty() {
        return Err(Error::Other(format!("no GGUF files found in {}", repo_id)));
    }

    let (filename, remote_size) = &files[0];
    let target_path = model_dir.join(filename);

    if target_path.exists() {
        let existing_size = std::fs::metadata(&target_path).map(|m| m.len()).unwrap_or(0);
        if existing_size == *remote_size {
            return Ok(PulledModel {
                path: target_path,
                repo_id: repo_id.to_string(),
                file_size: existing_size,
            });
        }
    }

    let download_url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, filename);
    let resp = client
        .get(&download_url)
        .send()
        .await
        .map_err(|e| Error::Other(format!("download failed: {e}")))?;

    let total = resp.content_length().unwrap_or(*remote_size);
    on_progress(0, total);

    let mut file = std::fs::File::create(&target_path).map_err(Error::Io)?;
    let mut downloaded: u64 = 0;
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| Error::Other(format!("download error: {e}")))?;
        use std::io::Write;
        file.write_all(&chunk).map_err(Error::Io)?;
        downloaded += chunk.len() as u64;
        on_progress(downloaded, total);
    }

    Ok(PulledModel {
        path: target_path,
        repo_id: repo_id.to_string(),
        file_size: downloaded,
    })
}

async fn fetch_gguf_files(client: &reqwest::Client, repo_id: &str) -> Result<Vec<(String, u64)>> {
    let api_url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
    let resp = client
        .get(&api_url)
        .send()
        .await
        .map_err(|e| Error::Other(format!("failed to query HF API: {e}")))?;

    let entries: Vec<serde_json::Value> = resp
        .json()
        .await
        .map_err(|e| Error::Other(format!("failed to parse HF API response: {e}")))?;

    let files: Vec<(String, u64)> = entries
        .iter()
        .filter(|e| e["type"].as_str() == Some("file"))
        .filter(|e| e["path"].as_str().is_some_and(|p| p.ends_with(".gguf")))
        .filter_map(|e| {
            let path = e["path"].as_str()?.to_string();
            let size = e["size"].as_u64().unwrap_or(0);
            Some((path, size))
        })
        .collect();

    Ok(files)
}


