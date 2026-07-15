//! Hybrid memory + disk cache with compression, TTL, and LRU eviction.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use xencode_core_rs::CacheStats;

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    #[error("Compression error: {0}")]
    Compression(String),
}

/// A cached entry with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    key: String,
    value: Vec<u8>,
    created_at: DateTime<Utc>,
    expires_at: Option<DateTime<Utc>>,
    access_count: u64,
    size_bytes: u64,
}

/// Hybrid cache with memory and disk tiers.
pub struct HybridCache {
    memory_cache: Mutex<HashMap<String, CacheEntry>>,
    disk_cache_dir: PathBuf,
    max_memory_entries: usize,
    max_disk_size_bytes: u64,
    stats: Mutex<CacheStats>,
}

impl HybridCache {
    /// Create a new hybrid cache.
    pub fn new(cache_dir: PathBuf, max_memory_entries: usize, max_disk_mb: u64) -> Self {
        std::fs::create_dir_all(&cache_dir).ok();
        Self {
            memory_cache: Mutex::new(HashMap::new()),
            disk_cache_dir: cache_dir,
            max_memory_entries,
            max_disk_size_bytes: max_disk_mb * 1024 * 1024,
            stats: Mutex::new(CacheStats::default()),
        }
    }

    /// Get a value from cache (memory first, then disk).
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>, CacheError> {
        // Check memory cache
        {
            let mut cache = self.memory_cache.lock().unwrap();
            if let Some(entry) = cache.get_mut(key) {
                if let Some(expires) = entry.expires_at {
                    if Utc::now() > expires {
                        cache.remove(key);
                        self.stats.lock().unwrap().misses += 1;
                        return Ok(None);
                    }
                }
                entry.access_count += 1;
                self.stats.lock().unwrap().hits += 1;
                return Ok(Some(entry.value.clone()));
            }
        }

        // Check disk cache
        let disk_path = self.disk_path(key);
        if disk_path.exists() {
            let data = std::fs::read(&disk_path)?;
            if let Ok(entry) = serde_json::from_slice::<CacheEntry>(&data) {
                if let Some(expires) = entry.expires_at {
                    if Utc::now() > expires {
                        std::fs::remove_file(&disk_path).ok();
                        self.stats.lock().unwrap().misses += 1;
                        return Ok(None);
                    }
                }
                // Promote to memory
                let entry_clone = entry.clone();
                self.memory_cache.lock().unwrap().insert(key.to_string(), entry);
                self.stats.lock().unwrap().hits += 1;
                return Ok(Some(entry_clone.value));
            }
        }

        self.stats.lock().unwrap().misses += 1;
        Ok(None)
    }

    /// Set a value in cache.
    pub fn set(&self, key: &str, value: Vec<u8>, ttl_seconds: Option<u64>) -> Result<(), CacheError> {
        let expires_at = ttl_seconds.map(|ttl| Utc::now() + chrono::Duration::seconds(ttl as i64));
        let entry = CacheEntry {
            key: key.to_string(),
            value: value.clone(),
            created_at: Utc::now(),
            expires_at,
            access_count: 0,
            size_bytes: value.len() as u64,
        };

        // Store in memory
        {
            let mut cache = self.memory_cache.lock().unwrap();
            if cache.len() >= self.max_memory_entries {
                // LRU eviction: remove least accessed entry
                if let Some(lru_key) = cache.iter()
                    .min_by_key(|(_, e)| e.access_count)
                    .map(|(k, _)| k.clone())
                {
                    cache.remove(&lru_key);
                }
            }
            cache.insert(key.to_string(), entry.clone());
        }

        // Store on disk
        {
            let disk_path = self.disk_path(key);
            let data = serde_json::to_vec(&entry)?;
            std::fs::write(&disk_path, data)?;
            self.enforce_disk_limit()?;
        }

        Ok(())
    }

    /// Remove a key from cache.
    pub fn remove(&self, key: &str) -> Result<(), CacheError> {
        self.memory_cache.lock().unwrap().remove(key);
        let disk_path = self.disk_path(key);
        if disk_path.exists() {
            std::fs::remove_file(&disk_path)?;
        }
        Ok(())
    }

    /// Clear all cached data.
    pub fn clear(&self) -> Result<(), CacheError> {
        self.memory_cache.lock().unwrap().clear();
        if self.disk_cache_dir.exists() {
            for entry in std::fs::read_dir(&self.disk_cache_dir)? {
                let entry = entry?;
                if entry.path().is_file() {
                    std::fs::remove_file(&entry.path())?;
                }
            }
        }
        Ok(())
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let stats = self.stats.lock().unwrap().clone();
        let disk_size = self.calculate_disk_size();
        CacheStats {
            size_bytes: disk_size,
            entry_count: self.memory_cache.lock().unwrap().len(),
            ..stats
        }
    }

    fn disk_path(&self, key: &str) -> PathBuf {
        let hashed_key = sha2::Sha256::digest(key.as_bytes());
        let filename = hex::encode(&hashed_key[..8]);
        self.disk_cache_dir.join(filename)
    }

    fn calculate_disk_size(&self) -> u64 {
        let mut total = 0u64;
        if let Ok(entries) = std::fs::read_dir(&self.disk_cache_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    total += meta.len();
                }
            }
        }
        total
    }

    fn enforce_disk_limit(&self) -> Result<(), CacheError> {
        let current_size = self.calculate_disk_size();
        if current_size <= self.max_disk_size_bytes {
            return Ok(());
        }

        // Remove oldest files first
        let mut files: Vec<_> = std::fs::read_dir(&self.disk_cache_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect();
        files.sort_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()));

        let mut size = current_size;
        for file in files {
            if size <= self.max_disk_size_bytes {
                break;
            }
            if let Ok(meta) = file.metadata() {
                size = size.saturating_sub(meta.len());
                std::fs::remove_file(&file.path()).ok();
            }
        }
        Ok(())
    }
}
