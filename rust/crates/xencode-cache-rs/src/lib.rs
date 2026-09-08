use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A cached response entry with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    response: String,
    model: String,
    prompt_hash: String,
    timestamp: f64,
    hit_count: u64,
}

/// Statistics about cache usage.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub evictions: u64,
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hit_rate = if self.hits + self.misses > 0 {
            (self.hits as f64 / (self.hits + self.misses) as f64) * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "entries: {}, hits: {}, misses: {}, hit_rate: {:.1}%, evictions: {}",
            self.entries, self.hits, self.misses, hit_rate, self.evictions
        )
    }
}

/// Errors from cache operations.
#[derive(Debug)]
pub enum CacheError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NoHomeDir,
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::Io(source) => write!(f, "cache I/O error: {source}"),
            CacheError::Json(source) => write!(f, "cache parse error: {source}"),
            CacheError::NoHomeDir => write!(f, "could not determine home directory"),
        }
    }
}

impl std::error::Error for CacheError {}

/// LRU response cache with TTL expiry and optional disk persistence.
///
/// Mirrors the Python `ResponseCache` from `xencode/core/cache.py`.
pub struct ResponseCache {
    entries: HashMap<String, CacheEntry>,
    max_size: usize,
    ttl_seconds: f64,
    cache_dir: Option<PathBuf>,
    stats: CacheStats,
}

impl ResponseCache {
    /// Create a new in-memory cache with the given capacity and TTL.
    pub fn new(max_size: usize, ttl_seconds: f64) -> Self {
        Self {
            entries: HashMap::new(),
            max_size,
            ttl_seconds,
            cache_dir: None,
            stats: CacheStats::default(),
        }
    }

    /// Create a cache with disk persistence in `~/.xencode/cache/`.
    pub fn with_persistence(max_size: usize, ttl_seconds: f64) -> Result<Self, CacheError> {
        let cache_dir = dirs::home_dir()
            .ok_or(CacheError::NoHomeDir)?
            .join(".xencode")
            .join("cache");
        std::fs::create_dir_all(&cache_dir).map_err(CacheError::Io)?;

        let mut cache = Self {
            entries: HashMap::new(),
            max_size,
            ttl_seconds,
            cache_dir: Some(cache_dir),
            stats: CacheStats::default(),
        };
        cache.load_from_disk()?;
        Ok(cache)
    }

    /// Look up a cached response by prompt and model.
    pub fn get(&mut self, prompt: &str, model: &str) -> Option<String> {
        let key = Self::cache_key(prompt, model);
        let now = current_timestamp();

        let expired = if let Some(entry) = self.entries.get(&key) {
            now - entry.timestamp > self.ttl_seconds
        } else {
            false
        };

        if expired {
            self.entries.remove(&key);
            self.stats.misses += 1;
            self.stats.entries = self.entries.len();
            return None;
        }

        if let Some(entry) = self.entries.get_mut(&key) {
            entry.hit_count += 1;
            self.stats.hits += 1;
            return Some(entry.response.clone());
        }

        self.stats.misses += 1;
        None
    }

    /// Store a response in the cache.
    pub fn set(&mut self, prompt: &str, model: &str, response: &str) {
        // Evict if at capacity (remove least-recently-used)
        if self.entries.len() >= self.max_size {
            self.evict_lru();
        }

        let key = Self::cache_key(prompt, model);
        let entry = CacheEntry {
            response: response.to_string(),
            model: model.to_string(),
            prompt_hash: key.clone(),
            timestamp: current_timestamp(),
            hit_count: 0,
        };

        self.entries.insert(key.clone(), entry);
        self.stats.entries = self.entries.len();

        // Persist to disk if enabled
        if self.cache_dir.is_some() {
            let _ = self.persist_entry(&key);
        }
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) -> Result<(), CacheError> {
        self.entries.clear();
        self.stats.entries = 0;

        if let Some(ref cache_dir) = self.cache_dir {
            if cache_dir.exists() {
                for entry in std::fs::read_dir(cache_dir).map_err(CacheError::Io)? {
                    let entry = entry.map_err(CacheError::Io)?;
                    let path = entry.path();
                    if path.extension().is_some_and(|ext| ext == "json") {
                        std::fs::remove_file(&path).map_err(CacheError::Io)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get current cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Generate a deterministic cache key from prompt + model.
    fn cache_key(prompt: &str, model: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        hasher.update(b"|");
        hasher.update(model.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Evict the least-recently-used entry (oldest timestamp with lowest hit_count).
    fn evict_lru(&mut self) {
        if let Some(key) = self
            .entries
            .iter()
            .min_by(|a, b| {
                a.1.timestamp
                    .partial_cmp(&b.1.timestamp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&key);
            self.stats.evictions += 1;

            // Remove from disk too
            if let Some(ref cache_dir) = self.cache_dir {
                let path = cache_dir.join(format!("{key}.json"));
                let _ = std::fs::remove_file(path);
            }
        }
    }

    /// Persist a single entry to disk.
    fn persist_entry(&self, key: &str) -> Result<(), CacheError> {
        if let Some(ref cache_dir) = self.cache_dir {
            if let Some(entry) = self.entries.get(key) {
                let path = cache_dir.join(format!("{key}.json"));
                let json = serde_json::to_string(entry).map_err(CacheError::Json)?;
                std::fs::write(path, json).map_err(CacheError::Io)?;
            }
        }
        Ok(())
    }

    /// Load all cached entries from disk.
    fn load_from_disk(&mut self) -> Result<(), CacheError> {
        if let Some(ref cache_dir) = self.cache_dir {
            if !cache_dir.exists() {
                return Ok(());
            }
            let now = current_timestamp();
            for entry in std::fs::read_dir(cache_dir).map_err(CacheError::Io)? {
                let entry = entry.map_err(CacheError::Io)?;
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "json") {
                    match std::fs::read_to_string(&path) {
                        Ok(content) => {
                            if let Ok(cached) = serde_json::from_str::<CacheEntry>(&content) {
                                // Skip expired entries
                                if now - cached.timestamp <= self.ttl_seconds {
                                    let key = path
                                        .file_stem()
                                        .unwrap_or_default()
                                        .to_string_lossy()
                                        .to_string();
                                    self.entries.insert(key, cached);
                                } else {
                                    // Clean up expired files
                                    let _ = std::fs::remove_file(&path);
                                }
                            }
                        }
                        Err(_) => continue,
                    }
                }
            }
            self.stats.entries = self.entries.len();
        }
        Ok(())
    }
}

fn current_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("xencode-cache-test-{stamp}"))
    }

    #[test]
    fn basic_set_and_get() {
        let mut cache = ResponseCache::new(10, 3600.0);
        cache.set("hello", "qwen:7b", "world");
        assert_eq!(cache.get("hello", "qwen:7b"), Some("world".to_string()));
    }

    #[test]
    fn miss_returns_none() {
        let mut cache = ResponseCache::new(10, 3600.0);
        assert_eq!(cache.get("missing", "model"), None);
    }

    #[test]
    fn different_models_are_separate_keys() {
        let mut cache = ResponseCache::new(10, 3600.0);
        cache.set("prompt", "model-a", "response-a");
        cache.set("prompt", "model-b", "response-b");
        assert_eq!(
            cache.get("prompt", "model-a"),
            Some("response-a".to_string())
        );
        assert_eq!(
            cache.get("prompt", "model-b"),
            Some("response-b".to_string())
        );
    }

    #[test]
    fn lru_eviction_when_full() {
        let mut cache = ResponseCache::new(2, 3600.0);
        cache.set("p1", "m", "r1");
        // Add a small delay to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.set("p2", "m", "r2");
        std::thread::sleep(std::time::Duration::from_millis(10));
        // This should evict p1 (oldest)
        cache.set("p3", "m", "r3");

        assert_eq!(cache.get("p1", "m"), None);
        assert_eq!(cache.get("p2", "m"), Some("r2".to_string()));
        assert_eq!(cache.get("p3", "m"), Some("r3".to_string()));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn ttl_expiry() {
        // Use 0 second TTL so everything expires immediately
        let mut cache = ResponseCache::new(10, 0.0);
        cache.set("prompt", "model", "response");
        // Should be expired
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert_eq!(cache.get("prompt", "model"), None);
    }

    #[test]
    fn stats_tracking() {
        let mut cache = ResponseCache::new(10, 3600.0);
        cache.set("p", "m", "r");
        cache.get("p", "m"); // hit
        cache.get("missing", "m"); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
    }

    #[test]
    fn clear_removes_all_entries() {
        let mut cache = ResponseCache::new(10, 3600.0);
        cache.set("p1", "m", "r1");
        cache.set("p2", "m", "r2");
        cache.clear().unwrap();
        assert_eq!(cache.get("p1", "m"), None);
        assert_eq!(cache.get("p2", "m"), None);
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn disk_persistence_roundtrip() {
        let dir = temp_dir();
        let cache_dir = dir.join("cache");
        fs::create_dir_all(&cache_dir).unwrap();

        // Create cache with manual cache_dir
        let mut cache = ResponseCache {
            entries: HashMap::new(),
            max_size: 10,
            ttl_seconds: 3600.0,
            cache_dir: Some(cache_dir.clone()),
            stats: CacheStats::default(),
        };

        cache.set("prompt", "model", "response");

        // Verify file was written
        let files: Vec<_> = fs::read_dir(&cache_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
            .collect();
        assert_eq!(files.len(), 1);

        // Create a new cache and load from disk
        let mut cache2 = ResponseCache {
            entries: HashMap::new(),
            max_size: 10,
            ttl_seconds: 3600.0,
            cache_dir: Some(cache_dir),
            stats: CacheStats::default(),
        };
        cache2.load_from_disk().unwrap();
        assert_eq!(cache2.get("prompt", "model"), Some("response".to_string()));

        fs::remove_dir_all(&dir).unwrap();
    }
}
