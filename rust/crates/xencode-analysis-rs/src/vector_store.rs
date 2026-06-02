use crate::indexer::DocumentChunk;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A vector store entry containing a document chunk and its embedding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreEntry {
    pub chunk: DocumentChunk,
    pub embedding: Vec<f32>,
}

/// A scored search result.
#[derive(Clone, Debug)]
pub struct ScoredEntry {
    pub entry: StoreEntry,
    pub score: f32,
}

/// Simple in-memory vector store with cosine similarity search.
#[derive(Clone, Serialize, Deserialize)]
pub struct VectorStore {
    dimensions: usize,
    entries: Vec<StoreEntry>,
}

impl VectorStore {
    /// Create a new vector store with the specified embedding dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            entries: Vec::new(),
        }
    }

    /// Insert a chunk with its embedding into the store.
    pub fn insert(&mut self, chunk: DocumentChunk, embedding: Vec<f32>) {
        self.entries.push(StoreEntry { chunk, embedding });
    }

    /// Insert multiple entries at once.
    pub fn insert_batch(&mut self, entries: Vec<(DocumentChunk, Vec<f32>)>) {
        for (chunk, embedding) in entries {
            self.entries.push(StoreEntry { chunk, embedding });
        }
    }

    /// Search for the top-k most similar entries by cosine similarity.
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<ScoredEntry> {
        let mut scored: Vec<ScoredEntry> = self
            .entries
            .iter()
            .map(|entry| {
                let score = cosine_similarity(query_embedding, &entry.embedding);
                ScoredEntry {
                    entry: entry.clone(),
                    score,
                }
            })
            .collect();

        // Sort by score descending (higher = more similar)
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Persist the store to a JSON file.
    pub fn persist(&self, path: &Path) -> Result<(), StoreError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| StoreError::SerializeError(e.to_string()))?;
        std::fs::write(path, &json)
            .map_err(|e| StoreError::WriteError(path.display().to_string(), e.to_string()))?;
        Ok(())
    }

    /// Load the store from a JSON file.
    pub fn load(path: &Path) -> Result<Self, StoreError> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| StoreError::ReadError(path.display().to_string(), e.to_string()))?;
        let store: VectorStore = serde_json::from_str(&json)
            .map_err(|e| StoreError::ParseError(e.to_string()))?;
        Ok(store)
    }

    /// Number of entries in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the embedding dimension.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get all entries (for inspection/export).
    pub fn entries(&self) -> &[StoreEntry] {
        &self.entries
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("Failed to serialize store: {0}")]
    SerializeError(String),
    #[error("Failed to write store to {0}: {1}")]
    WriteError(String, String),
    #[error("Failed to read store from {0}: {1}")]
    ReadError(String, String),
    #[error("Failed to parse store JSON: {0}")]
    ParseError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_store_search() {
        let mut store = VectorStore::new(3);
        store.insert(
            DocumentChunk {
                file_path: "a.txt".into(),
                start_line: 1,
                end_line: 1,
                content: "hello world".into(),
                language: "text".into(),
                chunk_id: "1".into(),
            },
            vec![1.0, 0.0, 0.0],
        );
        store.insert(
            DocumentChunk {
                file_path: "b.txt".into(),
                start_line: 1,
                end_line: 1,
                content: "goodbye world".into(),
                language: "text".into(),
                chunk_id: "2".into(),
            },
            vec![0.0, 1.0, 0.0],
        );

        let results = store.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.chunk.chunk_id, "1");
        assert!(results[0].score > 0.99);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut store = VectorStore::new(2);
        store.insert(
            DocumentChunk {
                file_path: "test.rs".into(),
                start_line: 1,
                end_line: 5,
                content: "fn test() {}".into(),
                language: "rust".into(),
                chunk_id: "test-1".into(),
            },
            vec![0.5, 0.5],
        );

        let json = serde_json::to_string(&store).unwrap();
        let restored: VectorStore = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.entries()[0].chunk.file_path, "test.rs");
    }
}
