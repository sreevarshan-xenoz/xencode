use serde::{Deserialize, Serialize};

/// Client for generating text embeddings via Ollama's embedding API.
pub struct EmbeddingClient {
    ollama_url: String,
    model: String,
    client: reqwest::Client,
}

impl EmbeddingClient {
    /// Create a new embedding client pointing to an Ollama server.
    pub fn new(ollama_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            ollama_url: ollama_url.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Generate an embedding vector for a single text string.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let url = format!("{}/api/embeddings", self.ollama_url);
        let body = EmbeddingRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

        let data: EmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        Ok(data.embedding)
    }

    /// Generate embeddings for multiple texts in batch.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Get the dimension of the embedding model.
    pub fn dimensions(&self) -> usize {
        match self.model.as_str() {
            "nomic-embed-text" | "all-minilm" => 768,
            "llama3.2" => 3072,
            _ => 768, // default assumption
        }
    }
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Failed to parse response: {0}")]
    ParseError(String),
}
