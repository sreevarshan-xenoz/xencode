pub mod analyzer;
pub mod embeddings;
pub mod indexer;
pub mod issues;
pub mod security;
pub mod vector_store;

pub use analyzer::CodeAnalyzer;
pub use embeddings::EmbeddingClient;
pub use indexer::{ChunkIndexer, DocumentChunk};
pub use issues::{
    AnalysisReport, AnalysisSummary, CodeIssue, IssueType, SecurityFinding, Severity,
};
pub use security::VulnerabilityScanner;
pub use vector_store::{cosine_similarity, VectorStore};
