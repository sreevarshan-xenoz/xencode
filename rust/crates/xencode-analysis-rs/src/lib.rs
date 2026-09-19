pub mod analyzer;
pub mod embeddings;
pub mod images;
pub mod indexer;
pub mod issues;
pub mod security;
pub mod vector_store;
pub mod web;

pub use analyzer::CodeAnalyzer;
pub use embeddings::EmbeddingClient;
pub use images::{
    analyze_image, detect_format, dimensions, inspect_bytes, is_image_path, to_data_url,
    ImageError, ImageFormat, ImageMeta, MAX_IMAGE_BYTES,
};
pub use web::{extract_text, fetch_url, FetchError, FetchedPage, FETCH_TIMEOUT_SECS, MAX_PAGE_BYTES};
pub use indexer::{ChunkIndexer, DocumentChunk};
pub use issues::{
    AnalysisReport, AnalysisSummary, CodeIssue, IssueType, SecurityFinding, Severity,
};
pub use security::VulnerabilityScanner;
pub use vector_store::{cosine_similarity, VectorStore};
