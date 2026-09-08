use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;

/// A chunk of a document ready for indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub content: String,
    pub language: String,
    pub chunk_id: String,
}

/// Splits source files into semantically meaningful chunks for RAG indexing.
pub struct ChunkIndexer;

impl ChunkIndexer {
    /// Chunk a file into semantically meaningful pieces.
    /// Python: splits at function/class boundaries, then line-based fallback.
    /// Rust: splits at fn/impl boundaries, then line-based.
    /// Generic: line-based chunking with overlap.
    pub fn chunk_file(path: &Path) -> Result<Vec<DocumentChunk>, IndexerError> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| IndexerError::ReadError(path.display().to_string(), e.to_string()))?;

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let language = match ext.as_str() {
            "py" => "python",
            "rs" => "rust",
            "js" | "jsx" => "javascript",
            "ts" | "tsx" => "typescript",
            "md" | "mdx" => "markdown",
            "json" => "json",
            "toml" => "toml",
            "yaml" | "yml" => "yaml",
            "html" => "html",
            "css" => "css",
            _ => "text",
        };

        let file_path = path.display().to_string();
        let chunks = match ext.as_str() {
            "py" => Self::chunk_python(&source, &file_path, language),
            "rs" => Self::chunk_rust(&source, &file_path, language),
            _ => Self::chunk_by_lines(&source, &file_path, language, 50, 5),
        };

        Ok(chunks)
    }

    /// Chunk Python source at function/class boundaries with line-based fallback.
    fn chunk_python(source: &str, file_path: &str, language: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = source.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Start a new chunk at class or function definition
            if line.starts_with("class ") || line.starts_with("def ") {
                let start_line = (i + 1) as u32;
                // Find the end of this function/class (next top-level def/class or end of file)
                let mut end = i + 1;
                let indent_level = lines[i]
                    .chars()
                    .position(|c| !c.is_whitespace())
                    .unwrap_or(0);
                while end < lines.len() {
                    let next_line = lines[end].trim();
                    if !next_line.is_empty()
                        && !next_line.starts_with('#')
                        && !next_line.starts_with('"')
                        && !next_line.starts_with('\'')
                    {
                        let next_indent = lines[end]
                            .chars()
                            .position(|c| !c.is_whitespace())
                            .unwrap_or(0);
                        if next_indent <= indent_level
                            && (next_line.starts_with("class ") || next_line.starts_with("def "))
                        {
                            break;
                        }
                    }
                    end += 1;
                }

                let content = lines[i..end].join("\n");
                chunks.push(DocumentChunk {
                    file_path: file_path.to_string(),
                    start_line,
                    end_line: end as u32,
                    content,
                    language: language.to_string(),
                    chunk_id: Uuid::new_v4().to_string(),
                });
                i = end;
            } else {
                i += 1;
            }
        }

        // If no semantic chunks found, fall back to line-based chunking
        if chunks.is_empty() {
            return Self::chunk_by_lines(source, file_path, language, 50, 5);
        }

        chunks
    }

    /// Chunk Rust source at fn/impl boundaries.
    fn chunk_rust(source: &str, file_path: &str, language: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = source.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let trimmed = lines[i].trim();
            if trimmed.starts_with("pub fn")
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("pub struct")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("pub enum")
                || trimmed.starts_with("enum ")
                || trimmed.starts_with("pub trait")
                || trimmed.starts_with("trait ")
                || trimmed.starts_with("impl")
            {
                let start_line = (i + 1) as u32;
                let mut end = i + 1;
                let mut open_braces = if trimmed.contains('{') { 1 } else { 0 };

                while end < lines.len() {
                    let l = lines[end];
                    open_braces += l.matches('{').count() as i32;
                    open_braces -= l.matches('}').count() as i32;
                    end += 1;
                    if open_braces <= 0 && l.trim().ends_with(';') {
                        break;
                    }
                    // Safety: if we've gone too far without closing, break
                    if end - i > 200 {
                        break;
                    }
                }

                let content = lines[i..end].join("\n");
                chunks.push(DocumentChunk {
                    file_path: file_path.to_string(),
                    start_line,
                    end_line: end as u32,
                    content,
                    language: language.to_string(),
                    chunk_id: Uuid::new_v4().to_string(),
                });
                i = end;
            } else {
                i += 1;
            }
        }

        if chunks.is_empty() {
            return Self::chunk_by_lines(source, file_path, language, 50, 5);
        }
        chunks
    }

    /// Simple line-based chunking with overlap.
    fn chunk_by_lines(
        source: &str,
        file_path: &str,
        language: &str,
        chunk_size: usize,
        overlap: usize,
    ) -> Vec<DocumentChunk> {
        let lines: Vec<&str> = source.lines().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < lines.len() {
            let end = std::cmp::min(start + chunk_size, lines.len());
            let content = lines[start..end].join("\n");
            chunks.push(DocumentChunk {
                file_path: file_path.to_string(),
                start_line: (start + 1) as u32,
                end_line: end as u32,
                content,
                language: language.to_string(),
                chunk_id: Uuid::new_v4().to_string(),
            });

            if end == lines.len() {
                break;
            }
            start += chunk_size - overlap;
        }

        chunks
    }

    /// Chunk a raw text string without file access.
    pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<DocumentChunk> {
        let lines: Vec<&str> = text.lines().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < lines.len() {
            let end = std::cmp::min(start + chunk_size, lines.len());
            let content = lines[start..end].join("\n");
            chunks.push(DocumentChunk {
                file_path: String::new(),
                start_line: (start + 1) as u32,
                end_line: end as u32,
                content,
                language: "text".to_string(),
                chunk_id: Uuid::new_v4().to_string(),
            });

            if end == lines.len() {
                break;
            }
            start += chunk_size - overlap;
        }

        chunks
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IndexerError {
    #[error("Failed to read file {0}: {1}")]
    ReadError(String, String),
}
