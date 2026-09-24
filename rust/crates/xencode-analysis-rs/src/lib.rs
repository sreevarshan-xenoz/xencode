pub mod analyzer;
pub mod images;
pub mod issues;
pub mod security;
pub mod web;

pub use analyzer::CodeAnalyzer;
pub use images::{
    analyze_image, detect_format, dimensions, inspect_bytes, is_image_path, prepare_for_send,
    to_data_url, ImageError, ImageFormat, ImageMeta, PreparedImage, JPEG_QUALITY, MAX_IMAGE_BYTES,
    MAX_IMAGE_EDGE,
};
pub use issues::{
    AnalysisReport, AnalysisSummary, CodeIssue, IssueType, SecurityFinding, Severity,
};
pub use security::VulnerabilityScanner;
pub use web::{
    extract_text, fetch_url, FetchError, FetchedPage, FETCH_TIMEOUT_SECS, MAX_PAGE_BYTES,
};
