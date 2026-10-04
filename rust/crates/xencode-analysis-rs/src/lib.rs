pub mod advisories;
pub mod analyzer;
pub mod covdiff;
pub mod deps;
pub mod envdrift;
pub mod images;
pub mod issues;
pub mod mutation;
#[cfg(test)]
pub mod property_eval;
#[cfg(test)]
pub mod property_eval_intersect;
pub mod runtime_hazards;
pub mod security;
pub mod toolchain;
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
    cap_chars, extract_text, fetch_url, fetch_url_guarded, guard_destination, FetchError,
    FetchedPage, DEFAULT_TEXT_CAP_CHARS, FETCH_TIMEOUT_SECS, MAX_PAGE_BYTES, MAX_REDIRECTS,
};
