pub mod advisories;
pub mod analyzer;
pub mod compete;
pub mod covdiff;
pub mod deps;
pub mod envdrift;
pub mod images;
pub mod issues;
pub mod merge_decision;
pub mod mutation;
#[cfg(test)]
pub mod property_eval;
#[cfg(test)]
pub mod property_eval_intersect;
pub mod runtime_hazards;
pub mod search;
pub mod security;
pub mod toolchain;
pub mod web;

pub use compete::{
    format_competing_table, list_competing_runs, load_competing_run, pick_arm, run_competing_arms,
    save_competing_run, ArmCheckRow, ArmResult, CandidateArmSpec, CompetingConfig, CompetingReport,
    PickOutcome,
};
pub use merge_decision::{
    build_merge_plan, execute_merge, precheck_branch, BranchCheck, BranchSpec, BranchVerdict,
    HumanMergeApproval, IntegrationCheck, MergeOutcome, MergePlan, MergePrecheck,
};

pub use analyzer::CodeAnalyzer;
pub use images::{
    analyze_image, detect_format, dimensions, inspect_bytes, is_image_path, prepare_for_send,
    to_data_url, ImageError, ImageFormat, ImageMeta, PreparedImage, JPEG_QUALITY, MAX_IMAGE_BYTES,
    MAX_IMAGE_EDGE,
};
pub use issues::{
    AnalysisReport, AnalysisSummary, CodeIssue, IssueType, SecurityFinding, Severity,
};
pub use search::{
    search as search_web, Hit as SearchHit, Provider as SearchProvider, SearchError,
    MAX_RESULTS as MAX_SEARCH_RESULTS,
};
pub use security::VulnerabilityScanner;
pub use web::{
    cap_chars, extract_text, fetch_url, fetch_url_guarded, guard_destination, llms_txt_url,
    FetchError, FetchedPage, DEFAULT_TEXT_CAP_CHARS, FETCH_TIMEOUT_SECS, MAX_PAGE_BYTES,
    MAX_REDIRECTS,
};
