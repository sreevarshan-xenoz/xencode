//! Document parsing into context — deterministic text extraction from
//! document files so they ride the context pipeline like any other text.
//!
//! v1 formats:
//!
//!   - PDF via `pdf-extract` (pure Rust, no external binaries).
//!   - DOCX by unzipping `word/document.xml` and reading `<w:t>` runs.
//!
//!   - [`is_document_path`] — extension pre-filter (`.pdf`, `.docx`).
//!   - [`parse_document_bytes`] — pure core over bytes.
//!   - [`parse_document`] — file entry point with a size cap.
//!
//! A [`MAX_DOC_CHARS`] cap keeps a 500-page manual from blowing up the
//! prompt (truncation is marked, never silent). Scanned/image-only PDFs
//! extract to nothing — the empty text is returned as-is so the caller can
//! say so instead of pretending the document was read.
//!
//! Follow-ups (not here): CSV/TSV tables, OCR for scanned pages.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Refuse documents larger than this before parsing.
pub const MAX_DOC_BYTES: usize = 20 * 1024 * 1024;

/// Longest extracted text kept; the rest is cut with a marked trailer.
pub const MAX_DOC_CHARS: usize = 100_000;

/// Document formats the pipeline understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocKind {
    Pdf,
    Docx,
}

#[derive(Debug, thiserror::Error)]
pub enum DocError {
    #[error("cannot read {0}: {1}")]
    ReadError(String, String),
    #[error("{0} exceeds the {1}-byte document cap")]
    TooLarge(String, usize),
    #[error("{0} is not a supported document (expected .pdf or .docx)")]
    UnknownType(String),
    #[error("cannot parse {0} as {1}: {2}")]
    ParseError(String, String, String),
}

/// Extracted document text — serializable for CLI JSON output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocText {
    pub path: String,
    pub kind: DocKind,
    pub text: String,
    /// True when the text hit [`MAX_DOC_CHARS`] and was cut.
    pub truncated: bool,
    pub bytes: u64,
}

/// True for supported document extensions (case-insensitive). Content
/// dispatch happens in [`parse_document_bytes`]; this is the cheap pre-filter.
pub fn is_document_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .as_deref(),
        Some("pdf" | "docx")
    )
}

/// Classify `bytes` for `path` into a [`DocText`]. Pure over the bytes
/// apart from no I/O at all — unit-tested.
pub fn parse_document_bytes(path: &str, bytes: &[u8]) -> Result<DocText, DocError> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    let (kind, text) = match ext.as_str() {
        "pdf" => (DocKind::Pdf, extract_pdf(path, bytes)?),
        "docx" => (DocKind::Docx, extract_docx(path, bytes)?),
        _ => return Err(DocError::UnknownType(path.to_string())),
    };
    let (text, truncated) = truncate(&text);
    Ok(DocText {
        path: path.to_string(),
        kind,
        text,
        truncated,
        bytes: bytes.len() as u64,
    })
}

/// Read `path` off disk and parse it. Enforces [`MAX_DOC_BYTES`] before
/// parsing so a stray export can't blow up the pipeline.
pub fn parse_document(path: &Path) -> Result<DocText, DocError> {
    let bytes = std::fs::read(path)
        .map_err(|e| DocError::ReadError(path.display().to_string(), e.to_string()))?;
    if bytes.len() > MAX_DOC_BYTES {
        return Err(DocError::TooLarge(
            path.display().to_string(),
            MAX_DOC_BYTES,
        ));
    }
    parse_document_bytes(&path.display().to_string(), &bytes)
}

/// Cut `text` to [`MAX_DOC_CHARS`] with a marked trailer. Pure.
fn truncate(text: &str) -> (String, bool) {
    if text.chars().count() <= MAX_DOC_CHARS {
        return (text.to_string(), false);
    }
    let kept: String = text.chars().take(MAX_DOC_CHARS).collect();
    (
        format!("{kept}\n…[truncated to {MAX_DOC_CHARS} chars]"),
        true,
    )
}

fn extract_pdf(path: &str, bytes: &[u8]) -> Result<String, DocError> {
    pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| DocError::ParseError(path.to_string(), "pdf".to_string(), e.to_string()))
}

/// DOCX is a zip: `word/document.xml` holds `<w:t>` text runs inside
/// `<w:p>` paragraphs. Paragraph breaks become newlines; runs inside one
/// paragraph join directly (Word splits runs mid-word for styling).
fn extract_docx(path: &str, bytes: &[u8]) -> Result<String, DocError> {
    let fail = |detail: String| DocError::ParseError(path.to_string(), "docx".to_string(), detail);
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| fail(e.to_string()))?;
    let mut xml = String::new();
    {
        let mut entry = archive
            .by_name("word/document.xml")
            .map_err(|e| fail(e.to_string()))?;
        std::io::Read::read_to_string(&mut entry, &mut xml).map_err(|e| fail(e.to_string()))?;
    }
    // One pass in document order: paragraph closes become newlines, runs
    // append directly (Word splits runs mid-word for styling, so runs
    // inside one paragraph join without a separator).
    let token = regex::Regex::new("(?s)</w:p\\s*>|<w:t[^>]*>(.*?)</w:t\\s*>").unwrap();
    let mut out = String::new();
    for caps in token.captures_iter(&xml) {
        match caps.get(1) {
            Some(run) => out.push_str(&decode_xml_entities(run.as_str())),
            None => out.push('\n'),
        }
    }
    // Collapse whitespace runs but keep the paragraph newlines.
    let blank = regex::Regex::new("[ \\t\\r]+").unwrap();
    let out = blank.replace_all(&out, " ");
    let newlines = regex::Regex::new("\\n\\s*\\n+").unwrap();
    Ok(newlines.replace_all(out.trim(), "\n").into_owned())
}

/// The entities Word actually emits. Unknown ones pass through verbatim.
fn decode_xml_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal multi-page PDF in-test with computed xref offsets,
    /// so no fixture files are needed. `pages` holds one text line each;
    /// parens/backslash must not appear in them.
    fn minimal_pdf(pages: &[&str]) -> Vec<u8> {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        let mut emit = |pdf: &mut Vec<u8>, offsets: &mut Vec<usize>, body: &[u8]| {
            offsets.push(pdf.len());
            pdf.extend_from_slice(body);
        };
        // 1: catalog, 2: pages, then per page: page, content; F: font.
        let count = pages.len();
        let first_page_obj = 3;
        let font_obj = 3 + 2 * count;
        let kids: String = (0..count)
            .map(|i| format!("{} 0 R ", first_page_obj + 2 * i))
            .collect();
        emit(
            &mut pdf,
            &mut offsets,
            format!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n").as_bytes(),
        );
        emit(
            &mut pdf,
            &mut offsets,
            format!("2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {count} >>\nendobj\n")
                .as_bytes(),
        );
        for (i, text) in pages.iter().enumerate() {
            let page_obj = first_page_obj + 2 * i;
            let content_obj = page_obj + 1;
            emit(
                &mut pdf,
                &mut offsets,
                format!(
                    "{page_obj} 0 obj\n<< /Type /Page /Parent 2 0 R \
                     /MediaBox [0 0 612 792] /Contents {content_obj} 0 R \
                     /Resources << /Font << /F1 {font_obj} 0 R >> >> >>\nendobj\n"
                )
                .as_bytes(),
            );
            let stream = format!("BT /F1 24 Tf 72 720 Td ({text}) Tj ET\n");
            emit(
                &mut pdf,
                &mut offsets,
                format!(
                    "{content_obj} 0 obj\n<< /Length {} >>\nstream\n{stream}endstream\nendobj\n",
                    stream.len()
                )
                .as_bytes(),
            );
        }
        emit(
            &mut pdf,
            &mut offsets,
            format!("{font_obj} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
                .as_bytes(),
        );
        let total = offsets.len() + 1;
        let xref_at = pdf.len();
        pdf.extend_from_slice(format!("xref\n0 {total}\n0000000000 65535 f \n").as_bytes());
        for off in &offsets {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!("trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n")
                .as_bytes(),
        );
        pdf
    }

    #[test]
    fn extracts_pdf_pages_in_order() {
        let bytes = minimal_pdf(&["Alpha line", "Beta line"]);
        let doc = parse_document_bytes("paper.pdf", &bytes).unwrap();
        assert_eq!(doc.kind, DocKind::Pdf);
        assert!(!doc.truncated);
        assert_eq!(doc.bytes, bytes.len() as u64);
        let a = doc.text.find("Alpha line").expect("page one text");
        let b = doc.text.find("Beta line").expect("page two text");
        assert!(a < b, "pages in order: {doc:?}");
    }

    #[test]
    fn rejects_garbage_as_pdf() {
        assert!(parse_document_bytes("paper.pdf", b"not a pdf at all").is_err());
    }

    /// Minimal DOCX: a zip holding word/document.xml with two paragraphs.
    fn minimal_docx() -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            zip.start_file(
                "word/document.xml",
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
            std::io::Write::write_all(
                &mut zip,
                br#"<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body><w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>wor</w:t></w:r><w:r><w:t>ld</w:t></w:r></w:p>
<w:p><w:r><w:t>Fish &amp; Chips</w:t></w:r></w:p></w:body></w:document>"#,
            )
            .unwrap();
            zip.finish().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn extracts_docx_runs_joined_paragraphs_split() {
        let bytes = minimal_docx();
        let doc = parse_document_bytes("note.docx", &bytes).unwrap();
        assert_eq!(doc.kind, DocKind::Docx);
        assert_eq!(doc.text, "Hello world\nFish & Chips");
    }

    #[test]
    fn rejects_zip_without_document_xml() {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            zip.start_file("other.txt", zip::write::SimpleFileOptions::default())
                .unwrap();
            std::io::Write::write_all(&mut zip, b"hi").unwrap();
            zip.finish().unwrap();
        }
        assert!(parse_document_bytes("note.docx", &buf.into_inner()).is_err());
    }

    #[test]
    fn rejects_unknown_extensions() {
        assert!(matches!(
            parse_document_bytes("a.txt", b"hi"),
            Err(DocError::UnknownType(_))
        ));
    }

    #[test]
    fn truncate_marks_long_text() {
        let long = "x".repeat(MAX_DOC_CHARS + 10);
        let (kept, truncated) = truncate(&long);
        assert!(truncated);
        assert!(kept.ends_with(&format!("…[truncated to {MAX_DOC_CHARS} chars]")));
        assert!(kept.chars().count() <= MAX_DOC_CHARS + 40);
        let (short, t2) = truncate("tiny");
        assert!(!t2);
        assert_eq!(short, "tiny");
    }

    #[test]
    fn path_filter_is_case_insensitive() {
        assert!(is_document_path(Path::new("a.PDF")));
        assert!(is_document_path(Path::new("d/e/f.DocX")));
        assert!(!is_document_path(Path::new("main.rs")));
        assert!(!is_document_path(Path::new("paper.pdf.exe")));
    }
}