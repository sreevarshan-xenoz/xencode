//! Image input pipeline (multimodal step 1) — deterministic image intake.
//!
//! Zero-LLM handling of image files so they can enter the context pipeline:
//!
//!   - [`is_image_path`] — extension allowlist (case-insensitive).
//!   - [`detect_format`] — magic-byte sniffing, so a misnamed file is still
//!     classified by its content.
//!   - [`dimensions`] — width/height parsed from the file headers
//!     (PNG IHDR, JPEG SOF, GIF LSD, WebP VP8/VP8L/VP8X, BMP DIB, ICO entry).
//!     SVG is vector: detected, but has no intrinsic pixel dimensions.
//!   - [`analyze_image`] — file-level entry point with a size cap.
//!   - [`to_data_url`] — base64 data URL, the payload shape vision-capable
//!     providers accept when that support lands.
//!
//! Everything is pure over bytes and unit-tested with hand-crafted headers —
//! no fixture files, no network, no model calls.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Refuse images larger than this — a 20 MiB cap keeps a stray export from
/// blowing up the context pipeline or the data-URL payload.
pub const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// Raster/vector formats the pipeline understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Gif,
    WebP,
    Bmp,
    Ico,
    Svg,
}

impl ImageFormat {
    pub fn mime(self) -> &'static str {
        match self {
            ImageFormat::Png => "image/png",
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Gif => "image/gif",
            ImageFormat::WebP => "image/webp",
            ImageFormat::Bmp => "image/bmp",
            ImageFormat::Ico => "image/x-icon",
            ImageFormat::Svg => "image/svg+xml",
        }
    }

    /// Vector formats carry no intrinsic pixel dimensions.
    pub fn is_vector(self) -> bool {
        matches!(self, ImageFormat::Svg)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("cannot read {0}: {1}")]
    ReadError(String, String),
    #[error("{0} exceeds the {1}-byte image cap")]
    TooLarge(String, usize),
    #[error("{0} is not a recognized image")]
    UnknownFormat(String),
}

/// Image inventory record — serializable for the CLI's JSON output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageMeta {
    pub path: String,
    pub format: ImageFormat,
    /// Pixel dimensions; `None` for vector formats (SVG).
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub bytes: u64,
}

/// True for known image extensions (case-insensitive). Content sniffing via
/// [`detect_format`] is authoritative; this is the cheap pre-filter.
pub fn is_image_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "ico" | "svg")
    )
}

/// Classify image bytes by magic number. Returns `None` for non-images.
pub fn detect_format(bytes: &[u8]) -> Option<ImageFormat> {
    if bytes.len() >= 8 && bytes[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some(ImageFormat::Png);
    }
    if bytes.len() >= 3 && bytes[..3] == [0xFF, 0xD8, 0xFF] {
        return Some(ImageFormat::Jpeg);
    }
    if bytes.len() >= 6 && (bytes[..6] == *b"GIF87a" || bytes[..6] == *b"GIF89a") {
        return Some(ImageFormat::Gif);
    }
    if bytes.len() >= 12 && bytes[..4] == *b"RIFF" && bytes[8..12] == *b"WEBP" {
        return Some(ImageFormat::WebP);
    }
    if bytes.len() >= 2 && bytes[..2] == *b"BM" {
        return Some(ImageFormat::Bmp);
    }
    if bytes.len() >= 4 && bytes[..4] == [0x00, 0x00, 0x01, 0x00] {
        return Some(ImageFormat::Ico);
    }
    // SVG is text: leading whitespace/BOM tolerated, must contain <svg.
    let head = bytes.get(..256).unwrap_or(bytes);
    let text = String::from_utf8_lossy(head);
    let trimmed = text.trim_start_matches(|c: char| c.is_whitespace() || c == '\u{FEFF}');
    if trimmed.starts_with('<') && text.contains("<svg") {
        return Some(ImageFormat::Svg);
    }
    None
}

fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn u16be(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}

fn u32be(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// Pixel dimensions for `format`, or `None` for vector formats and for
/// truncated/corrupt headers. Never panics on short input.
pub fn dimensions(bytes: &[u8], format: ImageFormat) -> Option<(u32, u32)> {
    match format {
        ImageFormat::Png => {
            // Signature (8) + length (4) + "IHDR" + width (4) + height (4).
            if bytes.len() >= 24 && bytes[12..16] == *b"IHDR" {
                Some((u32be(&bytes[16..20]), u32be(&bytes[20..24])))
            } else {
                None
            }
        }
        ImageFormat::Jpeg => jpeg_dimensions(bytes),
        ImageFormat::Gif => {
            if bytes.len() >= 10 {
                Some((u16le(&bytes[6..8]) as u32, u16le(&bytes[8..10]) as u32))
            } else {
                None
            }
        }
        ImageFormat::WebP => webp_dimensions(bytes),
        ImageFormat::Bmp => {
            // BITMAPINFOHEADER (size 40): width/height i32 LE at 18/22.
            if bytes.len() >= 26
                && u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]) == 40
            {
                let w = i32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
                let h = i32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]);
                if w > 0 && h != 0 {
                    return Some((w as u32, h.unsigned_abs()));
                }
            }
            None
        }
        ImageFormat::Ico => {
            // First directory entry: width/height bytes at 6/7 (0 means 256).
            if bytes.len() >= 8 {
                let w = if bytes[6] == 0 { 256 } else { bytes[6] as u32 };
                let h = if bytes[7] == 0 { 256 } else { bytes[7] as u32 };
                Some((w, h))
            } else {
                None
            }
        }
        ImageFormat::Svg => None,
    }
}

/// Walk JPEG markers to the first Start-Of-Frame segment, which carries the
/// dimensions. Skips standalone markers (RSTn, SOI, EOI, TEM) and fill bytes.
fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let mut i = 2usize; // past SOI
    while i + 1 < bytes.len() {
        if bytes[i] != 0xFF {
            return None;
        }
        // Skip fill bytes; 0xFF 0x00 is escaped entropy data, not a marker.
        let mut marker = bytes[i + 1];
        let mut j = i + 1;
        while marker == 0xFF && j + 1 < bytes.len() {
            j += 1;
            marker = bytes[j];
        }
        i = j + 1;
        if marker == 0x00 || marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
            continue;
        }
        if i + 1 >= bytes.len() {
            return None;
        }
        let len = u16be(&bytes[i..i + 2]) as usize;
        if len < 2 || i + len > bytes.len() {
            return None;
        }
        // SOF0–SOF15 except DHT (C4), JPG (C8), DAC (CC).
        if matches!(
            marker,
            0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB | 0xCD | 0xCE | 0xCF
        ) && len >= 7
        {
            return Some((u16be(&bytes[i + 5..i + 7]) as u32, u16be(&bytes[i + 3..i + 5]) as u32));
        }
        i += len;
    }
    None
}

/// WebP dimensions by chunk: VP8 lossy frame header, VP8L lossless bits, or
/// VP8X extended canvas size.
fn webp_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 21 {
        return None;
    }
    match &bytes[12..16] {
        b"VP8 " => {
            // 3-byte frame tag + 3-byte start code (9D 01 2A), then u16 LE w/h.
            if bytes.len() >= 30 && bytes[23..26] == [0x9D, 0x01, 0x2A] {
                let w = u16le(&bytes[26..28]) & 0x3FFF;
                let h = u16le(&bytes[28..30]) & 0x3FFF;
                Some((w as u32, h as u32))
            } else {
                None
            }
        }
        b"VP8L" => {
            if bytes.len() >= 25 && bytes[20] == 0x2F {
                let bits = u32::from_le_bytes([bytes[21], bytes[22], bytes[23], bytes[24]]);
                let w = (bits & 0x3FFF) + 1;
                let h = ((bits >> 14) & 0x3FFF) + 1;
                Some((w, h))
            } else {
                None
            }
        }
        b"VP8X" => {
            if bytes.len() >= 30 {
                let w = ((bytes[24] as u32) << 16
                    | (bytes[25] as u32) << 8
                    | bytes[26] as u32)
                    + 1;
                let h = ((bytes[27] as u32) << 16
                    | (bytes[28] as u32) << 8
                    | bytes[29] as u32)
                    + 1;
                Some((w, h))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Classify `bytes` into an [`ImageMeta`] for `path`. Pure — unit-tested.
pub fn inspect_bytes(path: &str, bytes: &[u8]) -> Result<ImageMeta, ImageError> {
    let format =
        detect_format(bytes).ok_or_else(|| ImageError::UnknownFormat(path.to_string()))?;
    let (width, height) = dimensions(bytes, format).unzip();
    Ok(ImageMeta {
        path: path.to_string(),
        format,
        width,
        height,
        bytes: bytes.len() as u64,
    })
}

/// Read `path` off disk and classify it. Enforces [`MAX_IMAGE_BYTES`] before
/// sniffing so a stray export can't blow up the pipeline.
pub fn analyze_image(path: &Path) -> Result<ImageMeta, ImageError> {
    let bytes = std::fs::read(path)
        .map_err(|e| ImageError::ReadError(path.display().to_string(), e.to_string()))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(ImageError::TooLarge(
            path.display().to_string(),
            MAX_IMAGE_BYTES,
        ));
    }
    inspect_bytes(&path.display().to_string(), &bytes)
}

/// Base64 data URL for `bytes` — the payload shape vision-capable providers
/// accept. Kept beside the metadata so intake and transport agree on format.
pub fn to_data_url(format: ImageFormat, bytes: &[u8]) -> String {
    use base64::Engine as _;
    format!(
        "data:{};base64,{}",
        format.mime(),
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_file(w: u32, h: u32) -> Vec<u8> {
        let mut v = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        v.extend_from_slice(&13u32.to_be_bytes());
        v.extend_from_slice(b"IHDR");
        v.extend_from_slice(&w.to_be_bytes());
        v.extend_from_slice(&h.to_be_bytes());
        v.extend_from_slice(&[8, 2, 0, 0, 0]);
        v
    }

    #[test]
    fn detects_png_and_reads_ihdr_dimensions() {
        let bytes = png_file(800, 600);
        assert_eq!(detect_format(&bytes), Some(ImageFormat::Png));
        assert_eq!(dimensions(&bytes, ImageFormat::Png), Some((800, 600)));
        let meta = inspect_bytes("a.png", &bytes).unwrap();
        assert_eq!(meta.width, Some(800));
        assert_eq!(meta.bytes, bytes.len() as u64);
    }

    #[test]
    fn detects_gif_variants_and_lsd_dimensions() {
        for tag in [b"GIF87a".as_slice(), b"GIF89a".as_slice()] {
            let mut v = tag.to_vec();
            v.extend_from_slice(&320u16.to_le_bytes());
            v.extend_from_slice(&200u16.to_le_bytes());
            assert_eq!(detect_format(&v), Some(ImageFormat::Gif));
            assert_eq!(dimensions(&v, ImageFormat::Gif), Some((320, 200)));
        }
    }

    #[test]
    fn detects_jpeg_sof0_dimensions_across_app_segments() {
        let mut v = vec![0xFF, 0xD8]; // SOI
        v.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]); // APP0 len 16
        v.extend_from_slice(&[0u8; 14]);
        v.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x0B, 0x08]); // SOF0
        v.extend_from_slice(&96u16.to_be_bytes()); // height
        v.extend_from_slice(&64u16.to_be_bytes()); // width
        v.extend_from_slice(&[0x03, 0x01, 0x22, 0x00]);
        v.extend_from_slice(&[0xFF, 0xD9]); // EOI
        assert_eq!(detect_format(&v), Some(ImageFormat::Jpeg));
        assert_eq!(dimensions(&v, ImageFormat::Jpeg), Some((64, 96)));
    }

    #[test]
    fn detects_webp_lossy_and_lossless() {
        // VP8: RIFF + WEBP + VP8 chunk with start code 9D 01 2A.
        let mut v = b"RIFF".to_vec();
        v.extend_from_slice(&100u32.to_le_bytes());
        v.extend_from_slice(b"WEBPVP8 ");
        v.extend_from_slice(&20u32.to_le_bytes());
        v.extend_from_slice(&[0x10, 0x00, 0x00, 0x9D, 0x01, 0x2A]);
        v.extend_from_slice(&320u16.to_le_bytes());
        v.extend_from_slice(&240u16.to_le_bytes());
        assert_eq!(detect_format(&v), Some(ImageFormat::WebP));
        assert_eq!(dimensions(&v, ImageFormat::WebP), Some((320, 240)));

        // VP8L 16x8: packed bits 0x1C00F LE.
        let mut l = b"RIFF".to_vec();
        l.extend_from_slice(&50u32.to_le_bytes());
        l.extend_from_slice(b"WEBPVP8L");
        l.extend_from_slice(&10u32.to_le_bytes());
        l.extend_from_slice(&[0x2F, 0x0F, 0xC0, 0x01, 0x00]);
        assert_eq!(dimensions(&l, ImageFormat::WebP), Some((16, 8)));
    }

    #[test]
    fn detects_bmp_and_ico_dimensions() {
        let mut v = b"BM".to_vec();
        v.extend_from_slice(&70u32.to_le_bytes());
        v.extend_from_slice(&[0u8; 4]);
        v.extend_from_slice(&54u32.to_le_bytes());
        v.extend_from_slice(&40u32.to_le_bytes()); // BITMAPINFOHEADER
        v.extend_from_slice(&800i32.to_le_bytes());
        v.extend_from_slice(&600i32.to_le_bytes());
        v.extend_from_slice(&[1, 0, 24, 0]);
        assert_eq!(detect_format(&v), Some(ImageFormat::Bmp));
        assert_eq!(dimensions(&v, ImageFormat::Bmp), Some((800, 600)));

        let ico = [0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x20, 0x20];
        assert_eq!(detect_format(&ico), Some(ImageFormat::Ico));
        assert_eq!(dimensions(&ico, ImageFormat::Ico), Some((32, 32)));
    }

    #[test]
    fn detects_svg_without_dimensions() {
        let svg = b"<?xml version=\"1.0\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>";
        assert_eq!(detect_format(svg), Some(ImageFormat::Svg));
        assert_eq!(dimensions(svg, ImageFormat::Svg), None);
        assert!(ImageFormat::Svg.is_vector());
    }

    #[test]
    fn rejects_non_images_and_truncated_headers() {
        assert_eq!(detect_format(b"hello world"), None);
        assert_eq!(detect_format(b""), None);
        assert_eq!(detect_format(b"\x89PNG\r\n"), None);
        assert!(inspect_bytes("x.bin", b"not an image").is_err());
        // PNG signature but truncated IHDR → recognized, no dimensions.
        let mut v = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        v.extend_from_slice(b"IHDR");
        assert_eq!(dimensions(&v, ImageFormat::Png), None);
    }

    #[test]
    fn path_filter_is_case_insensitive() {
        assert!(is_image_path(Path::new("shot.PNG")));
        assert!(is_image_path(Path::new("a/b/c.JpEg")));
        assert!(!is_image_path(Path::new("main.rs")));
        assert!(!is_image_path(Path::new("photo.png.exe")));
    }

    #[test]
    fn data_url_round_trips_through_base64() {
        let bytes = png_file(1, 1);
        let url = to_data_url(ImageFormat::Png, &bytes);
        assert!(url.starts_with("data:image/png;base64,"));
        use base64::Engine as _;
        let raw = &url["data:image/png;base64,".len()..];
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(raw)
                .unwrap(),
            bytes
        );
    }
}