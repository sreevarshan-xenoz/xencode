//! Resource URIs and prompt rendering for `xencode mcp serve` (MS-1).
//!
//! The server publishes two kinds of real MCP resource:
//!
//! * `xencode://workspace/<path>` — a file that actually exists under the
//!   workspace the server was started with, read from disk on `resources/read`;
//! * `xencode://config/summary` — a description of this running server, built
//!   from its own launch state rather than from a canned string.
//!
//! Both `resources/read` and the prompt renderer are deliberately pure here so
//! the URI boundary and the template rendering can be tested without a client.
//! The handler in [`crate::mcp_serve`] does the actual file I/O using the same
//! `path_allowed` gate the tools use, so a URI cannot reach outside the
//! workspace any more than a `path` argument can.

/// Prefix of a URI that names a real file under the workspace.
pub const WORKSPACE_PREFIX: &str = "xencode://workspace/";
/// URI of the synthesized description of this server's launch.
pub const CONFIG_SUMMARY_URI: &str = "xencode://config/summary";
/// The one prompt template xencode publishes.
pub const REVIEW_DIFF_PROMPT: &str = "review-diff";

/// The shipped explain-diff skill is a real file in this repository; embedding
/// it at build time keeps the rendered prompt identical to the guidance xencode
/// already gives an agent asked to explain a diff, without needing that file to
/// be present in whichever workspace happens to be served.
const REVIEW_DIFF_TEMPLATE: &str = include_str!("../../../../skills/explain-diff/SKILL.md");

/// Build the URI a client uses to name a workspace file.
pub fn workspace_uri(relative: &str) -> String {
    let mut uri = String::from(WORKSPACE_PREFIX);
    for byte in relative.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'/' => {
                uri.push(byte as char)
            }
            _ => uri.push_str(&format!("%{byte:02X}")),
        }
    }
    uri
}

/// Decode the `%XX` escapes in a URI's path portion. An invalid escape is an
/// error so a malformed URI can never be silently interpreted as a different
/// path.
fn percent_decode(raw: &str) -> Result<String, String> {
    let bytes = raw.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                let hex = bytes
                    .get(i + 1)
                    .zip(bytes.get(i + 2))
                    .ok_or_else(|| "truncated percent escape in resource URI".to_string())?;
                let pair = [*hex.0, *hex.1];
                let value = u8::from_str_radix(
                    std::str::from_utf8(&pair).map_err(|_| "invalid percent escape".to_string())?,
                    16,
                )
                .map_err(|_| "invalid percent escape in resource URI".to_string())?;
                decoded.push(value);
                i += 3;
            }
            byte => {
                decoded.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8(decoded).map_err(|_| "resource URI is not valid UTF-8".to_string())
}

/// Turn a `xencode://workspace/...` URI back into a relative path. Percent
/// escapes are decoded *before* the check, so an encoded `..` or `/` cannot slip
/// past it. Anything that is not a plain relative path inside the workspace is
/// refused with a reason the client can show.
pub fn workspace_relative_from_uri(uri: &str) -> Result<String, String> {
    let raw = uri.strip_prefix(WORKSPACE_PREFIX).ok_or_else(|| {
        format!("unsupported resource URI `{uri}`; expected {WORKSPACE_PREFIX}<path> or {CONFIG_SUMMARY_URI}")
    })?;
    let decoded = percent_decode(raw)?;
    if decoded.trim().is_empty() {
        return Err("resource URI names no file".to_string());
    }
    let escapes = std::path::Path::new(&decoded)
        .components()
        .any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        });
    if escapes {
        return Err(format!(
            "resource URI `{uri}` leaves the workspace; path traversal is refused"
        ));
    }
    Ok(decoded)
}

/// Render the shipped explain-diff guidance around a caller-supplied diff.
/// The diff is text the client actually supplied, not a placeholder answer.
pub fn render_review_diff(diff: &str) -> String {
    format!(
        "{}\n\n## Diff to review\n\n```diff\n{}\n```\n",
        REVIEW_DIFF_TEMPLATE.trim_end(),
        diff.trim_end()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_workspace_uri_round_trips_a_real_relative_path() {
        let uri = workspace_uri("src/main.rs");
        assert_eq!(uri, "xencode://workspace/src/main.rs");
        assert_eq!(workspace_relative_from_uri(&uri).unwrap(), "src/main.rs");
    }

    #[test]
    fn spaces_are_encoded_and_decoded() {
        let uri = workspace_uri("my notes.md");
        assert_eq!(uri, "xencode://workspace/my%20notes.md");
        assert_eq!(workspace_relative_from_uri(&uri).unwrap(), "my notes.md");
    }

    #[test]
    fn raw_and_encoded_traversal_are_both_refused() {
        for uri in [
            "xencode://workspace/../secret",
            "xencode://workspace/%2e%2e/secret",
            "xencode://workspace/%2E%2E%2Fsecret",
            "xencode://workspace//etc/passwd",
            "xencode://workspace/%2fetc/passwd",
        ] {
            assert!(
                workspace_relative_from_uri(uri).is_err(),
                "{uri} escaped the workspace"
            );
        }
    }

    #[test]
    fn a_foreign_scheme_or_malformed_escape_is_refused() {
        assert!(workspace_relative_from_uri("file:///etc/passwd").is_err());
        assert!(workspace_relative_from_uri("xencode://workspace/%ZZ").is_err());
        assert!(workspace_relative_from_uri("xencode://workspace/").is_err());
    }

    #[test]
    fn the_rendered_prompt_uses_the_shipped_template_and_the_callers_diff() {
        let diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new";
        let rendered = render_review_diff(diff);
        assert!(rendered.contains("# Explain a diff"), "{rendered}");
        assert!(
            rendered.contains(diff),
            "the caller's diff was not rendered"
        );
    }
}
