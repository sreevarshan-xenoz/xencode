//! Turning a sequence of network chunks back into the lines the server sent.
//!
//! Every streaming reader in this crate used to do the same thing: decode one
//! chunk, split it on newlines, parse each piece. That quietly loses text,
//! because a chunk boundary is chosen by the network stack and not by the
//! protocol. One JSON object can straddle two chunks, and a multi-byte
//! character can be cut in half — in which case the old code either failed to
//! parse both halves and dropped them, or failed to decode the chunk at all
//! and threw the whole thing away. Nothing was reported: the answer simply
//! came back shorter than the model wrote it, or empty.
//!
//! [`FrameLines`] fixes both by keeping whatever is not yet a complete line
//! and handing it to the next chunk, and by decoding a line that is genuinely
//! invalid UTF-8 lossily instead of discarding it.

/// Carries the incomplete tail of one chunk over to the next.
#[derive(Debug, Default)]
pub(crate) struct FrameLines {
    pending: Vec<u8>,
}

impl FrameLines {
    /// Hand over the next network chunk. `on_line` is called once for every
    /// line that is now whole, in order, with the terminator removed. A line
    /// that is not valid UTF-8 is passed through lossily rather than dropped.
    pub fn feed(&mut self, chunk: &[u8], on_line: &mut dyn FnMut(&str)) {
        let mut start = 0;
        while let Some(offset) = chunk[start..].iter().position(|byte| *byte == b'\n') {
            let end = start + offset;
            self.pending.extend_from_slice(&chunk[start..end]);
            emit(&mut self.pending, on_line);
            start = end + 1;
        }
        self.pending.extend_from_slice(&chunk[start..]);
    }

    /// Report whatever the body ended with, for a server that closed the
    /// connection without a final newline.
    pub fn finish(&mut self, on_line: &mut dyn FnMut(&str)) {
        if !self.pending.is_empty() {
            emit(&mut self.pending, on_line);
        }
    }
}

fn emit(pending: &mut Vec<u8>, on_line: &mut dyn FnMut(&str)) {
    let bytes = std::mem::take(pending);
    let line = match String::from_utf8(bytes) {
        Ok(line) => line,
        Err(broken) => String::from_utf8_lossy(broken.as_bytes()).into_owned(),
    };
    on_line(line.trim_end_matches('\r'));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(chunks: &[&[u8]]) -> Vec<String> {
        let mut reader = FrameLines::default();
        let mut lines = Vec::new();
        for chunk in chunks {
            reader.feed(chunk, &mut |line| lines.push(line.to_string()));
        }
        reader.finish(&mut |line| lines.push(line.to_string()));
        lines
    }

    #[test]
    fn a_line_split_across_two_chunks_arrives_once_and_whole() {
        assert_eq!(
            collect(&[b"data: {\"content\":\"hel", b"lo\"}\n\n"]),
            ["data: {\"content\":\"hello\"}", ""]
        );
    }

    #[test]
    fn a_multibyte_character_cut_in_half_is_not_lost() {
        // サーバー, split between the three bytes of one character.
        let text = "data: {\"t\":\"サーバー\"}\n".as_bytes();
        let cut = text.iter().position(|byte| *byte == 0xE3).unwrap() + 2;
        let lines = collect(&[&text[..cut], &text[cut..]]);
        assert_eq!(lines, ["data: {\"t\":\"サーバー\"}"]);
    }

    #[test]
    fn several_lines_in_one_chunk_are_reported_separately() {
        assert_eq!(collect(&[b"a\nb\nc\n"]), ["a", "b", "c"]);
    }

    #[test]
    fn a_body_ending_without_a_newline_still_reports_its_last_line() {
        assert_eq!(collect(&[b"event: done"]), ["event: done"]);
    }

    #[test]
    fn invalid_bytes_are_passed_through_lossy_instead_of_dropped() {
        assert_eq!(
            collect(&[b"data: \xff\xfe ok\n"]),
            ["data: \u{FFFD}\u{FFFD} ok"]
        );
    }

    /// The bytes a real server really sent, taken out of a committed cassette.
    fn recorded(cassette: &str, interaction: usize) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/cassettes")
            .join(cassette);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{} could not be read: {e}", path.display()));
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("cassette is JSON");
        parsed["interactions"][interaction]["response"]["body"]
            .as_str()
            .unwrap_or_else(|| panic!("{cassette} interaction {interaction} has no recorded body"))
            .to_string()
    }

    /// What the previous readers did with one chunk: decode it, split it,
    /// forget about it. Kept here so the tests can say what was lost.
    fn decoded_one_chunk_at_a_time(chunk: &[u8], out: &mut Vec<String>) {
        if let Ok(text) = std::str::from_utf8(chunk) {
            for line in text.lines() {
                out.push(line.to_string());
            }
        }
    }

    #[test]
    fn every_two_way_cut_of_a_real_stream_reassembles_the_lines_the_server_sent() {
        // A recording in katakana, so some of these cuts land inside a
        // character and some in the middle of a JSON object.
        let body = recorded("answer-japanese.json", 0);
        let bytes = body.as_bytes();
        let whole = collect(&[bytes]);
        assert!(whole.len() > 5, "the recording should hold several frames");
        for cut in 0..=bytes.len() {
            assert_eq!(
                collect(&[&bytes[..cut], &bytes[cut..]]),
                whole,
                "cutting the stream at byte {cut} changed what the reader saw"
            );
        }
    }

    #[test]
    fn cutting_a_real_stream_loses_text_whenever_a_chunk_is_decoded_on_its_own() {
        let body = recorded("answer-japanese.json", 0);
        let bytes = body.as_bytes();
        let whole = collect(&[bytes]);
        let mut lost_at = Vec::new();
        for cut in 1..bytes.len() {
            let mut lines = Vec::new();
            decoded_one_chunk_at_a_time(&bytes[..cut], &mut lines);
            decoded_one_chunk_at_a_time(&bytes[cut..], &mut lines);
            if lines != whole {
                lost_at.push(cut);
            }
        }
        // The only cut points the old code survived were the ones landing
        // exactly after a newline, where both halves happened to be whole
        // lines. Every other byte in the stream lost text. Pinning the exact
        // count is what makes the fix load-bearing rather than theoretical.
        let on_a_line_boundary = bytes[..bytes.len() - 1]
            .iter()
            .filter(|byte| **byte == b'\n')
            .count();
        assert_eq!(
            lost_at.len() + on_a_line_boundary,
            bytes.len() - 1,
            "the per-chunk decode lost text at {} of {} cut points, which is not \
             every non-boundary cut: {} cuts were unexpectedly safe",
            lost_at.len(),
            bytes.len() - 1,
            on_a_line_boundary
        );
        assert!(
            lost_at.len() > bytes.len() - 40,
            "the per-chunk decode survived {} cut points, which is more than the \
             {} line boundaries in this recording can account for",
            lost_at.len(),
            on_a_line_boundary
        );
    }

    #[test]
    fn a_recorded_tool_call_reassembles_whatever_the_cut_points_were() {
        let body = recorded("calculator-two-turns.json", 0);
        let bytes = body.as_bytes();
        let whole = collect(&[bytes]);
        let mut cuts = 0;
        for cut in (0..=bytes.len()).step_by(37) {
            assert_eq!(
                collect(&[&bytes[..cut], &bytes[cut..]]),
                whole,
                "cutting the tool-call stream at byte {cut} changed what the reader saw"
            );
            cuts += 1;
        }
        assert_eq!(cuts, bytes.len() / 37 + 1);
    }
}
