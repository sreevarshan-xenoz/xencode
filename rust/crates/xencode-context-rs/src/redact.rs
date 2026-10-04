//! Deterministic egress redaction of the *dynamic* context tiers (PR-3).
//!
//! The context engine assembles a turn in two halves with opposite rules. The
//! **stable head** (system prompt + trusted `AGENTS.md` + anchor) is sent
//! byte-for-byte on every turn so a local server can reuse its key/value cache;
//! redacting it would break that reuse *and* trip the `/ctx kv` drift check, so
//! this module never touches it. The **dynamic tiers** — task state, git facts,
//! the repo map, retrieved file bodies, the prior conversation, and the current
//! prompt — change every turn and have no cache to lose. Those are what gets
//! redacted here before they leave the machine.
//!
//! It is a *placeholder/restore* map, not a one-way scrub: each distinct secret
//! is replaced by a stable token (`«xencode-secret-1»`, numbered by first
//! appearance so the same value in two tiers collapses to one token), and the
//! real value is kept in a [`Vault`] for the one place it is needed again — when
//! the model writes a command that names the token, [`Vault::restore`] puts the
//! secret back so the command actually runs, without the plaintext ever having
//! crossed to the provider.
//!
//! **Honesty about recall.** This is pattern-based: it removes the credential
//! *shapes* xencode knows (private-key blocks, bearer tokens, prefixed API keys,
//! secret-named assignments). A secret that is not shaped like one of those — a
//! bare high-entropy string with no name beside it, a value in an unusual
//! format — passes through untouched. A redaction that reads as complete is
//! more dangerous than no redaction if it buys false reassurance, so this is a
//! best-effort reduction of what leaves the machine, not a guarantee, and the
//! real wall stays SE-4's approval gate and SE-7's sandbox.

use crate::trace::secret_spans;
use std::collections::HashMap;

/// The secret values a redaction pass took out, in the order first seen, so a
/// placeholder the model hands back can be turned into the real value at the
/// point of execution. Empty means nothing was redacted and restore is a no-op.
#[derive(Debug, Clone, Default)]
pub struct Vault {
    /// `placeholder -> real value`. Kept as a Vec (not a map) so
    /// [`Vault::placeholders`] reads back in the numbered order.
    entries: Vec<(String, String)>,
}

impl Vault {
    /// Nothing was redacted.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many distinct secrets were taken out. This is the number a run
    /// reports ("N secrets held back from this turn") without ever naming them.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The placeholder tokens handed to the model, in order. Never the values.
    pub fn placeholders(&self) -> impl Iterator<Item = &str> {
        self.entries
            .iter()
            .map(|(placeholder, _)| placeholder.as_str())
    }

    /// Put the redacted values back wherever their placeholder appears in
    /// `text`. A no-op on text with no placeholders, and safe to call on
    /// arbitrary command lines.
    pub fn restore(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (placeholder, secret) in &self.entries {
            if out.contains(placeholder.as_str()) {
                out = out.replace(placeholder.as_str(), secret);
            }
        }
        out
    }

    /// Restore across every string leaf of a tool call's JSON arguments,
    /// leaving numbers, booleans and structure untouched. This is the form the
    /// executor uses: a `command` or `path` that names a placeholder gets its
    /// real value back before the call is classified and run.
    pub fn restore_value(&self, value: serde_json::Value) -> serde_json::Value {
        if self.entries.is_empty() {
            return value;
        }
        match value {
            serde_json::Value::String(text) => serde_json::Value::String(self.restore(&text)),
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.into_iter().map(|i| self.restore_value(i)).collect())
            }
            serde_json::Value::Object(map) => serde_json::Value::Object(
                map.into_iter()
                    .map(|(k, v)| (k, self.restore_value(v)))
                    .collect(),
            ),
            other => other,
        }
    }
}

/// Builds a [`Vault`] by redacting text tier by tier. One redactor spans a whole
/// turn so a secret that appears in both the retrieval block and the prompt
/// collapses to a single placeholder rather than two, and the numbering is
/// decided by first appearance (left to right), never by hash or map order —
/// which is what makes the output deterministic and KV-cache-irrelevant.
#[derive(Debug, Default)]
pub struct Redactor {
    /// `real value -> placeholder`, so a repeat of a value reuses its token.
    by_secret: HashMap<String, String>,
    entries: Vec<(String, String)>,
}

impl Redactor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return `text` with every credential-shaped slice replaced by its
    /// placeholder, registering each distinct secret the first time it is seen.
    pub fn redact(&mut self, text: &str) -> String {
        let spans = secret_spans(text);
        if spans.is_empty() {
            return text.to_string();
        }
        let bytes = text.as_bytes();
        let mut out = String::with_capacity(text.len());
        let mut cursor = 0usize;
        for (start, end) in spans {
            // Spans are byte offsets from the regexes; every slice we copy is
            // therefore a whole char boundary on both ends.
            out.push_str(&text[cursor..start]);
            let secret = String::from_utf8_lossy(&bytes[start..end]).into_owned();
            let placeholder = self.intern(secret);
            out.push_str(&placeholder);
            cursor = end;
        }
        out.push_str(&text[cursor..]);
        out
    }

    /// Whether anything was taken out.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Freeze the collected secrets into a [`Vault`] for the run to restore from.
    pub fn into_vault(self) -> Vault {
        Vault {
            entries: self.entries,
        }
    }

    /// Register `secret` (or reuse its existing token) and return the
    /// placeholder to write in its place.
    fn intern(&mut self, secret: String) -> String {
        if let Some(existing) = self.by_secret.get(&secret) {
            return existing.clone();
        }
        let placeholder = format!("«xencode-secret-{}»", self.entries.len() + 1);
        self.by_secret.insert(secret.clone(), placeholder.clone());
        self.entries.push((placeholder.clone(), secret));
        placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_named_secret_becomes_a_placeholder_and_restores() {
        let mut redactor = Redactor::new();
        let line = "curl -H \"Authorization: Bearer sk-abc123DEF456ghi789\"";
        let redacted = redactor.redact(line);
        assert!(
            redacted.contains("«xencode-secret-"),
            "the credential must be gone: {redacted}"
        );
        assert!(
            !redacted.contains("sk-abc123"),
            "the raw token must not survive: {redacted}"
        );
        assert!(
            redacted.contains("Bearer") || redacted.contains("Authorization"),
            "the readable shape stays: {redacted}"
        );
        let vault = redactor.into_vault();
        assert_eq!(vault.len(), 1);
        assert_eq!(vault.restore(&redacted), line);
    }

    #[test]
    fn the_same_secret_in_two_tiers_shares_one_placeholder() {
        let mut redactor = Redactor::new();
        let a = redactor.redact("token=\"hunter2secretvalue\"");
        let b = redactor.redact("and again token=\"hunter2secretvalue\"");
        let vault = redactor.into_vault();
        assert_eq!(vault.len(), 1, "one distinct secret, one token");
        assert!(a.contains("«xencode-secret-1»"), "{a}");
        assert!(b.contains("«xencode-secret-1»"), "{b}");
        assert_eq!(vault.restore(&b), "and again token=\"hunter2secretvalue\"");
    }

    #[test]
    fn ordinary_text_is_returned_byte_for_byte() {
        let mut redactor = Redactor::new();
        let plain = "fn main() {\n    let count = 3; // a key insight\n}";
        assert_eq!(redactor.redact(plain), plain);
        assert!(redactor.into_vault().is_empty());
    }

    #[test]
    fn restore_is_a_noop_when_nothing_was_redacted() {
        let vault = Vault::default();
        assert!(vault.is_empty());
        assert_eq!(vault.restore("echo hi"), "echo hi");
        let value = serde_json::json!({"command": "ls", "n": 3, "nested": ["a", "b"]});
        assert_eq!(vault.restore_value(value.clone()), value);
    }

    #[test]
    fn restore_reaches_into_nested_json_arguments() {
        let mut redactor = Redactor::new();
        // The whole assignment line is what leaves the machine; only the value
        // slice is replaced, and the real secret is held for the run.
        let shown = redactor.redact("AWS_SECRET_ACCESS_KEY=\"wJalrXUtnFEMI\"");
        assert!(
            !shown.contains("wJalrXUtnFEMI"),
            "the secret must not be in the offered text: {shown}"
        );
        let vault = redactor.into_vault();
        let args = serde_json::json!({
            "command": format!("aws s3 ls --key {shown}"),
            "background": false,
            "timeout": 30,
        });
        // The model, asked to run the command, gets the placeholder back; the
        // executor restores the real value before it runs.
        let restored = vault.restore_value(args);
        assert_eq!(
            restored["command"].as_str().unwrap(),
            "aws s3 ls --key AWS_SECRET_ACCESS_KEY=\"wJalrXUtnFEMI\""
        );
        assert_eq!(restored["background"], false, "non-strings pass through");
        assert_eq!(restored["timeout"], 30);
    }
}
