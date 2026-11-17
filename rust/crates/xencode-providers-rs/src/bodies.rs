//! Bounds on how much of a provider's response this process will hold.
//!
//! The listing fetch has always had a cap. Answers and streamed frames did not:
//! a misbehaving endpoint, a proxy that streams forever, or a user-supplied
//! `remote:` URL could grow a `String` without limit. Every response now spends
//! from the same fixed budget before any of its bytes are kept.

use crate::ProviderError;

/// The most response body any one call will keep. The price-listing fetch has
/// used 8 MiB since it was added; answers get the same generous bound.
pub const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// Counts the raw bytes kept from one response while it is being read.
#[derive(Debug, Default)]
pub struct ByteBudget {
    used: usize,
}

impl ByteBudget {
    pub fn new() -> Self {
        Self::default()
    }

    /// Charge incoming bytes before keeping them, so an over-large response
    /// fails without ever being fully buffered.
    pub fn charge(&mut self, bytes: usize, label: &str) -> Result<(), ProviderError> {
        if bytes.saturating_add(self.used) > MAX_RESPONSE_BYTES {
            return Err(ProviderError::Network(format!(
                "{label} response exceeded the {MAX_RESPONSE_BYTES}-byte limit"
            )));
        }
        self.used = self.used.saturating_add(bytes);
        Ok(())
    }
}

/// Read a whole non-streaming response as text, refusing anything larger than
/// the budget. `response.text()` buffers first and asks questions later; this
/// reads in chunks and stops as soon as the limit is crossed.
pub async fn limited_text(
    mut response: reqwest::Response,
    label: &str,
) -> Result<String, ProviderError> {
    let mut budget = ByteBudget::new();
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| ProviderError::Network(format!("{label} response read failed: {e}")))?
    {
        budget.charge(chunk.len(), label)?;
        bytes.extend_from_slice(&chunk);
    }
    String::from_utf8(bytes)
        .map_err(|e| ProviderError::Parse(format!("{label} response was not UTF-8: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_budget_refuses_before_it_would_exceed_the_cap() {
        let mut budget = ByteBudget::new();
        budget.charge(MAX_RESPONSE_BYTES, "Test").unwrap();
        let err = budget.charge(1, "Test").unwrap_err();
        assert!(err.to_string().contains("exceeded"), "{err}");
    }
}
