use std::time::Duration;

use tokio::time::sleep;

use crate::ProviderError;

/// Configuration for retry behaviour when calling provider APIs.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (not counting the initial attempt).
    pub max_retries: u32,
    /// Base delay in milliseconds for exponential backoff.
    pub base_delay_ms: u64,
    /// Maximum delay in milliseconds (caps exponential growth).
    pub max_delay_ms: u64,
    /// Multiply delay by this factor each retry.
    pub backoff_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 500,
            max_delay_ms: 10_000,
            backoff_factor: 2.0,
        }
    }
}

impl RetryConfig {
    /// Create a conservative retry config (few retries, shorter delays).
    pub fn conservative() -> Self {
        Self {
            max_retries: 2,
            base_delay_ms: 200,
            max_delay_ms: 5_000,
            backoff_factor: 2.0,
        }
    }

    /// Create an aggressive retry config (more retries, longer delays).
    pub fn aggressive() -> Self {
        Self {
            max_retries: 5,
            base_delay_ms: 1_000,
            max_delay_ms: 30_000,
            backoff_factor: 3.0,
        }
    }

    /// Compute the delay for the nth retry (0-based).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay = self.base_delay_ms as f64 * self.backoff_factor.powi(attempt as i32);
        let capped = delay.min(self.max_delay_ms as f64) as u64;
        Duration::from_millis(capped)
    }
}

/// Determines whether an error is retriable.
///
/// Retriable errors:
/// - Network errors (connectivity, timeout, DNS resolution)
/// - 5xx server errors (service temporarily unavailable)
/// - Rate limiting (429)
///
/// Non-retriable errors:
/// - 4xx client errors (except 429)
/// - Parse errors (malformed response — retrying won't help)
pub fn is_retriable(err: &ProviderError) -> bool {
    match err {
        // Network errors are always retriable
        ProviderError::Network(msg) => {
            // Don't retry if the message suggests a permanent DNS failure,
            // but retry everything else (timeouts, connection resets, etc.)
            !msg.contains("dns error")
                && !msg.contains("dns")
                && !msg.contains("Name or service not known")
        }
        // API errors: retry 429 (rate limit), 5xx (server errors), 503 (unavailable)
        ProviderError::Api(msg) => {
            msg.contains("429")         // rate limited
                || msg.contains("500")  // internal server error
                || msg.contains("502")  // bad gateway
                || msg.contains("503")  // service unavailable
                || msg.contains("504")  // gateway timeout
                || msg.contains("529")  // rate limited (Anthropic-specific)
        }
        // Parse errors are not retriable — the response came back but couldn't be parsed
        ProviderError::Parse(_) => false,
    }
}

/// Execute an async operation with retry logic.
///
/// Retries on network errors and 5xx/429 API errors with exponential backoff.
///
/// # Example
///
/// ```ignore
/// use crate::retry::{retry_async, RetryConfig};
///
/// let result = retry_async(
///     RetryConfig::default(),
///     || async { provider.generate(model, messages).await },
/// ).await;
/// ```
pub async fn retry_async<F, Fut, T>(config: &RetryConfig, operation: F) -> Result<T, ProviderError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, ProviderError>>,
{
    let mut last_error: Option<ProviderError> = None;

    for attempt in 0..=config.max_retries {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                last_error = Some(err);

                if let Some(ref err) = last_error {
                    if !is_retriable(err) {
                        // Non-retriable error — bail out immediately
                        return Err(std::mem::take(&mut last_error).unwrap());
                    }
                }

                if attempt < config.max_retries {
                    let delay = config.delay_for_attempt(attempt);
                    sleep(delay).await;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        ProviderError::Network("exhausted all retry attempts".to_string())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.base_delay_ms, 500);
        assert_eq!(cfg.max_delay_ms, 10_000);
        assert!((cfg.backoff_factor - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn delay_increases_exponentially() {
        let cfg = RetryConfig::default();
        let d0 = cfg.delay_for_attempt(0);
        let d1 = cfg.delay_for_attempt(1);
        let d2 = cfg.delay_for_attempt(2);

        assert_eq!(d0.as_millis(), 500);
        assert_eq!(d1.as_millis(), 1000);
        assert_eq!(d2.as_millis(), 2000);
    }

    #[test]
    fn delay_capped_at_max() {
        let cfg = RetryConfig {
            max_retries: 10,
            base_delay_ms: 1000,
            max_delay_ms: 5_000,
            backoff_factor: 3.0,
        };
        let d3 = cfg.delay_for_attempt(3); // 1000 * 3^3 = 27000, capped to 5000
        assert_eq!(d3.as_millis(), 5_000);
    }

    #[test]
    fn is_retriable_network_error() {
        let err = ProviderError::Network("connection reset by peer".to_string());
        assert!(is_retriable(&err));
    }

    #[test]
    fn is_retriable_429() {
        let err = ProviderError::Api("429 Too Many Requests".to_string());
        assert!(is_retriable(&err));
    }

    #[test]
    fn is_retriable_503() {
        let err = ProviderError::Api("503 Service Unavailable".to_string());
        assert!(is_retriable(&err));
    }

    #[test]
    fn is_not_retriable_400() {
        let err = ProviderError::Api("400 Bad Request: invalid model".to_string());
        assert!(!is_retriable(&err));
    }

    #[test]
    fn is_not_retriable_401() {
        let err = ProviderError::Api("401 Unauthorized: bad key".to_string());
        assert!(!is_retriable(&err));
    }

    #[test]
    fn is_not_retriable_parse_error() {
        let err = ProviderError::Parse("unexpected end of JSON".to_string());
        assert!(!is_retriable(&err));
    }

    #[tokio::test]
    async fn retry_succeeds_on_first_try() {
        let cfg = RetryConfig::default();
        let result: Result<i32, ProviderError> = retry_async(&cfg, || async {
            Ok::<i32, ProviderError>(42)
        }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn retry_fails_fast_on_non_retriable() {
        let cfg = RetryConfig::default();
        let result: Result<String, ProviderError> = retry_async(&cfg, || async {
            Err::<String, ProviderError>(ProviderError::Api("400 Bad Request".to_string()))
        }).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("400"));
    }

    #[tokio::test]
    async fn retry_succeeds_after_retries() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let counter = AtomicU32::new(0);

        let cfg = RetryConfig::default();
        let result: Result<String, ProviderError> = retry_async(&cfg, || async {
            let attempt = counter.fetch_add(1, Ordering::SeqCst);
            if attempt < 2 {
                Err(ProviderError::Network("timeout".to_string()))
            } else {
                Ok("success".to_string())
            }
        }).await;
        assert_eq!(result.unwrap(), "success");
        assert_eq!(counter.load(Ordering::SeqCst), 3); // initial + 2 retries
    }

    #[tokio::test]
    async fn retry_exhaustion() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let counter = AtomicU32::new(0);

        let cfg = RetryConfig {
            max_retries: 2,
            base_delay_ms: 10,
            max_delay_ms: 100,
            backoff_factor: 2.0,
        };
        let result: Result<String, ProviderError> = retry_async(&cfg, || async {
            counter.fetch_add(1, Ordering::SeqCst);
            Err::<String, ProviderError>(ProviderError::Network("always fails".to_string()))
        }).await;
        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 3); // initial + 2 retries
    }
}
