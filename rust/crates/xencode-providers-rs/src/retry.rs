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
        ProviderError::Network(msg) => !is_name_resolution_failure(msg),
        // Decided on the status, not on the text of the body. Searching the
        // message for "500" retried permanent 4xx responses whose body merely
        // mentioned the digits.
        ProviderError::Api { status, .. } => match status {
            Some(429) | Some(500) | Some(502) | Some(503) | Some(504) => true,
            // 529 is Anthropic's "overloaded".
            Some(529) => true,
            Some(_) => false,
            // No HTTP status behind it — a missing key, an unusable response
            // shape. Retrying cannot change the outcome.
            None => false,
        },
        // Parse errors are not retriable — the response came back but couldn't be parsed
        ProviderError::Parse(_) => false,
    }
}

/// Whether a network error message describes a name that will not resolve.
///
/// Still string matching, because `ProviderError::Network` carries only the
/// formatted message, but at least it covers the platforms we run on: the
/// previous check tested `"Name or service not known"` (glibc) and would not
/// match macOS, so permanent DNS failures there burned the whole retry
/// schedule. `"dns error"` was also redundant with `"dns"`.
fn is_name_resolution_failure(msg: &str) -> bool {
    let msg = msg.to_ascii_lowercase();
    msg.contains("dns")
        || msg.contains("name or service not known")      // glibc
        || msg.contains("nodename nor servname provided") // macOS
        || msg.contains("no such host")                   // Windows / hyper
        || msg.contains("failed to lookup address")
}

/// Spread a backoff delay over `[delay/2, delay]` ("equal jitter").
///
/// Without this, clients that hit the same rate limit at the same moment all
/// back off by the identical amount and retry in lockstep, re-triggering it.
/// Half the delay is kept fixed so a retry still waits a sensible minimum.
///
/// Entropy comes from the clock's sub-millisecond noise rather than a new
/// dependency — decorrelating retries does not need a good RNG.
fn jittered(delay: Duration) -> Duration {
    let half = delay / 2;
    let span = delay - half;
    if span.is_zero() {
        return delay;
    }
    let noise = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.subsec_nanos() as u64)
        .unwrap_or(0);
    // Nanosecond arithmetic, not milliseconds: rounding to whole milliseconds
    // lands below `delay / 2` for delays under 2ms.
    half + Duration::from_nanos(noise % (span.as_nanos() as u64 + 1))
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
    retry_async_with_guard(config, || false, operation).await
}

/// Execute an async operation with retry logic and an early-stop predicate.
///
/// Identical to [`retry_async`] except that `should_stop` is consulted before
/// each attempt (and again before sleeping after a retriable error): when it
/// returns `true`, retrying stops and the most recent error is returned.
///
/// This is for operations with irreversible side effects — e.g. a stream that
/// has already delivered tokens to a consumer. Re-running such an operation
/// would duplicate the side effects, so the predicate lets the caller cut the
/// retry loop short once delivery has begun.
pub async fn retry_async_with_guard<F, Fut, T>(
    config: &RetryConfig,
    mut should_stop: impl FnMut() -> bool,
    operation: F,
) -> Result<T, ProviderError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, ProviderError>>,
{
    let mut last_error: Option<ProviderError> = None;

    for attempt in 0..=config.max_retries {
        // Belt-and-suspenders safety: the primary stop check happens in the
        // error branch below (before sleeping), but this guard guarantees the
        // operation is never re-invoked once the caller's side effects have
        // begun (e.g. tokens already delivered), even if the error path were
        // ever refactored to miss that check.
        if attempt > 0 && should_stop() {
            break;
        }

        match operation().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                // Stop on non-retriable errors, after the final attempt, or
                // when the caller's side effects have started — checking here
                // (before sleeping) avoids a wasted backoff delay.
                let retriable = is_retriable(&err);
                last_error = Some(err);

                if !retriable || attempt == config.max_retries || should_stop() {
                    break;
                }

                sleep(jittered(config.delay_for_attempt(attempt))).await;
            }
        }
    }

    Err(last_error
        .unwrap_or_else(|| ProviderError::Network("exhausted all retry attempts".to_string())))
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
        let err = ProviderError::api("Test", 429u16, "Too Many Requests");
        assert!(is_retriable(&err));
    }

    #[test]
    fn is_retriable_503() {
        let err = ProviderError::api("Test", 503u16, "Service Unavailable");
        assert!(is_retriable(&err));
    }

    #[test]
    fn is_not_retriable_400() {
        let err = ProviderError::api("Test", 400u16, "Bad Request: invalid model");
        assert!(!is_retriable(&err));
    }

    #[test]
    fn is_not_retriable_401() {
        let err = ProviderError::api("Test", 401u16, "Unauthorized: bad key");
        assert!(!is_retriable(&err));
    }

    /// The regression: retriability was decided by searching the message for
    /// status digits, so a permanent 4xx whose body merely mentioned them was
    /// retried three times with backoff before surfacing the same error.
    #[test]
    fn a_permanent_error_is_not_retried_because_its_body_mentions_a_5xx_number() {
        for body in [
            "max_tokens: 500 exceeds the model's limit",
            "model gpt-4-0502 not found",
            "you have 503 credits remaining",
            "request_id req_429ab3c1 was rejected",
            "invalid value for parameter 'top_k': 504",
        ] {
            let err = ProviderError::api("Test", 400u16, body);
            assert!(
                !is_retriable(&err),
                "retried a permanent 400 because its body said: {body}"
            );
        }
    }

    /// The mirror image: a genuine server error whose body happens not to
    /// repeat the status must still be retried.
    #[test]
    fn a_server_error_is_retried_even_when_its_body_omits_the_status() {
        let err = ProviderError::api("Test", 502u16, "upstream connect failure");
        assert!(is_retriable(&err));
    }

    #[test]
    fn an_api_error_with_no_status_is_not_retriable() {
        // A missing key, an unusable response shape — retrying changes nothing.
        let err = ProviderError::api_message("Anthropic API key not configured");
        assert!(!is_retriable(&err));
    }

    #[test]
    fn name_resolution_failures_are_not_retried_on_any_platform() {
        for msg in [
            "error trying to connect: dns error: failed to lookup address information",
            "failed to lookup address information: Name or service not known", // glibc
            "nodename nor servname provided, or not known",                    // macOS
            "no such host is known",                                           // Windows
        ] {
            let err = ProviderError::Network(msg.to_string());
            assert!(
                !is_retriable(&err),
                "retried a permanent DNS failure: {msg}"
            );
        }
    }

    #[test]
    fn transient_network_failures_are_still_retried() {
        for msg in [
            "connection reset by peer",
            "operation timed out",
            "connection refused",
        ] {
            let err = ProviderError::Network(msg.to_string());
            assert!(is_retriable(&err), "did not retry a transient error: {msg}");
        }
    }

    #[test]
    fn jitter_stays_within_half_the_delay_and_the_delay() {
        for millis in [1u64, 2, 100, 500, 10_000] {
            let base = Duration::from_millis(millis);
            for _ in 0..200 {
                let got = jittered(base);
                assert!(
                    got >= base / 2 && got <= base,
                    "jittered({millis}ms) = {got:?}, outside [half, full]"
                );
            }
        }
    }

    #[test]
    fn jitter_leaves_a_zero_delay_alone() {
        assert_eq!(jittered(Duration::ZERO), Duration::ZERO);
    }

    #[test]
    fn is_not_retriable_parse_error() {
        let err = ProviderError::Parse("unexpected end of JSON".to_string());
        assert!(!is_retriable(&err));
    }

    #[tokio::test]
    async fn retry_succeeds_on_first_try() {
        let cfg = RetryConfig::default();
        let result: Result<i32, ProviderError> =
            retry_async(&cfg, || async { Ok::<i32, ProviderError>(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn retry_fails_fast_on_non_retriable() {
        let cfg = RetryConfig::default();
        let result: Result<String, ProviderError> = retry_async(&cfg, || async {
            Err::<String, ProviderError>(ProviderError::api("Test", 400u16, "Bad Request"))
        })
        .await;
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
        })
        .await;
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
        })
        .await;
        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 3); // initial + 2 retries
    }

    // --- retry_async_with_guard (used by streaming: never re-deliver tokens) ---

    /// Fast config so guard tests run quickly.
    fn fast_config() -> RetryConfig {
        RetryConfig {
            max_retries: 3,
            base_delay_ms: 1,
            max_delay_ms: 5,
            backoff_factor: 2.0,
        }
    }

    /// A stream that fails midway must NOT be retried, because tokens have
    /// already been delivered to the consumer — retrying would duplicate them.
    #[tokio::test]
    async fn guard_does_not_retry_after_tokens_emitted() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

        let calls = AtomicU32::new(0);
        let emitted = AtomicBool::new(false);

        let result: Result<String, ProviderError> = retry_async_with_guard(
            &fast_config(),
            || emitted.load(Ordering::SeqCst),
            || async {
                let n = calls.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    emitted.store(true, Ordering::SeqCst);
                    Err(ProviderError::Network("mid-stream disconnect".to_string()))
                } else {
                    unreachable!("operation must not be re-invoked after tokens emitted")
                }
            },
        )
        .await;

        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// Before any token is emitted, transient failures should still be retried
    /// (the guard is false, so the operation runs up to max_retries + 1 times).
    #[tokio::test]
    async fn guard_retries_when_nothing_emitted() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

        let calls = AtomicU32::new(0);
        let emitted = AtomicBool::new(false);

        let result: Result<String, ProviderError> = retry_async_with_guard(
            &fast_config(),
            || emitted.load(Ordering::SeqCst),
            || async {
                let n = calls.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(ProviderError::Network("transient".to_string()))
                } else {
                    Ok("success".to_string())
                }
            },
        )
        .await;

        assert_eq!(result.unwrap(), "success");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// The guard is also consulted *before* sleeping after a retriable error,
    /// so a mid-stream failure returns immediately instead of paying the
    /// backoff delay. A 1s base delay makes a broken guard (which would sleep
    /// first) fail this test by taking far longer than the 250ms bound.
    #[tokio::test]
    async fn guard_avoids_wasted_backoff_sleep() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        use std::time::Instant;

        let emitted = AtomicBool::new(false);
        let attempts = AtomicU32::new(0);

        let cfg = RetryConfig {
            max_retries: 3,
            base_delay_ms: 1_000,
            max_delay_ms: 1_000,
            backoff_factor: 2.0,
        };

        let start = Instant::now();
        let result: Result<String, ProviderError> = retry_async_with_guard(
            &cfg,
            || emitted.load(Ordering::SeqCst),
            || async {
                attempts.fetch_add(1, Ordering::SeqCst);
                emitted.store(true, Ordering::SeqCst);
                Err(ProviderError::Network("mid-stream disconnect".to_string()))
            },
        )
        .await;
        let elapsed = start.elapsed();

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert!(
            elapsed.as_millis() < 250,
            "expected no backoff sleep, took {elapsed:?}"
        );
    }
}
