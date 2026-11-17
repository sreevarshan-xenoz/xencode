use axum::extract::FromRequestParts;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use chrono::{DateTime, Duration, Utc};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration as StdDuration, Instant};

use crate::ws::AppState;

/// Who a valid token belongs to.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Principal {
    pub username: String,
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

/// In-memory bearer tokens. A restart invalidates every session — clients
/// re-login, which is what `POST /auth/login` is for.
///
/// The trust model is local-first: login is an identity claim, not an
/// authentication against a password store. The bind surface is the perimeter
/// (see the CLI's `--host`/`--cert` refusal rule); tokens stop a stranger on
/// the same network from impersonating a peer, not the person who owns the
/// account the server runs under.
/// A store cannot grow past this many live tokens. Login is an identity claim
/// with no password, so without a cap anyone who can reach the server could
/// issue tokens forever and make every later request walk a longer store.
pub const MAX_TOKENS: usize = 4096;
/// At most this many logins in [`LOGIN_WINDOW`] for one claimed username. A
/// shared counter would let a single client exhaust it and lock out everyone.
pub const LOGIN_BURST: usize = 120;
pub const LOGIN_WINDOW: StdDuration = StdDuration::from_secs(60);

#[derive(Default)]
pub struct TokenStore {
    by_token: HashMap<String, Principal>,
    recent_logins: HashMap<String, VecDeque<Instant>>,
    /// Insertion order of tracked names, so the map stays bounded even when
    /// callers invent a new identity on every login.
    login_order: VecDeque<String>,
}

impl TokenStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `username` may log in right now. Old timestamps are dropped
    /// first, so a quiet client gets a fresh burst. The number of tracked
    /// names is bounded, so inventing identities cannot grow this map forever.
    pub fn allow_login(&mut self, username: &str) -> bool {
        let now = Instant::now();
        for times in self.recent_logins.values_mut() {
            while times
                .front()
                .is_some_and(|at| now.duration_since(*at) >= LOGIN_WINDOW)
            {
                times.pop_front();
            }
        }
        let inactive: Vec<String> = self
            .recent_logins
            .iter()
            .filter(|(_, times)| times.is_empty())
            .map(|(name, _)| name.clone())
            .collect();
        for name in inactive {
            self.recent_logins.remove(&name);
            self.login_order.retain(|entry| entry != &name);
        }
        if !self.recent_logins.contains_key(username) {
            while self.recent_logins.len() >= MAX_TOKENS {
                match self.login_order.pop_front() {
                    Some(oldest) => {
                        self.recent_logins.remove(&oldest);
                    }
                    None => break,
                }
            }
            self.login_order.push_back(username.to_string());
            self.recent_logins
                .insert(username.to_string(), VecDeque::new());
        }
        let times = self
            .recent_logins
            .get_mut(username)
            .expect("the name was added above");
        if times.len() >= LOGIN_BURST {
            return false;
        }
        times.push_back(now);
        true
    }

    /// Whether the store is at its hard size limit. Expired tokens are removed
    /// before the count, so a stalled server recovers without a restart.
    pub fn at_capacity(&mut self) -> bool {
        self.prune_expired();
        self.by_token.len() >= MAX_TOKENS
    }

    /// Issue a token for `username` with the default 24 h TTL.
    pub fn issue(&mut self, username: &str) -> Principal {
        self.issue_with_ttl(username, Duration::hours(24))
    }

    /// Issue with an explicit TTL — the test seam for expiry behaviour.
    pub fn issue_with_ttl(&mut self, username: &str, ttl: Duration) -> Principal {
        self.prune_expired();
        let token = format!("xencode_{}", uuid::Uuid::new_v4().simple());
        let principal = Principal {
            username: username.to_string(),
            token: token.clone(),
            expires_at: Utc::now() + ttl,
        };
        self.by_token.insert(token, principal.clone());
        principal
    }

    /// Return the principal for a presented token, or None if unknown or
    /// expired. A lookup is a single hash-map probe; it no longer sweeps the
    /// whole store, which made every authenticated request as slow as the
    /// number of tokens ever issued. There is no constant-time comparison to
    /// lean on here: the result of the map lookup decides the answer directly,
    /// and claiming otherwise would be misleading.
    pub fn lookup(&mut self, token: &str) -> Option<Principal> {
        let well_formed = token.starts_with("xencode_") && token.len() > "xencode_".len();
        let found = if well_formed {
            self.by_token.get(token).cloned()
        } else {
            None
        };
        let principal = found?;
        if principal.expires_at <= Utc::now() {
            self.by_token.remove(token);
            return None;
        }
        Some(principal)
    }

    pub fn revoke(&mut self, token: &str) -> bool {
        self.by_token.remove(token).is_some()
    }

    fn prune_expired(&mut self) {
        let now = Utc::now();
        self.by_token.retain(|_, p| p.expires_at > now);
    }
}

/// An extractor that rejects unauthenticated requests before the handler runs.
///
/// Reads `Authorization: Bearer <token>`. The token value is never logged.
pub struct Authed(pub Principal);

impl FromRequestParts<Arc<AppState>> for Authed {
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<AppState>,
    ) -> Result<Self, Self::Rejection> {
        let header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok());
        let Some(token) = header.and_then(parse_bearer) else {
            return Err(unauthorized());
        };
        match state.tokens.lock().await.lookup(token) {
            Some(principal) => Ok(Authed(principal)),
            None => Err(unauthorized()),
        }
    }
}

/// Extract the credential from an `Authorization` header value per RFC 7235
/// (case-sensitive scheme, optional whitespace). Returns None for anything
/// that is not a well-formed bearer presentation — including `Bearer` alone
/// or `Bearer<no-space>`.
fn parse_bearer(header: &str) -> Option<&str> {
    let rest = header.strip_prefix("Bearer")?;
    let trimmed = rest.trim_start();
    if trimmed.is_empty() || trimmed.len() == rest.len() {
        return None;
    }
    Some(trimmed.trim_end())
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        axum::Json(serde_json::json!({"error": "unauthorized"})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issued_token_looks_up_to_the_same_principal() {
        let mut store = TokenStore::new();
        let issued = store.issue("alice");
        let found = store.lookup(&issued.token).expect("token should resolve");
        assert_eq!(issued, found);
    }

    #[test]
    fn unknown_token_is_rejected() {
        let mut store = TokenStore::new();
        store.issue("alice");
        assert!(store.lookup("xencode_not-a-real-token").is_none());
    }

    /// The regression this guards: the old /auth/verify accepted ANY string
    /// that merely looked like a token. A well-formed guess must now fail.
    #[test]
    fn a_well_formed_but_unissued_token_is_rejected() {
        let mut store = TokenStore::new();
        store.issue("alice");
        let forged = format!("xencode_{}", uuid::Uuid::new_v4().simple());
        assert!(store.lookup(&forged).is_none());
    }

    #[test]
    fn expired_tokens_are_pruned_on_lookup() {
        let mut store = TokenStore::new();
        let issued = store.issue_with_ttl("alice", Duration::seconds(-1));
        assert!(store.lookup(&issued.token).is_none());
        assert!(
            store.by_token.is_empty(),
            "expired entry was left in the store"
        );
    }

    #[test]
    fn lookup_removes_an_expired_token_without_sweeping_the_store() {
        let mut store = TokenStore::new();
        let live = store.issue("alice");
        let stale = store.issue_with_ttl("bob", Duration::seconds(-1));
        assert!(store.lookup(&stale.token).is_none());
        assert!(
            store.by_token.contains_key(&live.token),
            "an unrelated token was dropped by a single lookup"
        );
    }

    #[test]
    fn login_burst_is_bounded() {
        let mut store = TokenStore::new();
        for _ in 0..LOGIN_BURST {
            assert!(store.allow_login("alice"));
        }
        assert!(
            !store.allow_login("alice"),
            "an unlimited burst was accepted"
        );
        assert!(store.allow_login("bob"), "one client locked out another");
        assert!(store.recent_logins.len() <= MAX_TOKENS);
    }

    #[test]
    fn expired_tokens_are_pruned_on_issue() {
        let mut store = TokenStore::new();
        let stale = store.issue_with_ttl("alice", Duration::seconds(-1));
        store.issue("bob");
        assert!(
            !store.by_token.contains_key(&stale.token),
            "issuing did not prune the expired entry"
        );
    }

    #[test]
    fn revoke_makes_a_token_unusable() {
        let mut store = TokenStore::new();
        let issued = store.issue("alice");
        assert!(store.revoke(&issued.token));
        assert!(store.lookup(&issued.token).is_none());
        assert!(!store.revoke(&issued.token));
    }

    #[test]
    fn tokens_are_unguessable_and_distinct() {
        let mut store = TokenStore::new();
        let a = store.issue("alice");
        let b = store.issue("alice");
        assert_ne!(a.token, b.token);
        assert!(a.token.starts_with("xencode_"));
        // xencode_ + 32 hex chars from a v4 UUID.
        assert_eq!(a.token.len(), "xencode_".len() + 32);
    }

    #[test]
    fn lookup_rejects_malformed_presentations_without_touching_the_store() {
        let mut store = TokenStore::new();
        let issued = store.issue("alice");
        // A substring of a real token must not resolve.
        let prefix = &issued.token[..issued.token.len() - 4];
        assert!(store.lookup(prefix).is_none());
        assert!(store.lookup("").is_none());
        assert!(store.lookup("xencode_").is_none());
        assert!(store.lookup(&issued.token).is_some());
    }

    #[test]
    fn bearer_parser_accepts_the_scheme_and_ignores_surrounding_space() {
        assert_eq!(parse_bearer("Bearer abc"), Some("abc"));
        assert_eq!(parse_bearer("Bearer  abc "), Some("abc"));
        assert_eq!(parse_bearer("Basic abc"), None);
        assert_eq!(parse_bearer("Bearer"), None);
        assert_eq!(parse_bearer("Bearer "), None);
        assert_eq!(parse_bearer("Bearerabc"), None);
    }
}
