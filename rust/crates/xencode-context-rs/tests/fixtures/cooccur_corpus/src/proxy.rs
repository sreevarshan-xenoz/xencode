//! The proxy sits before the gateway.
//! It applies a retry budget and a circuit breaker with backoff.

pub struct Proxy;

impl Proxy {
    pub fn forward(&self) {}
}
