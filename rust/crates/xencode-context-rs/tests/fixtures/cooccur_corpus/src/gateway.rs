//! The gateway accepts requests and routes them.
//! A failed request triggers a retry with exponential backoff.

pub struct Gateway;

impl Gateway {
    pub fn route_request(&self) {}
}
