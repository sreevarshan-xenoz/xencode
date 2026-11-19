//! A store implementation with two public functions and two named tests.
//!
//! `impl Store for MemoryStore` is the implementation-to-interface relation:
//! the trait is declared in `traits.rs`, so `build_graph` writes a `via: impl Store`
//! edge from this file to that one.

use crate::traits::Store;

pub struct MemoryStore;

impl Store for MemoryStore {
    fn persist(&self, _key: &str) {}
}

pub fn save(store: &MemoryStore) {
    store.persist("key");
}

pub fn load(_store: &MemoryStore) -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_persists_value() {
        let store = MemoryStore;
        save(&store);
    }

    #[test]
    fn load_missing_returns_none() {
        let store = MemoryStore;
        assert_eq!(load(&store), None);
    }
}
