//! A consumer of the store: a real `use` dependency on `store.rs`.

use crate::store::{load, save};

pub fn sync() {
    let store = crate::store::MemoryStore;
    save(&store);
    let _ = load(&store);
}
