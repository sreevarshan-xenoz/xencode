use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// A Last-Writer-Wins Register for simple collaborative data structures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LWWRegister<T: Clone> {
    pub value: T,
    pub timestamp: u64,
    pub peer_id: String,
}

impl<T: Clone> LWWRegister<T> {
    pub fn new(value: T, peer_id: &str) -> Self {
        Self {
            value,
            timestamp: current_time(),
            peer_id: peer_id.to_string(),
        }
    }

    /// Set a new value with the current timestamp.
    pub fn set(&mut self, value: T) {
        self.value = value;
        self.timestamp = current_time();
    }

    /// Get the current value.
    pub fn get(&self) -> &T {
        &self.value
    }

    /// LWW merge rule: the value with the higher timestamp wins.
    /// If timestamps are equal, the higher peer_id wins (deterministic tie-break).
    pub fn merge(&mut self, other: &Self) {
        match self.timestamp.cmp(&other.timestamp) {
            Ordering::Less => {
                self.value = other.value.clone();
                self.timestamp = other.timestamp;
                self.peer_id = other.peer_id.clone();
            }
            Ordering::Equal => {
                if other.peer_id > self.peer_id {
                    self.value = other.value.clone();
                    self.peer_id = other.peer_id.clone();
                }
            }
            Ordering::Greater => {} // Keep our value
        }
    }
}

/// A simple Grow-Only Set (G-Set) for tracking membership.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GSet<T: Clone + Ord> {
    elements: Vec<T>,
}

impl<T: Clone + Ord> GSet<T> {
    pub fn new() -> Self {
        Self { elements: Vec::new() }
    }

    pub fn add(&mut self, element: T) {
        if !self.elements.contains(&element) {
            self.elements.push(element);
        }
    }

    pub fn contains(&self, element: &T) -> bool {
        self.elements.contains(element)
    }

    pub fn get_all(&self) -> &[T] {
        &self.elements
    }

    pub fn merge(&mut self, other: &Self) {
        for elem in &other.elements {
            if !self.elements.contains(elem) {
                self.elements.push(elem.clone());
            }
        }
    }
}

fn current_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lww_register_basic() {
        let mut reg = LWWRegister::new("hello".to_string(), "alice");
        assert_eq!(*reg.get(), "hello");

        reg.set("world".to_string());
        assert_eq!(*reg.get(), "world");
    }

    #[test]
    fn test_lww_register_merge() {
        let mut alice = LWWRegister::new("alice_value".to_string(), "alice");
        let bob = LWWRegister::new("bob_value".to_string(), "bob");

        // Give bob's register a higher timestamp for testing
        std::thread::sleep(std::time::Duration::from_millis(2));

        let mut bob_later = LWWRegister::new("bob_later".to_string(), "bob");
        alice.merge(&bob_later);
        assert_eq!(*alice.get(), "bob_later");
    }

    #[test]
    fn test_gset_basic() {
        let mut set = GSet::new();
        set.add("alice");
        set.add("bob");
        set.add("alice"); // Duplicate, should not add

        assert!(set.contains(&"alice"));
        assert!(set.contains(&"bob"));
        assert_eq!(set.get_all().len(), 2);
    }

    #[test]
    fn test_gset_merge() {
        let mut set1 = GSet::new();
        set1.add("alice");
        set1.add("bob");

        let mut set2 = GSet::new();
        set2.add("bob");
        set2.add("carol");

        set1.merge(&set2);
        assert_eq!(set1.get_all().len(), 3);
        assert!(set1.contains(&"carol"));
    }
}
