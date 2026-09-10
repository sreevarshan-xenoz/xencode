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

    /// Set a new value on behalf of `peer_id`.
    ///
    /// The writer must be named. `peer_id` is the register's *last writer*, and
    /// `merge` overwrites it with whoever won — so after merging a value from
    /// another peer, the field no longer identifies this replica. A `set` that
    /// left it alone would attribute this write to that other peer, and the
    /// equal-timestamp tie-break would then compare the wrong identity and let
    /// two replicas settle on different values.
    pub fn set(&mut self, value: T, peer_id: &str) {
        self.value = value;
        self.timestamp = self.next_timestamp();
        self.peer_id = peer_id.to_string();
    }

    /// Get the current value.
    pub fn get(&self) -> &T {
        &self.value
    }

    /// A timestamp strictly greater than this register's current one.
    ///
    /// Wall-clock alone is not enough: `merge` may have adopted a timestamp
    /// from a peer whose clock runs ahead, and a local write made *after*
    /// observing that value must still order after it. Taking the max with
    /// `timestamp + 1` keeps writes causally ordered regardless of skew, while
    /// staying a readable millisecond clock in the common case.
    fn next_timestamp(&self) -> u64 {
        current_time().max(self.timestamp + 1)
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
        Self {
            elements: Vec::new(),
        }
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

impl<T: Clone + Ord> Default for GSet<T> {
    fn default() -> Self {
        Self::new()
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

        reg.set("world".to_string(), "alice");
        assert_eq!(*reg.get(), "world");
    }

    #[test]
    fn test_lww_register_merge() {
        let mut alice = LWWRegister::new("alice_value".to_string(), "alice");

        // Give bob's register a higher timestamp for testing
        std::thread::sleep(std::time::Duration::from_millis(2));

        let bob_later = LWWRegister::new("bob_later".to_string(), "bob");
        alice.merge(&bob_later);
        assert_eq!(*alice.get(), "bob_later");
    }

    /// The regression: after merging bob's value, alice's register carries
    /// `peer_id: "bob"`. A `set` that doesn't reclaim it attributes alice's
    /// own write to bob.
    #[test]
    fn set_reclaims_authorship_after_a_merge_from_another_peer() {
        let mut alice = LWWRegister::new("a".to_string(), "alice");
        let mut bob = LWWRegister::new("b".to_string(), "bob");
        bob.timestamp = alice.timestamp + 10;

        alice.merge(&bob);
        assert_eq!(alice.peer_id, "bob", "merge should adopt the winner's id");

        alice.set("a2".to_string(), "alice");
        assert_eq!(
            alice.peer_id, "alice",
            "alice's own write is still attributed to bob"
        );
    }

    /// A local write made after observing a peer's value must order after it,
    /// even when that peer's clock is far ahead of ours.
    #[test]
    fn a_write_after_a_merge_wins_despite_a_peer_clock_running_ahead() {
        let mut alice = LWWRegister::new("a".to_string(), "alice");

        // Bob's clock is an hour fast.
        let mut bob = LWWRegister::new("b".to_string(), "bob");
        bob.timestamp = current_time() + 3_600_000;

        alice.merge(&bob);
        assert_eq!(*alice.get(), "b");

        // Alice now writes, having seen bob's value. Her write is later in
        // real time, so it must win — a bare wall-clock stamp would lose.
        alice.set("a2".to_string(), "alice");
        assert!(
            alice.timestamp > bob.timestamp,
            "local write did not order after the value it observed"
        );

        // And it survives a re-merge of bob's older value.
        alice.merge(&bob);
        assert_eq!(*alice.get(), "a2");
    }

    #[test]
    fn successive_writes_strictly_increase_the_timestamp() {
        let mut reg = LWWRegister::new(0u32, "alice");
        let mut last = reg.timestamp;
        for i in 1..100 {
            reg.set(i, "alice");
            assert!(
                reg.timestamp > last,
                "timestamp did not advance on write {i}"
            );
            last = reg.timestamp;
        }
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
