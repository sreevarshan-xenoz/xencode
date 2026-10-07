//! Typed agent event bus (`AF-2`).
//!
//! Provides in-process broadcast channel publication and subscription for
//! [`AgentEvent`] values, decoupling engine activity from UI reduction.

use tokio::sync::broadcast;
use xencode_agents_rs::protocol::AgentEvent;

/// Broadcast event bus for agent events.
#[derive(Clone, Debug)]
pub struct EventBus {
    sender: broadcast::Sender<AgentEvent>,
}

impl EventBus {
    /// Create a new event bus with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Publish an event to all subscribers.
    pub fn publish(&self, event: AgentEvent) {
        let _ = self.sender.send(event);
    }

    /// Subscribe to the event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.sender.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xencode_agents_rs::protocol::Origin;

    #[test]
    fn publish_and_subscribe_delivers_typed_agent_events() {
        let bus = EventBus::new(16);
        let mut rx = bus.subscribe();

        bus.publish(AgentEvent::PermissionDenied {
            tool: "write_file".to_string(),
            call_id: None,
            reason: Some("write_file src/lib.rs".to_string()),
            origin: Origin::Observed,
        });

        let received = rx.try_recv().expect("event delivered");
        assert_eq!(received.name(), "permission_denied");
        if let AgentEvent::PermissionDenied { tool, reason, origin, .. } = received {
            assert_eq!(tool, "write_file");
            assert_eq!(reason, Some("write_file src/lib.rs".to_string()));
            assert_eq!(origin, Origin::Observed);
        } else {
            panic!("unexpected event kind");
        }
    }
}
