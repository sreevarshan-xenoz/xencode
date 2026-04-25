use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Health status of a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Error,
    Unavailable,
    Unknown,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Error => write!(f, "error"),
            HealthStatus::Unavailable => write!(f, "unavailable"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Health information for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHealth {
    pub status: HealthStatus,
    pub response_time: f64,
    pub last_check: f64,
    pub error_message: Option<String>,
}

/// Tracks health status across multiple models.
pub struct HealthTracker {
    models: HashMap<String, ModelHealth>,
}

impl HealthTracker {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Update the health record for a model.
    pub fn update(&mut self, model: &str, health: ModelHealth) {
        self.models.insert(model.to_string(), health);
    }

    /// Get the health record for a model.
    pub fn get(&self, model: &str) -> Option<&ModelHealth> {
        self.models.get(model)
    }

    /// Get all tracked models and their health.
    pub fn all(&self) -> &HashMap<String, ModelHealth> {
        &self.models
    }

    /// Get all healthy models.
    pub fn healthy_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|(_, h)| h.status == HealthStatus::Healthy)
            .map(|(name, _)| name.clone())
            .collect()
    }
}

impl Default for HealthTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the current Unix timestamp as f64.
pub fn current_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_update_and_get() {
        let mut tracker = HealthTracker::new();
        let health = ModelHealth {
            status: HealthStatus::Healthy,
            response_time: 0.5,
            last_check: current_timestamp(),
            error_message: None,
        };
        tracker.update("test-model", health);
        let retrieved = tracker.get("test-model").unwrap();
        assert_eq!(retrieved.status, HealthStatus::Healthy);
        assert!((retrieved.response_time - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn healthy_models_filters_correctly() {
        let mut tracker = HealthTracker::new();
        tracker.update(
            "healthy-1",
            ModelHealth {
                status: HealthStatus::Healthy,
                response_time: 0.3,
                last_check: current_timestamp(),
                error_message: None,
            },
        );
        tracker.update(
            "broken",
            ModelHealth {
                status: HealthStatus::Error,
                response_time: 0.0,
                last_check: current_timestamp(),
                error_message: Some("failed".to_string()),
            },
        );
        tracker.update(
            "healthy-2",
            ModelHealth {
                status: HealthStatus::Healthy,
                response_time: 0.4,
                last_check: current_timestamp(),
                error_message: None,
            },
        );

        let healthy = tracker.healthy_models();
        assert_eq!(healthy.len(), 2);
        assert!(healthy.contains(&"healthy-1".to_string()));
        assert!(healthy.contains(&"healthy-2".to_string()));
    }

    #[test]
    fn missing_model_returns_none() {
        let tracker = HealthTracker::new();
        assert!(tracker.get("nonexistent").is_none());
    }

    #[test]
    fn health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Error.to_string(), "error");
        assert_eq!(HealthStatus::Unavailable.to_string(), "unavailable");
        assert_eq!(HealthStatus::Unknown.to_string(), "unknown");
    }
}
