//! Engine composition discipline and static subsystem selection (AF-3).
//!
//! Subsystems are compiled into the tree and configured without dynamic loading
//! or reflection. Profiles define permission sets over CAP-1 capability vocabulary
//! (`filesystem.read`, `filesystem.write`, `shell.execute`, `network.request`, `external.mcp`).
//!
//! Composition can be inspected and diffed via `xencode --dump-config`.

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

/// Statically known capability names from CAP-1 vocabulary.
pub const KNOWN_CAPABILITIES: &[&str] = &[
    "filesystem.read",
    "filesystem.write",
    "shell.execute",
    "network.request",
    "external.mcp",
];

/// A composition profile over CAP-1 capability grants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompositionProfile {
    pub name: String,
    pub description: String,
    pub capabilities: BTreeMap<String, String>,
}

impl CompositionProfile {
    pub fn for_name(name: &str) -> Option<Self> {
        match name {
            "minimal" => Some(Self {
                name: "minimal".to_string(),
                description: "Read-only inspection without file mutations or network egress".to_string(),
                capabilities: BTreeMap::from([
                    ("filesystem.read".to_string(), "allow".to_string()),
                    ("filesystem.write".to_string(), "deny".to_string()),
                    ("shell.execute".to_string(), "deny".to_string()),
                    ("network.request".to_string(), "deny".to_string()),
                    ("external.mcp".to_string(), "deny".to_string()),
                ]),
            }),
            "coding" => Some(Self {
                name: "coding".to_string(),
                description: "Standard local development with file edits and test execution".to_string(),
                capabilities: BTreeMap::from([
                    ("filesystem.read".to_string(), "allow".to_string()),
                    ("filesystem.write".to_string(), "allow".to_string()),
                    ("shell.execute".to_string(), "allow".to_string()),
                    ("network.request".to_string(), "deny".to_string()),
                    ("external.mcp".to_string(), "deny".to_string()),
                ]),
            }),
            "autonomous" => Some(Self {
                name: "autonomous".to_string(),
                description: "Autonomous local loop with external tool integration".to_string(),
                capabilities: BTreeMap::from([
                    ("filesystem.read".to_string(), "allow".to_string()),
                    ("filesystem.write".to_string(), "allow".to_string()),
                    ("shell.execute".to_string(), "allow".to_string()),
                    ("network.request".to_string(), "deny".to_string()),
                    ("external.mcp".to_string(), "allow".to_string()),
                ]),
            }),
            "research" => Some(Self {
                name: "research".to_string(),
                description: "Repository exploration and network search without local file mutation".to_string(),
                capabilities: BTreeMap::from([
                    ("filesystem.read".to_string(), "allow".to_string()),
                    ("filesystem.write".to_string(), "deny".to_string()),
                    ("shell.execute".to_string(), "deny".to_string()),
                    ("network.request".to_string(), "allow".to_string()),
                    ("external.mcp".to_string(), "allow".to_string()),
                ]),
            }),
            "local-only" => Some(Self {
                name: "local-only".to_string(),
                description: "Local machine confinement with strict network isolation".to_string(),
                capabilities: BTreeMap::from([
                    ("filesystem.read".to_string(), "allow".to_string()),
                    ("filesystem.write".to_string(), "allow".to_string()),
                    ("shell.execute".to_string(), "allow".to_string()),
                    ("network.request".to_string(), "deny".to_string()),
                    ("external.mcp".to_string(), "deny".to_string()),
                ]),
            }),
            _ => None,
        }
    }

    pub fn known_profiles() -> Vec<&'static str> {
        vec!["minimal", "coding", "autonomous", "research", "local-only"]
    }
}

/// Resulting composition of the engine, emitted by `--dump-config`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositionSummary {
    pub config_version: u32,
    pub profile: String,
    pub capabilities: BTreeMap<String, String>,
    pub computer_backend: String,
    pub available_computer_backends: Vec<String>,
    pub worker_adapter: String,
    pub available_worker_adapters: Vec<String>,
    pub known_plugin_permissions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_profiles_yield_valid_capability_maps() {
        for name in CompositionProfile::known_profiles() {
            let profile = CompositionProfile::for_name(name).expect("profile must exist");
            assert_eq!(profile.name, name);
            for cap in KNOWN_CAPABILITIES {
                assert!(
                    profile.capabilities.contains_key(*cap),
                    "profile {name} missing cap {cap}"
                );
            }
        }
    }

    #[test]
    fn unknown_profile_returns_none() {
        assert!(CompositionProfile::for_name("unknown-arbitrary").is_none());
    }
}
