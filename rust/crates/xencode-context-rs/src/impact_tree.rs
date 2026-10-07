//! The stable result model behind a blast-radius surface (`QD-2`).
//!
//! [`change_impact`] reports three layers — crates, files, churn — kept apart
//! because each is a different strength of evidence. A panel that draws those
//! layers as one tree needs a shape to walk, and the shape it needs is not the
//! raw report: it is consumers grouped by the crate they live in, with the
//! hop-capped file graph and the churn coupling attached per node. This module
//! is that projection.
//!
//! Why it lives here and not in a TUI file: the same model is meant to be read
//! by the Coding-mode panel and by the Orchestrator-mode fleet card, and any
//! future non-interactive surface. Both consumers must agree on what "3 crates,
//! 17 files, 2 high-churn" means, so the arithmetic happens once, in one place,
//! with no dependency on ratatui, crossterm, or how a row is eventually painted.
//!
//! The projection changes nothing about QD-1's evidence: it reuses the same
//! hop caps, the same "an edge is a resolved name, not a call site" basis, and
//! it never promotes a predicted consumer to a proven one. Where QD-1 says
//! "unknown", this module says "unknown" too.

use super::impact::ChangeImpact;
use std::collections::BTreeMap;

/// The crate at the top of a branch of the fan-out: a workspace member that
/// links to the target's crate, and every consumer file that lives inside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactGroup {
    /// The workspace member's name as `cargo metadata` gives it.
    pub crate_name: String,
    /// Hops along the crate graph from the target's crate to this one. `1` is
    /// a direct dependent; higher values are transitive.
    pub crate_hop: usize,
    /// Present in `direct_crates` — the strongest crate-level evidence QD-1 has.
    pub direct: bool,
    /// The dependency kind of that direct edge (normal, dev, build), when there
    /// is one. A transitive-only crate reports `None` rather than guessing.
    pub kind: Option<String>,
    /// Files in this crate that the file graph says link to the target, closest
    /// first, each carrying its own evidence.
    pub files: Vec<ImpactFile>,
}

/// One consumer file inside a group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactFile {
    /// Path as QD-1's file layer reports it (workspace-relative).
    pub path: String,
    /// Hops along the *file* graph. A file at crate hop 2 may sit at file hop 1
    /// if it names the target directly — the two depths measure different edges
    /// and are shown side by side, never collapsed into one number.
    pub file_hops: usize,
    /// Every `use`/`mod`/`impl` payload that resolved between the target and
    /// this file, so the link can be checked rather than trusted.
    pub via: Vec<String>,
    /// Commits this file appeared in alongside the target. `None` when history
    /// is not readable at all — a co-change count of zero from a repo with no
    /// git is not the same claim as a co-change count of zero from a repo that
    /// has it. `Some(0)` means "we looked, they never co-changed".
    pub churn: Option<u32>,
}

/// What the panel's footer reports: three numbers with a common caveat that
/// none of them is derived from reading a file body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChurnSummary {
    /// Commits the target itself appeared in. Meaningless when `known` is false,
    /// so `known` gates whether the panel prints it at all.
    pub own_commits: u32,
    /// Partner files QD-1 reported on the churn layer.
    pub partner_count: usize,
    /// Commits across every partner. This is the "142" a footer shows — the
    /// size of the coupling, not a score.
    pub total_partner_commits: u32,
    /// Whether git was readable here. `false` means the whole churn layer is
    /// unknown, not empty.
    pub known: bool,
}

/// The whole fan-out for one target, in a shape a renderer walks row by row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactTree {
    /// The file being probed, as QD-1 widened it.
    pub target: String,
    /// The workspace member the target lives in, when it lives in one.
    pub target_crate: Option<String>,
    /// Every crate that holds at least one consumer file, ordered by crate hop
    /// then name. A crate on the reverse closure with no consumer file in the
    /// symbol graph does not appear here — it is a fact about the crate graph,
    /// not about the fan-out.
    pub groups: Vec<ImpactGroup>,
    /// The largest file hop any group contains, so the footer can say "max 3
    /// hops" truthfully rather than echo `IMPACT_MAX_HOPS` whether or not any
    /// consumer reached it.
    pub max_hops: usize,
    /// Distinct crates with consumer files. Not the same as the crate layer's
    /// `reverse_crates.len()` — see [`ImpactTree::groups`].
    pub crate_count: usize,
    /// Consumer files across every group.
    pub file_count: usize,
    pub churn: ChurnSummary,
    /// The confidence statement QD-1 owes the reader, kept verbatim so any
    /// surface prints the same sentence, not one it reworded.
    pub basis: String,
}

impl ImpactTree {
    /// Flatten the tree into the rows a keyboard cursor walks. The target line
    /// comes first, then for each crate group a header row and its files. This
    /// is the *only* ordering a panel may render, so cursor arithmetic and
    /// paint order can never disagree about what row N is.
    pub fn rows(&self) -> Vec<ImpactRow> {
        let mut out = vec![ImpactRow::Target {
            file: self.target.clone(),
            crate_name: self.target_crate.clone(),
        }];
        for group in &self.groups {
            out.push(ImpactRow::Crate {
                crate_name: group.crate_name.clone(),
                crate_hop: group.crate_hop,
                direct: group.direct,
                kind: group.kind.clone(),
                file_count: group.files.len(),
            });
            for f in &group.files {
                out.push(ImpactRow::File {
                    path: f.path.clone(),
                    crate_name: group.crate_name.clone(),
                    file_hops: f.file_hops,
                    via: f.via.clone(),
                    churn: f.churn,
                });
            }
        }
        out
    }
}

/// One selectable line. A renderer draws the variants; nothing here knows how
/// many columns each takes or which glyph marks the cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImpactRow {
    Target {
        file: String,
        crate_name: Option<String>,
    },
    Crate {
        crate_name: String,
        crate_hop: usize,
        direct: bool,
        kind: Option<String>,
        file_count: usize,
    },
    File {
        path: String,
        crate_name: String,
        file_hops: usize,
        via: Vec<String>,
        churn: Option<u32>,
    },
}

/// Project a QD-1 report into the tree a panel renders. Pure function over the
/// already-computed layers — no subprocess, no filesystem, no clock. Every
/// consumer file the file graph names is placed under the crate that holds it
/// (via `ChangeImpact::file_crates`); a file the manifest does not claim falls
/// into a named synthetic group at the end so it is *shown* rather than lost.
pub fn impact_tree(change: &ChangeImpact) -> ImpactTree {
    let direct_by_crate: BTreeMap<&str, &str> = change
        .direct_crates
        .iter()
        .map(|(name, kind)| (name.as_str(), kind.as_str()))
        .collect();
    let hop_by_crate: BTreeMap<&str, usize> = change
        .reverse_crates
        .iter()
        .map(|(name, hop)| (name.as_str(), *hop))
        .collect();

    // Bucket consumer files by the crate they live in. Files with no crate land
    // in the "" bucket, which becomes a single synthetic group below.
    let mut buckets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, f) in change.files.files.iter().enumerate() {
        let key = change
            .file_crates
            .get(&f.file)
            .cloned()
            .unwrap_or_else(|| UNCLAIMED.to_string());
        buckets.entry(key).or_default().push(i);
    }

    let churn_by_file: BTreeMap<&str, u32> = change
        .cochange
        .iter()
        .map(|(p, n)| (p.as_str(), *n))
        .collect();

    // Order groups by crate hop (target's own crate first at hop 0, then
    // dependents by depth), then by name. Unclaimed files sort last, whatever
    // their name, because "no crate claims it" is not information about
    // ordering.
    let own_crate = change.crate_name.clone().unwrap_or_default();
    let mut ordered: Vec<(String, &str)> = buckets
        .keys()
        .map(|name| (name.clone(), name.as_str()))
        .collect();
    ordered.sort_by(|a, b| {
        let key = |name: &str| -> (u8, usize, String) {
            if name == UNCLAIMED {
                (1, usize::MAX, name.to_string())
            } else if name == own_crate {
                (0, 0, name.to_string())
            } else {
                (
                    0,
                    hop_by_crate.get(name).copied().unwrap_or(usize::MAX),
                    name.to_string(),
                )
            }
        };
        key(a.1).cmp(&key(b.1))
    });

    let mut groups = Vec::new();
    let mut max_hops = 0usize;
    let mut file_count = 0usize;
    for (display_name, bucket_key) in ordered {
        let indices = &buckets[bucket_key];
        let files: Vec<ImpactFile> = indices
            .iter()
            .map(|&i| {
                let f = &change.files.files[i];
                if f.hops > max_hops {
                    max_hops = f.hops;
                }
                ImpactFile {
                    path: f.file.clone(),
                    file_hops: f.hops,
                    via: f.via.clone(),
                    churn: if change.history_known {
                        Some(churn_by_file.get(f.file.as_str()).copied().unwrap_or(0))
                    } else {
                        None
                    },
                }
            })
            .collect();
        file_count += files.len();
        groups.push(ImpactGroup {
            crate_name: display_name,
            crate_hop: if bucket_key == UNCLAIMED {
                0
            } else {
                hop_by_crate.get(bucket_key).copied().unwrap_or(0)
            },
            direct: bucket_key != UNCLAIMED && direct_by_crate.contains_key(bucket_key),
            kind: if bucket_key == UNCLAIMED {
                None
            } else {
                direct_by_crate.get(bucket_key).map(|s| s.to_string())
            },
            files,
        });
    }

    let crate_count = groups.iter().filter(|g| g.crate_name != UNCLAIMED).count();

    ImpactTree {
        target: change.target.clone(),
        target_crate: change.crate_name.clone(),
        groups,
        max_hops,
        crate_count,
        file_count,
        churn: ChurnSummary {
            own_commits: change.own_commits,
            partner_count: change.cochange.len(),
            total_partner_commits: change.cochange.iter().map(|(_, n)| *n).sum(),
            known: change.history_known,
        },
        basis: change.files.basis(),
    }
}

/// Synthetic group name for a consumer file that no workspace member claims.
/// Rendering them under a named row keeps them visible and honest: a projection
/// that silently drops a file the manifest does not own is a lie about count.
pub const UNCLAIMED: &str = "(outside any member)";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impact::{ChangeImpact, ImpactReport, ImpactedFile};

    /// A report of one file consumer, at hop `hops`, with `via` naming the link.
    fn consumer(path: &str, hops: usize, via: &[&str]) -> ImpactedFile {
        ImpactedFile {
            file: path.into(),
            via: via.iter().map(|s| s.to_string()).collect(),
            hops,
            uses_symbol: false,
        }
    }

    /// A minimal `ChangeImpact` for projection tests. Callers fill in only what
    /// the row they are pinning depends on; the rest is present so the struct
    /// compiles and reads as a whole answer, not a mock.
    fn base() -> ChangeImpact {
        ChangeImpact {
            target: "crates/alpha/src/lib.rs".into(),
            crate_name: Some("alpha".into()),
            reverse_crates: Vec::new(),
            direct_crates: Vec::new(),
            files: ImpactReport {
                target: "crates/alpha/src/lib.rs".into(),
                symbol: None,
                declared: Vec::new(),
                declared_more: 0,
                files: Vec::new(),
                indexed_files: 10,
                edges: 20,
            },
            cochange: Vec::new(),
            own_commits: 0,
            history_known: true,
            file_crates: Default::default(),
            cli_impact: None,
        }
    }

    #[test]
    fn a_consumer_lands_in_the_group_for_its_crate_with_hops_and_via_intact() {
        let mut c = base();
        c.files.files = vec![
            consumer("crates/beta/src/lib.rs", 1, &["use alpha::thing"]),
            consumer("crates/gamma/src/other.rs", 2, &["use alpha::lib::thing"]),
        ];
        c.file_crates
            .insert("crates/beta/src/lib.rs".into(), "beta".into());
        c.file_crates
            .insert("crates/gamma/src/other.rs".into(), "gamma".into());
        c.reverse_crates = vec![("beta".into(), 1), ("gamma".into(), 1)];
        c.direct_crates = vec![("beta".into(), "normal".into())];

        let tree = impact_tree(&c);
        assert_eq!(tree.crate_count, 2);
        assert_eq!(tree.file_count, 2);
        assert_eq!(tree.max_hops, 2, "the deepest consumer, not the cap");
        assert_eq!(tree.groups[0].crate_name, "beta");
        assert!(tree.groups[0].direct, "beta is on the direct list");
        assert_eq!(tree.groups[0].kind.as_deref(), Some("normal"));
        assert_eq!(
            tree.groups[0].files[0].via,
            vec!["use alpha::thing".to_string()]
        );
        assert_eq!(tree.groups[1].crate_name, "gamma");
        assert!(
            !tree.groups[1].direct,
            "gamma is only on the transitive closure"
        );
        assert_eq!(tree.groups[1].files[0].file_hops, 2);
    }

    #[test]
    fn groups_order_by_crate_hop_then_name_not_by_which_file_arrived_first() {
        let mut c = base();
        c.files.files = vec![
            consumer("crates/zeta/src/lib.rs", 1, &["use alpha::thing"]),
            consumer("crates/alpha/src/inner.rs", 1, &["use crate::thing"]),
            consumer("crates/near/src/lib.rs", 1, &["use alpha::thing"]),
        ];
        c.file_crates
            .insert("crates/zeta/src/lib.rs".into(), "zeta".into());
        c.file_crates
            .insert("crates/alpha/src/inner.rs".into(), "alpha".into());
        c.file_crates
            .insert("crates/near/src/lib.rs".into(), "near".into());
        c.reverse_crates = vec![("zeta".into(), 2), ("near".into(), 1)];

        let tree = impact_tree(&c);
        let names: Vec<&str> = tree.groups.iter().map(|g| g.crate_name.as_str()).collect();
        // alpha is the target's own crate, hop 0 by definition; near is 1; zeta is 2.
        assert_eq!(names, vec!["alpha", "near", "zeta"], "{tree:?}");
    }

    #[test]
    fn a_consumer_no_manifest_claims_is_shown_in_a_named_bucket_never_dropped() {
        let mut c = base();
        c.files.files = vec![consumer("stray.rs", 1, &["use alpha::thing"])];
        // Deliberately no file_crates entry.
        let tree = impact_tree(&c);
        assert_eq!(tree.file_count, 1, "the file is not lost");
        assert_eq!(tree.crate_count, 0, "an unclaimed file is not a crate");
        assert_eq!(tree.groups[0].crate_name, UNCLAIMED);
        assert_eq!(tree.groups[0].files[0].path, "stray.rs");
    }

    #[test]
    fn churn_is_a_real_number_when_history_exists_and_unknown_when_it_does_not() {
        let mut c = base();
        c.files.files = vec![
            consumer("crates/beta/src/lib.rs", 1, &["use alpha::thing"]),
            consumer("crates/beta/src/other.rs", 1, &["use alpha::thing"]),
        ];
        c.file_crates
            .insert("crates/beta/src/lib.rs".into(), "beta".into());
        c.file_crates
            .insert("crates/beta/src/other.rs".into(), "beta".into());
        c.cochange = vec![("crates/beta/src/lib.rs".into(), 5u32)];

        c.history_known = true;
        let tree = impact_tree(&c);
        assert_eq!(
            tree.groups[0].files[0].churn,
            Some(5),
            "a partner the churn layer named has a count"
        );
        assert_eq!(
            tree.groups[0].files[1].churn,
            Some(0),
            "a consumer with history that never co-changed is zero, not unknown"
        );
        assert_eq!(tree.churn.partner_count, 1);
        assert_eq!(tree.churn.total_partner_commits, 5);
        assert!(tree.churn.known);

        c.history_known = false;
        let tree = impact_tree(&c);
        assert_eq!(tree.groups[0].files[0].churn, None, "no git, no claim");
        assert_eq!(tree.groups[0].files[1].churn, None);
        assert!(!tree.churn.known);
    }

    #[test]
    fn rows_flatten_target_then_groups_files_in_the_same_order_the_tree_holds() {
        let mut c = base();
        c.files.files = vec![
            consumer("crates/beta/src/lib.rs", 1, &["use alpha::thing"]),
            consumer("crates/gamma/src/other.rs", 2, &["use alpha::thing"]),
            consumer("crates/beta/src/other.rs", 1, &["use alpha::thing"]),
        ];
        c.file_crates
            .insert("crates/beta/src/lib.rs".into(), "beta".into());
        c.file_crates
            .insert("crates/beta/src/other.rs".into(), "beta".into());
        c.file_crates
            .insert("crates/gamma/src/other.rs".into(), "gamma".into());
        let tree = impact_tree(&c);
        let rows = tree.rows();
        // 1 target + (1 group header + 2 files) + (1 group header + 1 file)
        assert_eq!(rows.len(), 6);
        assert!(matches!(rows[0], ImpactRow::Target { .. }));
        assert!(matches!(&rows[1], ImpactRow::Crate { crate_name, .. } if crate_name == "beta"));
        assert!(
            matches!(&rows[2], ImpactRow::File { path, .. } if path == "crates/beta/src/lib.rs")
        );
        assert!(
            matches!(&rows[3], ImpactRow::File { path, .. } if path == "crates/beta/src/other.rs")
        );
        assert!(matches!(&rows[4], ImpactRow::Crate { crate_name, .. } if crate_name == "gamma"));
        assert!(
            matches!(&rows[5], ImpactRow::File { path, .. } if path == "crates/gamma/src/other.rs")
        );
    }
}
