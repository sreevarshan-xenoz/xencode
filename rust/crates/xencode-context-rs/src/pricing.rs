//! Turning recorded tokens into money — from a price table the user supplies
//! (L-9, folded into CX-1).
//!
//! Nothing here knows what any model costs. There is no price list in the
//! binary, because a list baked in goes stale silently and would be presented as
//! fact. Prices come from `pricing.json` next to the metrics, which is ordinary
//! data: editing it changes what is reported without recompiling anything, and a
//! model that is not in it has an unknown price and is reported as unknown.
//!
//! Because a price is not stored on the records, the same records answer a
//! different question after the table is edited — which is the point.

use crate::rollup::TokenTotals;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What one model costs per million tokens, in US dollars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPrice {
    pub input_usd_per_mtok: f64,
    pub output_usd_per_mtok: f64,
    /// Cache-read input tokens, if the provider prices them separately.
    #[serde(default)]
    pub cached_input_usd_per_mtok: Option<f64>,
}

impl ModelPrice {
    fn is_usable(&self) -> bool {
        self.input_usd_per_mtok.is_finite()
            && self.output_usd_per_mtok.is_finite()
            && self.input_usd_per_mtok >= 0.0
            && self.output_usd_per_mtok >= 0.0
            && self
                .cached_input_usd_per_mtok
                .map(|p| p.is_finite() && p >= 0.0)
                .unwrap_or(true)
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct PriceFile {
    #[serde(default)]
    models: BTreeMap<String, ModelPrice>,
}

/// The table as loaded, plus what could not be loaded with it.
#[derive(Debug, Clone, Default)]
pub struct PriceTable {
    pub models: BTreeMap<String, ModelPrice>,
    /// Whether `pricing.json` was there at all. A missing file is not an error;
    /// it just means nothing is priced.
    pub file_present: bool,
    /// Lines that were unreadable, with the reason, so a typo in a price says so
    /// on the report instead of quietly making a model free.
    pub rejected: Vec<String>,
    pub path: PathBuf,
}

pub fn pricing_path(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join("pricing.json")
}

impl PriceTable {
    /// Read the table from disk. A file that does not parse as the expected
    /// shape yields no prices and one rejection line, never a panic.
    pub fn load(xencode_dir: &Path) -> Self {
        let path = pricing_path(xencode_dir);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(_) => {
                return Self {
                    path,
                    ..Default::default()
                }
            }
        };
        match serde_json::from_str::<PriceFile>(&text) {
            Ok(file) => {
                let mut models = BTreeMap::new();
                let mut rejected = Vec::new();
                for (name, price) in file.models {
                    if price.is_usable() {
                        models.insert(name, price);
                    } else {
                        rejected.push(format!(
                            "{name}: prices must be finite numbers of at least 0"
                        ));
                    }
                }
                Self {
                    models,
                    file_present: true,
                    rejected,
                    path,
                }
            }
            Err(e) => Self {
                file_present: true,
                rejected: vec![format!("{} could not be read: {e}", path.display())],
                path,
                ..Default::default()
            },
        }
    }

    pub fn price_for(&self, model: &str) -> Option<&ModelPrice> {
        self.models.get(model)
    }
}

/// What one model's recorded tokens cost, or why that cannot be said.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCost {
    /// The model id as the records name it; empty when a record named none.
    pub model: String,
    pub tokens: TokenTotals,
    /// `None` means unknown — never a zero, which would read as "free".
    pub micros: Option<u64>,
    /// Why the cost is unknown, in words for the report.
    pub unknown_because: Option<String>,
    /// True when cache reads were billed at the full input price because the
    /// table gives no separate rate — an upper bound, not the invoice.
    pub cache_billed_at_input_price: bool,
    /// The rates the figure was built from, for the line that shows the number.
    pub rates: Option<ModelPrice>,
}

/// The cost of everything in a rollup, as reported.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CostReport {
    pub per_model: Vec<ModelCost>,
    /// Sum over the priced models only.
    pub known_micros: u64,
    /// Models with no price, whose tokens are therefore not in `known_micros`.
    pub unpriced: Vec<String>,
}

impl CostReport {
    /// Whether `known_micros` is the whole story.
    pub fn complete(&self) -> bool {
        self.unpriced.is_empty()
    }
}

/// Price every model group in a rollup against the table. Pure arithmetic over
/// what was recorded — it reads nothing.
pub fn cost_of(by_model: &BTreeMap<String, TokenTotals>, table: &PriceTable) -> CostReport {
    let mut per_model = Vec::with_capacity(by_model.len());
    let mut known_micros = 0u64;
    let mut unpriced = Vec::new();

    for (model, tokens) in by_model {
        let (micros, unknown_because, rates, cache_at_input) = match table.price_for(model) {
            Some(price) => {
                // With no separate cache rate, cache reads are billed as input:
                // an upper bound the report marks as such.
                let cached_rate = price
                    .cached_input_usd_per_mtok
                    .unwrap_or(price.input_usd_per_mtok);
                let fresh = tokens.prompt_tokens.saturating_sub(tokens.cached_tokens) as f64
                    * price.input_usd_per_mtok;
                let cached = tokens.cached_tokens.min(tokens.prompt_tokens) as f64 * cached_rate;
                let completion = tokens.completion_tokens as f64 * price.output_usd_per_mtok;
                (
                    Some((fresh + cached + completion).max(0.0).round() as u64),
                    None,
                    Some(price.clone()),
                    price.cached_input_usd_per_mtok.is_none(),
                )
            }
            None => (
                None,
                Some(if model.is_empty() {
                    "records that do not name a model".to_string()
                } else {
                    format!("no price for {model} in pricing.json")
                }),
                None,
                false,
            ),
        };
        match micros {
            Some(value) => known_micros += value,
            None => unpriced.push(model.clone()),
        }
        per_model.push(ModelCost {
            model: model.clone(),
            tokens: tokens.clone(),
            micros,
            unknown_because,
            cache_billed_at_input_price: cache_at_input,
            rates,
        });
    }

    CostReport {
        per_model,
        known_micros,
        unpriced,
    }
}

/// Micro-dollars as `$0.0021`, with enough digits that a small local figure is
/// not rendered as `$0.00` and read as free.
pub fn format_usd(micros: u64) -> String {
    let whole = micros / 1_000_000;
    let frac = micros % 1_000_000;
    if frac == 0 {
        return format!("${whole}");
    }
    let mut digits = format!("{frac:06}");
    while digits.ends_with('0') {
        digits.pop();
    }
    format!("${whole}.{digits}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn temp_dir() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-pricing-test-{unique}"))
    }

    fn totals(prompt: u64, cached: u64, completion: u64) -> TokenTotals {
        TokenTotals {
            requests: 1,
            prompt_tokens: prompt,
            cached_tokens: cached,
            completion_tokens: completion,
        }
    }

    fn write_pricing(dir: &Path, text: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(pricing_path(dir), text).unwrap();
    }

    #[test]
    fn a_missing_table_prices_nothing_and_says_nothing_was_read() {
        let dir = temp_dir();
        let table = PriceTable::load(&dir);
        assert!(!table.file_present);
        assert!(table.models.is_empty());
        assert!(table.rejected.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn prices_are_read_from_the_file_and_applied_to_the_tokens_on_record() {
        let dir = temp_dir();
        // $0.30 / $0.60 per million tokens, cache reads at $0.06.
        write_pricing(
            &dir,
            r#"{"models":{"qwen2.5:7b":{"input_usd_per_mtok":0.3,"output_usd_per_mtok":0.6,"cached_input_usd_per_mtok":0.06}}}"#,
        );
        let table = PriceTable::load(&dir);
        let mut by_model = BTreeMap::new();
        // 1000 prompted of which 400 cached, 500 generated:
        // 600 × 0.3 + 400 × 0.06 + 500 × 0.6 = 180 + 24 + 300 = 504 per million.
        by_model.insert("qwen2.5:7b".to_string(), totals(1000, 400, 500));
        let report = cost_of(&by_model, &table);
        assert_eq!(report.known_micros, 504);
        assert!(report.complete());
        assert_eq!(report.per_model[0].micros, Some(504));
        assert!(!report.per_model[0].cache_billed_at_input_price);
        assert_eq!(format_usd(504), "$0.000504");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_unpriced_model_is_unknown_cost_not_zero_cost() {
        let dir = temp_dir();
        write_pricing(
            &dir,
            r#"{"models":{"other":{"input_usd_per_mtok":1.0,"output_usd_per_mtok":1.0}}}"#,
        );
        let table = PriceTable::load(&dir);
        let mut by_model = BTreeMap::new();
        by_model.insert("llama3.1:8b".to_string(), totals(1000, 0, 100));
        by_model.insert("other".to_string(), totals(1000, 0, 100));
        let report = cost_of(&by_model, &table);
        assert_eq!(report.unpriced, vec!["llama3.1:8b".to_string()]);
        assert!(!report.complete());
        // The priced model's 1100 tokens at $1/M: 1100 micro-dollars — the
        // unpriced model contributes nothing rather than a made-up figure.
        assert_eq!(report.known_micros, 1100);
        let unknown = report
            .per_model
            .iter()
            .find(|c| c.model == "llama3.1:8b")
            .unwrap();
        assert_eq!(unknown.micros, None);
        assert_eq!(
            unknown.unknown_because.as_deref(),
            Some("no price for llama3.1:8b in pricing.json")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn editing_the_table_changes_the_answer_without_anything_else_changing() {
        let dir = temp_dir();
        let mut by_model = BTreeMap::new();
        by_model.insert("qwen2.5:7b".to_string(), totals(1_000_000, 0, 0));

        write_pricing(
            &dir,
            r#"{"models":{"qwen2.5:7b":{"input_usd_per_mtok":0.1,"output_usd_per_mtok":0.1}}}"#,
        );
        let before = cost_of(&by_model, &PriceTable::load(&dir));
        write_pricing(
            &dir,
            r#"{"models":{"qwen2.5:7b":{"input_usd_per_mtok":0.2,"output_usd_per_mtok":0.1}}}"#,
        );
        let after = cost_of(&by_model, &PriceTable::load(&dir));
        assert_eq!(before.known_micros, 100_000);
        assert_eq!(after.known_micros, 200_000);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_model_without_a_cache_rate_is_billed_at_the_input_price_and_says_so() {
        let dir = temp_dir();
        write_pricing(
            &dir,
            r#"{"models":{"qwen2.5:7b":{"input_usd_per_mtok":1.0,"output_usd_per_mtok":1.0}}}"#,
        );
        let table = PriceTable::load(&dir);
        let mut by_model = BTreeMap::new();
        by_model.insert("qwen2.5:7b".to_string(), totals(1000, 900, 0));
        let report = cost_of(&by_model, &table);
        assert_eq!(report.known_micros, 1000);
        assert!(report.per_model[0].cache_billed_at_input_price);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_broken_or_nonsense_price_is_reported_rather_than_ignored() {
        let dir = temp_dir();
        write_pricing(&dir, r#"{"models": {"qwen2.5:7b""#);
        let table = PriceTable::load(&dir);
        assert!(table.models.is_empty());
        assert_eq!(table.rejected.len(), 1);

        write_pricing(
            &dir,
            r#"{"models":{"bad":{"input_usd_per_mtok":-1,"output_usd_per_mtok":0.5},"good":{"input_usd_per_mtok":0.5,"output_usd_per_mtok":0.5}}}"#,
        );
        let table = PriceTable::load(&dir);
        assert!(table.models.contains_key("good"));
        assert!(!table.models.contains_key("bad"));
        assert_eq!(table.rejected.len(), 1, "{:?}", table.rejected);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn records_that_name_no_model_are_grouped_as_unknown() {
        let dir = temp_dir();
        let table = PriceTable::load(&dir);
        let mut by_model = BTreeMap::new();
        by_model.insert(String::new(), totals(1000, 0, 0));
        let report = cost_of(&by_model, &table);
        assert_eq!(
            report.per_model[0].unknown_because.as_deref(),
            Some("records that do not name a model")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn money_is_printed_with_enough_digits_to_tell_zero_from_free() {
        assert_eq!(format_usd(0), "$0");
        assert_eq!(format_usd(1), "$0.000001");
        assert_eq!(format_usd(1_500_000), "$1.5");
        assert_eq!(format_usd(2_140_000), "$2.14");
    }
}
