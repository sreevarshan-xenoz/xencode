//! Turning recorded tokens into money — from a price table the user supplies
//! (L-9, folded into CX-1), optionally filled out by a catalogue fetched at
//! runtime (CX-4).
//!
//! Nothing here knows what any model costs. There is no price list in the
//! binary, because a list baked in goes stale silently and would be presented as
//! fact. Prices come from `pricing.json` next to the metrics, which is ordinary
//! data: editing it changes what is reported without recompiling anything, and a
//! model that is not in it has an unknown price and is reported as unknown.
//!
//! Because a price is not stored on the records, the same records answer a
//! different question after the table is edited — which is the point.
//!
//! A model the file does not name can be looked up instead, off a provider's
//! public catalogue, when `price_lookup` is set — see [`PriceLookup`]. That list is
//! cached on disk with the moment it was read and is outranked by anything written
//! by hand, so the two ways a price arrives never disagree without saying which is
//! which. The cache is only read for [`PRICE_TTL_DAYS`] days: past that it is a
//! document about what something used to cost, and a model it names comes back
//! unpriced until `xencode prices fetch` is run again.

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
    /// Prices read off a fetched catalogue, used only for models this file does
    /// not name, and only when `price_lookup` is set. `None` is the ordinary case:
    /// nobody fetched anything.
    pub lookup: Option<PriceLookup>,
    /// True when the listing on disk is older than [`PRICE_TTL_DAYS`]. It stays on
    /// the table so a report can name its age, and no price is read from it.
    pub listing_expired: bool,
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
                    lookup: None,
                    listing_expired: false,
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

    /// The table as a cost report should read it: `pricing.json` first, and — only
    /// for a model the file does not name — the fetched listing, when one is on
    /// disk, `price_lookup` is set, and the copy is inside [`PRICE_TTL_DAYS`]. With
    /// the flag off this is `load`, which is the behaviour every project had before
    /// a listing could be fetched at all.
    pub fn load_with_lookup(xencode_dir: &Path, use_listing: bool) -> Self {
        let mut table = Self::load(xencode_dir);
        if use_listing {
            let lookup = PriceLookup::load(xencode_dir);
            table.listing_expired = lookup
                .as_ref()
                .is_some_and(|lookup| lookup.stale(crate::conversation::now_millis()));
            table.lookup = lookup;
        }
        table
    }

    /// Why the listing on disk is not priced from, in the same words wherever a
    /// report says it. `None` when there is no listing or the one there is fits
    /// inside the age limit.
    pub fn listing_expired_note(&self, now_ms: u64) -> Option<String> {
        let lookup = self.lookup.as_ref()?;
        self.listing_expired.then_some(format!(
            "  • the listing on disk is {} days old, past the {PRICE_TTL_DAYS} days a looked-up \
             rate is taken for — nothing is priced from it, and `xencode prices fetch` reads \
             them again",
            lookup.age_days(now_ms)
        ))
    }

    /// A price for the model as the records name it, and which document it came
    /// from. `Listing` carries the id it was matched under, because a price that
    /// arrived by matching a different name has to be checkable.
    pub fn price_for_model(&self, model: &str) -> Option<PriceSource<'_>> {
        if let Some(price) = self.models.get(model) {
            return Some(PriceSource::File(price));
        }
        let lookup = self.lookup.as_ref()?;
        if self.listing_expired {
            return None;
        }
        lookup
            .listing_keys(model)
            .into_iter()
            .filter_map(|key| lookup.models.get(&key).map(|p| (key, p)))
            .next()
            .map(|(id, price)| PriceSource::Listing { id, price })
    }
}

/// Where one model's price came from.
#[derive(Debug, Clone)]
pub enum PriceSource<'a> {
    /// A rate written in `pricing.json`, which outranks everything else.
    File(&'a ModelPrice),
    /// A rate read off a fetched listing, under the listing's own id.
    Listing { id: String, price: &'a ModelPrice },
}

impl<'a> PriceSource<'a> {
    pub fn price(&self) -> &'a ModelPrice {
        match self {
            PriceSource::File(price) => price,
            PriceSource::Listing { price, .. } => price,
        }
    }
}

/// Where a fetched price list is kept. It sits in the cache directory with the
/// other things derived from somewhere else: losing it costs a re-fetch, and
/// editing it by hand would be editing somebody else's numbers.
pub fn lookup_path(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join("cache").join("price-lookup.json")
}

/// How many days a fetched list is priced from before a report stops reading it.
/// A scraped price goes stale without announcing it, and nothing here is told when
/// the catalogue changes, so past this age the rates on disk are treated as
/// unknown rather than trusted a little: the cost report says which models have no
/// price, and `xencode prices fetch` makes them priced again.
pub const PRICE_TTL_DAYS: i64 = 7;

/// The listing xencode knows how to read today: OpenRouter's public model
/// catalogue, which names a price per model and needs no account to read.
pub const OPENROUTER_SOURCE: &str = "openrouter";

/// Prices read off a provider's public catalogue, kept until they are fetched
/// again. Nothing about this is authoritative: it is a copy of what a third party
/// charged the day it was read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceLookup {
    /// Which catalogue this came from, as [`OPENROUTER_SOURCE`] or a name a later
    /// source uses. Refused at load when empty, because an unattributed price has
    /// no honest place in a report.
    pub source: String,
    pub fetched_at_unix_ms: u64,
    /// Keyed by the catalogue's own model id, which is not always the name the
    /// records go by — see [`PriceLookup::listing_keys`].
    pub models: BTreeMap<String, ModelPrice>,
    /// Entries the catalogue listed in a shape no price could be read out of.
    /// Counted rather than dropped quietly, so a catalogue that changes format
    /// shows up as a number on the report instead of as missing prices.
    #[serde(default)]
    pub unreadable: usize,
}

impl PriceLookup {
    /// Read OpenRouter's `/api/v1/models` body into a list of prices.
    ///
    /// Its rates are US dollars per single token, as a string; ours are per
    /// million, so the only arithmetic here is a scale by a million. Cache reads
    /// come from `input_cache_read`. What the catalogue also publishes and this
    /// does not read: a price for writing to the cache, a higher one for the
    /// long-context tiers, and rates for audio, images and web searches. The
    /// recorded tokens count none of those, so a figure built from them would be
    /// invented rather than looked up.
    pub fn parse_openrouter(body: &str, fetched_at_unix_ms: u64) -> Result<Self, String> {
        let listing: OpenRouterListing = serde_json::from_str(body)
            .map_err(|e| format!("the price listing could not be read: {e}"))?;
        let mut models = BTreeMap::new();
        let mut unreadable = 0usize;
        for entry in listing.data {
            let id = entry.id.trim();
            let input = entry.pricing.get("prompt").and_then(per_mtok);
            let output = entry.pricing.get("completion").and_then(per_mtok);
            match (id, input, output) {
                (id, Some(input), Some(output)) if !id.is_empty() => {
                    let price = ModelPrice {
                        input_usd_per_mtok: input,
                        output_usd_per_mtok: output,
                        cached_input_usd_per_mtok: entry
                            .pricing
                            .get("input_cache_read")
                            .and_then(per_mtok),
                    };
                    if price.is_usable() {
                        models.insert(id.to_string(), price);
                    } else {
                        unreadable += 1;
                    }
                }
                _ => unreadable += 1,
            }
        }
        if models.is_empty() {
            // An empty list would read as "nothing is priced", which is what a
            // changed catalogue format looks like from here. Refuse the write and
            // keep the last real list.
            return Err(format!(
                "the listing carried no usable prices ({unreadable} entries could not be read)"
            ));
        }
        Ok(Self {
            source: OPENROUTER_SOURCE.to_string(),
            fetched_at_unix_ms,
            models,
            unreadable,
        })
    }

    /// The list on disk, if there is one that can be read. Missing is the normal
    /// state — this file only exists because somebody asked for it.
    pub fn load(xencode_dir: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(lookup_path(xencode_dir)).ok()?;
        let lookup: Self = serde_json::from_str(&text).ok()?;
        if lookup.source.is_empty() || lookup.fetched_at_unix_ms == 0 || lookup.models.is_empty() {
            return None;
        }
        Some(lookup)
    }

    pub fn write(&self, xencode_dir: &Path) -> std::io::Result<PathBuf> {
        let path = lookup_path(xencode_dir);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = serde_json::to_vec(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, body)?;
        Ok(path)
    }

    /// Whole days between when this was fetched and `now_ms`. A clock that moved
    /// backwards reports 0 rather than a negative age, which would print as a
    /// subtraction nobody asked for.
    pub fn age_days(&self, now_ms: u64) -> i64 {
        (now_ms.saturating_sub(self.fetched_at_unix_ms) / 86_400_000) as i64
    }

    pub fn stale(&self, now_ms: u64) -> bool {
        self.age_days(now_ms) >= PRICE_TTL_DAYS
    }

    /// The names worth trying against this list for a model the records call
    /// `model`, most specific first.
    ///
    /// A model id is whatever the config went by, so the same model reaches here
    /// as `anthropic/claude-x`, `openrouter:anthropic/claude-x`, or
    /// `openrouter/anthropic/claude-x`. All three mean the catalogue's two-part
    /// id; an id with no slash in it — a local tag like `qwen3-0.6b`, or an Ollama
    /// one like `qwen2.5:7b` — is offered nothing, because guessing that a local
    /// model is some distant one with a similar name is how a wrong price gets
    /// believed.
    pub fn listing_keys(&self, model: &str) -> Vec<String> {
        let model = model.trim();
        let mut keys = Vec::new();
        let mut offer = |candidate: String| {
            if !candidate.is_empty() && !keys.contains(&candidate) {
                keys.push(candidate);
            }
        };
        offer(model.to_string());
        // `provider:id` — the prefix is how the config names a route, not part of
        // the catalogue's name. Only worth stripping when what follows still
        // looks like a catalogue id.
        if let Some((_, rest)) = model.split_once(':') {
            if rest.contains('/') {
                offer(rest.to_string());
            }
        }
        // `a/b/c` — drop the leading route segment and try the last two.
        let parts: Vec<&str> = model.split('/').collect();
        if parts.len() >= 3 {
            offer(format!(
                "{}/{}",
                parts[parts.len() - 2],
                parts[parts.len() - 1]
            ));
        }
        keys.retain(|key| self.models.contains_key(key));
        keys
    }

    /// How many models on this list xencode can actually price with.
    pub fn priced_models(&self) -> usize {
        self.models.len()
    }
}

/// One entry of the catalogue, read loosely: only the two fields that must be
/// right are typed, and the rates are kept as the JSON gave them.
#[derive(Debug, Deserialize)]
struct OpenRouterListing {
    #[serde(default)]
    data: Vec<OpenRouterModel>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModel {
    #[serde(default)]
    id: String,
    #[serde(default)]
    pricing: BTreeMap<String, serde_json::Value>,
}

/// A per-token rate — a number, or the decimal string the catalogue usually
/// sends — turned into dollars per million tokens.
fn per_mtok(value: &serde_json::Value) -> Option<f64> {
    let per_token = match value {
        serde_json::Value::String(text) => text.trim().parse::<f64>().ok()?,
        serde_json::Value::Number(number) => number.as_f64()?,
        _ => return None,
    };
    if !per_token.is_finite() || per_token < 0.0 {
        return None;
    }
    Some(per_token * 1_000_000.0)
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
    /// Set when the rate was read off a fetched catalogue rather than
    /// `pricing.json`, holding the catalogue's own id for it — so a price that
    /// arrived by matching a differently-spelled name can be checked against the
    /// one it matched.
    pub listing_id: Option<String>,
}

/// The cost of everything in a rollup, as reported.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CostReport {
    pub per_model: Vec<ModelCost>,
    /// Sum over the priced models only.
    pub known_micros: u64,
    /// Models with no price, whose tokens are therefore not in `known_micros`.
    pub unpriced: Vec<String>,
    /// Models priced from the fetched listing rather than `pricing.json`. Reported
    /// so a figure built off somebody else's catalogue can be seen to be one.
    pub priced_from_listing: Vec<String>,
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
    let mut priced_from_listing = Vec::new();

    for (model, tokens) in by_model {
        let (micros, unknown_because, rates, cache_at_input, listing_id) = match table
            .price_for_model(model)
        {
            Some(source) => {
                let price = source.price();
                let id = match &source {
                    PriceSource::File(_) => None,
                    PriceSource::Listing { id, .. } => Some(id.clone()),
                };
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
                    id,
                )
            }
            None => (
                None,
                Some(if model.is_empty() {
                    "records that do not name a model".to_string()
                } else if table.lookup.is_some() {
                    format!("no price for {model} in pricing.json or the fetched listing")
                } else {
                    format!("no price for {model} in pricing.json")
                }),
                None,
                false,
                None,
            ),
        };
        match micros {
            Some(value) => known_micros += value,
            None => unpriced.push(model.clone()),
        }
        if listing_id.is_some() {
            priced_from_listing.push(model.clone());
        }
        per_model.push(ModelCost {
            model: model.clone(),
            tokens: tokens.clone(),
            micros,
            unknown_because,
            cache_billed_at_input_price: cache_at_input,
            rates,
            listing_id,
        });
    }

    CostReport {
        per_model,
        known_micros,
        unpriced,
        priced_from_listing,
    }
}

/// One line saying where the looked-up prices came from and when they were read,
/// or `None` when nothing was looked up.
///
/// This exists because a fetched price is the one number in a cost report that
/// can be wrong without anything being broken: the catalogue it came from changed
/// and nobody here was told. So the age rides along with the figure, in the same
/// words wherever a report shows it. A listing older than [`PRICE_TTL_DAYS`]
/// contributes no prices at all, which is what
/// [`PriceTable::listing_expired_note`] is for.
pub fn listing_provenance(
    lookup: Option<&PriceLookup>,
    from_listing: &[String],
    now_ms: u64,
) -> Option<String> {
    let lookup = lookup?;
    if from_listing.is_empty() {
        return None;
    }
    let fetched = crate::rollup::local_day_key(lookup.fetched_at_unix_ms);
    let when = if fetched.is_empty() {
        "at a time this file did not record".to_string()
    } else {
        format!("on {fetched}, {} days ago", lookup.age_days(now_ms))
    };
    Some(format!(
        "  • {} price{} read off the {} catalogue {when}",
        from_listing.len(),
        if from_listing.len() == 1 { "" } else { "s" },
        lookup.source,
    ))
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

    /// A body in the shape OpenRouter's `/api/v1/models` answers with, cut to the
    /// parts this reader looks at. The rates are the real ones from the listing
    /// read on 2026-10-03: dollars per single token, sent as strings.
    const LISTING: &str = r#"{
        "data": [
            {"id":"anthropic/claude-sonnet-4.5","pricing":{
                "prompt":"0.000003","completion":"0.000015",
                "web_search":"0.01","input_cache_read":"0.0000003",
                "input_cache_write":"0.00000375",
                "overrides":[{"min_prompt_tokens":200000,"prompt":"0.000006"}]}},
            {"id":"openai/gpt-5","pricing":{"prompt":"0.00000125","completion":"0.00001"}},
            {"id":"broken/broken","pricing":{"prompt":"not a number","completion":"0.000001"}},
            {"id":"missing/missing","pricing":{"completion":"0.000001"}}
        ]
    }"#;

    #[test]
    fn a_catalogues_rates_are_read_in_its_units_and_written_in_ours() {
        let lookup = PriceLookup::parse_openrouter(LISTING, 1_760_000_000_000).unwrap();
        assert_eq!(lookup.source, OPENROUTER_SOURCE);
        // $0.000003 a token is $3.00 a million of them.
        let claude = lookup.models.get("anthropic/claude-sonnet-4.5").unwrap();
        assert_eq!(claude.input_usd_per_mtok, 3.0);
        assert_eq!(claude.output_usd_per_mtok, 15.0);
        // The cache-read rate is the only one of its cache fields the recorded
        // tokens can be weighed with; the write rate and the long-context tier in
        // `overrides` are left in the listing, unread and unclaimed.
        assert_eq!(claude.cached_input_usd_per_mtok, Some(0.3));
        // A model with no cache rate at all stays `None` rather than borrowing the
        // input price here — that decision belongs to the cost, which reports it.
        assert_eq!(
            lookup
                .models
                .get("openai/gpt-5")
                .unwrap()
                .cached_input_usd_per_mtok,
            None
        );
        // Two entries could not be priced and are counted, not dropped in silence.
        assert_eq!(lookup.unreadable, 2);
        assert_eq!(lookup.priced_models(), 2);
    }

    #[test]
    fn a_listing_with_nothing_readable_in_it_is_refused_rather_than_saved() {
        // The shape a catalogue format change looks like from here. Writing it
        // would turn every price on the report into "unknown" without saying so.
        let problem = PriceLookup::parse_openrouter(r#"{"data":[]}"#, 1).unwrap_err();
        assert!(problem.contains("no usable prices"), "{problem}");
        assert!(PriceLookup::parse_openrouter("not json", 1).is_err());
    }

    #[test]
    fn a_model_is_tried_against_the_listing_under_each_name_the_config_might_give_it() {
        let lookup = PriceLookup::parse_openrouter(LISTING, 1).unwrap();
        // The catalogue's own id, and the two ways this project's config spells
        // the same model with a route in front of it.
        for named in [
            "anthropic/claude-sonnet-4.5",
            "openrouter:anthropic/claude-sonnet-4.5",
            "openrouter/anthropic/claude-sonnet-4.5",
        ] {
            assert_eq!(
                lookup.listing_keys(named),
                vec!["anthropic/claude-sonnet-4.5".to_string()],
                "{named}"
            );
        }
        // A local model is not some distant one with a similar name. Nothing is
        // offered for it, so it stays unpriced and says so.
        assert!(lookup.listing_keys("llamacpp:qwen3-0.6b").is_empty());
        assert!(lookup.listing_keys("qwen2.5:7b").is_empty());
        // An id the listing has never heard of matches nothing either.
        assert!(lookup.listing_keys("mistral/mistral-large-3").is_empty());
    }

    #[test]
    fn a_price_written_by_hand_beats_a_fetched_one_and_the_flag_decides_whether_fetching_counts() {
        let dir = temp_dir();
        write_pricing(
            &dir,
            r#"{"models":{"anthropic/claude-sonnet-4.5":{"input_usd_per_mtok":9.9,"output_usd_per_mtok":9.9}}}"#,
        );
        let fetched = crate::conversation::now_millis() - 2 * 86_400_000;
        PriceLookup::parse_openrouter(LISTING, fetched)
            .unwrap()
            .write(&dir)
            .unwrap();

        // Off by default: with the flag unset the fetched file might as well not
        // exist, and nothing dials anybody.
        let off = PriceTable::load_with_lookup(&dir, false);
        assert!(off.lookup.is_none());
        assert!(off.price_for_model("openai/gpt-5").is_none());

        let on = PriceTable::load_with_lookup(&dir, true);
        assert_eq!(on.lookup.as_ref().unwrap().priced_models(), 2);
        match on.price_for_model("anthropic/claude-sonnet-4.5").unwrap() {
            PriceSource::File(price) => assert_eq!(price.input_usd_per_mtok, 9.9),
            other => panic!("the hand-written rate lost: {other:?}"),
        }
        match on.price_for_model("openrouter:openai/gpt-5").unwrap() {
            PriceSource::Listing { id, price } => {
                assert_eq!(id, "openai/gpt-5");
                assert_eq!(price.input_usd_per_mtok, 1.25);
            }
            other => panic!("the listing was not consulted: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_cost_priced_from_the_listing_says_so_and_an_unpriced_one_names_both_documents() {
        let dir = temp_dir();
        let fetched = crate::conversation::now_millis() - 2 * 86_400_000;
        PriceLookup::parse_openrouter(LISTING, fetched)
            .unwrap()
            .write(&dir)
            .unwrap();
        let table = PriceTable::load_with_lookup(&dir, true);
        let mut by_model = BTreeMap::new();
        // 1000 tokens in, 500 out, at $3 and $15 per million.
        by_model.insert(
            "openrouter:anthropic/claude-sonnet-4.5".to_string(),
            totals(1000, 0, 500),
        );
        by_model.insert("mistral/mistral-large-3".to_string(), totals(1000, 0, 0));
        let report = cost_of(&by_model, &table);
        // 1000 × $3 + 500 × $15 per million = 3000 + 7500 micro-dollars.
        assert_eq!(report.known_micros, 10_500);
        assert_eq!(
            report.priced_from_listing,
            vec!["openrouter:anthropic/claude-sonnet-4.5".to_string()]
        );
        let priced = report
            .per_model
            .iter()
            .find(|c| c.model == "openrouter:anthropic/claude-sonnet-4.5")
            .unwrap();
        assert_eq!(
            priced.listing_id.as_deref(),
            Some("anthropic/claude-sonnet-4.5")
        );
        let unknown = report
            .per_model
            .iter()
            .find(|c| c.model == "mistral/mistral-large-3")
            .unwrap();
        assert_eq!(
            unknown.unknown_because.as_deref(),
            Some("no price for mistral/mistral-large-3 in pricing.json or the fetched listing")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_listing_outlives_its_welcome_and_says_what_to_do_about_it() {
        let lookup = PriceLookup::parse_openrouter(LISTING, 1_760_000_000_000).unwrap();
        let fetched = lookup.fetched_at_unix_ms;
        let day_ms = 86_400_000u64;

        // Nothing was priced from it, so there is nothing to attribute.
        assert!(listing_provenance(Some(&lookup), &[], fetched).is_none());
        assert!(
            listing_provenance(None, &["openai/gpt-5".to_string()], fetched).is_none(),
            "no listing on disk means no line about one"
        );

        let one = vec!["openai/gpt-5".to_string()];
        let fresh = listing_provenance(Some(&lookup), &one, fetched + 2 * day_ms).unwrap();
        assert!(
            fresh.contains("1 price read off the openrouter catalogue"),
            "{fresh}"
        );
        assert!(fresh.contains("2 days ago"), "{fresh}");
        assert_eq!(
            lookup.age_days(fetched - day_ms),
            0,
            "a clock that went backwards"
        );

        // Past the age limit the copy stops being a price source: the models it
        // used to price come back unpriced, and the note names the cure.
        let dir = temp_dir();
        let written = PriceLookup::parse_openrouter(LISTING, fetched - 9 * day_ms)
            .unwrap()
            .write(&dir)
            .unwrap();
        assert!(written.exists());
        let table = PriceTable::load_with_lookup(&dir, true);
        assert!(table.listing_expired);
        assert!(
            table.price_for_model("openai/gpt-5").is_none(),
            "a nine-day-old rate is not a rate"
        );
        let note = table
            .listing_expired_note(crate::conversation::now_millis())
            .expect("the listing was loaded but not read");
        assert!(note.contains("nothing is priced from it"), "{note}");
        assert!(note.contains("xencode prices fetch"), "{note}");
        // A listing inside the limit has nothing to explain.
        let fresh_dir = temp_dir();
        PriceLookup::parse_openrouter(LISTING, crate::conversation::now_millis() - day_ms)
            .unwrap()
            .write(&fresh_dir)
            .unwrap();
        let fresh_table = PriceTable::load_with_lookup(&fresh_dir, true);
        assert!(!fresh_table.listing_expired);
        assert!(fresh_table
            .listing_expired_note(crate::conversation::now_millis())
            .is_none());
        assert!(fresh_table.price_for_model("openai/gpt-5").is_some());
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(fresh_dir);
    }
}
