//! The line protocol behind `xencode query --format ndjson`.
//!
//! Every line is one object carrying `"v": 1`. The version is written per line
//! rather than once at the top because a script that joins mid-stream (a pipe
//! opened on a partial run, a `tail -f`) must still be able to tell which
//! schema it is reading. A consumer that meets a higher `v` than 1 must stop
//! rather than guess; a consumer that meets an unknown `type` at a `v` it does
//! know must skip the line.
//!
//! The one guarantee worth stating, because a script is built on it: the
//! `text` fields of every `token` line, concatenated in order, equal the
//! `answer` field of the terminating `done` line. That holds for a cached
//! answer too, which is why the cached path emits the answer as a `token` line
//! instead of only in `done`.

use serde_json::Value;

/// Bumped only when a line written by the previous version would be read
/// wrongly, not when a field is added.
pub const VERSION: u8 = 1;

fn event(kind: &str) -> Value {
    serde_json::json!({ "v": VERSION, "type": kind })
}

/// Emitted before any model output, once the model and route are settled.
/// `source` is the same word the metrics rows use: `local` or `cloud`.
pub fn start(model: &str, provider: &str, source: &str, session: Option<&str>) -> Value {
    let mut line = event("start");
    line["model"] = model.into();
    line["provider"] = provider.into();
    line["source"] = source.into();
    line["session"] = match session {
        Some(id) => Value::String(id.to_string()),
        None => Value::Null,
    };
    line
}

/// A piece of the answer, as it arrived. The split is the network's, not
/// the model's: pieces do not align with words.
pub fn token(text: &str) -> Value {
    let mut line = event("token");
    line["text"] = text.into();
    line
}

/// The answer, plus how it was obtained. A route reports token counts only
/// when its response carries them, and a llama.cpp server reports them only
/// when its stream ends with a usage chunk; `tokens_generated` and
/// `tokens_per_second` are null whenever nothing was reported — including
/// on a cached answer, where no generation happened. `elapsed_ms` is always
/// this command's own wall clock.
pub fn done(
    answer: &str,
    cached: bool,
    elapsed_ms: u64,
    tokens_generated: Option<u64>,
    tokens_per_second: Option<f64>,
) -> Value {
    let mut line = event("done");
    line["answer"] = answer.into();
    line["cached"] = cached.into();
    line["elapsed_ms"] = elapsed_ms.into();
    line["tokens_generated"] = match tokens_generated {
        Some(n) => n.into(),
        None => Value::Null,
    };
    line["tokens_per_second"] = match tokens_per_second {
        Some(rate) => serde_json::json!((rate * 100.0).round() / 100.0),
        None => Value::Null,
    };
    line
}

/// The last line of a run that produced no answer. A failure before the
/// model was dialed (a bad `--json-schema`, say) still ends this way, so a
/// script can rely on exactly one terminating line per run.
pub fn error(message: &str) -> Value {
    let mut line = event("error");
    line["message"] = message.into();
    line
}
