//! The part of JSON Schema this program needs, and nothing more (MI-1).
//!
//! Two jobs. [`flatten`] resolves local `"$ref"` pointers into the schema that
//! holds them, so what goes out to a server is self-contained — a server is
//! entitled to reject a reference it cannot follow, and the local model server was
//! seen doing exactly that with an unresolvable one. [`check`] answers the question
//! a server cannot answer for us: did the answer we got back actually match the
//! shape we asked for.
//!
//! The supported vocabulary is the one the tool definitions in this workspace are
//! written in — `type`, `enum`, `required`, `properties`, `items`, `minItems`,
//! `maxItems`, `additionalProperties`, `anyOf` — plus `"$ref"` into `"$defs"` or
//! `"definitions"`. A schema that uses anything else is checked as far as it can
//! be and says nothing about the rest, which is the honest behaviour for a
//! pre-flight check: it exists to stop a half-formed tool call from being executed
//! as though it had said nothing, not to be a conformance suite.
//!
//! No dependency on a schema crate. This workspace builds offline against what is
//! vendored, and this is a fraction of the specification rather than all of it.

use serde_json::Value;

/// How deep [`flatten`] follows references before it stops. A recursive schema is
/// legal, so this is a guard rail rather than a limit anyone should reach.
const MAX_REF_DEPTH: usize = 16;

/// Resolve every local `"$ref"` — `#`, `#/$defs/name`, `#/definitions/name` — into
/// the document that holds it, so the result needs nothing from whoever reads it.
///
/// A reference that cannot be resolved is left as it was rather than quietly
/// replaced: the schema still says what it meant, and a server that dislikes it
/// says so out loud, which beats this function inventing a substitute. A reference
/// that comes back around on itself becomes `{}`, which imposes nothing — the
/// alternative is never finishing, and a self-similar branch is rare enough not to
/// be worth constraining here.
pub fn flatten(schema: &Value) -> Value {
    walk(schema, schema, &mut Vec::new())
}

fn walk(node: &Value, root: &Value, following: &mut Vec<String>) -> Value {
    match node {
        Value::Array(items) => items
            .iter()
            .map(|item| walk(item, root, following))
            .collect(),
        Value::Object(map) => {
            if let Some(pointer) = map.get("$ref").and_then(Value::as_str) {
                if following.len() >= MAX_REF_DEPTH || following.iter().any(|seen| seen == pointer)
                {
                    return Value::Object(serde_json::Map::new());
                }
                let Some(target) = resolve(pointer, root) else {
                    return Value::Object(map.clone());
                };
                following.push(pointer.to_string());
                let out = walk(&target, root, following);
                following.pop();
                return out;
            }
            let mut out = serde_json::Map::new();
            for (key, value) in map {
                out.insert(key.clone(), walk(value, root, following));
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// `#/definitions/thing`, read against the document that contains it. Escaped
/// `~0` and `~1` are undone the way a JSON pointer says.
fn resolve(pointer: &str, root: &Value) -> Option<Value> {
    let path = pointer.strip_prefix('#')?;
    let path = path.strip_prefix('/').unwrap_or(path);
    if path.is_empty() {
        return Some(root.clone());
    }
    let mut node = root;
    for segment in path.split('/') {
        let key = segment.replace("~1", "/").replace("~0", "~");
        node = match node {
            Value::Object(map) => map.get(&key)?,
            Value::Array(items) => items.get(key.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(node.clone())
}

/// Whether `value` is what `schema` asked for. The `Err` holds one reason, worded
/// for the model that produced the value to act on, because it goes back to it as
/// the result of its own call.
pub fn check(schema: &Value, value: &Value) -> Result<(), String> {
    let schema = flatten(schema);
    let mut where_we_are: Vec<String> = Vec::new();
    match verify(&schema, value, &mut where_we_are) {
        Ok(()) => Ok(()),
        Err(reason) => Err(if where_we_are.is_empty() {
            format!("that is not what was asked for: {reason}")
        } else {
            format!(
                "that is not what was asked for: {} is {reason}",
                where_we_are.join(".")
            )
        }),
    }
}

/// Read a model's answer as the JSON one of these schemas describes, or say
/// where it does not fit. Nothing is repaired on the way: stripping a code fence
/// or a leading sentence would be guessing at where the answer ends, and the
/// point of a declared schema is that a wrong answer is visible.
pub fn read_answer(text: &str, schema: &Value) -> Result<Value, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("the answer was empty".to_string());
    }
    let value: Value = serde_json::from_str(trimmed)
        .map_err(|error| format!("the answer was not JSON: {error}"))?;
    check(schema, &value)?;
    Ok(value)
}

fn verify(schema: &Value, value: &Value, at: &mut Vec<String>) -> Result<(), String> {
    types(schema, value)?;
    listed(schema, value)?;

    if let Some(cases) = schema.get("anyOf").and_then(Value::as_array) {
        if cases.is_empty() || cases.iter().all(|case| verify(case, value, at).is_err()) {
            let names: Vec<&str> = cases
                .iter()
                .filter_map(|case| case.get("type").and_then(Value::as_str))
                .collect();
            return Err(if names.len() == cases.len() {
                format!("one of: {}", names.join(", "))
            } else {
                format!("one of {} different forms", cases.len())
            });
        }
        return Ok(());
    }

    match value {
        Value::Object(map) => {
            if let Some(required) = schema.get("required").and_then(Value::as_array) {
                for key in required.iter().filter_map(Value::as_str) {
                    if !map.contains_key(key) {
                        at.push(key.to_string());
                        return Err("missing, and it was asked for".to_string());
                    }
                }
            }
            let properties = schema.get("properties").and_then(Value::as_object);
            if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
                if let Some(properties) = properties {
                    if let Some(extra) = map
                        .keys()
                        .find(|key| !properties.contains_key(key.as_str()))
                    {
                        at.push(extra.clone());
                        return Err("not a field that was asked for".to_string());
                    }
                }
            }
            let keys: Vec<String> = map.keys().cloned().collect();
            for key in keys {
                let Some(case) = properties.and_then(|p| p.get(&key)) else {
                    continue;
                };
                let Some(held) = map.get(&key) else { continue };
                at.push(key);
                match verify(case, held, at) {
                    Ok(()) => {
                        at.pop();
                    }
                    // The name stays on the trail, because it is the part the
                    // model needs to hear.
                    Err(reason) => return Err(reason),
                }
            }
        }
        Value::Array(items) => {
            if let Some(min) = schema.get("minItems").and_then(Value::as_u64) {
                if (items.len() as u64) < min {
                    return Err(format!("at least {min} of them, this had {}", items.len()));
                }
            }
            if let Some(max) = schema.get("maxItems").and_then(Value::as_u64) {
                if (items.len() as u64) > max {
                    return Err(format!("at most {max} of them, this had {}", items.len()));
                }
            }
            let Some(case) = schema.get("items") else {
                return Ok(());
            };
            for (index, item) in items.iter().enumerate() {
                at.push(format!("[{index}]"));
                match verify(case, item, at) {
                    Ok(()) => {
                        at.pop();
                    }
                    Err(reason) => return Err(reason),
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// The `type` keyword, which is either one name or a list of them.
fn types(schema: &Value, value: &Value) -> Result<(), String> {
    let names: Vec<&str> = match schema.get("type") {
        Some(Value::String(one)) => vec![one.as_str()],
        Some(Value::Array(list)) => list.iter().filter_map(Value::as_str).collect(),
        _ => return Ok(()),
    };
    if names.iter().any(|name| is_a(name, value)) {
        return Ok(());
    }
    Err(format!(
        "{} (this was {})",
        if names.len() == 1 {
            format!("a {}", names[0])
        } else {
            format!("one of {}", names.join(" or "))
        },
        kind_of(value)
    ))
}

fn is_a(name: &str, value: &Value) -> bool {
    match name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "integer" => value.is_i64() || value.is_u64() || digits_in_text(value),
        "number" => value.is_number() || digits_in_text(value),
        _ => true,
    }
}

/// A weak model quotes a number as readily as writing it, and the readers in this
/// program accept both spellings on purpose, so a check that rejected the quoted
/// one would send back a reply that works.
fn digits_in_text(value: &Value) -> bool {
    match value.as_str().map(str::trim) {
        Some(text) => {
            !text.is_empty()
                && text
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
        }
        None => false,
    }
}

fn listed(schema: &Value, value: &Value) -> Result<(), String> {
    let Some(allowed) = schema.get("enum").and_then(Value::as_array) else {
        return Ok(());
    };
    if allowed.iter().any(|option| option == value) {
        return Ok(());
    }
    if allowed.iter().any(|option| {
        option.is_number()
            && matches!(
                (option, value.as_str().map(str::trim)),
                (Value::Number(wanted), Some(text)) if wanted.to_string() == text
            )
    }) {
        return Ok(());
    }
    let quoted: Vec<String> = allowed
        .iter()
        .map(|option| match option {
            Value::String(text) => format!("\"{text}\""),
            other => other.to_string(),
        })
        .collect();
    Err(format!("one of: {}", quoted.join(", ")))
}

/// What a value is, in the words a schema uses.
pub(crate) fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "nothing at all",
        Value::Bool(_) => "true or false",
        Value::Number(_) => "a number",
        Value::String(_) => "a piece of text",
        Value::Array(_) => "a list",
        Value::Object(_) => "a set of fields",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_local_reference_is_written_out_into_the_schema_that_used_it() {
        let schema = json!({
            "$defs": {"pick": {"type": "string", "enum": ["yes", "no"]}},
            "type": "object",
            "properties": {"answer": {"$ref": "#/$defs/pick"}},
            "required": ["answer"],
        });
        let flat = flatten(&schema);
        assert_eq!(
            flat["properties"]["answer"],
            json!({"type": "string", "enum": ["yes", "no"]}),
            "the reference should be gone, its target in place"
        );
        assert!(flat["properties"]["answer"].get("$ref").is_none());
        // The definition itself is harmless to leave in, and dropping it would
        // change what the caller wrote.
        assert!(schema["$defs"].is_object(), "the input is not touched");
    }

    #[test]
    fn definitions_is_another_name_for_the_same_place() {
        let schema = json!({
            "definitions": {"n": {"type": "integer"}},
            "type": "object",
            "properties": {"count": {"$ref": "#/definitions/n"}},
        });
        assert_eq!(
            flatten(&schema)["properties"]["count"],
            json!({"type": "integer"})
        );
    }

    #[test]
    fn a_reference_that_cannot_be_resolved_is_left_as_written() {
        let schema = json!({"type": "object", "properties": {"a": {"$ref": "#/$defs/gone"}}});
        let flat = flatten(&schema);
        assert_eq!(flat["properties"]["a"], json!({"$ref": "#/$defs/gone"}));
        // And a check against it says nothing, rather than failing everything.
        assert!(check(&schema, &json!({"a": "anything"})).is_ok());
    }

    #[test]
    fn a_schema_that_refers_to_itself_stops_instead_of_nesting_forever() {
        let schema = json!({
            "$defs": {"node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/node"}, "v": {"type": "integer"}},
                "required": ["v"],
            }},
            "type": "object",
            "properties": {"tree": {"$ref": "#/$defs/node"}},
            "required": ["tree"],
        });
        let flat = flatten(&schema);
        assert_eq!(
            flat["properties"]["tree"]["properties"]["child"],
            json!({}),
            "the second visit to the same reference imposes nothing"
        );
        assert!(check(
            &schema,
            &json!({"tree": {"v": 1, "child": {"v": 2, "child": {"v": 3}}}})
        )
        .is_ok());
    }

    #[test]
    fn a_field_that_was_asked_for_and_not_given_is_named_in_the_answer() {
        let schema = json!({
            "type": "object",
            "properties": {"path": {"type": "string"}, "mode": {"type": "string", "enum": ["once", "all"]}},
            "required": ["path", "mode"],
        });
        let error = check(&schema, &json!({"mode": "all"})).unwrap_err();
        assert!(error.contains("path"), "{error}");
        assert!(error.contains("missing"), "{error}");
        let error = check(&schema, &json!({"path": "a.rs", "mode": "always"})).unwrap_err();
        assert!(error.contains("once"), "{error}");
        assert!(error.contains("mode"), "{error}");
        assert!(check(&schema, &json!({"path": "a.rs", "mode": "all"})).is_ok());
    }

    #[test]
    fn a_wrong_type_is_reported_as_the_thing_it_should_have_been() {
        let schema = json!({"type": "object", "properties": {"count": {"type": "integer"}}});
        let error = check(&schema, &json!({"count": "many"})).unwrap_err();
        assert!(error.contains("count"), "{error}");
        assert!(error.contains("integer"), "{error}");
        // The spelling the readers in this program take, they keep taking.
        assert!(check(&schema, &json!({"count": "3"})).is_ok());
        assert!(check(&schema, &json!({"count": 3})).is_ok());
    }

    #[test]
    fn lists_are_checked_item_by_item_and_against_their_length() {
        let schema = json!({
            "type": "object",
            "properties": {
                "options": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
            },
            "required": ["options"],
        });
        assert!(check(&schema, &json!({"options": ["a", "b"]})).is_ok());
        let error = check(&schema, &json!({"options": ["a"]})).unwrap_err();
        assert!(error.contains("at least 2"), "{error}");
        let error = check(&schema, &json!({"options": ["a", 2]})).unwrap_err();
        assert!(error.contains("[1]"), "{error}");
    }

    #[test]
    fn a_field_nobody_asked_for_is_only_a_problem_when_it_says_it_is_not() {
        let open = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        assert!(check(&open, &json!({"a": "x", "b": 1})).is_ok());
        let closed = json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": false,
        });
        let error = check(&closed, &json!({"a": "x", "b": 1})).unwrap_err();
        assert!(error.contains("b"), "{error}");
    }

    #[test]
    fn anything_else_a_schema_might_say_is_not_our_business() {
        // A keyword this module does not implement must not turn a good answer
        // into a rejection.
        let schema = json!({
            "type": "object",
            "properties": {"a": {"type": "string", "pattern": "^x[0-9]+$"}},
            "required": ["a"],
        });
        assert!(check(&schema, &json!({"a": "nope"})).is_ok());
        // Nor does an unknown type name.
        assert!(check(&json!({"type": "duration"}), &json!("3s")).is_ok());
    }

    #[test]
    fn any_of_holds_if_one_of_its_cases_does() {
        let schema = json!({
            "type": "object",
            "properties": {
                "answer": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            },
            "required": ["answer"],
        });
        assert!(check(&schema, &json!({"answer": 2})).is_ok());
        assert!(check(&schema, &json!({"answer": "b"})).is_ok());
        let error = check(&schema, &json!({"answer": ["b"]})).unwrap_err();
        assert!(error.contains("integer"), "{error}");
        assert!(error.contains("string"), "{error}");
        assert!(error.contains("answer"), "{error}");
    }

    #[test]
    fn an_answer_is_read_as_the_json_it_claims_to_be() {
        let schema = json!({
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}}
        });
        // The whitespace a stream leaves behind is not part of the answer.
        assert_eq!(
            read_answer("  {\"answer\": \"yes\"}\n", &schema).unwrap()["answer"],
            "yes"
        );
        // A sentence in front of the JSON stays a sentence: nothing is cut out
        // of it to make the answer fit.
        let prose = read_answer("Sure! {\"answer\": \"yes\"}", &schema).unwrap_err();
        assert!(prose.contains("not JSON"), "{prose}");
        assert_eq!(
            read_answer("", &schema).unwrap_err(),
            "the answer was empty"
        );
        // Right shape of thing, wrong contents: said differently, and named.
        let missing = read_answer(r#"{"confident": true}"#, &schema).unwrap_err();
        assert!(missing.contains("answer"), "{missing}");
    }
}
