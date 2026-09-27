//! Replace one declaration's body, and only if the result still parses.
//!
//! The plain text edit tool asks for a string to find and a string to put in its
//! place, so a model has to reproduce the exact bytes of what it is replacing —
//! including indentation it cannot see reliably — and any edit that lands inside a
//! string literal or a comment rather than the code it meant is accepted without
//! complaint. This asks for a name instead. The declaration is found in the parse
//! tree, so `fn total` means the function called `total` and not the three times
//! that text appears in comments and log messages, and the replacement is confined
//! to that declaration's own body.
//!
//! The second half is the reason this lives beside the parser rather than in the
//! editor: a tree-sitter parse never fails. A file with a brace missing still
//! produces a tree, because error recovery invents a plausible shape for the broken
//! text — so "it parsed" proves nothing, and every result is checked for the nodes
//! recovery created (`ERROR`, and zero-width `MISSING` placeholders) before it is
//! returned. Both the file as it stands and the file as it would be are held to
//! that, so a damaged file cannot be made more damaged quietly.

use crate::parse::{
    error_nodes, find_named_declaration, item_names, line_of_byte, parse, Declaration,
};

/// Why an edit was refused. Every case is a message for the model that asked, so
/// it says what to send instead rather than only what went wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditFailure {
    /// No declaration of that name is in the file.
    NoSuchSymbol {
        symbol: String,
        /// Names the file does declare, to point a wrong guess at a right one.
        nearby: Vec<String>,
    },
    /// The name is declared more than once, so the call cannot say which one.
    Ambiguous { symbol: String, lines: Vec<usize> },
    /// The declaration exists but has no body to replace — a `mod x;` or a
    /// trait method signature.
    NoBody { symbol: String, kind: String },
    /// The proposed body is not a block, so there is nothing to swap in.
    NotABlock { received: String },
    /// A parse problem, in the file as it is or as the edit would leave it.
    Broken {
        /// `current` for the file as it stands, `proposed` for the edited one.
        which: &'static str,
        line: usize,
        detail: String,
    },
}

impl EditFailure {
    /// The instruction-shaped message handed back to the model.
    pub fn to_message(&self) -> String {
        match self {
            EditFailure::NoSuchSymbol { symbol, nearby } => {
                let mut message = format!("no declaration named `{symbol}` in this file");
                if !nearby.is_empty() {
                    let names = nearby.join(", ");
                    message.push_str(&format!(" — it does declare: {names}"));
                }
                message
            }
            EditFailure::Ambiguous { symbol, lines } => format!(
                "`{symbol}` is declared {} times here, on line(s) {} — this tool \
                 edits one declaration by name, so use edit_file with the exact \
                 text of the one you mean",
                lines.len(),
                lines
                    .iter()
                    .map(|line| line.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            EditFailure::NoBody { symbol, kind } => format!(
                "`{symbol}` is a {kind} with no body to replace — this tool only \
                 edits a declaration whose braced block is in this file"
            ),
            EditFailure::NotABlock { received } => format!(
                "the new body has to be the whole replacement block, braces \
                 included, and it is not — it starts with {:?}. Send \
                 `{{ … }}` as the body text.",
                received.chars().take(12).collect::<String>()
            ),
            EditFailure::Broken {
                which,
                line,
                detail,
            } => match *which {
                "current" => format!(
                    "refused: the file as it stands does not parse — {detail} at \
                     line {line}. Nothing was changed; repair that first, or edit \
                     by exact text with edit_file"
                ),
                _ => format!(
                    "refused: that new body does not leave valid Rust — {detail} at \
                     line {line}. Nothing was changed; send a body that parses"
                ),
            },
        }
    }
}

/// Replace the braced body of the declaration named `symbol` with `new_body`,
/// returning the whole new file text. Nothing is written by this function — it
/// takes text and returns text, and the caller decides whether the result is kept.
///
/// `new_body` replaces the body region exactly, so it must be a complete block
/// including its braces; whitespace before it is whatever the declaration already
/// had.
pub fn replace_symbol_body(
    content: &str,
    symbol: &str,
    new_body: &str,
) -> Result<String, EditFailure> {
    if symbol.is_empty() {
        return Err(EditFailure::NoSuchSymbol {
            symbol: symbol.to_string(),
            nearby: Vec::new(),
        });
    }
    let trimmed = new_body.trim();
    if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return Err(EditFailure::NotABlock {
            received: new_body.to_string(),
        });
    }
    let tree = parse(content).ok_or_else(|| EditFailure::Broken {
        which: "current",
        line: 1,
        detail: "the Rust grammar would not load".to_string(),
    })?;
    let root = tree.root_node();
    if let Some(first) = error_nodes(content, root).into_iter().next() {
        return Err(EditFailure::Broken {
            which: "current",
            line: line_of_byte(content, first.0),
            detail: first.1,
        });
    }

    let matches = find_named_declaration(content, symbol);
    let editable: Vec<&Declaration> = matches
        .iter()
        .filter(|found| found.body.is_some())
        .collect();
    match editable.len() {
        0 if matches.is_empty() => Err(EditFailure::NoSuchSymbol {
            symbol: symbol.to_string(),
            nearby: nearby_names(content),
        }),
        0 => {
            let found = &matches[0];
            Err(EditFailure::NoBody {
                symbol: symbol.to_string(),
                kind: item_kind(&found.kind),
            })
        }
        1 => {
            let found = editable[0];
            let (body_start, body_end) = found.body.expect("filtered above");
            let updated = format!(
                "{}{}{}",
                &content[..body_start],
                trimmed,
                &content[body_end..]
            );
            let tree = parse(&updated).ok_or_else(|| EditFailure::Broken {
                which: "proposed",
                line: 1,
                detail: "the Rust grammar would not load".to_string(),
            })?;
            if let Some(first) = error_nodes(&updated, tree.root_node()).into_iter().next() {
                return Err(EditFailure::Broken {
                    which: "proposed",
                    line: line_of_byte(&updated, first.0),
                    detail: first.1,
                });
            }
            // The declaration has to still be the declaration it replaced. Error
            // recovery can absorb a name into whatever follows it and still hand
            // back a tree, so a body that swallows its own item is caught here
            // rather than written out.
            let after = find_named_declaration(&updated, symbol);
            let after_editable: Vec<&Declaration> =
                after.iter().filter(|found| found.body.is_some()).collect();
            if after_editable.len() != 1 || after_editable[0].kind != found.kind {
                return Err(EditFailure::Broken {
                    which: "proposed",
                    line: line_of_byte(&updated, body_start),
                    detail: format!(
                        "`{symbol}` is no longer the one {} it was",
                        item_kind(&found.kind)
                    ),
                });
            }
            Ok(updated)
        }
        _ => Err(EditFailure::Ambiguous {
            symbol: symbol.to_string(),
            lines: editable
                .iter()
                .map(|found| line_of_byte(content, found.decl_start))
                .collect(),
        }),
    }
}

/// The grammar's name for a shape, said the way a person would say it.
fn item_kind(kind: &str) -> String {
    kind.trim_end_matches("_item").replace('_', " ")
}

/// The names a file declares, for a "no declaration named X" message to point at.
/// Capped and sorted, because the point is a hint and not a listing of a large file.
fn nearby_names(content: &str) -> Vec<String> {
    let mut names = item_names(content);
    names.sort();
    names.dedup();
    names.truncate(12);
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str =
        "//! One function.\n\nfn total(readings: &[u32]) -> u32 {\n    readings.iter().sum()\n}\n";

    #[test]
    fn a_declaration_is_replaced_by_its_name_without_reproducing_its_text() {
        let updated =
            replace_symbol_body(SOURCE, "total", "{\n    readings.len() as u32\n}").unwrap();
        assert_eq!(
            updated,
            "//! One function.\n\nfn total(readings: &[u32]) -> u32 {\n    readings.len() as u32\n}\n"
        );
        // The signature and everything outside the declaration survived untouched.
        assert!(updated.contains("fn total(readings: &[u32]) -> u32 {"));
    }

    #[test]
    fn a_name_that_is_not_declared_is_refused_and_says_what_the_file_does_declare() {
        let failure = replace_symbol_body(SOURCE, "count", "{ }").unwrap_err();
        assert!(matches!(failure, EditFailure::NoSuchSymbol { .. }));
        let message = failure.to_message();
        assert!(
            message.contains("no declaration named `count`"),
            "{message}"
        );
        assert!(message.contains("total"), "{message}");
    }

    #[test]
    fn a_symbol_mentioned_in_a_comment_is_not_a_declaration_to_edit() {
        // The whole point of reading the parse tree: the text `fn tally` appears
        // here, and editing it is not possible because it is prose.
        let source = "//! Rewrite `fn tally` later.\n\nfn real() {}\n";
        let failure = replace_symbol_body(source, "tally", "{ }").unwrap_err();
        assert!(matches!(failure, EditFailure::NoSuchSymbol { .. }));
    }

    #[test]
    fn a_body_has_to_be_a_block_including_its_braces() {
        let failure = replace_symbol_body(SOURCE, "total", "    0\n").unwrap_err();
        assert!(matches!(failure, EditFailure::NotABlock { .. }));
        assert!(failure.to_message().contains("braces"));
    }

    #[test]
    fn one_name_declared_twice_in_one_file_is_refused_rather_than_guessed() {
        let source = "impl Thing {\n    fn total(&self) -> u32 { 1 }\n}\n\nimpl Other {\n    fn total(&self) -> u32 { 2 }\n}\n";
        let failure = replace_symbol_body(source, "total", "{ 3 }").unwrap_err();
        let EditFailure::Ambiguous { lines, .. } = failure else {
            panic!("expected an ambiguous name, got {failure:?}");
        };
        // Both are `fn total`, one in each block: the call cannot say which.
        assert_eq!(lines, vec![2, 6]);
    }

    #[test]
    fn a_declaration_with_no_body_in_this_file_is_not_edited() {
        let source = "mod helpers;\n\nfn real() {}\n";
        let failure = replace_symbol_body(source, "helpers", "{ }").unwrap_err();
        assert!(matches!(failure, EditFailure::NoBody { .. }), "{failure:?}");
        assert!(
            failure.to_message().contains("no body to replace"),
            "{}",
            failure.to_message()
        );
    }

    #[test]
    fn a_new_body_that_does_not_parse_is_refused_and_nothing_is_produced() {
        // The done-when for this tool: recovery would happily build a tree out of
        // a statement with no left-hand side, so the check is for the nodes
        // recovery invented, not for whether parsing succeeded.
        let failure = replace_symbol_body(SOURCE, "total", "{\n    let = 4;\n}").unwrap_err();
        let EditFailure::Broken { which, line, .. } = failure else {
            panic!("expected a refused edit, got {failure:?}");
        };
        assert_eq!(which, "proposed");
        // The body region starts on line 3 of SOURCE and the bad statement is the
        // first line inside it, so the fault is on line 4 of the file as it would be.
        assert_eq!(line, 4, "the reported line is {line}");
        assert!(failure.to_message().contains("Nothing was changed"));
    }

    #[test]
    fn a_file_that_does_not_parse_already_is_not_edited_either() {
        let broken = "fn total() -> u32 {\n    let = 4;\n}\n";
        let failure = replace_symbol_body(broken, "total", "{ 0 }").unwrap_err();
        let EditFailure::Broken { which, .. } = failure else {
            panic!("expected a refused edit, got {failure:?}");
        };
        assert_eq!(which, "current");
        assert!(failure.to_message().contains("as it stands"));
    }

    #[test]
    fn an_edit_that_leaves_the_name_declared_twice_is_refused() {
        // Valid Rust, and still refused: the result no longer answers the call the
        // model just made, so repeating it would edit something else. The check is
        // that the one declaration edited is still the one declaration found, after
        // the edit rather than before it.
        let failure =
            replace_symbol_body(SOURCE, "total", "{ fn total() -> u32 { 0 } }").unwrap_err();
        let EditFailure::Broken { which, detail, .. } = failure else {
            panic!("expected a refused edit, got {failure:?}");
        };
        assert_eq!(which, "proposed");
        assert!(detail.contains("no longer the one"), "{detail}");
    }

    #[test]
    fn a_method_inside_an_impl_is_found_wherever_it_sits() {
        let source =
            "struct Counter;\n\nimpl Counter {\n    fn bump(&self) -> u32 {\n        1\n    }\n}\n";
        let updated = replace_symbol_body(source, "bump", "{ 2 }").unwrap();
        assert!(
            updated.contains("    fn bump(&self) -> u32 { 2 }\n}"),
            "{updated}"
        );
    }

    #[test]
    fn the_body_of_a_struct_and_of_an_enum_can_be_replaced_too() {
        let source = "pub struct Point {\n    x: u32,\n}\n";
        let updated =
            replace_symbol_body(source, "Point", "{\n    x: u32,\n    y: u32,\n}").unwrap();
        assert_eq!(updated, "pub struct Point {\n    x: u32,\n    y: u32,\n}\n");
    }

    #[test]
    fn a_file_written_in_another_language_is_refused_as_the_file_it_is() {
        // There is no language check in front of this: the Rust grammar is what is
        // loaded, so another language arrives at the same door a damaged Rust file
        // does. The message it gets is the honest one — the file does not parse —
        // rather than a claim that the name is missing.
        let source = "def total(readings):\n    return sum(readings)\n";
        let failure = replace_symbol_body(source, "total", "{ 0 }").unwrap_err();
        assert!(
            matches!(
                failure,
                EditFailure::Broken {
                    which: "current",
                    ..
                }
            ),
            "{failure:?}"
        );
        assert!(
            failure
                .to_message()
                .contains("the file as it stands does not parse"),
            "{}",
            failure.to_message()
        );
    }
}
