//! The Rust symbol tier, read off a parse tree.
//!
//! The inventory this replaces was nine regular expressions matching lines that
//! began with `struct`, `fn`, `use` and so on. That tier could not tell a
//! declaration from a line that merely looks like one. A `pub struct` written on
//! its own line inside a block comment, inside a multi-line string literal, or
//! inside a macro's token tree was reported as a declaration this file owns — and
//! a macro body is not one, because it becomes a declaration wherever the macro is
//! invoked, which the declaring file cannot say. The other direction was lost too:
//! a method written on its trait's own line — `pub trait Speak { fn say(&self) ->
//! String; }` — was no function at all. Parsing the file answers
//! both, because a comment and a token tree are nodes with their own kinds and a
//! declaration has named fields.
//!
//! Only Rust is parsed. The graph this feeds is a Rust module graph — file→file
//! edges from `use`, `mod` and `impl Trait for Type` — and nothing in the
//! workspace reads symbols in another language, so a second grammar would be a
//! build-time C dependency bought for no reader.

use crate::symbols::{cap_doc_text, doc_line, export_names, trait_impl_target, PerFileSymbols};
use tree_sitter::{Node, Parser};

/// Parse `content` and collect its symbol inventory.
///
/// A file that will not parse contributes no symbols rather than a guessed set.
/// That is a real state, not a hypothetical one: the indexer is handed buffers
/// mid-edit, and a half-written file is a file whose declarations are not decided
/// yet.
pub fn extract(content: &str) -> PerFileSymbols {
    let mut collector = Collector::default();
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .is_ok()
    {
        if let Some(tree) = parser.parse(content, None) {
            collector.walk(content, tree.root_node());
        }
    }
    collector.finish()
}

#[derive(Default)]
struct Collector {
    structs: Vec<String>,
    functions: Vec<String>,
    imports: Vec<String>,
    exports: Vec<String>,
    mods: Vec<String>,
    enums: Vec<String>,
    traits: Vec<String>,
    impls: Vec<String>,
    types: Vec<String>,
    tests: Vec<String>,
    docs: Vec<String>,
}

impl Collector {
    /// Visit every node, including the ones inside a function body: the tier this
    /// replaces matched lines wherever they sat, and a nested `fn` or an `impl`
    /// inside a module block is a declaration the file owns.
    fn walk(&mut self, content: &str, node: Node) {
        self.visit(content, node);
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            self.walk(content, child);
        }
    }

    fn visit(&mut self, content: &str, node: Node) {
        match node.kind() {
            "struct_item" => push(&mut self.structs, named_item(content, node)),
            "enum_item" => push(&mut self.enums, named_item(content, node)),
            "trait_item" => push(&mut self.traits, named_item(content, node)),
            "type_item" => push(&mut self.types, named_item(content, node)),
            "function_item" | "function_signature_item" => {
                if is_test_item(content, node) {
                    push(&mut self.tests, named_item(content, node));
                }
                push(&mut self.functions, named_item(content, node));
            }
            "mod_item" => {
                // `mod name;` declares a file. `mod name { … }` opens a namespace
                // inside this one and declares nothing, which is why the block form
                // is not collected — reading it as a file edge attached every
                // `mod tests` to a `tests.rs` that did not exist.
                if node.child_by_field_name("body").is_none() {
                    push(&mut self.mods, named_item(content, node));
                }
            }
            "impl_item" => {
                // Only the trait of a `impl Trait for Type`: an inherent `impl Type`
                // names something this file can already reach, so it is not a
                // dependency. The field is absent when there is no `for`.
                if let Some(trait_node) = node.child_by_field_name("trait") {
                    let path = text(content, trait_node);
                    push(&mut self.impls, trait_impl_target(&path));
                }
            }
            "use_declaration" => {
                let Some(raw) = use_payload(content, node) else {
                    return;
                };
                self.imports.push(raw.clone());
                // Only a bare `pub` re-exports. `pub(crate) use` makes a name
                // visible inside the crate and exports nothing from it.
                if node
                    .children(&mut node.walk())
                    .find(|child| child.kind() == "visibility_modifier")
                    .is_some_and(|v| text(content, v).trim() == "pub")
                {
                    self.exports.extend(export_names(&raw));
                }
            }
            "line_comment" => {
                // The marker test is the one the text tier used, but it now runs on
                // nodes the parser called comments — never on a string literal that
                // happens to begin with three slashes.
                if let Some(doc) = doc_line(&text(content, node)) {
                    self.docs.push(doc.to_string());
                }
            }
            _ => {}
        }
    }

    fn finish(self) -> PerFileSymbols {
        let sorted = |mut names: Vec<String>| {
            names.sort();
            names.dedup();
            names
        };
        PerFileSymbols {
            structs: sorted(self.structs),
            functions: sorted(self.functions),
            imports: sorted(self.imports),
            exports: sorted(self.exports),
            mods: sorted(self.mods),
            enums: sorted(self.enums),
            traits: sorted(self.traits),
            impls: sorted(self.impls),
            types: sorted(self.types),
            tests: sorted(self.tests),
            docs: cap_doc_text(self.docs),
        }
    }
}

fn push(list: &mut Vec<String>, value: Option<String>) {
    if let Some(value) = value {
        list.push(value);
    }
}

/// The declared name of an item node, or `None` for a node the grammar could not
/// finish — an unterminated `struct Foo` mid-edit has no name to claim.
fn named_item(content: &str, node: Node) -> Option<String> {
    node.child_by_field_name("name")
        .map(|child| text(content, child))
}

fn text(content: &str, node: Node) -> String {
    content[node.byte_range()].to_string()
}

/// The path a `use` statement carries, without the `use` keyword and without the
/// closing semicolon, exactly as the line-matching tier captured it — including
/// the interior whitespace of a brace group written over several lines, which the
/// import parser below splits itself.
fn use_payload(content: &str, node: Node) -> Option<String> {
    let argument = node
        .named_children(&mut node.walk())
        .find(|child| child.kind() != "visibility_modifier")?;
    let payload = text(content, argument);
    let raw = payload.trim();
    (!raw.is_empty()).then(|| raw.to_string())
}

/// Whether a function is the one a `#[test]` attribute belongs to. An attribute is
/// a sibling of the item it decorates, so the test is of the run of attributes
/// directly above this function: `#[test]`, or `#[test(…)`, with any further
/// attributes between it and here. A `#[test]` written inside a doc comment sits in
/// no such run of siblings, which is how the previous tier came to call the next
/// function a test.
fn is_test_item(content: &str, node: Node) -> bool {
    let mut sibling = node.prev_named_sibling();
    while let Some(current) = sibling {
        if current.kind() != "attribute_item" {
            return false;
        }
        let attribute = text(content, current);
        if attribute == "#[test]" || attribute.starts_with("#[test(") {
            return true;
        }
        sibling = current.prev_named_sibling();
    }
    false
}

/// Byte ranges the parser says are not declarations of this file: comments, string
/// and character literals, and the token trees of a macro definition or invocation.
///
/// This exists for the comparison test, which needs to tell two kinds of
/// disagreement apart. A name the line matchers picked up out of one of these
/// ranges was never a symbol — it was prose, or a shape a macro will expand
/// somewhere this file cannot say — so the parser refusing it is the fix. A name
/// picked up anywhere else is a declaration the parser is required to still find.
#[cfg(test)]
pub(crate) fn non_code_spans(content: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .is_ok()
    {
        if let Some(tree) = parser.parse(content, None) {
            gather(tree.root_node(), &mut spans);
        }
    }
    spans
}

#[cfg(test)]
fn gather(node: Node, spans: &mut Vec<(usize, usize)>) {
    if matches!(
        node.kind(),
        "line_comment"
            | "block_comment"
            | "string_literal"
            | "raw_string_literal"
            | "char_literal"
            | "macro_definition"
            | "macro_invocation"
    ) {
        spans.push((node.start_byte(), node.end_byte()));
        return;
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        gather(child, spans);
    }
}
