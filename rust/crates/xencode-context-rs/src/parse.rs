//! The small amount of parse-tree plumbing shared by reading a Rust file and
//! editing one.
//!
//! [`crate::tsymbols`] walks the tree to inventory a file and
//! [`crate::editing`] walks it to replace one declaration's body. Both need the
//! same three things — a tree that may not exist, a byte turned into a line a
//! person can read, and a node turned into the text it covers — so they live here
//! rather than being written twice, with the risk that the two copies disagree
//! about what the grammar says.

use tree_sitter::{Node, Tree};

/// Parse `content` as Rust. `None` means the grammar would not load, which is a
/// build or ABI problem rather than a problem with this file.
///
/// A file that does not parse still produces a tree here: tree-sitter recovers.
/// Callers that care about that ask for [`error_nodes`].
pub fn parse(content: &str) -> Option<Tree> {
    let mut parser = tree_sitter::Parser::new();
    if parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .is_err()
    {
        return None;
    }
    parser.parse(content, None)
}

/// The line a byte offset falls on, counted the way an editor counts: one-based,
/// so the number is something a reader can act on.
pub fn line_of_byte(content: &str, byte: usize) -> usize {
    1 + content.as_bytes()[..byte.min(content.len())]
        .iter()
        .filter(|ch| **ch == b'\n')
        .count()
}

/// What recovery invented between `node` and the end of its subtree: a list of
/// `(byte, description)`, empty for a file that is genuinely well-formed.
///
/// An `ERROR` node is text the grammar could not place. A `MISSING` node is the
/// opposite — a piece the grammar needed and there was none of, which it reports
/// with a zero-width node. Both are why "the file parsed" is not the same question
/// as "the file is valid".
pub fn error_nodes(content: &str, node: Node) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    gather_errors(content, node, &mut found);
    found
}

fn gather_errors(content: &str, node: Node, found: &mut Vec<(usize, String)>) {
    if node.kind() == "ERROR" {
        let text = &content[node.byte_range()];
        let excerpt: String = text.trim().chars().take(24).collect();
        found.push((
            node.start_byte(),
            if excerpt.is_empty() {
                "an unrecognised construct".to_string()
            } else {
                format!("an unrecognised construct around {excerpt:?}")
            },
        ));
    } else if node.is_missing() {
        found.push((
            node.start_byte(),
            format!("a required {} is missing", human_kind(node.parent())),
        ));
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        gather_errors(content, child, found);
    }
}

/// The shape a missing node belongs to, phrased for a person: the grammar names a
/// zero-width placeholder after what it wanted, and the parent says where.
fn human_kind(parent: Option<Node>) -> String {
    match parent.map(|node| node.kind()) {
        Some("function_item") | Some("function_signature_item") => "function body".to_string(),
        Some("impl_item") | Some("trait_item") => "braced block".to_string(),
        Some("use_declaration") => "use path".to_string(),
        Some(other) if !other.is_empty() => other.replace('_', " "),
        _ => "piece of syntax".to_string(),
    }
}

/// One declaration in a file that carries a name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    /// The grammar's own name for the shape: `function_item`, `struct_item`.
    pub kind: String,
    /// Where the declaration starts, which is the line a reader would find.
    pub decl_start: usize,
    /// The braced block it owns, if the text holds one at all: a `mod x;` or a
    /// trait's `fn sig(&self);` is declared here but written elsewhere.
    pub body: Option<(usize, usize)>,
}

/// Every declaration named `symbol`, anywhere in the file — including one nested
/// in a module block or an `impl`, since that is still a declaration this text
/// owns and the one an edit to the name would land on.
pub fn find_named_declaration(content: &str, symbol: &str) -> Vec<Declaration> {
    let Some(tree) = parse(content) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    collect_named(content, tree.root_node(), symbol, &mut found);
    found
}

fn collect_named(content: &str, node: Node, symbol: &str, found: &mut Vec<Declaration>) {
    if let Some(declaration) = declaration_named(content, node, symbol) {
        found.push(declaration);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_named(content, child, symbol, found);
    }
}

/// Whether this node is an item declaring `symbol`, and what body comes with it.
fn declaration_named(content: &str, node: Node, symbol: &str) -> Option<Declaration> {
    if !node.kind().ends_with("_item") {
        return None;
    }
    let name = node.child_by_field_name("name")?;
    if &content[name.byte_range()] != symbol {
        return None;
    }
    let body = node
        .child_by_field_name("body")
        .filter(|block| content[block.byte_range()].starts_with('{'))
        .map(|block| (block.start_byte(), block.end_byte()));
    Some(Declaration {
        kind: node.kind().to_string(),
        decl_start: node.start_byte(),
        body,
    })
}

/// The innermost function whose text holds `byte`, named the way a reader would
/// look for it: `Type::method` inside an `impl`, the bare name otherwise. A use
/// inside a closure belongs to the function the closure is written in. `None`
/// when the byte is outside every function — a `use` line, a field's type, a
/// `const`.
pub fn enclosing_function(content: &str, byte: usize) -> Option<String> {
    let tree = parse(content)?;
    let mut node = tree.root_node().descendant_for_byte_range(byte, byte)?;
    loop {
        if node.kind() == "function_item" {
            let name = &content[node.child_by_field_name("name")?.byte_range()];
            let mut up = node.parent();
            while let Some(parent) = up {
                if parent.kind() == "impl_item" {
                    if let Some(ty) = parent.child_by_field_name("type") {
                        return Some(format!("{}::{name}", &content[ty.byte_range()]));
                    }
                }
                if parent.kind() == "function_item" {
                    break;
                }
                up = parent.parent();
            }
            return Some(name.to_string());
        }
        node = node.parent()?;
    }
}

/// The names of every item this file declares, in the order they were found.
pub fn item_names(content: &str) -> Vec<String> {
    let Some(tree) = parse(content) else {
        return Vec::new();
    };
    let mut names = Vec::new();
    collect_names(content, tree.root_node(), &mut names);
    names
}

fn collect_names(content: &str, node: Node, names: &mut Vec<String>) {
    if node.kind().ends_with("_item") {
        if let Some(name) = node.child_by_field_name("name") {
            names.push(content[name.byte_range()].to_string());
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_names(content, child, names);
    }
}
