//! TypeScript's lexical symbol tier.
//!
//! This is deliberately a lexical tier, not a parse tree: no TypeScript grammar
//! is linked into this build and no TypeScript compiler is invoked. One pass
//! marks every byte that belongs to a comment, a string or a template literal,
//! keeping newlines so line numbers survive, and lifts JSDoc text out while it
//! does so. A second pass reads declarations from the remaining code. Every list
//! is sorted and deduped, so the same text always produces the same inventory.
//!
//! A tier that reads text rather than a grammar is weaker than Rust's parse
//! tree, and the difference is part of the contract: there is no type
//! resolution, no file-to-file dependency graph from TypeScript imports, and no
//! name-based symbol edit. Compiler diagnostics need a real language server,
//! which this build does not ship. What the tier does promise is that a keyword
//! written in a comment, a quoted string or a template literal is not reported
//! as a declaration.

use crate::symbols::{cap_doc_text, PerFileSymbols};
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// A comment/string/template-aware view of a file. Non-code bytes become spaces,
/// newlines are preserved, and JSDoc is collected on the way past.
struct Masked {
    code: Vec<u8>,
    docs: Vec<String>,
}

#[derive(Clone, Copy)]
enum Ctx {
    Code,
    /// Inside a template literal. `Some(depth)` means we are within a `${…}`
    /// expression, counting braces until it closes; `None` is its body.
    Template {
        expr_depth: Option<usize>,
    },
    Quote(u8),
    LineComment,
    BlockComment {
        start: usize,
    },
    Regex {
        class: bool,
    },
}

/// Collect a TypeScript file's symbol inventory from its text.
pub fn extract_typescript_symbols(content: &str) -> PerFileSymbols {
    let masked = mask(content);
    let mut collector = Collector {
        docs: masked.docs,
        ..Collector::default()
    };
    let code = std::str::from_utf8(&masked.code).expect("masking preserves UTF-8");
    collector.scan(content, code);
    collector.finish()
}

#[derive(Default)]
struct Collector {
    structs: Vec<String>,
    functions: Vec<String>,
    imports: Vec<String>,
    exports: Vec<String>,
    enums: Vec<String>,
    traits: Vec<String>,
    types: Vec<String>,
    docs: Vec<String>,
}

impl Collector {
    /// `content` is the original text (module specifiers live inside strings,
    /// which are masked out of `code`); `code` is the masked text declarations
    /// are read from.
    fn scan(&mut self, content: &str, code: &str) {
        let p = patterns();
        let lines: Vec<&str> = code.lines().collect();
        let mut depth: usize = 0;
        // Brace depths that open a class or interface body. A method is a
        // declaration only at one of those depths; the same shape elsewhere is
        // a call or a control statement, which this tier does not claim.
        let mut class_depths: HashSet<usize> = HashSet::new();
        let mut pending_class = false;

        for (index, line) in lines.iter().enumerate() {
            let line_depth = depth;
            let mut declared: Option<String> = None;

            if let Some(c) = p.class.captures(line) {
                let name = c[1].to_string();
                self.structs.push(name.clone());
                pending_class = true;
                declared = Some(name);
            } else if let Some(c) = p.interface.captures(line) {
                let name = c[1].to_string();
                self.traits.push(name.clone());
                pending_class = true;
                declared = Some(name);
            } else if let Some(c) = p.enum_.captures(line) {
                let name = c[1].to_string();
                self.enums.push(name.clone());
                declared = Some(name);
            } else if let Some(c) = p.type_alias.captures(line) {
                let name = c[1].to_string();
                self.types.push(name.clone());
                declared = Some(name);
            }

            if let Some(c) = p.function.captures(line) {
                let name = c[1].to_string();
                self.functions.push(name.clone());
                declared = Some(name);
            } else if let Some(c) = p.binding.captures(line) {
                // `const enum Foo` is an enum, not a variable called `enum`.
                if &c[1] != "enum" {
                    let name = c[1].to_string();
                    if is_bound_function(
                        line,
                        c.get(1).map_or(0, |m| m.end()),
                        lines.get(index + 1).copied(),
                    ) {
                        self.functions.push(name.clone());
                    }
                    declared = Some(name);
                }
            }

            if class_depths.contains(&line_depth) {
                if let Some(c) = p.method.captures(line) {
                    // Group 1 is an optional `get`/`set`; group 2 is the name.
                    let name = c[2].to_string();
                    if !CONTROL_WORDS.contains(&name.as_str()) {
                        self.functions.push(name);
                    }
                }
            }

            if let Some(name) = declared {
                if line.trim_start().starts_with("export ") {
                    self.exports.push(name);
                }
            } else if line.trim_start().starts_with("export default") {
                // A default expression has no declared name of its own.
                self.exports.push("default".to_string());
            }

            if let Some(c) = p.export_list.captures(line) {
                for item in c[1].split(',') {
                    let item = item.trim();
                    let name = item
                        .rsplit_once(" as ")
                        .map_or(item, |(_, alias)| alias.trim());
                    if is_ident(name) {
                        self.exports.push(name.to_string());
                    }
                }
            }

            for byte in line.bytes() {
                match byte {
                    b'{' => {
                        depth += 1;
                        if pending_class {
                            class_depths.insert(depth);
                            pending_class = false;
                        }
                    }
                    b'}' => {
                        depth = depth.saturating_sub(1);
                        class_depths.retain(|d| *d <= depth);
                    }
                    _ => {}
                }
            }
            if pending_class && line.trim_end().ends_with(';') {
                pending_class = false;
            }
        }

        // Imports are read from the original text because the module specifier
        // is a string; the `import` keyword itself must still be code, or a
        // comment or template literal containing one would be reported.
        let masked = code.as_bytes();
        for re in [&p.import_from, &p.import_side] {
            for c in re.captures_iter(content) {
                let Some(kw) = c.name("kw") else { continue };
                if import_is_code(content, masked, kw.start()) {
                    self.imports.push(c[3].to_string());
                }
            }
        }
    }

    fn finish_lists(&mut self) {
        for list in [
            &mut self.structs,
            &mut self.functions,
            &mut self.imports,
            &mut self.exports,
            &mut self.enums,
            &mut self.traits,
            &mut self.types,
        ] {
            list.sort();
            list.dedup();
        }
    }

    fn finish(mut self) -> PerFileSymbols {
        self.finish_lists();
        PerFileSymbols {
            structs: self.structs,
            functions: self.functions,
            imports: self.imports,
            exports: self.exports,
            mods: Vec::new(),
            enums: self.enums,
            traits: self.traits,
            impls: Vec::new(),
            types: self.types,
            tests: Vec::new(),
            docs: cap_doc_text(self.docs),
        }
    }
}

/// Control keywords cannot be method names; everything else at class-body depth
/// followed by `(` is taken as a method at this lexical tier.
const CONTROL_WORDS: &[&str] = &[
    "if", "for", "while", "switch", "catch", "return", "throw", "else", "do", "new",
];

struct Patterns {
    class: Regex,
    interface: Regex,
    enum_: Regex,
    type_alias: Regex,
    function: Regex,
    binding: Regex,
    method: Regex,
    export_list: Regex,
    import_from: Regex,
    import_side: Regex,
}

fn patterns() -> &'static Patterns {
    static P: OnceLock<Patterns> = OnceLock::new();
    P.get_or_init(|| Patterns {
        class: Regex::new(
            r"^[ \t]*(?:(?:export|default|declare|abstract)\s+)*class\s+([A-Za-z_$][A-Za-z0-9_$]*)",
        )
        .unwrap(),
        interface: Regex::new(
            r"^[ \t]*(?:(?:export|default|declare)\s+)*interface\s+([A-Za-z_$][A-Za-z0-9_$]*)",
        )
        .unwrap(),
        enum_: Regex::new(
            r"^[ \t]*(?:(?:export|declare|const)\s+)*enum\s+([A-Za-z_$][A-Za-z0-9_$]*)",
        )
        .unwrap(),
        type_alias: Regex::new(
            r"^[ \t]*(?:(?:export|declare)\s+)*type\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:<[^>\n]*>)?\s*=",
        )
        .unwrap(),
        function: Regex::new(
            r"^[ \t]*(?:(?:export|default|declare|async)\s+)*function\s*\*?\s*([A-Za-z_$][A-Za-z0-9_$]*)",
        )
        .unwrap(),
        binding: Regex::new(
            r"^[ \t]*(?:(?:export|declare)\s+)*(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
        )
        .unwrap(),
        method: Regex::new(
            r"^[ \t]*(?:(?:public|private|protected|static|abstract|override|readonly|async|declare)\s+)*(?:(get|set)\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\s*[?!]?\s*(?:<[^>\n]*>)?\s*\(",
        )
        .unwrap(),
        export_list: Regex::new(r"^[ \t]*export\s+(?:type\s+)?\{([^}]*)\}").unwrap(),
        import_from: Regex::new(
            r#"(?m)^(?P<space>[ \t]*)(?P<kw>import)\b[\s\S]*?\bfrom\s*['"]([^'"]+)['"]"#,
        )
        .unwrap(),
        import_side: Regex::new(r#"(?m)^(?P<space>[ \t]*)(?P<kw>import)\s+['"]([^'"]+)['"]"#)
            .unwrap(),
    })
}

fn import_is_code(content: &str, masked: &[u8], keyword_at: usize) -> bool {
    let line_start = content[..keyword_at]
        .rfind('\n')
        .map_or(0, |newline| newline + 1);
    masked[line_start..keyword_at]
        .iter()
        .all(|b| *b == b' ' || *b == b'\t')
        && masked[keyword_at..].starts_with(b"import")
}

/// Whether the value bound to a `const`/`let`/`var` is a function: an arrow or a
/// function expression, on this line or on the next one when the `=` ends the
/// line. Assignment is found by skipping `=>` and comparison operators.
fn is_bound_function(line: &str, from: usize, next: Option<&str>) -> bool {
    let tail = &line[from..];
    let Some(eq) = assignment_index(tail) else {
        return false;
    };
    let rhs = &tail[eq + 1..];
    let trimmed = rhs.trim_start();
    if rhs.contains("=>")
        || trimmed.starts_with("function")
        || trimmed.starts_with("async function")
    {
        return true;
    }
    next.is_some_and(|line| line.trim_start().starts_with('(') && line.contains("=>"))
}

fn assignment_index(tail: &str) -> Option<usize> {
    let bytes = tail.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'=' => {
                if bytes.get(i + 1) == Some(&b'>') {
                    i += 2; // a `=>` in a type annotation, not the assignment
                } else if i > 0 && b"=!<>+-*/%&|^".contains(&bytes[i - 1]) {
                    i += 1; // a comparison or compound assignment
                } else if bytes.get(i + 1) == Some(&b'=') {
                    i += 2;
                } else {
                    return Some(i);
                }
            }
            b';' => return None,
            _ => i += 1,
        }
    }
    None
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' || c == '$' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

/// Replace non-code bytes with spaces, keeping newlines and collecting JSDoc.
fn mask(content: &str) -> Masked {
    let bytes = content.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut docs = Vec::new();
    let mut stack = vec![Ctx::Code];
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        let ctx = *stack.last().expect("context stack is never empty");
        match ctx {
            Ctx::BlockComment { start } => {
                out.push(b' ');
                if b == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    out.push(b' ');
                    docs.extend(jsdoc(&content[start..i + 2]));
                    stack.pop();
                    i += 2;
                } else {
                    if b == b'\n' {
                        out[i] = b'\n';
                    }
                    i += 1;
                }
            }
            Ctx::LineComment => {
                if b == b'\n' {
                    out.push(b'\n');
                    stack.pop();
                } else {
                    out.push(b' ');
                }
                i += 1;
            }
            Ctx::Quote(quote) => {
                out.push(if b == b'\n' { b'\n' } else { b' ' });
                if b == b'\n' {
                    stack.pop();
                    i += 1;
                } else if b == b'\\' && i + 1 < bytes.len() {
                    out.push(b' ');
                    i += 2;
                } else {
                    if b == quote {
                        stack.pop();
                    }
                    i += 1;
                }
            }
            Ctx::Regex { class } => {
                out.push(if b == b'\n' { b'\n' } else { b' ' });
                if b == b'\n' {
                    stack.pop();
                    i += 1;
                } else if b == b'\\' && i + 1 < bytes.len() {
                    out.push(b' ');
                    i += 2;
                } else {
                    match b {
                        b'[' => {
                            *stack.last_mut().expect("regex context") = Ctx::Regex { class: true }
                        }
                        b']' => {
                            *stack.last_mut().expect("regex context") = Ctx::Regex { class: false }
                        }
                        b'/' if !class => {
                            stack.pop();
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
            Ctx::Template { expr_depth: None } => {
                out.push(if b == b'\n' { b'\n' } else { b' ' });
                if b == b'\\' && i + 1 < bytes.len() {
                    out.push(b' ');
                    i += 2;
                } else if b == b'`' {
                    stack.pop();
                    i += 1;
                } else if b == b'$' && bytes.get(i + 1) == Some(&b'{') {
                    // Keep the brace so the line-depth scan stays balanced.
                    out.push(b'{');
                    *stack.last_mut().expect("template context") = Ctx::Template {
                        expr_depth: Some(1),
                    };
                    i += 2;
                } else {
                    i += 1;
                }
            }
            Ctx::Code
            | Ctx::Template {
                expr_depth: Some(_),
            } => {
                let in_expr = matches!(
                    ctx,
                    Ctx::Template {
                        expr_depth: Some(_)
                    }
                );
                if b.is_ascii_whitespace() {
                    out.push(b);
                    i += 1;
                    continue;
                }
                if b == b'/' && bytes.get(i + 1) == Some(&b'/') {
                    out.push(b' ');
                    out.push(b' ');
                    stack.push(Ctx::LineComment);
                    i += 2;
                    continue;
                }
                if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    out.push(b' ');
                    out.push(b' ');
                    stack.push(Ctx::BlockComment { start: i });
                    i += 2;
                    continue;
                }
                if b == b'"' || b == b'\'' {
                    out.push(b' ');
                    stack.push(Ctx::Quote(b));
                    i += 1;
                    continue;
                }
                if b == b'`' {
                    out.push(b' ');
                    stack.push(Ctx::Template { expr_depth: None });
                    i += 1;
                    continue;
                }
                if b == b'/' && starts_regex(&out) {
                    out.push(b' ');
                    stack.push(Ctx::Regex { class: false });
                    i += 1;
                    continue;
                }
                out.push(b);
                if in_expr {
                    let mut depth = match ctx {
                        Ctx::Template {
                            expr_depth: Some(d),
                        } => d,
                        _ => 0,
                    };
                    if b == b'{' {
                        depth += 1;
                    }
                    if b == b'}' {
                        depth = depth.saturating_sub(1);
                    }
                    *stack.last_mut().expect("template context") = Ctx::Template {
                        expr_depth: (depth > 0).then_some(depth),
                    };
                }
                i += 1;
            }
        }
    }
    Masked { code: out, docs }
}

/// A heuristic for the one ambiguous delimiter in JavaScript text: `/` starts a
/// regular expression after an operator or one of the keywords that introduces
/// an expression, and is division after a value. Kept deliberately small — a
/// lexical tier does not claim to parse regex literals, only to keep their
/// slashes from being mistaken for comments.
fn starts_regex(out: &[u8]) -> bool {
    let mut end = out.len();
    while end > 0 && out[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    if end == 0 {
        return true;
    }
    let prev = out[end - 1];
    if prev == b')' || prev == b']' || prev == b'}' {
        return false;
    }
    if b"=([{,:;!&|?+-*%~^<>".contains(&prev) {
        return true;
    }
    let mut start = end;
    while start > 0
        && (out[start - 1].is_ascii_alphanumeric()
            || out[start - 1] == b'_'
            || out[start - 1] == b'$')
    {
        start -= 1;
    }
    matches!(
        &out[start..end],
        b"return"
            | b"case"
            | b"throw"
            | b"yield"
            | b"typeof"
            | b"instanceof"
            | b"in"
            | b"of"
            | b"new"
            | b"delete"
            | b"void"
    )
}

/// JSDoc prose, with comment markers and leading stars removed.
fn jsdoc(raw: &str) -> Vec<String> {
    if !raw.starts_with("/**") || raw.len() < 5 {
        return Vec::new();
    }
    raw[3..raw.len() - 2]
        .lines()
        .filter_map(|line| {
            let text = line.trim().trim_start_matches('*').trim();
            (!text.is_empty()).then(|| text.to_string())
        })
        .collect()
}
