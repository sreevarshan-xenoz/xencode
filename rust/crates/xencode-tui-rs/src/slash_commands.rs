//! The slash commands the composer intercepts, and the helpers that recognise
//! and complete them.

/// Slash commands intercepted by `submit_message`, in handler order.
pub const SLASH_COMMANDS: &[&str] = &[
    "/init",
    "/files",
    "/changes",
    "/models",
    "/project",
    "/settings",
    "/ctx",
    "/advise",
    "/impact",
    "/bytebot",
    "/spawn",
    "/plan",
    "/rewind",
    "/lesson",
    "/gate",
    "/mcp",
    "/plugin",
    "/skills",
    "/trace",
    "/cost",
    "/doctor",
    "/verify",
    "/hotspots",
    "/agents",
    "/workers",
    "/orchestrator",
    "/trust",
    "/egress",
    "/goto",
    "/level",
    "/model",
    "/help",
];

/// The first word of `prompt` when it reads as a slash command (`/word`,
/// letters, digits and dashes only) — so an absolute path such as
/// `/usr/lib is missing` still reaches the model as a prompt.
pub fn unknown_slash_command(prompt: &str) -> Option<&str> {
    let word = prompt.split_whitespace().next()?;
    let name = word.strip_prefix('/')?;
    let looks_like_command = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    looks_like_command.then_some(word)
}

/// Complete a partially typed command token against `SLASH_COMMANDS`.
/// Returns the longest common prefix when it extends the token (pure —
/// unit-tested); None when ambiguous or already complete.
pub fn complete_slash_token(token: &str) -> Option<String> {
    if token.len() < 2 || !token.starts_with('/') {
        return None;
    }
    let matches: Vec<&'static str> = SLASH_COMMANDS
        .iter()
        .copied()
        .filter(|c| c.starts_with(token))
        .collect();
    if matches.is_empty() {
        return None;
    }
    let mut lcp = matches[0].to_string();
    for m in &matches[1..] {
        let keep = lcp
            .chars()
            .zip(m.chars())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| a)
            .collect::<String>();
        lcp = keep;
    }
    (lcp.len() > token.len()).then_some(lcp)
}
