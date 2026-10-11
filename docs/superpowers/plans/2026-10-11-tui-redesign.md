# TUI redesign (TD-1 … TD-6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace xencode's IDE-style three-pane screen with a conversation-first screen (style B), an always-ready input, a slim side panel, seven full-screen views behind one Ctrl+P palette, and eight retired panels.

**Architecture:** Tool calls become `role: "tool"` messages carrying a JSON `ToolEntry`, so every existing consumer of `messages` (engine view, replay, ACP, memory) carries them without a new field; a new `transcript` module renders messages as style-B lines; a new `shell` module draws the conversation frame (transcript, inline approval, input box, status line, side panel); a `screens` registry names the seven views and feeds the palette; the keymap makes the conversation input always ready. Old panels are reused as view bodies, then the retired ones and the preset layouts are deleted.

**Tech Stack:** Rust, ratatui 0.29, crossterm 0.28, nucleo-matcher (palette), unicode-width.

**Spec:** `docs/superpowers/specs/2026-10-11-tui-redesign-design.md`

## Global Constraints

- Rust only; **no mocks** in product code or tests (`AGENTS.md`). Render tests draw the real `ui::draw` into `ratatui::backend::TestBackend`; key tests drive the real `keymap::handle_key`.
- One commit per plan item (`TD-1` … `TD-6`), subject naming the ID, plain-English body, ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push unless the owner asks.
- Every `cargo test` run sets `XCODE_CONFIG_DIR` to a fresh temp folder and uses `timeout -k`; test the changed crate while iterating, `cargo test --workspace` before each commit; `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings -A clippy::format-in-format-args` clean on Windows **and** Linux (WSL clone `~/xencode`, script `C:/Users/sreev/.claude/jobs/73ce86b6/tmp/clippy-linux.sh`).
- Never run anything on the owner's local GPU (memory `ask-before-gpu`): no live model in tests.
- Glyphs, not emoji, in the new screens: `● ◐ ○ ✕ ✓ ⎿ ▍ ›`. Colours only from `app.theme` (`ThemeColors`); `NO_COLOR` must still produce a readable screen (`strip_colors` stays).
- Settings added: `side_panel` = `"auto" | "on" | "off"` (default `"auto"`, hides under 100 columns), `tool_output` = `"fold" | "expand"` (default `"fold"`), `status_line` = list of `model, provider, approval, context, cost, branch` (default all). Old config files load unchanged; `layout` presets are read and ignored after TD-5.
- Docs ride with features; manuals document only verified behaviour.

## Review Focus

1. **A very long single word or URL in a message** (no spaces, wider than the transcript): it must hard-wrap inside the indent, never overflow the frame or panic. Pinned in Task 1.
2. **Tool output with control characters, tabs or ANSI escapes** (from `cargo test`, `git`): the transcript shows plain text (tabs expanded, escapes stripped), not raw bytes that corrupt the screen. Pinned in Task 1.
3. **An approval arriving while the person has typed half a message:** the draft is kept, and keys go to the approval only while it is shown; after answering, the draft is still there. Pinned in Task 3.
4. **A terminal resized from 160 to 60 columns mid-session:** the side panel hides, the status line drops items from the right, the input stays on the last rows, nothing panics. Pinned in Task 2.
5. **An older window or replay file whose messages have no `tool` role** (and an engine newer than the window): old sessions still render, and an unknown role renders as a plain dim line. Pinned in Task 1.

---

## File Structure

- Create `rust/crates/xencode-tui-rs/src/transcript.rs` — `ToolEntry`, tool-message helpers, the style-B line renderer and indented wrapping.
- Create `rust/crates/xencode-tui-rs/src/shell.rs` — the conversation frame: areas, transcript drawing with scroll, inline approval box, input box, status line.
- Create `rust/crates/xencode-tui-rs/src/sidepanel.rs` — the five side-panel sections.
- Create `rust/crates/xencode-tui-rs/src/screens.rs` — `Screen` enum (Conversation, Files, Changes, Agents, Models, Project, Settings), names, keys, slash commands, which panels each hosts.
- Modify `src/app.rs` — `apply_token` turns `[TOOLCALL]`/`[TOOLEND]` into tool messages; `screen` field; model context skips `tool` messages; settings fields read from config.
- Modify `src/ui.rs` — `draw_frame` draws the shell for the Conversation screen and a view frame (tab bar + body) for the others; retired draw functions deleted in Task 5.
- Modify `src/keymap.rs` — always-ready input, new global chords, screen switching.
- Modify `src/palette.rs` — entries from `screens` + commands + settings.
- Modify `src/focus.rs`, `src/help.rs` — destinations and help from the screens registry; retired areas removed.
- Modify `rust/crates/xencode-config-rs/src/config.rs` — `side_panel`, `tool_output`, `status_line`.
- Tests: `rust/crates/xencode-tui-rs/tests/redesign_screens.rs` (new, render + key tests), existing `small_terminal_render.rs`, `first_screen.rs`, `disclosure_level.rs`, `fresh_start_level.rs` updated as panels change.

---

### Task 1: Transcript model and renderer (TD-1)

**Files:**
- Create: `rust/crates/xencode-tui-rs/src/transcript.rs`
- Modify: `rust/crates/xencode-tui-rs/src/lib.rs` (add `pub mod transcript;`)
- Modify: `rust/crates/xencode-tui-rs/src/app.rs` (`apply_token` at ~4434: `[TOOLCALL]`/`[TOOLEND]`; `[TOOL]` lines at ~4437; the model-context builder)
- Modify: `rust/crates/xencode-tui-rs/src/ui.rs` (`chat_lines` ~895 delegates to `transcript::lines`)
- Test: unit tests in `transcript.rs`; `tests/redesign_screens.rs`

**Interfaces:**
- Produces:
  - `pub const TOOL_ROLE: &str = "tool";`
  - `#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)] pub struct ToolEntry { pub id: String, pub name: String, pub arguments: serde_json::Value, pub summary: String, pub outcome: Option<String>, pub preview: String }`
  - `pub fn tool_message(entry: &ToolEntry) -> UiMessage` and `pub fn read_tool(msg: &UiMessage) -> Option<ToolEntry>`
  - `pub struct RenderOpts { pub width: u16, pub expand_tools: bool, pub expanded: Option<String> /* tool id */ }`
  - `pub fn lines(messages: &[UiMessage], theme: &ThemeColors, opts: &RenderOpts) -> Vec<Line<'static>>`
  - `pub fn wrap_indented(text: &str, width: usize, indent: usize) -> Vec<String>` (tabs → 4 spaces, ANSI/control chars stripped, hard-wraps words wider than the line)

- [ ] **Step 1: Write the failing unit tests** in `transcript.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::UiMessage;
    use crate::theme::ThemeColors;

    fn text(lines: &[Line]) -> Vec<String> {
        lines.iter().map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect()).collect()
    }

    #[test]
    fn wrapped_lines_keep_their_indent() {
        let out = wrap_indented("one two three four five six", 12, 2);
        assert!(out.len() > 1, "{out:?}");
        assert!(out.iter().all(|l| l.starts_with("  ")), "{out:?}");
        assert!(out.iter().all(|l| unicode_width::UnicodeWidthStr::width(l.as_str()) <= 12), "{out:?}");
    }

    #[test]
    fn a_word_wider_than_the_line_is_cut_inside_the_indent() {
        let url = "https://example.invalid/".to_string() + &"a".repeat(80);
        let out = wrap_indented(&url, 20, 2);
        assert!(out.len() >= 4, "{out:?}");
        assert!(out.iter().all(|l| l.starts_with("  ") && unicode_width::UnicodeWidthStr::width(l.as_str()) <= 20));
    }

    #[test]
    fn tool_output_is_plain_text() {
        let out = wrap_indented("a\tb\u{1b}[31mred\u{1b}[0m\u{7}", 40, 0);
        assert_eq!(out, vec!["a    bred".to_string()]);
    }

    #[test]
    fn each_speaker_gets_a_marker_and_a_name() {
        let msgs = vec![
            UiMessage { role: "user".into(), content: "fix it".into() },
            UiMessage { role: "assistant".into(), content: "done".into() },
        ];
        let t = text(&lines(&msgs, &ThemeColors::get("ocean"), &RenderOpts { width: 60, expand_tools: false, expanded: None }));
        assert!(t.iter().any(|l| l.starts_with("▍You")), "{t:?}");
        assert!(t.iter().any(|l| l.starts_with("▍xencode")), "{t:?}");
        assert!(t.iter().any(|l| l.contains("fix it")));
    }

    #[test]
    fn a_tool_call_is_one_line_and_long_output_folds() {
        let entry = ToolEntry {
            id: "c1".into(), name: "run_command".into(),
            arguments: serde_json::json!({"command": "cargo test"}),
            summary: "cargo test".into(), outcome: Some("failed".into()),
            preview: (1..=20).map(|n| format!("line {n}")).collect::<Vec<_>>().join("\n"),
        };
        let msgs = vec![tool_message(&entry)];
        let theme = ThemeColors::get("ocean");
        let folded = text(&lines(&msgs, &theme, &RenderOpts { width: 60, expand_tools: false, expanded: None }));
        assert!(folded.iter().any(|l| l.contains("⎿") && l.contains("cargo test")), "{folded:?}");
        assert!(folded.iter().any(|l| l.contains("+16 lines") && l.contains("ctrl+o")), "{folded:?}");
        assert!(!folded.iter().any(|l| l.contains("line 20")));
        let open = text(&lines(&msgs, &theme, &RenderOpts { width: 60, expand_tools: true, expanded: None }));
        assert!(open.iter().any(|l| l.contains("line 20")), "{open:?}");
    }

    #[test]
    fn an_unknown_role_is_a_plain_dim_line() {
        let msgs = vec![UiMessage { role: "observer".into(), content: "hello".into() }];
        let t = text(&lines(&msgs, &ThemeColors::get("ocean"), &RenderOpts { width: 40, expand_tools: false, expanded: None }));
        assert!(t.iter().any(|l| l.contains("hello")), "{t:?}");
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p xencode-tui-rs --lib transcript` (with `XCODE_CONFIG_DIR` set, `timeout -k 10 1800`)
Expected: FAIL — module `transcript` / functions not found.

- [ ] **Step 3: Implement `transcript.rs`.** Rules:
  - `wrap_indented`: strip `ESC [ … letter` sequences and other C0 controls except `\n`, expand `\t` to 4 spaces, split on `\n`, greedy word wrap at `width - indent` columns (unicode-width), a word wider than that is cut into chunks; every output line is `" ".repeat(indent) + chunk`; an empty input line stays an empty indented line.
  - `lines`: for each message by role —
    - `user`: blank line, then `▍` (theme.message_user, bold) + `You` (bold), then content wrapped at indent 2, each line prefixed by `▍ ` in theme.message_user.
    - `assistant`: blank line, `▍xencode` (theme.accent), then the existing markdown rendering (`crate::markdown`) of the content with each line prefixed `▍ ` in theme.accent; code blocks keep their highlighting.
    - `system`: dim (`theme.message_system`) line(s) `  · ` + wrapped text.
    - `tool` (via `read_tool`): one row `  ⎿ ` + summary (or `name` when empty) + `  ` + outcome glyph (`✓` ok, `✕` failed/denied/refused, `…` running) in success/danger/info; then preview lines (wrapped, dim, indent 4): all when `expand_tools` or `expanded == Some(id)`, else the first 4 and a row `    +N lines · ctrl+o`.
    - any other role: plain dim wrapped line.
  - `tool_message` serialises `ToolEntry` to JSON in `content` with role `TOOL_ROLE`; `read_tool` returns `None` for other roles or bad JSON.

- [ ] **Step 4: Wire the agent events.** In `App::apply_token` (~app.rs:4434): `[TOOLCALL]{json}` → parse `{id,name,arguments}`, push `transcript::tool_message(ToolEntry{summary: "", outcome: None, preview: ""…})`; `[TOOLEND]{json}` → find the last tool message with that `id`, set `outcome`, `preview`, rewrite its content. `[TOOL]→ summary` → if the last tool message has an empty `summary`, set it there and do **not** push the `⚙` system message; `[TOOL]← …` → drop when a tool message with a preview exists, else keep today's system line. Keep `live_tool_started` calls. In the model-context builder (search `context_messages` / where `self.messages` become provider messages), skip `role == "tool"`.

- [ ] **Step 5: Use it in the current chat pane.** `ui::chat_lines` (ui.rs:895) returns `transcript::lines(&app.messages, &app.theme, &RenderOpts{ width, expand_tools: app.config.tool_output == "expand", expanded: app.tool_expanded.clone() })`. Add `tool_expanded: Option<String>` to `App` (default `None`) and `tool_output: String` (default `"fold"`) to `XencodeConfig` with `#[serde(default = "default_tool_output")]`.

- [ ] **Step 6: Run the crate tests**

Run: `cargo test -p xencode-tui-rs` and `cargo test -p xencode-config-rs`
Expected: transcript tests pass; fix any existing render test whose expected text changed (old `🧑 You` / `🤖 Xencode` headers) to the new `▍You` / `▍xencode`.

- [ ] **Step 7: Engine test.** In `tests/redesign_screens.rs`, apply `[TOOLCALL]{"id":"c1","name":"read_file","arguments":{"path":"src/a.rs"}}` then `[TOOLEND]{"id":"c1","outcome":"ok","preview":"fn a() {}"}` through `app.apply_token` on `App::for_tests()` and assert exactly one `tool` message with outcome `ok`; render 100×30 and assert `⎿` appears once and no `⚙` line duplicates it.

- [ ] **Step 8: Commit** `TD-1: the transcript shows each speaker with a marker and each tool call as one folded line`.

### Task 2: The conversation shell and side panel (TD-2)

**Files:**
- Create: `src/shell.rs`, `src/sidepanel.rs`
- Modify: `src/ui.rs` (`draw_frame`), `src/app.rs` (`side_panel_open: bool`), config (`side_panel`, `status_line`)
- Test: `tests/redesign_screens.rs`

**Interfaces:**
- Consumes: `transcript::lines`, `RenderOpts` (Task 1)
- Produces:
  - `pub struct ShellAreas { pub transcript: Rect, pub approval: Option<Rect>, pub input: Rect, pub status: Rect, pub side: Option<Rect> }`
  - `pub fn areas(area: Rect, app: &App) -> ShellAreas` (side shown when `side_panel == "on"`, or `"auto"` and width ≥ 100 and `app.side_panel_open`; input height = 2 + lines of draft clamped 1..=6; approval box height from the pending approval's diff, clamped to half the transcript)
  - `pub fn draw(f: &mut Frame, app: &mut App, area: Rect)` (draws everything for the Conversation screen)
  - `pub fn status_items(app: &App, width: u16) -> Vec<String>` (left items in configured order, dropped from the right until the line fits with the right hint)
  - `sidepanel::sections(app: &App) -> Vec<Section>` with `pub struct Section { pub title: &'static str, pub rows: Vec<Line<'static>> }`; `sidepanel::draw(f, app, rect)`

- [ ] **Step 1: Failing render tests** (`tests/redesign_screens.rs`), helper `fn screen(app: &mut App<'static>, w: u16, h: u16) -> Vec<String>` returning each buffer row as a String:

```rust
#[test]
fn the_input_sits_on_the_last_rows_and_the_status_line_under_it() {
    let mut app = App::for_tests();
    let rows = screen(&mut app, 100, 30);
    assert!(rows[29].contains("ctrl+p"), "{:?}", rows[29]);           // status line
    assert!(rows[26].contains("╭") && rows[28].contains("╰"), "{rows:#?}"); // rounded input box
}

#[test]
fn the_side_panel_shows_at_wide_sizes_and_hides_under_100_columns() {
    let mut app = App::for_tests();
    app.side_panel_open = true;
    let wide = screen(&mut app, 160, 40).join("\n");
    assert!(wide.contains("Context") && wide.contains("Changes"), "{wide}");
    let narrow = screen(&mut app, 80, 24).join("\n");
    assert!(!narrow.contains("Context"), "{narrow}");
}

#[test]
fn a_resize_from_wide_to_narrow_keeps_the_frame_whole() {
    let mut app = App::for_tests();
    for w in [160u16, 120, 100, 99, 80, 60, 40] {
        let rows = screen(&mut app, w, 24);
        assert!(rows[23].trim_end().chars().count() <= w as usize);
        assert!(rows.iter().any(|r| r.contains("╰")), "input box lost at {w}");
    }
}

#[test]
fn an_approval_is_drawn_in_the_conversation_with_its_diff() {
    let mut app = App::for_tests();
    queue_edit_approval(&mut app); // pushes an ApprovalRequest for edit_file src/parser.rs with a +/- preview
    let text = screen(&mut app, 120, 36).join("\n");
    assert!(text.contains("src/parser.rs") && text.contains("+    self.close_level();"), "{text}");
    assert!(text.contains("y allow") && text.contains("n deny"), "{text}");
}
```

- [ ] **Step 2: Run to verify they fail** (`cargo test -p xencode-tui-rs --test redesign_screens`). Expected: FAIL on the status line / input box / Context assertions.

- [ ] **Step 3: Implement `shell.rs`:**
  - Layout: vertical — transcript (Min 1), approval (when pending), input (2 + draft lines), status (1). Horizontal split of the top area when the side panel shows: side panel 30 columns on the right.
  - Transcript: `transcript::lines`, bottom-anchored, scroll from `app.chat_scroll` (existing), "follow" when at the bottom; welcome line when no messages: `What should xencode do? · / commands · @ files · ? keys` and `model · project` dim.
  - Approval box: rounded border in `theme.warning`, title `Allow <class> to <subject>?`, the preview as a coloured diff (`+` success, `-` danger), last row `y allow · a always for this kind · n deny · tab add a note`.
  - Input: rounded border (`BorderType::Rounded`) in `theme.accent`, placeholder `Ask xencode…   / commands   @ files   ! shell` when empty; the existing `app.chat_input` TextArea drawn inside.
  - Status line: left `status_items` joined by ` · ` (accent for model), right `ctrl+p commands · ctrl+b panel · ? keys` dim; drop left items from the end until it fits.
- [ ] **Step 4: Implement `sidepanel.rs`:** rounded frame titled ` xencode · <branch> `; sections Changes (git-changed files with +/− from `git diff --numstat` cached in `app` — reuse the existing git status data; `+N more` past 6), Plan (`app.agent_plan` with glyphs), Context (bar from context fill, `tokens/limit`, model, cost from `app.spend`, approval mode), Agents (`app.team_shown()` + spawns, needs-you first, glyphs), Project (branch, dirty count, LSP/MCP status flags from existing app state). A section with nothing shows one dim line ("no changes yet").
- [ ] **Step 5:** `ui::draw_frame`: when `app.focus` is a chat focus (ChatInput) and no other screen is active, call `shell::draw` over the whole area instead of header + body + status; overlays (help, palette, toasts) still draw on top; the old approval overlay is skipped when the shell drew it inline.
- [ ] **Step 5b: Attention.** When a pending approval appears and the terminal reports focus lost (crossterm `Event::FocusLost`, enabled with `EnableFocusChange`), set the window title to `xencode · needs you` (crossterm `SetTitle`) and write an OSC 9 notification (`]9;xencode needs you`); restore the title `xencode · <project>` on focus gained or when the approval is answered. Test: `shell::attention_title(app)` returns `"xencode · needs you"` with an approval pending and `"xencode · <project>"` without.
- [ ] **Step 5c: Windows repaint.** On Windows, after a resize or every 2 s of streaming, call `terminal.clear()` before the next draw so ConPTY leaves no stale cells; record what is seen in the ledger (Ruling if flicker forces a different trigger).
- [ ] **Step 6:** Config: `side_panel: String` default `"auto"`, `status_line: Vec<String>` default `["model","provider","approval","context","cost","branch"]`. App: `side_panel_open: bool` default `true`.
- [ ] **Step 7: Run** the crate tests; update tests in `small_terminal_render.rs` / `first_screen.rs` / `no_color.rs` that asserted the old header or footer text for the chat screen to the new text. Expected: PASS.
- [ ] **Step 8: Commit** `TD-2: the conversation fills the screen with an input box, a status line and a side panel`.

### Task 3: Always-ready keys, screens registry and the palette (TD-3)

**Files:**
- Create: `src/screens.rs`
- Modify: `src/keymap.rs` (`handle_key`, `global_ctrl_chord`, `normal_key`, `editing_key`), `src/palette.rs` (`entries`), `src/app.rs` (`screen: Screen`), `src/ui.rs` (view frame), `src/help.rs`
- Test: `tests/redesign_screens.rs`

**Interfaces:**
- Produces:
  - `#[derive(Clone, Copy, PartialEq, Eq, Debug)] pub enum Screen { Conversation, Files, Changes, Agents, Models, Project, Settings }`
  - `impl Screen { pub const ALL: [Screen; 7]; pub fn name(self) -> &'static str; pub fn command(self) -> &'static str /* "files", "changes", … */; pub fn description(self) -> &'static str; pub fn panels(self) -> &'static [FocusArea] /* tabs */ }`
  - `App::open_screen(&mut self, s: Screen)`; `App::screen` field; `App::screen_tab: usize`
  - `PaletteTarget::Screen(Screen)` added to `palette.rs`

- [ ] **Step 1: Failing key tests:**

```rust
fn key(app: &mut App<'static>, code: KeyCode, mods: KeyModifiers) {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    xencode_tui_rs::keymap::handle_key(app, KeyEvent::new(code, mods), &tx);
}

#[test]
fn typing_goes_to_the_input_without_pressing_i() {
    let mut app = App::for_tests();
    for c in "fix q s m".chars() { key(&mut app, KeyCode::Char(c), KeyModifiers::NONE); }
    assert_eq!(app.chat_input.lines().join("\n"), "fix q s m");
    assert_eq!(app.screen, Screen::Conversation, "q/s/m are letters here, not shortcuts");
}

#[test]
fn ctrl_p_finds_every_view() {
    let mut app = App::for_tests();
    key(&mut app, KeyCode::Char('p'), KeyModifiers::CONTROL);
    assert!(app.palette_visible);
    for s in Screen::ALL {
        assert!(app.palette_entries_for("").iter().any(|e| e.name == s.command()), "{s:?}");
    }
}

#[test]
fn a_draft_survives_an_approval() {
    let mut app = App::for_tests();
    for c in "half a mess".chars() { key(&mut app, KeyCode::Char(c), KeyModifiers::NONE); }
    queue_edit_approval(&mut app);
    key(&mut app, KeyCode::Char('n'), KeyModifiers::NONE); // answers the approval
    assert!(app.pending_approval().is_none());
    assert_eq!(app.chat_input.lines().join(""), "half a mess");
}

#[test]
fn esc_esc_puts_the_last_message_back_in_the_input() {
    let mut app = App::for_tests();
    app.messages.push(UiMessage { role: "user".into(), content: "earlier ask".into() });
    key(&mut app, KeyCode::Esc, KeyModifiers::NONE);
    key(&mut app, KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(app.chat_input.lines().join("\n"), "earlier ask");
}

#[test]
fn esc_leaves_a_view_and_returns_to_the_conversation() {
    let mut app = App::for_tests();
    app.open_screen(Screen::Settings);
    key(&mut app, KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(app.screen, Screen::Conversation);
}
```

- [ ] **Step 2: Run to verify they fail.** Expected: FAIL (`screen` field, `Screen`, `palette_entries_for` missing).
- [ ] **Step 3: `screens.rs`** — the registry above; `panels()`: Conversation `[ChatInput]`, Files `[FileExplorer, CodeEditor]`, Changes `[ReviewDashboard, CodeReview, GitCommit]`, Agents `[WorkerPanel, WorktreePanel, TaskManager]`, Models `[ModelSelector, CustomModels, ProviderHealth]`, Project `[AdvisePanel, ImpactPanel, ProjectAnalyzer, SecurityAuditor, MultiLanguage]`, Settings `[Settings]`.
- [ ] **Step 4: App** — `screen: Screen` (default Conversation), `screen_tab: usize`; `open_screen` sets `screen`, `screen_tab = 0`, and `focus = screen.panels()[0]`; for Conversation sets `focus = ChatInput` and `input_mode = Editing`. Startup sets Editing for the conversation. `palette_entries_for(query)` returns ranked entries.
- [ ] **Step 5: Keymap.**
  - In the Conversation screen the input is always Editing: `editing_key` handles text; Esc stops a running turn, else on a second Esc within 600 ms copies the last user message into the input (`app.last_esc_at: Option<Instant>`); Esc never switches to Normal there.
  - Ctrl+C: clears a non-empty input; on an empty input, arms quit ("Ctrl+C again quits"); a running turn is stopped first (today's behaviour).
  - Global chords become: Ctrl+P palette (was ProjectAnalyzer), Ctrl+B side panel toggle (was ByteBot), Ctrl+O expand/fold the latest tool output (was Worktrees), Ctrl+L Models screen picker (was Insights), Ctrl+E Files screen, Ctrl+Space orchestrator toggle (kept), Shift+Tab (`KeyCode::BackTab`) cycles approval mode ask → auto-edit → plan (`config.agent_approval`), `?` on an empty input opens help. All other old chords are removed from the chat; their panels are reached through Ctrl+P and their screens.
  - In other screens: Esc returns to the Conversation (from the first tab) or the previous tab; Tab / Shift+Tab move between the screen's tabs; the panel's own key handler runs otherwise (existing `focus_key`).
- [ ] **Step 5b: Input prefixes and queued messages.**
  - A message sent while a turn runs is kept in `app.queued: Vec<String>`, shown in the transcript as a dim `queued: <text>` line, and submitted in order when `[DONE]` arrives. Test: submit while `is_generating`, assert nothing is sent and `queued.len() == 1`; apply `[DONE]`, assert the queue empties into a user message.
  - `!cmd` on Enter runs `cmd` as the agent's `run_command` would, through the same approval gate (class Shell), and shows its output as a tool message. Test: `!echo hi` produces a pending Shell approval naming `echo hi`.
  - `@` followed by text opens the existing file search over `app.file_tree` (fuzzy, nucleo); Enter inserts the path. Test: typing `@cargo` lists `Cargo.toml` first in `app.file_picker_matches()`.
- [ ] **Step 6: Palette** — `entries()` begins with one `Screen` entry per `Screen::ALL` (name = `command()`), then slash commands, then settings; `choose_palette_entry` opens the screen. Ctrl+X still opens it (kept as an alias).
- [ ] **Step 7: View frame** in `ui.rs`: for `screen != Conversation`, draw a one-row tab bar (`Changes ▸ Diff · Review · Commit` style, active tab in accent), the active panel's draw function into the remaining area (change each reused panel draw function to take an `area: Rect` and draw there instead of a centred popup), and the status line at the bottom.
- [ ] **Step 8: Help** — `help::help_lines` lists the global keys of spec §4.3 and the seven screens with their commands; no list of 25 chords.
- [ ] **Step 9: Run** crate tests; update `small_terminal_render.rs` FOCI sweep to also sweep every `Screen` × size; fix tests that pressed `i` first (they still pass because typing works anyway) and tests that relied on removed chords. Expected: PASS.
- [ ] **Step 10: Commit** `TD-3: the input is always ready, Ctrl+P opens any of seven views, and the new screen is the default`.

### Task 4: The merged views (TD-4)

**Files:**
- Modify: `src/ui.rs` (each reused panel's draw fn takes `area`), `src/screens.rs`, `src/sidepanel.rs` (Enter on a change opens Changes), `src/keymap.rs` (tab keys), `src/app.rs`
- Test: `tests/redesign_screens.rs`

**Interfaces:**
- Consumes: `Screen`, `open_screen`, tab bar (Task 3)

- [ ] **Step 1: Failing tests:** for each screen in `[Files, Changes, Agents, Models, Project, Settings]`, open it at 120×36 and assert the tab bar names its tabs and the body is not a centred popup (the first body row starts at column 0..2, the frame spans the full width); `/changes`, `/agents`, `/models`, `/project`, `/files`, `/settings` typed into the input open that screen; Enter on the side panel's first changed file opens Changes on the Diff tab.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement:** slash commands map to `open_screen`; Changes' tabs are the existing ReviewDashboard (diff vs base), CodeReview, GitCommit draws; Agents: WorkerPanel, WorktreePanel, TaskManager; Models: ModelSelector, CustomModels, ProviderHealth; Project: AdvisePanel, ImpactPanel, ProjectAnalyzer, SecurityAuditor, MultiLanguage; Files: explorer left (30%) and editor right in one frame (`[FileExplorer, CodeEditor]` share the screen, Tab moves between them). Styling of reused panels: rounded borders, no emoji in titles (replace the leading emoji in each panel title with nothing).
- [ ] **Step 4: Run** crate tests + the any-size sweep over every screen and tab. Expected: PASS.
- [ ] **Step 5: Commit** `TD-4: changes, agents, models and project each become one full-screen view`.

### Task 5: Retire eight panels and the preset layouts (TD-5)

**Files:**
- Modify: `src/focus.rs` (enum variants, `ALL`, `disclosure_level`, `DESTINATIONS`, `FEATURE_LIST`/`navigate_feature` removed), `src/ui.rs` (delete `draw_voice_interface`, `draw_learning_mode`, `draw_terminal_assistant`, `draw_performance_profiler` + `util_line`, `draw_performance_dashboard`, `draw_collaboration_hub` + `capitalize_role`, `draw_layout_panel` + `layout_detail_text`, `draw_feature_navigator`; the old header/body/footer path for chat), `src/keymap.rs` (their handlers, chords Ctrl+D/Ctrl+0/Ctrl+F/Ctrl+U/Ctrl+1..9 view recall, `cycle_layout`), `src/help.rs`, `src/app.rs` (state only those panels used; keep `voice.rs`, `collab_client.rs`, learning/profiler logic only if another caller exists — otherwise remove the TUI-only fields), tests that referenced them.
- Test: `small_terminal_render.rs` FOCI shrinks to the remaining 19; `the_sweep_covers_every_focus_area` still holds.

- [ ] **Step 1: Failing test:** in `tests/redesign_screens.rs` assert `FocusArea::ALL.len() == 19` and that `palette_entries_for("voice")`, `("learning")`, `("collab")`, `("profiler")`, `("dashboard")`, `("navigator")`, `("layout history")` return no screen or panel entry.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Delete** the eight panels everywhere the code map lists them (focus.rs, help.rs, keymap.rs close list / focus_key / on_esc, ui.rs overlay match and hints, tests FOCI, disclosure_level.rs, fresh_start_level.rs, zz-style captures). Remove the body presets: `templates::tree` stays for saved custom views, but `config.layout` no longer chooses a body for the Conversation screen; on load an old `layout` value is ignored with one notice the first time (`layout presets are gone; the conversation screen replaced them`). `/orchestrator status`'s section list and anything else counting panels are updated.
- [ ] **Step 4: Run** the full workspace tests (Windows + Linux) and compare with the known-failure baseline (`.superpowers/sdd/2026-10-10-team-orchestrator/baseline-failed.txt`). Expected: no new failures.
- [ ] **Step 5: Commit** `TD-5: eight rarely used panels and the old preset layouts leave the TUI`.

### Task 6: Manuals and real screens (TD-6)

**Files:** `README.md`, `QUICK_START.md`, `CLI_GUIDE.md`, `docs/USER_MANUAL.md`, `CHANGELOG.md`, `NEXT_PLAN_TASKS.md`; regenerate `docs/completions/*` and `docs/man/xencode.1` if any CLI help changed.

- [ ] **Step 1:** Render the new screens at 120×36 from the real code (a temporary capture test, not committed) and use the text in the manuals: the conversation screen with a tool call, an approval and the side panel; the palette; one view.
- [ ] **Step 2:** Rewrite the TUI sections: keys table (spec §4.3), the seven views and their commands, the settings `side_panel`, `tool_output`, `status_line`, and the CHANGELOG list of every retired panel and every moved chord (Ctrl+P, Ctrl+B, Ctrl+O, Ctrl+L, Ctrl+D, Ctrl+0, Ctrl+F, Ctrl+U, Ctrl+1..9, `i`).
- [ ] **Step 3:** NEXT_PLAN_TASKS: a TD section with TD-1 … TD-6 ticked and their evidence; counts updated from the final `cargo test --workspace`.
- [ ] **Step 4: Commit** `TD-6: the manuals describe the new conversation screen and its views`.
