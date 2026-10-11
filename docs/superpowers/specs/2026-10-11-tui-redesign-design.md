# TUI redesign (TD-1 … TD-6) — design

Date: 2026-10-11. Status: approved in conversation with the owner, who then said "decide stuffs
yourself … I will see things once completed"; decisions made after that point are recorded as
rulings in the plan's ledger.

## 1. What the owner asked for, and what was decided

The owner: "change the design of the tui also cause it looks like a slop now so think and check
how other agentic ides and cli tools are there and how we can change ours in a useful way for
every developer."

Decided with the owner, in order:

| Question | Decision |
|---|---|
| What the screen is built around | Conversation first, plus a slim side panel |
| Visual style (three mockups) | **B, "polished"**: coloured `▍` marker per speaker, no box per message, rounded input and approval boxes, framed side panel with headings (Crush / OpenCode family) |
| Built-in explorer and editor | Kept, as one toggleable **Files** view |
| The ~25 other panels | Become full-screen **views** reached from one palette, and the rarely used are **retired or merged** (table in §3) |
| Side panel content | Changes + Plan, Context + model, Agents/workers, Project health |
| Typing | **Always ready to type** — no `i`, no Normal/Editing modes |
| Build approach | **A1**: new focused modules, switch the default once whole, delete the old screens; no long-lived second UI |

## 2. Why: what the current screen does (rendered, not read)

Rendered with `TestBackend` at 120×36 and 80×24 on 2026-10-11:

- The default `classic` layout gives about 70% of the width to a file list and an empty code
  editor ("No file opened"); the conversation gets the right third — 21 columns at 80×24.
- Wrapped chat lines lose their indent and start at column 0; each speaker header's divider
  wraps onto a second line.
- Every pane is boxed, every title carries an emoji, panes carry two scrollbars, and the footer
  carries about twelve items (`?:help NORMAL [CODING] 🧰 main | ollama unknown 🔒 local only 0s
  📄 84 Δ 1 | i:edit m:models…`).
- The welcome text is a list of shortcuts; the help overlay lists about 25 Ctrl chords.
- Help, navigator, approvals and settings are all overlays drawn over the three panes.

What works and is kept: the approval prompt shows a real diff with `y`/`a`/`n`; the layout tree
(V-1), slash commands and the Ctrl+X palette (AG-3) are sound foundations.

## 3. Facts this design rests on (survey of 2026-10-11)

From a survey of Claude Code, Codex CLI, Gemini/Antigravity CLI, OpenCode, Kilo, Crush, Aider,
Amp, Goose, Warp, Cursor, Zed, VS Code agent mode, Windsurf, Antigravity, TUIOS, Claude Squad,
agent-deck, uzi and Conductor (sources in the session record). The patterns the best terminal
agents share, which this design adopts:

1. A transcript without a box per message; one input pinned to the bottom.
2. One quiet status line (model · context % · mode · branch), configurable.
3. `/` commands with fuzzy completion, `@` file mentions, `!` shell passthrough.
4. Shift+Tab cycles the approval mode, shown as a badge.
5. Esc interrupts; Esc Esc edits the previous message; the first Ctrl+C clears the input.
6. Tool calls folded to one line with an expand key — **with a setting to always expand**
   (Claude Code users objected when what the agent read was hidden).
7. Inline coloured diffs with line numbers; inline approvals with a short option list and a way
   to add a note.
8. "? for shortcuts" instead of a wall of hints; a Ctrl+P palette for everything else.
9. Messages typed during a turn are queued.
10. Terminal title and OSC 9 / BEL notification when the agent needs the person.
11. A toggleable side panel that hides on narrow terminals.

Anti-patterns avoided: a box around everything, several always-visible panels, banner art that
stays, a footer of key hints, emoji as status, a modal overlay for every action, auto-scroll
that pulls the view down while the person reads above, partial repaints that leave stale cells
in Windows Terminal (ConPTY).

## 4. The design

### 4.1 Views (27 destinations become 7 views + a palette)

| View | Built from today's panels | Purpose |
|---|---|---|
| **Conversation** (home) | Chat, ByteBot | The agent session. ByteBot's autonomous run is a mode of the conversation; its steps are the plan in the side panel. |
| **Files** | File Explorer, Code Editor | Browse and edit, one view. |
| **Changes** | Code Review, PR Review, Git Commit | Diff against the base file by file, AI review of it, stage and commit. |
| **Agents** | Worker Panel, Worktrees, Background Tasks | Team workers, `/spawn` runs, background tasks, their worktrees and costs. Orchestrator mode's rail is this view's data. |
| **Models** | Models, Custom Models, Provider Health | Pick a model, add an endpoint, provider health and latency. |
| **Project** | Insights, Impact, Project Analyzer, Security Auditor, Multi-Language | Refactor warnings, blast radius of a file, file breakdown, vulnerable dependencies, languages. |
| **Settings** | Settings | Restyled. |

Retired from the TUI (their logic stays in the crates; their CLI commands stay): Voice
Interface, Learning Mode, Terminal Assistant (replaced by `!cmd`), Performance Profiler
(`xencode perf`), Performance Dashboard (session stats move to the side panel and `/cost`),
Collaboration Hub (`xencode server`), Layout History, Feature Navigator (replaced by Ctrl+P).

Popups that remain: approvals (drawn inline in the conversation), the quick pickers (model,
theme, `@` file search, the agent picker) and the Ctrl+P palette.

### 4.2 The conversation screen

- **Transcript:** each turn starts with a coloured `▍` and the speaker's name; wrapped lines keep
  their indent; no box per message.
- **Tool calls:** one row each, `⎿ <verb> <subject> <short result>`; output past 4 lines folds
  to `+N lines · ctrl+o`. Setting `tool_output = "fold" | "expand"` (default fold).
- **Diffs and code:** inline, coloured, with line numbers; code blocks syntax-highlighted;
  reasoning ("thinking") folded by default.
- **Approval:** a rounded box in the accent-warning colour in the transcript flow: summary, the
  diff or command, and `y allow · a always for this kind · n deny · tab add a note`.
- **Queued input:** a message sent while a turn runs is shown as queued and sent when it ends.
- **Input:** a rounded box pinned to the bottom, 1–6 lines; `/` commands with fuzzy completion,
  `@` file search, `!cmd` shell passthrough (runs through the same approval gate as the agent's
  shell tool).
- **Status line (one row):** `model · local|cloud · approval mode · context % · cost · branch`,
  right side `ctrl+p commands · ctrl+b panel · ? keys`; items drop from the right as width
  shrinks. Setting `status_line` lists the items.
- **Side panel (Ctrl+B):** about 30 columns; sections Changes (files changed this session with
  +/− counts; Enter opens the diff in Changes), Plan (the agent's plan or ByteBot's steps with
  ✓ ◐ ○), Context (bar, tokens, model, cost, approval mode), Agents (workers with ● ◐ ○ ✕ ✓,
  needs-you first), Project (branch and dirty state, LSP and MCP status, diagnostics count).
  Long lists end `+N more`. Setting `side_panel = "auto" | "on" | "off"`; `auto` hides it under
  100 columns.
- **Attention:** when a turn waits on an approval and the terminal is unfocused, the window title
  becomes `xencode · needs you` and an OSC 9 notification (or BEL) is sent.
- **Look:** one accent colour plus muted greys; glyphs (● ◐ ○ ✕ ✓ ⎿ ▍) not emoji; the existing
  themes provide the colours; `NO_COLOR` respected; on Windows a full repaint per frame.
- **Empty state:** one line: `What should xencode do? · / commands · @ files · ? keys`, the
  model and the project — no list of shortcuts.

### 4.3 Keys

| Key | Action |
|---|---|
| typing | goes to the input (no mode) |
| Enter / Shift+Enter or Alt+Enter | send / new line |
| Esc | stop the running turn, else close the current view or popup |
| Esc Esc | edit the last message |
| Ctrl+C | clear the input; on an empty input, a second press quits |
| Ctrl+P | palette: views, commands, settings |
| Ctrl+B | side panel |
| Ctrl+O | expand / fold the selected or latest tool output |
| Ctrl+L | model picker |
| Ctrl+E | Files view |
| Shift+Tab | approval mode: ask → auto-edit → plan |
| Ctrl+Space | orchestrator mode (OM spec) |
| `?` on an empty input | shortcuts |
| PageUp / PageDown, wheel | scroll; scrolling up stops auto-follow until the end is reached |

Each view owns its own single-letter keys (as views do today); only the keys above are global.

## 5. Architecture (approach A1)

New modules in `xencode-tui-rs`, each with one job:

- `transcript` — `TranscriptItem` (Message, ToolCall {id, name, args, outcome, preview},
  Approval, Plan, Notice) built from the agent loop's existing `[TOOLCALL]`/`[TOOLEND]`/plan
  events, and a renderer `fn lines(items, width, opts) -> Vec<Line>`. The engine's `View`
  carries the items (`#[serde(default)]`, so older windows still read the view).
- `shell` — the frame: transcript area, input, status line, side panel; width split and hiding.
- `sidepanel` — five sections, each a function from `&App` to rows.
- `views` (registry) — the seven views: name, key, slash command, draw fn, key fn. The palette,
  `/help` and the docs read this one list.
- `keymap` — rewritten for always-ready typing; global keys per §4.3; views keep their own.

State a view owns moves out of `app.rs` with the view. The three body presets (`classic`,
`zen`, …) go; the layout tree and saved-layout machinery stay (orchestrator mode uses them).

## 6. Migration (each step green; the new screen is the default from step 3)

1. **TD-1** transcript model and renderer (Conversation draws from it).
2. **TD-2** shell: status line, input box, side panel.
3. **TD-3** always-ready keys, Ctrl+P over the views registry; Conversation, Files and Settings
   moved; the new screen becomes the default.
4. **TD-4** Changes, Agents, Models and Project views from the merged panels.
5. **TD-5** retire the eight panels; delete the old preset layouts and their code paths.
6. **TD-6** docs (README, QUICK_START, CLI_GUIDE, USER_MANUAL, CHANGELOG, NEXT_PLAN) with
   screens rendered from the real code.

## 7. Settings and compatibility

New: `side_panel` (auto/on/off), `tool_output` (fold/expand), `status_line` (item list).
Removed: the body preset `layout` (an old value is read and ignored, saying so once); keys
bound only to retired panels. Old config files load unchanged otherwise.

## 8. Testing and what counts as verified

No mocks (`AGENTS.md`). Render tests draw the real screens at 80×24, 100×30 and 160×48 and
assert on the text and positions (input on the last rows, wrapped lines keep the indent, the side
panel hidden under 100 columns, an approval shows its diff, the status line fits); the existing
any-size sweep keeps every view panic-free. Key tests drive the real key handler (typing without
`i`, Esc stopping a turn, Esc Esc editing the last message, Ctrl+P finding every view). A real
window linked to a real engine sees the structured transcript. Windows and Linux CI.

## 9. Risks

- The transcript's structured items must stay in step with the agent loop's events; the
  renderer falls back to text for an event it does not know.
- Removing modes changes muscle memory for current users; `?` and the empty-state line say what
  moved, and the CHANGELOG lists every retired key.
- `app.rs` is 22,000 lines; moving view state out is done view by view inside TD-3/TD-4, not as
  a separate refactor.
