# Orchestrator mode as a tiling agent workspace (OM-1 … OM-6) — design

Date: 2026-10-11. Builds on the TUI redesign (`2026-10-11-tui-redesign-design.md`, TD-*), which
lands first. Status: approved in conversation (layout B with tweaks, approach B1); the owner then
asked for the remaining decisions to be made without asking, so they are marked **Ruling** below
and repeated in the plan's ledger.

## 1. What the owner asked for, and what was decided

The owner: "in orchestrator mode the whole tui ui changes like tuios and opens all available
agents like claude, agy, codex, opencode, kilo … like in a tiling manner."

| Question | Decision |
|---|---|
| What runs in a tile | The agent's **real interface** (claude, codex, opencode, kilo, agy, gemini, crush, …) in an embedded terminal |
| Where each agent works | **Its own git worktree** and branch, like team workers |
| When you leave the mode or close xencode | **They keep running** in the project's engine; reopening shows each tile's current screen |
| Which tiles open on entry | **A picker** of installed agents (last choice remembered) |
| Layout | **B: a rail plus tiles**, `z` zooms the focused tile, `t` tiles every agent |
| Tweaks (approved as a set) | loud needs-you; attention from the agents' own signals; rich rail rows; xencode's own chat as tile 0; merge from the rail with the checked merge; `a` ask all; small-terminal fallback; a Ctrl+A prefix that passes a doubled Ctrl+A through |
| Where the processes live | **B1: in the per-project engine** |

## 2. Facts this design rests on (checked 2026-10-11)

- Today orchestrator mode only changes a status-bar label, shows a toast and unlocks the
  `/orchestrator …` verbs; the drawing code never reads the mode. The `Ctrl+T` terminal region
  says "Terminal emulation coming soon"; there is no PTY or VT parser in the workspace.
- `portable-pty` 0.9.0 (wezterm) gives PTYs on Unix and Windows (ConPTY). `vt100` 0.16.2 parses a
  byte stream into a screen with scrollback. `tui-term` 0.3.4 draws a `vt100` screen but builds
  on `ratatui-core` 0.1 (the ratatui 0.30 line) while xencode is on ratatui 0.29.
- ConPTY reports a wrapped line like a newline, conhost answers some queries itself, and Windows
  Terminal leaves stale cells on partial repaints.
- Multi-agent tools that work (Claude Code's agent view, agent-deck, Claude Squad, TUIOS) share:
  a list of sessions with status (working / needs you / idle / done / error), needs-you first, a
  number to jump, zoom one, an inbox of what waits on the person, one worktree per agent.
- Agents signal attention themselves: Codex sends OSC 9; Claude Code notifies by bell/OSC and has
  hooks.
- The engine (`xencode engine`) already outlives windows while it has work, hosts the team of
  worker agents (`engine/team.rs`), and has worktrees, the checked merge, the audit trail and the
  Agents data.

## 3. Shape

```
 window (TUI)                         engine (per project)
 ┌───────────────────────┐   Term*    ┌──────────────────────────────┐
 │ rail + tiles           │ ◄──────── │ TermHost                      │
 │ per tile: vt100 parser │  output   │  per session: PTY + child     │
 │ draws the screen       │ ────────► │  + vt100 screen (replay)      │
 │ keys → focused tile    │  input,   │  + attention state            │
 └───────────────────────┘  resize    │ Team (worktrees, merge, audit)│
                                      └──────────────────────────────┘
```

- A new crate **`xencode-term-rs`**: a PTY session (`portable-pty` spawn, reader thread, writer,
  resize, kill of the whole tree), a screen (`vt100::Parser`, 5,000 lines of scrollback), and an
  attention detector over the byte stream.
- The engine hosts a **`TermHost`**: sessions keyed by worker id; output bytes go to every window
  watching, the screen is kept for replay.
- Each tile is a **team worker of kind `terminal`** (the existing workers are kind `acp`), so its
  worktree, branch, `team_status`, `xencode team clean`, the checked merge, the audit line, the
  Agents view and the rail all apply unchanged.
- The window draws a tile from a `vt100` screen it keeps itself, fed from the stream.
  **Ruling:** xencode draws `vt100` cells into its ratatui buffer with its own small function
  rather than depending on `tui-term` — upgrading ratatui across the 32k-line UI for one widget is
  the larger risk.

## 4. Protocol (engine ↔ window)

New messages, in the engine protocol's existing style (JSON lines; byte payloads base64):

- `ClientMsg::Term { req, request }`, with `TermRequest`:
  `Agents` (installed agents for the picker), `Open { agent }` (makes the worktree and starts the
  agent; returns the worker id), `Input { id, bytes }`, `Resize { id, cols, rows }`,
  `Watch { ids }` (start receiving output; the reply carries each screen's replay),
  `Stop { id }`.
- `EngineMsg::TermOutput { id, bytes }` (only to windows watching that id),
  `EngineMsg::TermReply { req, ok, body }`, and the worker's state in the view as today.
- **Replay on attach:** `vt100::Screen::contents_formatted()` reproduces the visible screen; the
  window feeds it to a fresh parser, then appends the live stream.
- **Ruling — size:** a session's size is the size of the tile that last sent `Resize` (the focused
  window wins); other windows draw it scaled into their tile, clipped. Two windows at different
  sizes are rare; tmux makes the same trade.

## 5. The mode

- **Entry:** Ctrl+Space or `/orchestrator on`. If no terminal workers run, a picker lists the
  installed agents (from the existing roster: claude, codex, opencode, kilo, agy, gemini, crush,
  cursor-agent, cline, kiro-cli), each with installed/missing; the last selection is remembered
  in config (`orchestrator.agents`). Running sessions are always shown; the picker adds more.
- **Ruling — sign-in and posture:** a tile runs the vendor's own CLI interactively with the
  person's own environment and sign-in, exactly as if they opened a terminal and typed `claude`;
  picking it in the picker is the person's choice. The Local Only posture, which governs agents
  xencode delegates work to on its own, does not block a tile; the picker says
  "runs <vendor>'s own CLI with your sign-in". Keys are not stripped (the person's own session).
- **Layout (default):** a rail on the left (about 22 columns) and tiles. The rail groups agents
  as Needs you / Working / Idle / Done, each row: jump number, status glyph, name, elapsed, branch
  or task, diff size (+/−), cost when known; then `+ new agent (n)`; then INBOX with what waits on
  the person. Tiles: xencode's conversation is tile **0**; terminal workers follow. Up to 4 tiles
  are drawn in a grid by default (2×2); more agents stay in the rail, pinned ones drawn.
  - `z` zoom the focused tile full screen (a one-line strip of the others on top); again to
    return. `t` draw every agent as an equal tile.
  - A tile waiting on the person has a yellow frame and the rail row moves to Needs you; the
    window title reads `xencode · 1 needs you`; an OSC 9 notification is sent when unfocused.
  - Under 100 columns the rail shrinks to glyphs and only the focused tile is drawn.
- **Keys:** everything typed goes to the focused tile, except the prefix **Ctrl+A** (setting
  `orchestrator.prefix`). After the prefix: `0`–`9` focus that tile, arrows or `h j k l` move
  focus, `z` zoom, `t` tile all, `n` new agent (picker), `x` stop (asks), `m` review and merge,
  `a` ask all, `r` rail focus, `?` keys, `Ctrl+A` sends a literal Ctrl+A. A mouse click focuses a
  tile. Shift+PageUp/PageDown scroll the focused tile's scrollback.
- **Ask all (`a`):** one line of input; on Enter its text plus a carriage return is written to every
  selected tile's PTY. **Ruling:** "selected" means every working or idle terminal tile; there is
  no separate selection UI in the first version.
- **Merge (`m`):** opens the Changes view on that worker's worktree diff against its base; Enter
  runs the existing checked merge (`team_merge` semantics: checks must pass, the person's working
  copy must be clean, audit line). Merge is allowed while the agent idles; a working agent gets a
  confirmation first.
- **Stop (`x`):** asks, then ends the agent's process tree; the worktree is kept (clean or merge
  later).
- **Exit:** Ctrl+Space or Esc from the rail returns to the conversation screen; the agents keep
  running in the engine.

## 6. Attention and state

- **Working:** output arrived in the last 2 s. **Idle:** no output for 2 s. **Needs you:** the
  agent sent BEL or an OSC 9 / OSC 777 notification since the person last typed into it, or (for
  Claude Code) its notification hook fired. **Done / failed:** the process exited (0 / non-zero).
- **Ruling:** no screen-text heuristics ("Do you want to proceed?") in the first version — they
  break with every vendor release; the agents' own signals are used, and the person's input
  clears needs-you.

## 7. Lifetime

- Sessions live in the engine; the engine counts a running terminal session as work, so it does
  not exit while any agent runs (TM's `has_work`).
- Closing the window detaches; reopening the mode watches every session and replays its screen.
- An engine that ends (crash, reboot) ends its PTYs; their worktrees are adopted as stopped
  workers, as team workers already are.

## 8. Errors

| Problem | What happens |
|---|---|
| The agent's program is not installed | The picker shows it greyed, with the install line from the roster |
| The program exits at once | The tile shows its last output and `exited (code N)`; the rail row is ✕ |
| The PTY cannot be created | The tile says so in words; nothing else is affected |
| The engine is gone | The window's existing reconnect starts a new one; tiles show `engine restarting` and the rail re-reads |

## 9. Testing and what counts as verified

No mocks. The terminal crate is tested against a **real shell** in a real PTY on both platforms
(`cmd`/`powershell` on Windows, `sh` on Unix): output reaches the screen, input reaches the
program, resize changes the program's view of the size, BEL and OSC 9 set needs-you, a child that
exits reports its code, stop ends the process tree, replay reproduces the screen. The engine is
tested end to end: a window opens a terminal worker on a real shell, watches it, types into it,
closes, reopens and sees the same screen. The tiling layout and keys are tested with render and
key tests like the TD ones. A live run with the installed agents (claude, codex, …) is watched
and written down; it is not a CI test (it needs the person's sign-in).

## 10. Stages

| Stage | What | Done when |
|---|---|---|
| OM-1 | `xencode-term-rs`: PTY session, screen, attention, renderer to a ratatui buffer | a real shell runs in a PTY on Windows and Linux; its screen draws into a buffer |
| OM-2 | Engine `TermHost`, terminal workers in the Team, protocol | a window opens, watches, types into, detaches from and re-attaches to a real shell session |
| OM-3 | The mode: picker, rail + tiles, focus, prefix keys, zoom, tile all | the layout and keys work over real sessions |
| OM-4 | Attention, inbox, window title and notification | a BEL or OSC 9 from a real program raises needs-you |
| OM-5 | Merge from the rail, stop, ask all, xencode tile 0, small terminals | a real worker's worktree change lands through the checked merge from the rail |
| OM-6 | Manuals and a watched live run with the installed agents | docs from the real behaviour; the run written down |

## 11. Risks

- ConPTY differences (line wraps, query answers) can make an agent's screen look slightly off
  on Windows; xencode repaints fully on Windows and documents what differs.
- Vendor TUIs change; xencode only passes bytes through and reads their standard attention
  signals, so a vendor change cannot break xencode, only that agent's own screen.
- Many agents at once cost memory and attention; the rail shows all, the grid draws four.
