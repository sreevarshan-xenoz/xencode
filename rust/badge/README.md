# xencode-badge

A small round xencode logo that floats above other windows and shows what every
running xencode terminal session is doing, so you can work in other programs
and still see when the agent needs you or has finished.

It is its own Cargo workspace, separate from `rust/`, so ordinary xencode builds
never compile the GPU interface library it uses (GPUI, from the Zed editor,
Apache-2.0, pinned to one commit in `Cargo.toml`).

## Build and run

```
cargo build --release --manifest-path rust/badge/Cargo.toml
```

Put `rust/badge/target/release/xencode-badge` (`.exe` on Windows) next to the
`xencode` executable or on `PATH`, then either run `xencode badge` or turn on
Settings → `Floating Badge` in the terminal app to start it with every session.
Only one badge runs per user; starting another exits straight away.

## What it shows

Each running session writes a small status file to the `live/` folder of
xencode's state directory (`xencode paths` prints where that is). The badge
reads that folder every second and shows the most urgent state:

| Badge | Meaning |
|---|---|
| Dim grey `x` | No session is doing anything |
| Blue `x` with a ring of dots stepping round | A session is working |
| Amber `?`, gently pulsing | A session is waiting for you, for example to allow a tool call |
| Green `✓` | A session finished in the last minute |
| Red `!` | A session failed; it stops showing once you have hovered over it |
| Small grey dot on the logo | A session stopped sending its heartbeat (it may have been closed or crashed) |

Hover over the badge for a card with one row per session: project, state in
words, model, what it is doing, and how long ago that changed. Drag the badge
to move it; the place is remembered in `badge.json` in the settings directory.
Right-click it to close it.

## Platforms

Watched working on Windows 11 (2026-10-09): every state above, the hover card,
dragging and the saved position, right-click to close, the one-badge rule, and
a real terminal session appearing in the card. On Windows the badge asks the
window manager not to draw its usual border, rounded corners and shadow.

macOS is built and tested in CI but has not yet been watched running on a Mac.
