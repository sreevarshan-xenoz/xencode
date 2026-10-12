//! The engine's terminal agents (OM-2): each one a real program — a coding
//! agent's own interface, or a shell — in a pseudo-terminal, in a git worktree
//! of its own, kept running by the engine while windows come and go.
//!
//! The engine keeps every tile's screen, so a window that starts watching
//! later is sent the screen as it stands and then the live output.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use xencode_team_rs::worktree;
use xencode_term_rs::{Attention, CommandSpec, Screen, Session, Size, TileState};

/// What a window is told about one tile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileInfo {
    pub id: String,
    pub agent: String,
    pub branch: String,
    pub worktree: String,
    /// `working`, `idle`, `needs_you`, `done` or `failed`.
    pub state: String,
    pub cols: u16,
    pub rows: u16,
    /// The exit code once the program has ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit: Option<i32>,
}

struct Tile {
    agent: String,
    branch: String,
    worktree: String,
    session: Session,
    screen: Screen,
    attention: Attention,
    size: Size,
    exit: Option<i32>,
}

impl Tile {
    fn state(&self) -> TileState {
        self.attention.state(Instant::now(), self.exit)
    }
}

fn state_word(state: TileState) -> &'static str {
    match state {
        TileState::Working => "working",
        TileState::Idle => "idle",
        TileState::NeedsYou => "needs_you",
        TileState::Done => "done",
        TileState::Failed => "failed",
    }
}

/// One entry of the picker: an agent's name, whether its program is on this
/// machine, and what a tile runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PickerAgent {
    pub name: String,
    pub installed: bool,
    pub runs: String,
}

/// The shell a plain terminal tile runs.
fn shell_program() -> (String, Vec<String>) {
    if cfg!(windows) {
        ("cmd".to_string(), vec!["/Q".to_string()])
    } else {
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "sh".to_string());
        (shell, Vec::new())
    }
}

/// The agents a tile can run: a plain shell, then every agent on the roster
/// with whether its program is installed here.
pub fn picker() -> Vec<PickerAgent> {
    let mut out = vec![PickerAgent {
        name: "shell".to_string(),
        installed: true,
        runs: shell_program().0,
    }];
    for spec in xencode_agents_rs::roster::ROSTER {
        let found = spec
            .binaries
            .iter()
            .find_map(|b| xencode_agents_rs::roster::which(b));
        out.push(PickerAgent {
            name: spec.name.to_string(),
            installed: found.is_some(),
            runs: found
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| spec.binaries.first().copied().unwrap_or("").to_string()),
        });
    }
    out
}

/// The program a tile for `agent` runs: the vendor's own interactive CLI, as
/// if the person had typed its name in their own terminal.
fn program_for(agent: &str) -> Result<(String, Vec<String>), String> {
    if agent == "shell" {
        return Ok(shell_program());
    }
    let spec = xencode_agents_rs::roster::find(agent).ok_or_else(|| {
        format!("`{agent}` is not an agent xencode knows; `shell` runs a terminal")
    })?;
    let path = spec
        .binaries
        .iter()
        .find_map(|b| xencode_agents_rs::roster::which(b))
        .ok_or_else(|| {
            format!(
                "{agent} is not installed here (looked for {:?})",
                spec.binaries
            )
        })?;
    Ok((path.display().to_string(), Vec::new()))
}

/// Every terminal agent the engine runs.
#[derive(Default)]
pub struct TermHost {
    tiles: BTreeMap<String, Tile>,
    next: u32,
}

impl TermHost {
    /// Start `agent` in a new worktree of the project at `root`, in a
    /// terminal of `size`. Returns what windows are told about it.
    pub fn open(&mut self, root: &Path, agent: &str, size: Size) -> Result<TileInfo, String> {
        let (program, args) = program_for(agent)?;
        self.next += 1;
        // `t` for terminal, so a tile never shares an id with an ACP worker.
        let id = format!("t{}", self.next);
        let base = worktree::current_branch(root).unwrap_or_else(|| "HEAD".to_string());
        let (path, branch) = worktree::create(root, &id, &base)?;
        let session = Session::spawn(
            &CommandSpec {
                program,
                args,
                cwd: path.clone(),
                env: vec![("XENCODE_TILE".to_string(), id.clone())],
            },
            size,
        )
        .map_err(|e| e.to_string())?;
        let tile = Tile {
            agent: agent.to_string(),
            branch,
            worktree: path.display().to_string(),
            session,
            screen: Screen::new(size),
            attention: Attention::default(),
            size,
            exit: None,
        };
        self.tiles.insert(id.clone(), tile);
        Ok(self.info(&id).expect("just inserted"))
    }

    fn info(&self, id: &str) -> Option<TileInfo> {
        self.tiles.get(id).map(|t| TileInfo {
            id: id.to_string(),
            agent: t.agent.clone(),
            branch: t.branch.clone(),
            worktree: t.worktree.clone(),
            state: state_word(t.state()).to_string(),
            cols: t.size.cols,
            rows: t.size.rows,
            exit: t.exit,
        })
    }

    /// Every tile, in id order.
    pub fn tiles(&self) -> Vec<TileInfo> {
        self.tiles.keys().filter_map(|id| self.info(id)).collect()
    }

    fn tile(&mut self, id: &str) -> Result<&mut Tile, String> {
        self.tiles
            .get_mut(id)
            .ok_or_else(|| format!("there is no terminal agent `{id}`"))
    }

    /// Bytes typed into tile `id`. Typing answers whatever it was asking.
    pub fn input(&mut self, id: &str, bytes: &[u8]) -> Result<(), String> {
        let tile = self.tile(id)?;
        tile.attention.typed();
        tile.session.write(bytes).map_err(|e| e.to_string())
    }

    /// The size the window drawing tile `id` gives it; the program is told.
    pub fn resize(&mut self, id: &str, size: Size) -> Result<(), String> {
        let tile = self.tile(id)?;
        tile.size = size;
        tile.screen.resize(size);
        tile.session.resize(size).map_err(|e| e.to_string())
    }

    /// End tile `id`'s program and everything it started. Its worktree and
    /// branch stay, for a merge or a clean later.
    pub fn stop(&mut self, id: &str) -> Result<(), String> {
        self.tile(id)?.session.kill();
        Ok(())
    }

    /// The bytes that redraw tile `id`'s screen as it stands.
    pub fn replay(&self, id: &str) -> Option<Vec<u8>> {
        self.tiles.get(id).map(|t| t.screen.replay())
    }

    /// Take in what every program wrote: each tile's screen and attention are
    /// kept current, its questions are answered, and the output is returned
    /// per tile for the windows watching it.
    pub fn pump(&mut self) -> Vec<(String, Vec<u8>)> {
        let now = Instant::now();
        let mut out = Vec::new();
        for (id, tile) in self.tiles.iter_mut() {
            let bytes = tile.session.pump(&mut tile.screen);
            if !bytes.is_empty() {
                tile.attention.observe(&bytes, now);
                out.push((id.clone(), bytes));
            }
            if tile.exit.is_none() {
                tile.exit = tile.session.exit_status();
            }
        }
        out
    }

    /// Whether any program is still running, so the engine stays up for it.
    pub fn has_work(&self) -> bool {
        self.tiles.values().any(|t| t.exit.is_none())
    }
}
