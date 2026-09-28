# 🤖 Xencode CLI Guide

The command-line interface for the Xencode AI assistant (Rust binary).
Running `xencode` with no subcommand launches the TUI.

## 🚀 Installation

```bash
# From source
git clone https://github.com/sreevarshan-xenoz/xencode
cd xencode
./install.sh        # Linux/macOS — builds + installs the binary

# Or build directly
cd rust && cargo build --release -p xencode-cli
cp target/release/xencode ~/.local/bin/
```

## 🎯 Quick Start

```bash
# Launch the TUI (default)
xencode

# One-shot query
xencode query "Explain clean code principles"

# Analyze a path
xencode analyze ./src

# Show version
xencode --version
```

## 📋 Command Reference

### `xencode` / `xencode tui`
Launch the immersive terminal UI (default when no subcommand is given).

TUI keys — press `?` (or `F1`) in the TUI for the live, panel-aware
keybinding overlay; the authoritative list lives there. Essentials:
`Tab` cycles explorer/editor/chat · `i` edits chat (`Enter` sends,
`Alt+Enter`/`Ctrl+J` newline, `Alt+↑/↓` history, `Tab` completes `/`
commands) · `m` model selector · `s` settings · `e` edit focused file ·
`Ctrl+R` AI review · `Ctrl+Y` PR review · `Ctrl+K` background tasks · `Ctrl+O` worktrees · `Ctrl+L` insights · `Ctrl+B` ByteBot ·
`Ctrl+H` health check · `Ctrl+G` git refresh · `Ctrl+W` close panel ·
`Ctrl+C` or `q` quit. Slash commands: `/init`, `/ctx`, `/advise`,
`/bytebot`, `/plan` (pin or clear the agent's todo list),
`/rewind` (undo the agent's file changes for this session),
`/mcp` (connect every MCP server declared in config; `/mcp status`,
`/mcp stop`), `/plugin` (report which plugins loaded and what they changed;
`/plugin reload` re-scans the plugin directory), `/trace [turns]` (what the
recent agent turns did — rounds, tool calls with the arguments they were made
from and their outcome, the files the context put in front of the model, whether
the turn carried the `[d]` decision marker, and any token count a server
reported — read from `.xencode/cache/turns.jsonl` in the project,
so it answers with every model server down), `/cost` (tokens, KV-cache reuse,
p50/p95 speed and spend for the turns recorded in this project, read from
`.xencode/cache/metrics.jsonl` through its rollup sidecar and priced by
`.xencode/pricing.json`; a model with no price there is shown as unpriced, never
as free, and it too answers with every model server down), and
`/spawn <task> [#branch]` (run a subagent in a fresh
git worktree next to the project, e.g. `proj-spawn-1` on branch
`xencode/spawn-1`; a `#branch` suffix names the branch). The spawned
agent's live steps stream in the transcript, its final answer is posted
back with `(spawn #<id> · <task>)`, and `/spawn status` lists every
registered run with its worktree location. Your main chat keeps working
while the subagent works.

#### What the build tells the model: `/ctx prompts`

The instructions this program sends with every request are plain markdown files
under `rust/crates/xencode-context-rs/prompts/` — the agent system prompt, the
tool vocabulary, the transcript-folding prompt, and the two subagent briefs.
`/ctx prompts` lists them by name with the file each one lives in and a version,
which is a hash of that file's text, plus one digest for the set. Edit a file and
the version moves; there is no number to forget to bump, and no wording change
that can be recorded as "nothing changed". The files are compiled in, so a rebuild
is required — which also means a saved edit cannot change the instructions under a
running session, the property llama.cpp's prompt-cache reuse depends on.

The set digest is written into every metrics and turn row from now on
(`.xencode/cache/metrics.jsonl`, `.xencode/cache/turns.jsonl`), and `/ctx eval`
records each retrieval arm's score with it in `.xencode/cache/eval.jsonl`. The
eval panel then compares a score only with an earlier run of the same arm at the
same depth taken under the same prompts, and says so when it cannot:

```text
[CTX]🧪 Retrieval eval — 25 gold queries, top-5 (25 gold files) · prompts f6062cc81640
[CTX]   deterministic         previous run of these prompts: MRR 0.329 → 0.329 (+0.000)
[CTX]   + text (path+symbol)  no comparison: the last run of this arm used prompts 12ab34cd56ef, this one uses f6062cc81640
```

The first line is a repeat measurement of this repo's built-in gold set; the
second is what a run looks like after someone edited a prompt file in between.

Below the three arms, the same panel prices each retrieval bias on the questions
written for it, against those same questions with the bias switched off, on both
arms:

```text
[CTX]   shape biases, per partition · 893 test names over 101 files:
[CTX]     on the deterministic arm:
[CTX]       general  21 probes · control, no weight moves: MRR 0.347
[CTX]       bugfix    4 probes · MRR 0.050 → 0.237 (+0.188) with the bias
[CTX]     on the + text + doc prose arm:
[CTX]       general  21 probes · control, no weight moves: MRR 0.722
[CTX]       bugfix    4 probes · MRR 1.000 → 1.000 (+0.000) with the bias
[CTX]   improved: bugfix on deterministic
```

`general` is printed as the control it is — it moves no weight, so its two numbers
are always equal — and a bias that improves nothing says so on the last line
instead of disappearing from the panel. Measuring per partition is what keeps an
overall score from rising on one kind of question while a bias quietly damages
another. The sample above is a run of the same measurement from a clean tree, via
`cargo test -p xencode-context-rs --test gold_baseline -- --ignored --nocapture`;
`/ctx eval` also counts the files git reports as changed, so in a dirty working
tree its numbers can sit slightly differently.
A prompt edit and a retrieval change are two different things, and before this
they were easy to mistake for each other.
`cargo test -p xencode-context-rs --test gold_baseline -- --ignored --nocapture`
measures the same three arms from the command line and appends to the same log.

#### Reading a dependency's own source: `crate:<name>[/<path>]`

The three read tools the model is given — `read_file`, `list_dir`,
`search_files` — reach one place besides the workspace: the upstream source of a
crate this project's `Cargo.lock` pins, which cargo has already unpacked under
`$CARGO_HOME/registry/src`. No network is involved; `cargo fetch` is what put it
there. The address is `crate:` plus the package name, optionally with a path
inside it:

```text
crate:serde/src/de.rs
crate:adler2/README.md
```

Which version it means is decided by `Cargo.lock`, not by what happens to be on
disk — that directory holds several versions of the same crate side by side, and
reading the wrong one answers a question about a build this project does not
have. Every read therefore says where it came from, and the answers below were
produced on this machine:

```text
[adler2 2.0.1 — the version this project's Cargo.lock pins — read from crate:adler2/Cargo.toml, unpacked by cargo]
1   # THIS FILE IS AUTOMATICALLY GENERATED BY CARGO
…

error: crate:adler2 is a directory — ask for a file inside it, for example crate:adler2/Cargo.toml or crate:adler2/README.md

[adler2 2.0.1 — the version this project's Cargo.lock pins — read from crate:adler2, unpacked by cargo]
6 match(es):
crate:adler2/Cargo.lock:3:version = 4
crate:adler2/Cargo.lock:7:version = "2.0.1"
crate:adler2/Cargo.lock:14:version = "1.0.0"
crate:adler2/Cargo.toml:15:version = "2.0.1"
crate:adler2/Cargo.toml:87:version = "1.0.0"
crate:adler2/Cargo.toml.orig:3:version = "2.0.1"
```

What it does not do: it is not a second filesystem root for writes. A `crate:`
address in `write_file`, `edit_file`, `edit_symbol`, `ast_edit`, `codemod` or a shell command's working
directory is refused by the permission policy and again by the executor, in every
approval mode, even after edits have been granted for the session. A crate the
lock does not name is refused by name, an ambiguous one (pinned in two versions,
both unpacked) lists both directories instead of choosing, and one pinned but not
yet downloaded names `cargo fetch`. `what_breaks` and `repo_advise` do not read
dependencies at all: their answers are about this workspace, and reaching
outside it would let a dependency's code be presented as yours.

#### Reading a dependency's documentation: `read_docs`

Source is one thing; how the author documents the crate is another. `read_docs`
takes a package name and answers with the file the crate itself points at as its
readme — its `Cargo.toml` `readme = "…"` entry when it has one, otherwise the
conventional names in a fixed order — and takes an optional `path` for any other
document inside it (`CHANGELOG.md`, `docs/guide.md`). It reads the same unpacked
copy cargo already put on this machine, so no network is involved by default:

```text
[adler2 2.0.1 — the version this project's Cargo.lock pins — read from crate:adler2/README.md, unpacked by cargo]
# Adler-32 checksums for Rust
…
Other documentation in this crate: CHANGELOG.md, LICENSE-0BSD, LICENSE-APACHE, LICENSE-MIT, RELEASE_PROCESS.md — ask again with one of those paths.
```

That last line is the point of the tool: a model that reads one document is told
which others exist, in the form it needs to ask for one. A path the crate does
not have is answered the same way rather than as a missing file, and a `path`
that climbs out of the crate directory is refused before the disk is touched:

```text
error: adler2 2.0.1 has no "not-a-document.md"; documentation it does have: CHANGELOG.md, LICENSE-0BSD, LICENSE-APACHE, LICENSE-MIT, README.md, RELEASE_PROCESS.md — ask again with one of those paths
error: read_docs path "../../etc/passwd" reaches outside the crate; keep it relative and inside it
```

Long documents are cut at the front, at 8 192 bytes, and say so with the
instruction for getting the rest — the cut half of a readme is useless if the
model cannot tell it is a half.

When the crate is not on this machine at all, the answer gives the reason and
both ways to get it, and it does not dial out to find either:

```text
error: this project's Cargo.lock does not name not-a-crate-anywhere-here, so there is no version of it to read from here. read_docs reads only what cargo has already unpacked unless the user turns on allow_online_docs (`xencode config set allow_online_docs true`); a version named in Cargo.lock can also be unpacked on this machine with `cargo fetch`.
```

`allow_online_docs` is what opens the fetched half. With it on, and only where
there is no local copy, `read_docs` will take the version-pinned readme from
crates.io (`/api/v1/crates/<name>/<version>/readme`, which answers with rendered
markdown converted back to text, links kept) or a file from docs.rs
(`/crate/<name>/<version>/source/<path>`, from which the file's own text is
recovered). A version is mandatory for both — crates.io replies HTTP 400 to a
version-less readme request, so there is no "latest" to fall back to and the tool
says so rather than making the call. Each fetched answer says it was fetched and
why, which is the difference between reading what this project builds and reading
whatever the registry published:

```text
[serde 1.0.200 readme — fetched from https://crates.io/api/v1/crates/serde/1.0.200/readme, because cargo has not unpacked serde 1.0.200 on this machine]
Serde is a framework for serializing and deserializing Rust data structures efficiently and generically.
…
```

A version that is published but has no readme is reported as that, not as a
network failure: crates.io still redirects, and the object store it lands on
refuses, which the tool reads as "no readme published". `read_docs` is read-only
in every approval mode, like the three read tools, and it is not a way to write:
the only files it opens are documentation, and a fetched document is handed to the
model as text, never saved.

#### Asking what is known to be wrong with a dependency: `lookup_advisory`

The third thing a model gets wrong about a dependency is its history: asked
whether a version is affected by something, it answers from memory and names an
advisory number that was never published. `lookup_advisory` answers from the
corpus on this machine instead — the RustSec advisory database and Google's OSV
records for crates.io, both downloaded by `xencode advisories sync` (see below)
and both read from disk from then on. It is read-only in every approval mode,
and the tool itself has no network path at all: a turn inside the agent loop
cannot make a request under the name of a safety check.

Ask for a crate and it lists what the corpus holds; ask with a version, or leave
the version out in a project whose lock file pins one, and it judges that version.
Here it is in this project, with no version supplied — `lru` is what
`rust/Cargo.lock` pins:

```text
judging version 0.12.5, which this project's Cargo.lock pins
3 advisory record(s) for lru — corpus synced today, rustsec revision e2111519b
assessed against version 0.12.5:
  RUSTSEC-2026-0253 [rustsec] 2026-05-12: Potential use-after-free due to lack of panic safety in `LruCache::pop()` — informational: unsound
      see: https://github.com/jeromefroe/lru-rs/pull/238
      this version: AFFECTED here — the corpus offers 0.18.2 as safe
  RUSTSEC-2026-0002 [rustsec] 2026-01-07: `IterMut` violates Stacked Borrows by invalidating internal pointer — severity GHSA LOW — informational: unsound
      see: https://github.com/jeromefroe/lru-rs/pull/224
      this version: AFFECTED here — the corpus offers 0.16.3 as safe
  RUSTSEC-2021-0130 [rustsec] 2021-12-21: Use after free in lru crate — severity GHSA HIGH
      see: https://github.com/jeromefroe/lru-rs/issues/120
      this version: not affected (this version is at or above the 0.7.1 fix)
      affected function lru::LruCache::iter in < 0.7.1
      affected function lru::LruCache::iter_mut in < 0.7.1
```

The header says where the version came from, because "0.12.5 is affected" about
this project is a different claim from the same words about a version someone
typed. A record that names the functions it concerns prints them under the
verdict, up to four of them, and the pinned record is judged on its own facts: an
advisory that was fixed long ago is said to be fixed here, not feared.

Two answers are deliberately worded so they cannot be read as an all-clear. A
crate nobody has published an advisory for comes back as `no advisory in the
local corpus` followed by the size of the corpus, its sync date and the RustSec
revision it was taken at, and the note that absence of an advisory is not a
statement that the crate is safe. A machine that has never synced comes back
with nothing at all:

```text
error: no advisory corpus at /home/sree/.xencode/advisories — advisory state is unknown, not clean. Run `xencode advisories sync` (needs network once).
```

Where the two corpora overlap, the RustSec record is the one kept, since it is
curated; the OSV mirror of it is dropped, except that a rating the curated record
does not carry is transferred onto it — 380 of the 822 RustSec advisories with no
CVSS vector gain a one-word GHSA rating that way, which is why both corpora are
downloaded rather than one.

#### What a failing build answers with

A `cargo build` or `cargo check` the model asks for is run with
`--message-format=json`, and the answer is rustc's own account of the failure
rather than the tail of a text dump (measured below on a scratch crate, with the
real numbers from `cargo test -p xencode-core-rs --lib a_live_build -- --ignored
--nocapture`):

```text
$ cargo build --message-format=json
exit 101
1 error(s), 0 warning(s) from rustc:
  error E0308: mismatched types — src/lib.rs:1:27
      help: you can convert a `u32` to a `u64` ⇒ .into()
      rustc can apply this itself (src/lib.rs:1:28): .into()

What rustc's own error index says about E0308:
Expected type did not match the received type.
…
   Compiling probe v0.1.0 (…)
error: could not compile `probe` (lib) due to 1 previous error
```

An `E`-code carries the full text of its entry in the error index, which ships
inside the compiler: once per code, at most three codes, each cut at 1 200
characters. Twenty diagnostics is the list ceiling and the whole account is kept
under 6 KiB, so that what is left out is named instead of being cut from the
front by the 8 KiB tail that applies to every command's output. On the same
single error that is 1 432 bytes handed over in place of 11 252 bytes of JSON and
1 271 bytes of rustc's rendered text.

Only a plain, single `cargo build` or `cargo check` is asked this way. A command
that composes (`cargo build && cargo test`) would take the flag on the wrong
word, anything after `--` belongs to rustc rather than cargo, and a command that
already chose a format is left alone — those, `cargo test`, and a build started
with `background_start` behave exactly as before.

### `xencode query <prompt>`
Send a one-shot query to the configured model.

```bash
xencode query "Explain microservices architecture"

# Pin a model, skip the cache, attach a session
xencode query "Explain async programming" \
  --model qwen3:4b \
  --no-cache \
  --session demo

# llama.cpp sampling controls
xencode query "Write a haiku" \
  --temperature 0.7 \
  --top-k 40 \
  --min-p 0.05 \
  --max-tokens 256

# Structured output: --grammar takes a GBNF file or string, --json-schema a
# JSON schema. An invalid --json-schema is rejected up front (exit 1) rather
# than silently degrading to a plain completion.
xencode query "List three file formats" --json-schema '{"type":"object"}'
```
Sampling flags are read by a model served by llama.cpp, and — except `--grammar`
and `--mirostat`, which Ollama has no field for — by a model served by Ollama as
`options` on the request; see
[What a request to Ollama carries](#what-a-request-to-ollama-carries). The prompt
alone, with no flags, goes to the configured default model.

#### What `--json-schema` guarantees, and what it does not

The schema is sent to a llama.cpp server as `response_format` of type
`json_schema`, with every `$ref` written out by xencode first — a server that
has to resolve the references itself has been seen to give up and answer
unconstrained. Whatever route the model is on, the answer is then checked here,
against the schema that was asked for:

* it fits: exit 0, and the reply is cached and remembered like any other;
* it does not: the tokens have already been printed, so they are not unsaid, but
  the run ends with `error: the answer does not fit --json-schema: …` and exit 1,
  and nothing bad-tempered gets stored. Under `--format ndjson` the stream ends
  with the `error` event instead of `done`, which is what a script should be
  watching.

Nothing is repaired on the way: a reply wrapped in a code fence, or introduced by
a sentence, is reported rather than trimmed into fitting, because guessing where
the answer begins is the mistake this check exists to avoid. A cached reply is
only reused if it still fits the schema being declared now, so a second run with a
different schema re-asks instead of replaying.

Measured here on `llama-server` build 10809 with a 1.5B model, asking for
`{"answer": "yes"|"no", "reason": string}`: with `--max-tokens 60` the reply was
cut off inside the `reason` string and reported as
`the answer was not JSON: EOF while parsing a string at line 3 column 261`
with exit 1; with `--max-tokens 200` the same request answered
`{"answer": "yes", "reason": "Rust is designed for systems programming…"}` and
exit 0. A schema is not a substitute for a token budget.

On the agent side the same rule runs the other way: each tool call the model
requests is checked against the argument description that tool was offered with,
before the approval prompt is opened, and a call that does not fit is answered
with the mismatch instead of being run. That includes a call whose arguments
arrive as text that stops halfway, which used to be read as a call that asked for
nothing at all. `update_plan` is the one exception, deliberately: its reader has
always accepted bare strings and invented key names because that is what small
models write, and its worst outcome is a list the next call corrects.

#### Which context window a run budgets for

Before a run assembles its context, it asks how large the model's window is,
because that decides how much gets included and how much gets trimmed. For a
model served by llama.cpp the answer comes from the server rather than from a
guess: `xencode query` reads `/props` and takes
`default_generation_settings.n_ctx`, and says what it learned on stderr.

```console
$ xencode query --model llama:dolphin "Name one primary colour in one word."
context: 8192-token window reported by the server at http://localhost:8080
Red
```

That line is the server's number, not a constant. The server above was started
with `-c 8192`; restarting the same model with `-c 4096` and changing nothing in
`~/.xencode/config.json` made the same command print
`context: 4096-token window reported by the server…`. (Measured on
`llama-server` build 10809. Note for anyone parsing `/props` themselves: that
build reports no top-level `n_ctx` — the value lives in
`default_generation_settings`.)

What the server says wins over the model family's usual window, on purpose.
`llama:llama-3.1-8b` is a family documented at 128k, and a server started with
`-c 4096` has 4096 and no more; budgeting for the family would fill context the
model never sees. A number learned from a llama.cpp server is never applied to
another route — a `qwen2.5:7b` or `anthropic:…` run keeps its own answer, since
the reported window describes a process that run is not talking to.

A run whose model is served by Ollama has no `/props` to read, so its window comes
from a different question — what the model's own weights hold — and is then asked
of the server explicitly; see
[What a request to Ollama carries](#what-a-request-to-ollama-carries). What has not
changed: when a server does not answer, or answers without the field, the hardware
profile's default governs as before. In the TUI the number is refreshed at startup,
after a llama.cpp model load or swap, and at the start of each turn — so a server
restarted outside xencode takes effect from the next turn rather than the one
already in flight.

#### What a request to Ollama carries

The sampling flags and `--json-schema` are not llama.cpp-only: a run whose model is
an Ollama tag sends the same intentions in the words that server reads. `xencode
query` asks `/api/show` about the model before it asks for an answer, and builds
`/api/chat` out of what came back.

| Intention | In the Ollama request | Where the number comes from |
| --- | --- | --- |
| Context window | `options.num_ctx` | the smaller of what this machine's profile can serve and what the model's weights hold (`<architecture>.context_length` in `/api/show`) |
| Structured output | `format`, holding the whole JSON schema | `--json-schema` |
| Whether the model may think first | `think` | `ollama_reasoning`, and only sent as `true` when `/api/show` lists `thinking` |
| How long the model stays loaded | `keep_alive` | `ollama_keep_alive` |

Two settings choose the last two:

```bash
xencode config set ollama_reasoning off    # "think": false, even for a thinking model
xencode config set ollama_reasoning on     # "think": true, for a model that says it can
xencode config set ollama_reasoning auto   # no "think" field; the model's own default
xencode config set ollama_keep_alive 10m   # "keep_alive": "10m"
xencode config set ollama_keep_alive 0     # unload as soon as this answer is done
xencode config set ollama_keep_alive ""    # back to the server's own five minutes
```

`ollama_reasoning` takes only `auto`, `on` and `off`. Ollama has no way to ask for a
thinking *budget*, so a number there is refused with a message pointing at
`llama_cpp_reasoning`, which does have one. `ollama_keep_alive` has to contain a
digit, because `10m` and `30s` are the server's words, not ours.

What is said when the server cannot do what was asked is said in the transcript, as
`ℹ️` system lines in the TUI and `context:` lines from `xencode query`. The first and
third lines below came off this machine today; the middle one is the same sentence
with numbers this machine cannot produce, because the profile here asks for less
than either model's weights hold:

```
context: asking Ollama for a 8192-token window
context: asked Ollama for a 40960-token window; this model holds 32768, so that is what it was given
context: this model was asked to reason first but does not say it can, so nothing was asked of it
```

Three things about this route were measured on this machine against Ollama 0.34.4,
serving `qwen3-1.7b` and a `qwen25-0.5b` GGUF at `Q4_K_M`, and they are why the
shape above is what it is:

- **The window has to be asked for on every request.** A request that leaves
  `options.num_ctx` out is a different model configuration to Ollama: the server
  unloads the running model and reloads it at its own default. Starting from a model
  loaded at `-c 8192` by xencode, one plain `/api/chat` with no options logged
  `msg="unload completed"` and `starting llama-server … -c 4096`. So the window is
  decided once per session and carried by every request the session makes — chat
  turns, a one-off ask from the TUI, `/review`, and the eval judge — rather than
  being left to the server. That default is sized by free VRAM here
  (`total_vram="1.9 GiB" default_num_ctx=4096`), not by the model.
- **`think: true` is a hard failure when the model cannot do it.** Answering 400 with
  `"…" does not support thinking` is worse than answering without a preamble, so the
  ask is withdrawn when `/api/show` does not list `thinking`, and said out loud. A
  `false` is always accepted.
- **A schema and a tool list together are decided by the schema.** Asking a 1.7B
  model for JSON *and* offering it tools, it answers with valid JSON matching the
  schema and `tool_calls` stays null. `grammar` is not sent at all: Ollama's
  `/api/chat` has no field for a GBNF grammar alongside `format`, and a `mirostat`
  sampling mode is refused outright (`invalid option provided`), so those two
  llama.cpp controls stay llama.cpp-only. Asking for either on an Ollama model says
  so as it is dropped, rather than answering as if it had been honored:

  ```
  sampling: --grammar was not sent — Ollama takes a JSON schema in `format`, not a GBNF grammar
  sampling: --mirostat was not sent — a running Ollama answers it as `invalid option provided`
  ```

`xencode eval`'s ranking judge asks on the same terms, since it dials the same
server as the run it is judging.

#### Turn routing

A `model_profiles` entry can carry `for_task`, and `model_routing` decides whether
anything acts on it. Off — the default, and the value a config written before the
key existed gets — every profile applies only by hand in the Custom Models panel.
On, a turn is read for what it says about the code and the first profile marked for
that reading runs it:

```bash
xencode config set model_routing true
xencode config set model_routing false   # back to profiles applying by hand only
```

The reading has two values, because those are the two this project measured:
`bugfix` for a prompt carrying words for broken code (`fix`, `fails`, `crash`,
`broken` and similar, matched on whole words) and `general` for everything else,
which is genuinely wider than it sounds — a rename, a question and an edit as large
as any all read as `general`. That is what makes it the right mark for a cheaper
reading model and what makes it a second default. A mark naming a reading this
version does not have (`refactor`, `feature`, a misspelling) is kept as written and
matches nothing, so the file still loads and the profile stays applicable by hand.

Two things are refused rather than done quietly:

- **A llama.cpp swap.** A self-started `llama-server` holds one model at a time, so a
  profile that names a different one is not applied; the chat prints
  `profile <name> was not used: … a running llama.cpp server holds one model at a
  time` and the turn goes to the model already chosen. Moving from one llama.cpp
  model to another is a panel action, not something a word in a prompt should do to
  the VRAM budget.
- **Sampling numbers on a route that has nowhere for them.** `temperature` and
  `max_tokens` reach only llama.cpp, so a profile moving an Ollama turn says so:
  `which sets temperature 0.2 and 64 tokens at most — numbers that route has no
  place for, so the model's own defaults answer`.

`xencode query` follows the same rule, and `-m/--model` outranks it — naming a model
is an instruction, not a suggestion, so a run with `-m` prints nothing about
profiles. Otherwise the chosen model appears in the `start` event like any other
model, and the reason rides on standard error, out of the parseable stream:

```
profile: reader took this turn on q25local:latest, which sets temperature 0.2 and 64 tokens at most — numbers that route has no place for, so the model's own defaults answer — the prompt says fix, fails
```

That line came off this machine (Ollama 0.34.4, MX250 with 1,045 MiB of its 2,048
MiB already used) with the default model `q3local:latest` and a profile named
`reader` on `q25local:latest`, asked a prompt that says the tests fail. What the
swap costs here, measured with `num_predict: 1` so the answer itself was
instant: loading a model that was not resident took 1.7 s, loading a second model
while the first stayed resident took 3.8 s and `/api/ps` then listed only one of
them, and bringing the first back afterwards took 8.0 s — against 1.2–1.8 s for the
same turn once the model was loaded. That is the reason a rule is off by default and
the reason a llama.cpp move is refused.

#### Which hardware profile the budget spends against

The window says how much room there is. A second setting says how much of it a run
is allowed to fill with project context, how many retrieved files may ride along,
and how large each one may be — and until now that setting was fixed in the code at
the middle of its three choices, on every machine, whatever it was running on.

It is now decided per session and said out loud. `xencode query` prints the profile
it settled on and what settled it, as the first line it writes (the lines after it
depend on the model and are covered above and below):

```console
$ xencode query "what does this project do" -m llama:dolphin
hardware: BALANCED profile from 15.4 GiB of RAM
```

The memory figure is this machine's own, read from `MemTotal` in `/proc/meminfo`
(`16141080 kB`, which is 15.4 GiB). Below 8 GiB a run budgets as LOW, from 8 to 24
GiB as BALANCED, and 24 GiB and above as HIGH. Those boundaries are a reasoned
choice about how much memory a model and its context need together, not a
measurement, which is why the setting exists:

```console
$ xencode config set hardware_profile low
set hardware_profile = low
$ xencode query "hi" -m llama:dolphin
hardware: LOW profile set in config
```

A named profile is obeyed exactly, including when it is a worse guess than the
machine's — someone who knows their model's size knows more than a memory reading
does. The two failure modes are reported rather than absorbed. A word that is not a
profile is refused where it is set (`hardware_profile must be "auto", "low",
"balanced" or "high", not "banlanced"`), and one already sitting in the file is
named by the run that ignores it:

```console
hardware: BALANCED profile from 15.4 GiB of RAM, though the config said "banlanced", which is not a profile
```

And a machine that reports nothing — no `/proc/meminfo`, or one with no readable
`MemTotal` — gets the profile it got before any of this existed, with the reason
saying so. That case was checked by running the command with `/proc/meminfo` bound
to an empty file: `hardware: BALANCED profile this machine reported no memory size,
so the default applies`.

What the profile is not: a statement about the graphics card. Nothing here reads
`nvidia-smi`, because the window a run fills is the one asked of the server at
startup, and a card sitting in the machine may not be what serves the model. In
the TUI the same decision appears in `/ctx kv`, where the profile line carries its
reason and the retrieval it has decided on —
`🗂 Profile BALANCED (from 15.4 GiB of RAM) — ctx 8192 · utilization 75% ·
retrieval top-5 at 16000 characters each, from the profile's own numbers, with no
prompt measured yet` — and where `hardware_profile` in the config says which of
those numbers came from the machine and which from a word someone set.

The same profile now also decides how a server this program starts is launched:
`xencode llamacpp start` and the TUI's auto-start pass a preset that matches the
profile — `--ctx-size` at the window above, plus flash attention, KV-cache
quantization and batch size — and then ask the running server what it came up
with, so a flag that never took effect is reported instead of assumed. See
[`xencode llamacpp <action>`](#xencode-llamacpp-action).

#### What a prompt actually costs, counted by the model

Knowing the window is one thing; knowing what the prompt is worth inside it is
another. Until now the only figure was arithmetic on character counts — one
token per four characters of prose, one per three of code. `xencode query` now
also asks the llama.cpp server that is about to read the prompt what that prompt
counts as, through the server's own `/tokenize` endpoint, and prints the two
side by side:

```console
$ xencode query "Which module owns the caps?"
hardware: BALANCED profile from 15.4 GiB of RAM
retrieval: up to 5 files, 16000 characters each (character arithmetic, not measured)
read as general work — no word for broken code in the prompt, so the weights are used as they stand
context: 8192-token window reported by the server at http://127.0.0.1:8080
context: 1311 tokens counted by the server, 1313 by character arithmetic
```

The `retrieval:` line is said out loud because a command that runs once has no
earlier turn to size itself from: it uses the profile's own numbers and does not
pretend they were measured. An interactive session scales them instead — see
[How much retrieval gets](#how-much-retrieval-gets) below.

The `read as …` line says which retrieval weights the prompt was scored with, and
the words that decided it. A turn whose prompt says something is broken is read as
`bugfix work`, and a file holding a test whose name uses those same words is then
scored above one that merely shares a symbol name with the question; every other
prompt is `general work`, which changes no weight. The reading is a whole-word look
at the prompt and nothing else — no model call, no classifier — and asking for new
code or for a rename reads as `general`, because both were measured on this
repository's gold set at 0.000 and price no weight.

Neither number is simply the better one, which is why both are shown. The counted
number describes the text the turn is made of and nothing else: the chat
template's per-message markers are added by the server afterwards, so the count
is a floor. The character figure is what the trimmable parts were fitted to, and
it stops there — the question and any attached files are the parts xencode is not
allowed to trim, so a turn that overflows reports less than it costs. Started
against a server restarted with `-c 512`, a long repeated question gave three
numbers for one prompt:

```console
context: 566 tokens counted by the server, 384 by character arithmetic
warning: the prompt was counted at 566 tokens, which is more than the 512-token window this server is running with
error: Query failed: API error: llama.cpp 400 - {"error":{"code":400,"message":"request (579 tokens) exceeds the available context size (512 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":579,"n_ctx":512}}
```

All three are true: 384 is what the budgeter spent, 566 is what the prompt's text
is worth in this model's vocabulary, 579 is what the request cost once the
template's framing was added. The warning says "counted at 566" rather than
promising the request fails, because a count of the text cannot see the framing —
but it is the first moment this overflow can be named at all, since the server's
own refusal arrives only after the request.

The count is asked once per turn, of the server that turn goes to, and only when
the model is served by llama.cpp; an Ollama run has no counting endpoint and keeps
the character figure. A server that gives no count and a server that was never
asked are different answers, and each prints its own line rather than one
sentence covering both:

```console
context: 1313 tokens by character arithmetic (this server gave no count for /tokenize)
context: 1313 tokens by character arithmetic (no count could be asked of this server: llama.cpp API error: error sending request for url (http://127.0.0.1:8096/tokenize))
```

The first is a server that answered the request and refused it — including a build
that has never heard of the field the request uses — and the second is one that was
not reachable at all. Neither is reported as a prompt of zero tokens. Both lines
were run rather than reasoned about: the refusal against a server replying `404` to
every request, the unreachable one against a port nothing listens on.

In the TUI the same question is asked in the background, so no turn waits for it:
a `/ctx` preview prints the count next to its estimate, and a real turn stays
silent unless what was counted does not fit the window the server reported, in
which case a line says what was counted and against what.

What the counting showed about the estimator, measured on files from this
repository: the prose divisor is close, within about 10% on the documents tried
and in both directions. The code divisor is not — counting a token per three
characters priced Rust files at +26% to +40% above what this model's vocabulary
needs, because this vocabulary reads Rust at about four characters per token.
Nothing has been retuned on the strength of one vocabulary: a divisor fitted to
one model is wrong for the next, and the answer to a wrong divisor is the count
above rather than a better guess. The retrieval caps now read the server's own
number instead of the divisor — see below.

#### How much retrieval gets

Retrieval has to fit in what the rest of the prompt leaves, and until now it did
not look: the hardware profile fixed both numbers, five files on a balanced
machine and 16,000 characters of each, whatever the conversation already held.

An interactive session now scales those two numbers from what the server said the
last prompt cost. What counts as overhead is everything in a prompt that is not a
retrieved file body — the system head, the guidelines files, project state, the
git summary, the conversation so far, and the framing the chat template adds — and
its size is the server's own `prompt_tokens` for the request, with the retrieved
share taken out by proportion of characters. A turn that cost 5,766 tokens of
which 18,197 of 23,003 characters were file bodies leaves 1,204 tokens of
overhead; counting those same parts one at a time gave 1,166, so the split is
three percent high, where the character divisor it replaces was 32 % high on the
retrieved half alone (6,066 predicted for bodies the server read as 4,587).

That figure is averaged rather than taken from the last turn, and the average is
only ever reported rounded to a multiple of 256 tokens. Both are there for the
same reason: the caps are chosen before the next prompt is built, so a number that
moved with every reply would make retrieval oscillate between turns. A new reading
moves the average about a quarter of the way toward itself, and drift smaller than
a step changes nothing at all.

The room that is left then buys files at 512 tokens each: the file count is that
many tokens at a time, held between one and eight, and the characters kept from
each file are the remaining space divided at three characters per token, held
between 1,536 and 24,000. Space below the eight-file ceiling buys *more files* at
roughly the smallest excerpt worth sending; only past the ceiling does it buy
longer ones. So the character figure can go down while the budget goes up, and
what is guaranteed instead is that the whole retrieval cannot exceed the room it
was given.

In `/ctx kv`, and in the header of a `/ctx <query>` preview, this is visible in
words:

```console
🗂 Profile BALANCED (from 15.4 GiB of RAM) — ctx 8192 · utilization 75% · retrieval top-5 at 16000 characters each, from the profile's own numbers, with no prompt measured yet
🗂 Profile BALANCED (from 15.4 GiB of RAM) — ctx 8192 · utilization 75% · retrieval top-6 at 1536 characters each, from 3072 tokens of prompt the server measured
🎯 Retrieval (BALANCED profile, top-6 of 6, 1664 characters each, room left after 2816 tokens of prompt):
```

The first two are one session in this repository against a server started with
`--ctx-size 8192`, before and after its first real turn: 6144 tokens of fill target
minus a 3072-token prompt leaves 3072 for retrieval, which is six files, and each
file at the 1,536-character floor. That turn's metrics row read `BALANCED — prompt
3018 · cached 0 · reuse 0%`, and the turn after it reported 2841 of 5611 tokens read
from the cache. The third line is a `/ctx <query>` preview from an earlier session of
the same build, and its figures differ for the same reason the others do not repeat:
a prompt measured in another conversation costs a different amount.

A streamed answer carries no usage unless the request asks for it, and until now none
of them did, so the figures above are the first ones that came from the server rather
than from xencode's own arithmetic.

Because the overhead includes the conversation, a long chat narrows retrieval as
it fills up. That is the intended direction — the room is the room — but it is
also counted twice, since history has its own trimming, so a long conversation
errs toward fetching less rather than overflowing.

Neither scaling applies where there is nothing to average: `xencode query` runs
once and says so on its `retrieval:` line, and a model served by Ollama has no
usage reported on a stream, so that route keeps the profile's numbers.

#### What a small window is given instead: the repo map

A 4096-token model has no room to be shown a few files and told the rest of the
project exists somewhere. When a turn's budget is that tight — 2 457 tokens, the
figure a `Low` machine fills to — the prompt carries a map of names just before
the file bodies. Its rows are files that declare something, ranked by how near
they sit to the files the turn is already about (dependency edges, two hops at
most), the most depended-on first, three declared names each and `+N more` past
that. It never costs more than 300 tokens, a row is admitted whole or not at all
so a path is never cut in half, and its last line says how many named files went
unlisted:

```text
Repo map — files nearest the current work, most depended-on first, names only:
  • rust/crates/xencode-context-rs/src/index.rs [the current work]: FileEntry, FilesIndex, Manifest, +9 more
  • rust/crates/xencode-context-rs/src/symbols.rs [the current work]: DepEdge, PerFileSymbols, RegexCache, +35 more
  • rust/crates/xencode-context-rs/src/gitinfo.rs [the current work]: DiffFile, GitInfo, changed_paths_between, +13 more
  • rust/crates/xencode-context-rs/src/retrieve.rs [1 hop(s) from the current work]: RetrievalIndex, RetrieveOptions, RetrievedFile, +18 more
  • rust/crates/xencode-context-rs/src/init.rs [1 hop(s) from the current work]: InitSummary, ContextError, auto_gitignore_index_dir, +14 more
  • rust/crates/xencode-context-rs/src/advise.rs [1 hop(s) from the current work]: Advice, AdviceKind, advise, +9 more
  • rust/crates/xencode-context-rs/src/context.rs [1 hop(s) from the current work]: ChatAssembly, ChatInput, ChatTurn, +23 more
  • rust/crates/xencode-context-rs/src/embed.rs [1 hop(s) from the current work]: Bm25, build, file, +6 more
  … +126 more files in the index, not listed
```

That is the map this repository's index produces for the query
`where is the login handler?`, printed by
`cargo test -p xencode-context-rs --test repo_map_live -- --ignored --nocapture`,
and the `/ctx` preview summarises the tier it put in the prompt:

```console
[CTX]🗺 repo map tier: 8 files named in 283 tokens
```

What the tier buys is a name the model can ask for. Retrieval on that budget
sends three file bodies, so on a question whose answer is not one of them the
model either guesses or says it does not know where the code is; the map is what
lets it answer "read `rust/crates/…/auth.rs`" instead. Over the 25 gold queries
at Low's three-file budget, the bodies alone named the expected file in 7 and
the bodies plus the map named it in 9, for a median 283 tokens out of 2 457.

A wider budget leaves the tier out, because there the bodies themselves are the
orientation — which is why the line above does not appear in a `/ctx` preview on
this machine: it has enough memory to be a `Balanced` one.

#### Repeatable answers: `--seed`, and what it does not cover

`--seed <n>` sends the sampler seed to llama.cpp. Without it — and without
`--temperature 0` — the server draws a fresh seed for every request, so the same
prompt answers differently. Measured on this machine against `llama-server`
0.4.0-dev (build 10809) with a 1.5B model at `--temperature 1.5`: three runs with
`--seed 42` gave the same answer three times, and three runs with no seed gave
three different answers.

A seed pins the draws, not the whole run. Two things outside it can still change
the answer, and both bit during the measurement above:

- **The prompt has to be the same.** `xencode query` carries recent turns out of
  the shared conversation memory (`conversation_memory.json` under the config
  directory) into every request, so consecutive runs are not asking the same
  question. Set `memory_enabled: false`, or point `XCODE_CONFIG_DIR` at a fresh
  directory, when a run has to be reproducible.
- **So does the server's cache state.** llama.cpp keeps the prompt prefix it has
  already evaluated; the first request after the model loads re-runs the whole
  prompt and later ones continue from the cached part. That changes the arithmetic
  slightly, and at a high temperature a slightly different logit can flip the
  sampled token. Reproduce from the same state — or restart the server — before
  calling a difference a regression.

The same settings exist as config defaults (`llama_cpp_seed`,
`llama_cpp_temperature`), and the Settings panel has a **Llama Seed** row. The TUI
writes what each llama.cpp turn was asked to sample at into
`cache/metrics.jsonl`, so `/cost` can say how much of the recorded history could
actually be produced again rather than asserting it.

A seed pins what the sampler draws, not what the run did. To have a whole turn
come back the same, record it and replay it — see `xencode replay` below.

#### `xencode query --format ndjson` — the answer as events a script can read

The default format prints the model's words as they arrive, which is what
someone reading a terminal wants. `--format ndjson` prints one JSON object per
line instead, so a program can act on the answer while it is still coming:

```bash
xencode query "Summarise this crate" --format ndjson | while IFS= read -r line; do
    printf '%s' "$line" | jq -j '.text // empty'
done
```

There is no `--stream` flag: `--format ndjson` always streams, one line out as
each piece arrives, and the plain format already prints words as they land.

Every line carries `"v": 1`, and the keys of an object are written in
alphabetical order, so the same event always renders as the same bytes. Two
rules make the version worth having: a consumer that meets a `v` above the one
it understands must stop rather than guess, and a consumer that meets a known
`v` with an unfamiliar `"type"` must skip that line and keep reading.

| Line | Fields | Written when |
| --- | --- | --- |
| `start` | `model`, `provider`, `source`, `session` | Once, before any other line, as soon as the model and route are settled. `provider` is the client that was dialed (`ollama`, `llamacpp`, `remote`, `openrouter`, `qwen`, `anthropic`, `google_gemini`); `source` is `local` or `cloud`, spelled as the rows in `cache/metrics.jsonl` spell it; `session` is the conversation id, or `null` when memory is off or `--session` named one that does not exist. |
| `token` | `text` | Once per piece as the answer arrives. Where the pieces stop is the network's choice, not the model's: a piece can end mid-word or mid-character-boundary text. |
| `done` | `answer`, `cached`, `elapsed_ms`, `tokens_generated`, `tokens_per_second` | Once, last, on a run that produced an answer. `elapsed_ms` is this command's own wall clock. `tokens_generated` and `tokens_per_second` are `null` unless the route reported counts — a llama.cpp server only does so when its stream ends with a usage chunk, and the cached path never has any, because nothing was generated. |
| `error` | `message` | Once, last, instead of `done`. A run that failed before dialing the model (an invalid `--json-schema`, say) still ends with this line, so exactly one line per run says how it ended. |

**The one property worth building on:** the `text` of every `token` line,
concatenated in order, is exactly the `answer` in the `done` line — including
for a cached answer, which is why a cache hit is written as a `token` line as
well. A real run against a local llama.cpp server, 49 lines:

```jsonc
{"model":"llamacpp:/home/sree/.lmstudio/models/bartowski/Dolphin3.0-Qwen2.5-1.5B-GGUF/Dolphin3.0-Qwen2.5-1.5B-Q4_K_M.gguf","provider":"llamacpp","session":null,"source":"local","type":"start","v":1}
{"text":"Prime","type":"token","v":1}
{"text":" numbers","type":"token","v":1}
// … 45 more token lines, 47 in all …
{"answer":"Prime numbers are numbers that have exactly two distinct positive divisors: 1 and themselves. Below are three prime numbers:\n\n1. 2\n2. 3\n3. 5\n\nThese are the first three prime numbers.","cached":false,"elapsed_ms":4373,"tokens_generated":null,"tokens_per_second":null,"type":"done","v":1}
```

A failure writes the same shape, and the human-readable copy stays on stderr
where a parser will not trip over it. This is the whole stdout of a run whose
model server was not listening:

```text
$ xencode query "hi" --format ndjson; echo "exit=$?"
{"model":"qwen2.5:7b","provider":"ollama","session":null,"source":"local","type":"start","v":1}
{"message":"Query failed: network error: error sending request for url (http://127.0.0.1:9/api/chat)","type":"error","v":1}
exit=1
```

**A shell trap that is worth knowing before you write the consumer.** Reading a
token with `$(…)` throws the answer's line breaks away, because command
substitution strips trailing newlines — an answer of `1. 2\n2. 3\n3. 5` prints
as `1. 22. 33. 5`. Copy the bytes instead:

```bash
printf '%s' "$line" | jq -j '.text'      # right: no added or stripped newline
piece=$(printf '%s' "$line" | jq -r '.text')   # wrong for this stream
```

This format has no `tool` line. `xencode query` sends one request and does not
run the agent loop, so there is no tool call for it to report; tools are run by
the TUI's chat and by the WebSocket server.

### `xencode analyze <path> [--format text|json]`
Analyze a file or directory for code issues and vulnerabilities. Image
files take the intake path: format, dimensions, and byte size are reported
(`--format json` returns the `ImageMeta` for a single image).

Directory mode walks the full tree (junk dirs like `target/` skipped),
analyzes every non-image file, and reports skip counts. Directory JSON is
a documented object — single-file shapes are unchanged:

```bash
xencode analyze ./src
xencode analyze ./assets/logo.png --format json
xencode analyze ./src --format json | jq '{issues: (.issues|length), images: (.images|length), skipped}'
```

#### `--runtime`

A different question with a different engine, so it short-circuits the rest of
`analyze`. Reports the async and concurrency mistakes that compile cleanly,
produce no warning, and cost a production freeze or a silent task death:

| finding | why it matters |
|---|---|
| `std::fs::*`, `std::thread::sleep` inside an `async fn` | blocks the thread the runtime is running other work on, so every other task on that worker stops until it returns |
| `mpsc::unbounded_channel()` | nothing bounds the queue, so a producer faster than its consumer grows it until the process is killed |
| `tokio::spawn(…)` as a bare statement | the handle is dropped, so a panic or cancellation inside the task stops silently |

```bash
xencode analyze ./rust/crates --runtime
xencode analyze ./rust/crates --runtime --format json | jq '.findings[] | select(.test_only == false)'
```

Three things worth knowing before you trust it:

- **It needs the `ast-grep` binary**, and says so when it is missing — including
  that nothing is known about the code, and a non-zero exit. A missing engine
  never prints "none found", because "none found" and "did not run" must not
  read alike.
- **It deliberately does not duplicate clippy.** `await_holding_lock` already
  reports a lock guard held across an `.await`, correctly and with a suggestion,
  so that class is not reimplemented here. What this adds is the four classes
  clippy is silent about, all verified against the same file.
- **A finding in a `#[cfg(test)]` module is reported and labelled, not hidden.**
  Test findings sort after shipped ones, so a report about your code leads with
  your code. A test that blocks a reactor is still worth knowing about.

A blocking call inside `tokio::task::spawn_blocking` is **not** reported: that is
the correct place for one, and a lexical "is this inside an `async fn`" cannot
tell it apart from sleeping on the reactor.

### `xencode interop [--agent NAME]... [--timeout SECS] [--out PATH] [--format text|json]`

The `AR-1` probe: launch every installed coding-agent CLI headless on a read-only task in a
scratch git repository, and record what came back. The scratch directory is created in a temp
location and removed afterwards, so the probe never touches your workspace.

```bash
xencode interop
xencode interop --agent codex --agent cline
xencode interop --out probe.json --format json
```

The task asks one agent to read one file in the fixture and reply with the word in it. It asks
for no changes, so an agent that follows it modifies nothing — which also means an agent with
no account stops at its authentication check, and **that refusal is recorded as an
observation**, not as a failure of the probe.

What is captured per agent: the binary, its version, the argv executed, the working
directory, exit code, wall-clock, stdout and stderr, any event lines found on a
machine-readable stream, a session id if one appeared, and whether the run stopped on an
authentication check or showed a permission signal.

Three things about how to read the output:

- **A non-zero exit is not a broken probe.** `claude`, `gemini` and `crush` on a machine
  without accounts exit non-zero having refused to spend anything. That is the answer for
  those rows, and it is why the report distinguishes it from an agent that ran and failed.
- **Absence is stated, not left blank.** `kilo` and `agy` are named by the plan and are not
  installed here, so every claim about them stays unverified and the report says so.
- **Anything the run could not answer is listed** under "still unanswered", so the gaps are
  part of the output rather than something you have to notice are missing.

`--out` writes the full JSON report through the same atomic write as every other file, so a
report half-written is not a thing that can happen.

### `xencode scan [path] [--hidden] [--max-depth N] [--format text|json]`
List workspace entries (kind, size, path) as TSV or JSON.

```bash
xencode scan . --max-depth 2
xencode scan . --format json | jq '.[].path'
```

### `xencode models <action>`
Local model management.

```bash
xencode models list           # Ollama models + models served by llama.cpp
xencode models health <name>  # Check one model's health
xencode models default        # Show the smart-selected default
xencode models advice         # Which GGUF this machine can hold, and where to get it
```

### `xencode llamacpp <action>`
llama.cpp server management: `status`, `start`, `stop`, `load`, `unload`,
`list`, `set-path`.

```bash
xencode llamacpp status
xencode llamacpp start --model mymodel.gguf --port 8080 [--exec /path/to/llama-server]
xencode llamacpp set-path ~/models/mymodel.gguf   # persist the GGUF path
xencode llamacpp list                             # models on a running server
xencode llamacpp stop
```

`start` launches `llama-server` with a set of flags chosen for this machine's
memory profile (`auto`, `low`, `balanced` or `high` — see `hardware_profile`
below), and then asks the running server what it is actually serving, printing
both lines:

```
  flags: --flash-attn on --cache-type-k q8_0 --cache-type-v q4_0 --ctx-size 4096 --batch-size 512 --parallel 1
LOW preset: 4096 tokens of context in 1 slot(s), as asked
```

The check can only see what `/props` reports, which is the context window and
the slot count; the cache quantization and batch size are passed but not
reported back, so they are not claimed. Flags in `llama_cpp_args` are appended
after the profile's own, and `llama-server` takes the later of a repeated flag,
so overriding one is supported and shows up as a disagreement: setting
`llama_cpp_args` to `--ctx-size 2048` with the `low` profile printed
`LOW preset: 2048 tokens of context in 1 slot(s), not the 4096 tokens of
context in 1 slot(s) asked for — later flags win, so check llama_cpp_args`.
The same check is available in the TUI: it is the status line under the model
list (`m`).

**What the machine can hold is asked before the server is started**, from the
same three readings `xencode hw probe` uses: the device list off the server
binary, the model geometry out of the `.gguf` header, and the free memory out of
`/proc/meminfo`. Three outcomes, and a launch that is fine says nothing at all:

```text
# nothing on this machine can hold the weights — no server is started:
error: this model is 20480 MiB and the largest place to put it here is 10406 MiB
of system memory free, so no window makes it servable. A smaller quantization of
the same model is the usual answer: `xencode hw probe --model <file>` …

# the window asked for does not fit on any device — it is started shorter:
131072 tokens needs about 4018 MiB once the cache this launch carries is counted,
which is more than any device here can hold; the biggest is NVIDIA GeForce MX250
with 1677 MiB of its own memory free. Starting at 46080 tokens instead.
```

The cache is priced at the quantization the command line really carries — the
last `--cache-type-k`/`--cache-type-v` on it, which is the one the server obeys,
and 16-bit when nothing names one, since that is `llama-server`'s own default.

**A server that dies is reported as having died, with what it said.** Its error
output is kept, so a launch that fails says why in the first seconds instead of
after the whole deadline:

```text
error: llama-server stopped before it answered: exited with code 1
the server's own last 6 lines:
  0.00.087.316 E llama_model_load: error loading model: llama_model_loader: failed to load model from /tmp/nope.gguf
  …
  0.00.087.584 E srv  llama_server: exiting due to model loading error
```

Readiness is asked strictly: `llama-server` answers `/health` with 503
`Loading model` while it loads and 200 `{"status":"ok"}` when it is done, and the
first of those is not the second. That matters because the failure this path
exists for happens *during* the load: a key-value cache the card cannot allocate.
When what the server said was about memory, xencode starts it again at half the
window and says that it did —

```text
  llama-server ran out of memory at 46080 tokens; starting again at 22528.
```

— and stops after that one retry, naming what is actually there to try next
(`xencode hw probe --model <file>`, `xencode colab up`, or
`xencode config set remote_base_url <url>`). A server killed by memory is not a
missing file, so a download would fix nothing and is not suggested. All three of
those blocks above are output from real launches on this laptop, including the
restart at 22528, which came up and confirmed its own window.

**A model file that is not on disk is fetched, if you say from where.**
`llama_cpp_model_url` is the HTTPS URL of the `.gguf` itself — not of a page that
links it. `llamacpp start` with a missing file and no URL stops and prints the
`xencode config set llama_cpp_model_url <url>` line. With one, the disk is
checked before a single byte is written, using the file's own advertised size
minus whatever a stopped attempt already downloaded:

```text
error: the model download did not finish: the file is 18.5 GiB and the disk holding /boot/xencode-l10/model.gguf has 765.9 MiB free
```

Bytes land in `<path>.part` and move to the real path only once complete, so a
file that exists is always a whole one. Interrupting mid-download and re-running
continues from the last byte rather than restarting (measured here: a killed
468.6 MiB fetch resumed at 344.3 MiB and finished):

```text
l10-resume-test.gguf is not there yet, but a stopped download is: 344.3 MiB of its bytes are on disk.
  344.3 MiB of 468.6 MiB (73 %)
  …
  model ready: 468.6 MiB
  344.3 MiB of that came from the bytes the earlier attempt had already fetched.
```

A server that ignores range requests gets its partial file discarded and the
whole file re-fetched, with a note saying so. The TUI's auto-start fetches the
same way and shows a `⬇` progress line over the body while it runs.

**Pinning the bytes.** `llama_cpp_model_sha256` holds a digest you supply — from
`xencode models advice` below, or from wherever you chose the URL. The bytes are
hashed as they arrive and compared at the end, including whatever an interrupted
attempt had already written, so a resumed download is checked as a whole file. A
file that does not match is thrown away instead of being moved into place, and a
server is never started on it:

```text
error: refusing to start: /tmp/lf7e/model.gguf hashes to 74a4da8c, not the 00000000 this configuration expects. The file is not the one that was pinned — delete it and start again to fetch it fresh, or set llama_cpp_model_sha256 to the checksum you now want.
```

That check is one read of the file, and it happens on every launch. The TUI's
model panel does it too, on the `l` load of a file this machine holds (`m` opens
the panel); a load that names a model alias is left to the server, because there
are no bytes here to look at and calling a name `verified` would be a lie. A file
already on disk that nothing is pinned to is not a failure: the panel says
`unsigned` instead of pretending, and starts anyway.

**What the checksum proves, and what it does not.** It proves the transfer was
faithful to a number that came from outside the transfer. It does not prove who
the bytes came from: a digest taken from the same server that is serving the file
is circular, because whatever that host answers would be declared genuine. The
pin therefore lives in a dated table assembled separately, and a file xencode
fetched gets a `<path>.provenance.json` note beside it recording the size, the
checksum, the repository revision the host named on the way to the bytes, and
whether an outside digest was matched. That note is xencode's own record of what
it saw, not a certificate — which is why an unpinned file reads `unsigned` and
not `verified`.

```bash
xencode config set llama_cpp_model_sha256 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
xencode config set llama_cpp_model_sha256 ""   # stop checking this file
```

A value is accepted only as 64 hexadecimal characters, with an optional
`sha256-` prefix and any case; anything else is refused by `config set` with the
reason.

### `xencode models advice`

Which GGUF this machine can actually serve, taken from a table of sizes,
checksums and pinned addresses rather than from a list of model names baked into
the binary.

```bash
xencode models advice
```

```text
room:     10.1 GiB (10323 MiB of system memory free)
advice:   checked 2026-09-27
          0 days old
from:     the table shipped with xencode
tier:     large
          Qwen3 14B, 4-bit
            size     8.4 GiB
            url      https://huggingface.co/unsloth/Qwen3-14B-GGUF/resolve/a04a82c4739b3ef5fa6da7d10261db2c67dd1985/Qwen3-14B-Q4_K_M.gguf
            sha256   5eaa0870bd81ed3b58a630a271234cfa604e43ffb3a19cd68e54a80dd9d52a66

to serve one of these:
  xencode config set llama_cpp_model_url "…"
  xencode config set llama_cpp_model_sha256 "…"
  xencode config set llama_cpp_model_path "/tmp/lf7e/.xencode/models/Qwen3-14B-Q4_K_M.gguf"
  xencode llamacpp start
```

`room` is the largest single memory pool on this machine — a GPU's own VRAM when
one can hold it, otherwise system RAM — measured now, not estimated from a model
of the product line. Every `url` is a `/resolve/<revision>/` address, so the
bytes behind it cannot be changed underneath by someone pushing to the
repository's default branch.

The table ages in public: `advice` prints the date it was checked and how many
days old that is, and answers from a table older than six months say so. Replace
it by writing your own at `~/.xencode/model_advice.json` (same shape: `as_of`,
`ollama_preference`, `tiers[]` of `max_file_bytes` and `gguf` entries) — xencode
then reads yours and the `from:` line says which file it answered from. A user
file that fails to parse is refused out loud and the shipped table is used
instead, because a typo in a JSON file should not cost the answer. The same
preference list decides which installed Ollama tag `xencode models default`
picks.

**How much the model may think** is a launch setting too. `llama_cpp_reasoning`
takes `auto` (leave it to the model), `off`, or a token budget as a plain number:

```bash
xencode config set llama_cpp_reasoning off    # --reasoning off
xencode config set llama_cpp_reasoning 256    # --reasoning-budget 256
xencode config set llama_cpp_reasoning auto   # no flag at all
```

Anything else — a word that is not `off`/`auto`, a negative or fractional
number — is refused by `config set`, and a value typed into the JSON file by
hand is refused the same way by `llamacpp start` (the command stops rather than
starting a server that thinks). The TUI's auto-start reports such a value and
boots without the flag, because a start nobody is watching should not stall.

Two limits worth knowing, both measured against `llama-server` b10809 with
`unsloth/Qwen3-0.6B-GGUF` at `Q4_K_M`:

- These are launch flags only. Sending `reasoning_budget`, `reasoning_effort` or
  `chat_template_kwargs` per request is accepted with HTTP 200 and then ignored
  — three replies with different per-request values were byte-for-byte the same
  amount of thinking, so this product does not pretend to control them there.
- `/props` does not report the setting, so the `as asked` line above cannot
  confirm it. A too-small budget also does not fail loudly: it answers from a
  half-finished plan. With a budget of 32 the model ran out of thinking halfway
  through the sheep question and answered 8; left unrestricted it answered 8 as
  well; with `off` it answered 9 in 32 tokens. One question, one small model —
  a reason to treat a budget as a speed and length control, not a quality one.

### `xencode hw <action>`
Ask this machine what it can serve, instead of guessing.

```bash
xencode hw probe                        # configured model, llama-server on PATH
xencode hw probe --model ~/models/foo.gguf --exec /opt/llama.cpp/bin/llama-server
```

`probe` prints, in order: the RAM and core counts, the `llama-server` build it
found, the graphics devices the kernel sees, the compute devices the server can
actually use, the model file's own geometry, what its cache costs per token, how
much memory is free against how much the weights need, and the window the context
budget is currently working with. It ends with the flags to start the server with
and the `xencode config set llama_cpp_args "…"` line that keeps them. **It writes
nothing** and starts nothing.

Two of those sections carry a warning rather than a number, on purpose:

- **Video memory does not come from `lspci` or PCI config space.** On the machine
  this was written on, the graphics card's largest PCI window is 256 MiB and the
  card holds 2048 MiB. The sizes printed for the devices are read from
  `llama-server --list-devices`, which is the only source that both knows the
  memory and knows what the binary in front of you can offload to — a build with
  no CUDA support offloads over Vulkan, so reasoning from the vendor ID alone
  points at the wrong answer.
- **A device is not necessarily a graphics card with its own memory.** An
  integrated window reported here as 11822 MiB is three quarters of the machine's
  RAM and serving from it measured 15 tokens/s against 58 on the CPU. A device
  whose total reaches half the machine's memory is labelled as sharing system
  memory and is not chosen; the free memory the server reports is what the
  recommendation is built on.

The context size is where the arithmetic matters. The KV cache sits on top of the
weights and grows with every token, so a window that looks affordable by the file
size alone is the failure mode: with the model fully offloaded, this machine loaded
an 8192 token window and stopped with an out-of-memory error at 10240, with 1156 MiB
of device memory free. The probe holds a quarter of the free memory back for the
server's own buffers and rounds the window down, and it never recommends a window
larger than the context budget asked for — a bigger window that spends the memory
on cache instead of the model was measured answering at 43 tokens/s here, which is
slower than running the same model on the CPU. When the weights themselves do not
fit, it says so and says a smaller quant or a remote server is the way out, instead
of recommending a window small enough to squeeze the model in.

Two things it cannot tell you, both because the server does not say: which device a
running server put the model on (`/props` reports a context size, slots and build
info, and no device or offload field), and what a build competing for the same
memory will do to it. The `mmap` line states the relationship and leaves the
measurement to you.

### `xencode history <action>`
Ask how fast this repository's history is to query, and speed it up where an
index is missing.

```bash
xencode history status                       # current directory's repository
xencode history status --path ~/Projects/foo --json
xencode history setup                        # write the two indexes, then re-time
xencode history status --file src/main.rs    # blame probe on a file you care about
```

`status` prints four things and starts nothing but `git`: where the repository
data lives, how many commits are reachable, whether a **commit-graph** and a
**multi-pack-index** exist (with size, pack count, and whether
`git commit-graph verify` passed), and a `timed now:` table. Every number in that
table comes from a `git` process that just ran for this command — the commit
subjects, the commits with the paths they touched, the reachable-commit count, and
a blame of one file. Nothing is carried over from an earlier run, and the blame
probe only appears when a file is known: `--file` if you gave one, otherwise
`README.md`, otherwise the first tracked path.

`setup` writes `.git/objects/info/commit-graph` (`git commit-graph write
--reachable`) and `.git/objects/pack/multi-pack-index` (`git multi-pack-index
write`), then measures again. Both writes are idempotent and touch no commit, so
running it twice is safe and says `commit-graph rewritten`. Where git refuses, the
refusal is printed in git's own first error line rather than paraphrased — with
fewer than two packs, for instance, `git multi-pack-index write` exits 255 with
`error: no pack files to index.` and there is honestly no index to have.

**The comparison is where it gets blunt.** The two timings of one query on one
machine differ by a couple of milliseconds, so a claim has to clear both 20% and
2 ms; anything smaller is reported as no change. On this repository — 813
commits, 608 MiB of `.git`, 2 packs, git 2.55.0 — the indexes were not both
present to begin with: the commit-graph was missing, and writing it changed only
the commit-count query (2.7 ms → 2.0 ms). The other three did not move, and the
command prints that instead of a speed-up:

```
no query changed by more than both 20% and 2 ms — on 813 commits the commit chain
was not the cost, so these indexes are there for the queries built on history
rather than for the ones timed above
```

That is the honest reading of what a commit-graph does: it shortens walking the
commit chain, and on a repository this size the chain was never the expensive
part. What *is* expensive here, measured from the same shell, is `git log
--numstat` over the whole history at **11.3 s** and `git log -S <text> --all` at
**12.9 s** — neither of which these indexes fix. Cheap by comparison: `--follow`
on one file at 51 ms and `git blame -L 1,120` on one file at 10 ms.

`status` also annotates two repository shapes that change what those numbers mean.
A **shallow** clone (one started with `--depth`) is marked, because there is no
history to walk and `git fetch --unshallow` is the way to get one. A **partial
clone** (`--filter=blob:none`) is marked with a warning that blame and `log -S`
fetch a blob from the remote for every commit they visit — which is what turns
those two queries from milliseconds into minutes, and is worth knowing before
reading a timing as "git is slow".

### `xencode colab <action>`
Google Colab bridge: run the inference server on a Colab VM (T4 GPU etc.)
and reach it from this machine. The only supported transport is the official
`google-colab-cli` `colab ssh --proxy-mode` WebSocket SSH bridge — never a
public tunnel (Colab's free tier forbids ngrok/cloudflared-style tunnels and
suspends accounts that use them). The CLI is Linux/macOS only; Windows users
type a public-tunnel URL (paid tier) into Settings → Remote URL instead.

```bash
xencode config set colab_enabled true  # opt in — `up` refuses while the bridge is off
xencode colab preflight                # is the bridge usable? (exit 0 when green)
xencode colab preflight --generate-key # also create ~/.xencode/colab_ed25519 if missing
xencode colab up                       # create the VM, install the runtime, hold the tunnel
xencode colab up --reconnect           # rebuild a broken bridge from colab.json (one key)
xencode colab status                   # is the forward/session/endpoint alive?
xencode colab down                     # kill the forward, colab stop, clear state
```

`preflight` checks in one pass: the `colab` CLI on PATH, version >= 0.7.0
(0.6.0 shipped without the `ssh` subcommand), backend auth via
`colab sessions`, ssh/ssh-keygen on PATH, and the ed25519 key pair. Every
failing check prints a runnable fix line.

Prerequisites — the bridge rides two tools Xencode does not ship:

```bash
# 1. the official Google CLI (>= 0.7.0), however you install it, on PATH
colab --version
# 2. Google application-default credentials, with all four scopes the two
#    backends need — userinfo.email for the session backend and colaboratory
#    for the keep-alive RPC, or calls fail with 401/403 that look like a
#    permissions bug. gcloud refuses a list missing cloud-platform.
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
```

`xencode colab preflight` is the source of truth for what is missing; run it
before blaming the tunnel.

`up` is the happy-path bring-up: `colab new --gpu <gpu> -s <name>` when the
session is absent, pushes an ssh bootstrap that installs the runtime bound to
`127.0.0.1` only *inside* the VM, holds an `ssh -N -l root -L` forward, waits
until `/v1/models` answers, and writes `~/.xencode/colab.json` — then points
the provider URLs at the forward (`llama_cpp_url`/`ollama_url` for the runtime,
`remote_base_url` for the OpenAI-compatible remote). The ssh user is `root`
because Colab injects the bridge key for root only. Flags override config;
`config colab_*` keys fill the rest:

```bash
xencode colab up                      # uses colab.session / colab.runtime / colab.model
xencode colab up --reconnect          # rebuild a broken bridge from colab.json (one key)
xencode colab up --runtime ollama     # tag flow into the model picker; respins the VM
xencode colab up --gpu L4 --model Qwen/Qwen2.5-7B-Instruct-GGUF
xencode colab up --local-port 18001   # laptop side of the forward
xencode colab up --remote-port 18080  # VM-side port (0 = runtime-native)
xencode colab up --weights hf         # llama.cpp weights from Hugging Face
xencode colab up --quant Q6_K         # which GGUF quant to serve (default Q4_K_M)
```

`up` refuses unless the bridge is switched on (`xencode config set
colab_enabled true`), and defaults the session name to `xencode-vm` and the
model to `Qwen/Qwen2.5-7B-Instruct-GGUF` when neither a flag nor config supplies
one.

`up` is patient by design: Colab gives a runtime exactly one SSH bridge and the
slot of a bridge that just died takes a while to free, so both the bootstrap and
the forward retry through that window (`Already-active SSH session` /
`banner exchange` errors, up to 8 attempts 20 s apart). `READY` from the VM
means the server actually serves — after the download it polls `/v1/models`
inside the VM before reporting, so a model that needs ~40 s to load on a T4
cannot look up-and-fail. The bootstrap budget is 40 minutes.

`up --reconnect` is the one-key repair path driven by `colab.json`: if the
forward's endpoint already answers `/v1/models` it returns immediately (no
colab or ssh calls at all — a dead forward pid is re-spawned and re-probed
before anything is re-fetched); otherwise it re-creates the session if the VM
was reaped server-side (never when the session still exists), re-runs the
bootstrap, and re-spawns the forward, then re-probes and rewrites state.
Without a `colab.json` it errors with a pointer to a full `xencode colab up`.

`runtime` chooses what is installed on the VM: `llama.cpp` (pinned prebuilt
llama.cpp release — CUDA build when `nvidia-smi` answers, the plain x64 build
otherwise — serving one GGUF fetched from Hugging Face) or `ollama`
(`ollama serve` + the pull — its tags then flow into the model picker for free
via the existing provider list). llama.cpp listens on `127.0.0.1:18080` by
default because Colab's own runtime proxy permanently holds `8080` on the VM.
`weights` is `hf` for llama.cpp; `drive`/`gcs` are refused with a fix message.
Session names are validated before they touch a shell (`[A-Za-z0-9_-]`,
1–64 chars).

`status` never fails hard — it reports three cells (forward pid alive,
session listed by `colab sessions`, and a `/v1/models` probe on the forward)
so it stays scriptable while fully degraded. When the VM is older than 12
hours and the endpoint is down it flags a likely Colab reaper and prints the
one-key fix (`xencode colab up --reconnect`). `down` is idempotent: kills the
recorded forward pid, runs `colab stop -s <name>`, and clears state; with no
`colab.json` it reports `nothing to tear down`.

Once `up` is green the tunnel is an ordinary provider — there is no Colab-specific
client path. `up` writes `llama_cpp_url` (or `ollama_url` for that runtime) and
`remote_base_url = <forward>/v1`, so the `remote:` prefix, the TUI model picker
and the Remote row in Provider Health (Ctrl+F) all speak through the same
forward:

```bash
xencode query -m 'remote:/root/xencode-llama/model.gguf' "Capital of France?"
curl -s http://127.0.0.1:18000/v1/models     # what the VM actually serves
```

llama.cpp reports the GGUF path it was started with as its model id, so on the
`llama.cpp` runtime that id is `$HOME/xencode-llama/model.gguf` inside the VM —
`/root/...` because Colab injects the bridge key for root. Read it from
`/v1/models` rather than assuming it.

### `xencode config <action>`
Configuration management. Config lives in `~/.xencode/config.json`;
set `XCODE_CONFIG_DIR` to point Xencode at a different directory.

```bash
xencode config show
xencode config set default_model qwen3:4b
xencode config set mcp_timeout 30
xencode config set llama_cpp_args "--n-gpu-layers all --device Vulkan1"
xencode config reset
```

A value that begins with a dash is taken as the value rather than as an option to
`config set`, because the line `xencode hw probe` hands back to paste starts with
`--n-gpu-layers` and quoting it was refused until this was fixed.

`config set` keys (values are validated; `config show` prints the JSON):
`mcp_servers`, `agent_hooks` and `model_profiles` are nested structures, so they are edited directly in the JSON instead, or managed in the TUI where a panel exists for them.

| Key | Type | Notes |
|-----|------|-------|
| `default_model` | string | e.g. `qwen3:4b` |
| `ollama_url`, `llama_cpp_url` | string | provider endpoints |
| `remote_url`, `remote_key` | string | Remote / Colab endpoint (any OpenAI-compatible server, e.g. `http://127.0.0.1:18000/v1` + its bearer token); empty `remote_key` clears it |
| `colab_enabled` | bool | Gates the whole Colab bridge; `false` → `xencode colab *` refuses |
| `colab_session` | string | Session name for `xencode colab up`; empty = create one |
| `colab_runtime` | string | `llama.cpp` or `ollama` — what gets installed on the VM |
| `colab_model` | string | HF GGUF repo (llama.cpp) or ollama tag |
| `colab_quant` | string | GGUF quant to serve; empty = `Q4_K_M` |
| `colab_weights_source` | string | `hf` (llama.cpp); `drive`/`gcs` accepted but refused at bring-up |
| `colab_local_port`, `colab_remote_port` | number | Laptop side of the forward / VM-side port (`0` = runtime-native: llama.cpp `18080`, ollama `11434`) |
| `colab_auto_connect` | bool | Persisted but not acted on yet — nothing reconnects without an explicit `xencode colab up` |
| `llama_cpp_model_path`, `llama_cpp_executable` | string | llama.cpp paths |
| `llama_cpp_model_url` | string | HTTPS URL of the GGUF file itself; `llamacpp start` fetches the model into `llama_cpp_model_path` from here when it is missing — disk-priced first, resumable — see [xencode llamacpp](#xencode-llamacpp-action) |
| `llama_cpp_model_sha256` | string | the digest these bytes are pinned to: 64 hexadecimal characters, an optional `sha256-` prefix, any case; empty turns the check off. Checked as a download arrives and again before any server is started on the file — see [xencode llamacpp](#xencode-llamacpp-action), "Pinning the bytes". `xencode models advice` prints one for every model it suggests |
| `llama_cpp_args` | string | split on whitespace; passed to a self-started `llama-server` after the hardware profile's own flags, so a repeated flag is decided here |
| `llama_cpp_reasoning` | string | how much a local model may think before answering: `auto` or empty for no flag, `off` for `--reasoning off`, or a token budget as a number for `--reasoning-budget`. A launch setting — see [How much the model may think](#how-much-the-model-may-think) |
| `ollama_reasoning` | string | whether a model served by Ollama may think before answering: `auto` or empty to leave the model's own default in charge, `off` for `"think": false`, `on` for `"think": true` — which is only sent when `/api/show` says the model can think. A number is refused: Ollama has no thinking budget, so use `llama_cpp_reasoning` for that — see [What a request to Ollama carries](#what-a-request-to-ollama-carries) |
| `ollama_keep_alive` | string | how long the server keeps this model loaded after an answer, in the server's own words (`10m`, `30s`, `0` to unload immediately). Must contain a digit; empty leaves out the field and the server's five minutes rule. Decides whether the next request pays for a reload — see [What a request to Ollama carries](#what-a-request-to-ollama-carries) |
| `max_cache_size`, `response_timeout`, `max_memory_items` | number | |
| `cost_budget_usd_micros` | number | Warning threshold for one conversation's spend, in millionths of a dollar ($5.00 = `5000000`). Unset by default; it warns in the status bar and never refuses a request. Spend is priced from `.xencode/pricing.json` in the project — see `/cost`. **Not a `config set` key** — edit it in the JSON. |
| `llama_cpp_temperature`, `llama_cpp_top_k`, `llama_cpp_min_p`, `llama_cpp_max_tokens`, `llama_cpp_seed` | number | llama.cpp sampling defaults, read from the JSON; `config set` does not accept them, and the TUI's Settings panel covers the same fields. An unset one sends nothing and the server decides — see "Repeatable answers" above. |
| `cache_enabled`, `memory_enabled` | bool | `true`/`false` |
| `layout` | string | TUI body preset: `classic`, `chat-first`, `zen` (unknown → classic at render) |
| `active_theme` | string | UI theme: `ocean`, `midnight`, `forest`, `terminal`, `dracula`, `solarized`, `nord`, `light` (default `ocean`); cycled live in the TUI. **Not a `config set` key** — `xencode config set` has no arm for it, so edit it in the JSON. An unrecognised name is *not* rejected: `ThemeColors::get` falls through to the `ocean` palette, so a typo renders as ocean and reads back as the typo. Check the spelling against the eight above. |
| `hardware_profile` | string | How much project context a run may spend: `auto` (default — chosen from this machine's memory), `low`, `balanced`, `high`; anything else is rejected by `config set` and reported by a run if it is already in the file. See [Which hardware profile the budget spends against](#which-hardware-profile-the-budget-spends-against) |
| `rounded_borders` | bool | rounded panel corners |
| `show_scrollbars` | bool | scrollbars on chat & explorer panes |
| `show_line_numbers` | bool | editor line-number gutter + current-line highlight |
| `agent_approval` | string | agent tool-approval mode: `ask`, `edit-allow`, `all-allow` (unknown → `ask`) |
| `agent_max_rounds` | integer | assistant→tool rounds allowed per chat turn before the model must answer in prose (`1`–`64`, default `16`) |
| `agent_command_timeout` | integer | seconds the agent's foreground `run_command` may take before it is killed (`1`–`600`, default `30`); slow work belongs in `background_start` |
| `agent_fallback_models` | list | comma-separated ordered alternates for the agent's turns (I4-01), e.g. `xencode config set agent_fallback_models "qwen2.5:14b,google_gemini:gemini-2.0-flash"`. The configured default model is always tried first, so this list holds only fallbacks (duplicates of it are dropped). A candidate is abandoned — and the chain moves on — only when it failed **before emitting any token** and the error is not our own response-decode failure; a token already on screen, or a `Parse` error, fixes the model in place. Each candidate gets one attempt per step and the transcript records a `[FALLBACK]` line when the chain moves. A candidate that would send the conversation somewhere the primary would not — a cloud API as the alternate for a local model, or the reverse — is never tried, and the transcript names it as skipped instead; a `remote:` endpoint counts as local only when its configured URL points at this machine (`localhost`, `127.x`, `::1`, `.local`). `xencode query` is single-shot and does not use this chain. An empty list (the default) disables fallback. |
| `session_recording` | bool | Write down every model call of an agent turn — the request, the response bytes as they arrived, and what each tool returned — to `.xencode/cache/sessions/<run-id>.jsonl`, so `xencode replay` can run that turn again. Off by default. Only the routes whose bytes this program reads itself are recordable: Ollama, llama.cpp, a `remote:` endpoint and OpenRouter. Asking for a recording of an Anthropic, Gemini or Qwen model is refused with the reason, because those have their own readers and a "recording" of them would be a paraphrase. |
| `allow_cloud_models` | bool | Whether a prompt may reach an internet service at all. Off by default — and off for a config written before the key existed — so `qwen:…`, `google_gemini:…`, an OpenRouter-style `vendor/model` when an OpenRouter key is set, and a `remote:` endpoint whose URL is not this machine are refused before a connection is opened, with the refusal naming this key. A key in `api_keys` is not permission for the trip; it identifies you to the provider. The TUI status bar prints the rule in force (`🔒 local only` / `🌐 cloud allowed`) and Settings → Providers has a **Cloud Models** row that toggles it. |
| `allow_online_docs` | bool | Whether the agent's `read_docs` tool may fetch a crate's documentation when cargo has not unpacked it on this machine. Off by default, and independent of `allow_cloud_models` — turning one on does not turn on the other, because one is a prompt leaving and the other is a text file arriving. With it off, `read_docs` answers from cargo's own copy and says what else would be needed to get more. Open it with `xencode config set allow_online_docs true`. |
| `mcp_timeout` | integer | seconds a server may take to handshake and answer before it is reported failed (`1`–`300`, default `30`) |
| `model_profiles` | list of objects | saved profiles the TUI's Custom Models panel (J-05) shows: `{ "name": "...", "model": "ollama:qwen2.5:7b", "temperature": 0.2, "max_tokens": 2048, "for_task": "bugfix" }`. `temperature` and `max_tokens` are optional — omit them and the panel renders "unset — the server decides" and sends nothing. `model` takes exactly the form `default_model` does. `Enter` applies a profile to the next turn; `s` in the panel writes the whole list back here; `f` cycles `for_task` through `bugfix`, `general` and none. There is no `top_p`: no provider path in this workspace sends it, and only llama.cpp receives these two knobs in the request body |
| `model_routing` | bool | Whether a profile's `for_task` mark is acted on by itself. Off by default, so a marked profile still only applies by hand. On, the first profile whose mark matches the turn runs that turn on its model: `bugfix` for a prompt that says something is broken (`fix`, `fails`, `crash` and similar words), `general` for every other prompt, and a mark naming a reading this version does not have (or no mark at all) matches nothing. A profile that would move a llama.cpp model is refused instead — a running `llama-server` holds one model at a time — and the chat prints why. See [Turn routing](#turn-routing) |
| `mcp_servers` | object | MCP stdio servers to offer as tools: `"name" → { "command": "...", "args": [...], "env": {...} }` (credentials go in `env`, never `args`); nothing is started until you run `/mcp` |
| `agent_hooks` | object | shell hooks around **approved** agent tool calls: `"before"` and `"after"` maps from an exact tool name (or `"*"` for every tool) to a command run via `sh -c` in the workspace root. A failing `before` hook vetoes the call (nothing runs, no rewind point, output shown as `error: pre-hook vetoed this call`); a passing one has its output prepended to the result. The `after` hook always runs and its output is appended. Hook output is capped like `run_command` (stderr merged, tail kept) |

Edit on `agent_hooks` directly in the JSON (`config set` has no nested-map key):

```json
{
  "agent_hooks": {
    "before": {
      "write_file": "git status --short",
      "run_command": "echo 'about to run a shell command'"
    },
    "after": {
      "*": "true"
    }
  }
}
```

### `xencode cache <action>`
Response cache management: `stats`, `clear`.

```bash
xencode cache stats
xencode cache clear
```

### `xencode replay <run-id> [--run-tools]`
Run a recorded agent turn again from the bytes it was made of. Turn on
`session_recording` (see the config table) and each model call of a turn is
written to `.xencode/cache/sessions/<run-id>.jsonl`: the request, the response
bytes as they arrived on the socket, what each tool returned, and the clock
reading at the time. A replay serves those bytes again on a loopback port while
the real agent loop runs against them — the HTTP client, the stream reader that
has to reassemble a tool call arriving in fragments, the permission gate, and the
tools themselves, which execute for real. No model answers a replay, so it needs
neither a server nor a provider account.

```bash
xencode replay --list                  # what has been recorded, newest first
xencode replay 1790240197              # the id, or enough of its start to be unique
xencode replay 1790240197 --run-tools  # and let the recorded commands run again
xencode replay 1790240197 --tool-root /tmp/scratch --out /tmp/check
```

```
recording 1790240197-eee44c61: 2 model calls, 1 tool call
replay: 2 of 2 model calls answered, 0 requests the recording could not answer
tools: 1 of 1 recorded calls asked again (outcomes: done=1)
answers: the same bytes the recording holds, reassembled from a socket
clock: every time in the ledger is the recorded one; nothing here reads the clock, which is why two replays of one run can be compared at all
ledger: <project>/.xencode/cache/replays/1790240197-eee44c61/tool_calls.jsonl
```

(Only the last line is shortened here — the command prints the ledger's full
path.)

`--list` names each recording, what it asked for, and how many calls and tools it
holds. `--tool-root` is the tree the replay's tool calls work against (the current
project by default); `--out` is where the ledger and the replay's own recording go
(`<project>/.xencode/cache/replays/<run id>` by default), and it cannot be pointed
at the directory holding the recording being replayed.

**The gate stays in charge.** Without `--run-tools` nothing is approved silently:
no one is there to answer the approval prompt, so a call the recording shows was
gated comes back `denied`, the next model call has no request to be answered with,
and the report says `1 of 2 model calls answered` and exits non-zero.
`--run-tools` is the only thing that sets the loop to allow everything, and only
because a caller asked to live through the recorded commands again.

What the ledger holds is one line per recorded tool call: the tool's name and
arguments, its outcome, how many characters the result was and a SHA-256 of it,
which model call it came from, and the time fields taken from the recording
(`recorded_ts_unix_ms`, `recorded_duration_ms`) rather than from the replay's own
clock — which is what lets two replays be compared at all. Lines the replay never
reached are written too, with `"replayed": false`.

A recording is only as good as its being repeatable, so the second turn is matched
on the tool's own output bytes: a replay whose command printed something different
is refused rather than answered with a recording made for a different answer. A
recorded command must therefore be one that gives the same output every time —
`date`, `git log` and anything over the network make a recording that can only ever
report the request it could not answer.

### `xencode audit verify [PATH]`
Check the session server's audit log for records that were changed after they
were written. Each record carries a digest of its own contents and the digest of
the record before it, so editing, removing or moving a line is reported on a
specific line. Defaults to `~/.xencode/audit.jsonl`. Exits non-zero when
something does not add up.

```bash
xencode audit verify
xencode audit verify /path/to/audit.jsonl
```

What it cannot tell you: a log that someone truncated at the end still verifies,
because nothing outside the file says how long it should be, and anyone willing
to recompute every digest can rewrite the whole file. It catches an edit, not a
rewrite.

### `xencode eval <action>`
Score the agent against defects that were seeded on purpose. `list` prints the
eight shapes and every run recorded so far; `run` writes each selected shape into
its own fresh git repository, hands the real agent loop the `task.md` that
describes the bug, and then grades what the run left on disk.

```bash
xencode eval list
xencode eval run -m llamacpp:dolphin --llamacpp-url http://127.0.0.1:8099
xencode eval run -c off-by-one -c stale-cache --repeats 3 --out /tmp/eval
xencode eval run --allow-shell            # the shell class is otherwise refused
xencode eval run --judge                  # rank the near misses after the grading
xencode eval run --judge --judge-model llamacpp:bigger  # rank with another model
```

What counts as a pass is deliberately narrow: the case's own `cargo test --offline`
has to go green **and** the run has to change exactly the file the reference fix
changes. A green grader reached by editing `tests/behaviour.rs` is reported as
`changed its own test` and is never a pass. Every run appends one line to
`.xencode/cache/task_eval.jsonl` with its model, prompt digest, permission
posture, sampling pins and per-case verdicts, so a rate is only ever printed next
to a previous rate taken under the same rules. A case whose model request failed
is `not run` and stays out of the denominator — an unreachable server is not
evidence about a model.

Flags worth knowing: `-c/--case` (repeatable, `off-by-one` and `Off_By_One` both
work), `-m/--model`, `--repeats`, `--max-rounds`, `--allow-shell`, `--out`,
`--ollama-url`, `--llamacpp-url`, `--timeout` (default 120 s per request),
`--max-tokens` (default 1024; `0` leaves the limit to the server), `--judge` and
`--judge-model`. The answer length is capped because a small model that has
started repeating itself will otherwise hold one case for minutes.

`--judge` asks a model, afterwards, which of the attempts that *failed* came
closest to a correct fix. It is a ranking and nothing else: the judge is shown
only the near misses — a case that never ran, one that passed, one that changed
no files, and one that edited its own grader are all left out — and the request
has no field in which to call anything a pass, so `pass rate` is computed the
same whether or not a judge was asked. Three bias controls are in code rather
than in the wording: the attempts are listed in an order derived from their
identities and then the same question is asked again with the list reversed, and
if the two answers differ the ranking is discarded and the report says so; a
candidate is shown as its change and the grader's last lines, never as the agent's
prose, with the change capped at 4,000 characters, so a wall of edits cannot
outshout a small one; and nothing in the request names a model, with
`--judge-model` letting a different one do the reading. What is *not* solved is a
judge recognising its own style when it is the same model that wrote the attempts
— relabelling hides the name, not the handwriting. A run holding more than 26
near misses shows the first 26 and says how many were left out.

What it has been asked of, so far, is a 1.5B model on a local `llama-server`, and
twice the answer was that there was nothing to rank: judged over a whole run it
reported `no case was a near miss`, because that model leaves every file untouched,
and put the two questions to it directly over a real socket it replied `unsure` and
named nothing. The plumbing is proven; no ordering this tool has printed yet came
from a model big enough to compare two diffs. A judged run prints the version of the
ranking instruction it used, and adding it moved the digest `xencode eval list`
shows for every run, so a score from before it is not offered as a comparison.

What this number is *not*: it is one model, one set of instructions and eight
shapes of defect. Eight cases is below the ten-to-thirty the plan asked for, and
`--repeats 3` reaches twenty-four by running the same eight more often, which is
not the same as twenty-four different defects. The agent works in a repository
that contains its own grader, so the expected values are readable by it; nothing
here stops a model from reading them, and the diff check is what notices.

### `xencode memory <action>`
Conversation memory (persisted under `~/.xencode`): `list`, `show <session>`.

```bash
xencode memory list
xencode memory show <session-id>
```

### `xencode tasks <action>`
File-backed background tasks. State lives in `.xencode/tasks/` under the
current directory, so tasks started here are visible to later `xencode
tasks` runs in the same project (the TUI's `Ctrl+K` panel keeps its own
in-process registry). Status is derived on read from the task's exit file,
killed flag, and process liveness.

```bash
xencode tasks start "cargo test" --name tests   # → "started task 1 (pid …)"
xencode tasks list                              # table; --json for machine output
xencode tasks poll 1 --lines 20                 # status + trailing stdout/stderr
xencode tasks stop 1                            # signal a running task
xencode tasks rm 1                              # forget a finished task (refuses running)
```

### `xencode worktree <action>`
Git worktrees of the repository at the current directory: `list`,
`add <path> [<branch>]` (existing branch or commit to check out; when
omitted git names the new branch after the directory), `remove <path>`
(dirty worktrees are refused by git itself, the main checkout is never
removable).

```bash
xencode worktree add ../feature-x        # new branch "feature-x"
xencode worktree add ../hotfix main      # check out existing branch
xencode worktree list
xencode worktree remove ../feature-x
```

### `xencode advise [FILTER] [--json] [--limit 40]`
Repository insights from the `.xencode` snapshot written by the TUI's
`/init`: broken imports, import cycles, hub files and orphans.
The dependency graph behind them is built from `use` statements, `mod`
declarations and `impl Trait for Type` blocks, so a file an orphan report
names is one none of those reach in either direction.
`FILTER` is a positional substring matched against each finding's file
path; `--limit 0` shows everything. Errors with exit 1 when the project
has no index yet.

```bash
xencode advise                      # top 40 findings
xencode advise src/auth             # only findings touching that path
xencode advise --json --limit 0     # full machine-readable report
```

### `xencode server [OPTIONS]`
Start the collaboration HTTP/WebSocket server. Sessions live in memory
(the audit log is the only thing that survives a restart); clients
authenticate on the WebSocket with a token from `POST /auth/login`, and
the first WS frame must be the `auth` frame — the URL carries no
identity (`/ws/{session_id}`).

```bash
xencode server                          # http://127.0.0.1:8765, ws://
xencode server --port 9000 --audit-path none
xencode server --host 0.0.0.0 --cert fullchain.pem --key privkey.pem   # https + wss
```

| Flag | Meaning |
|---|---|
| `--port <PORT>` | Listen port (default `8765`) |
| `--host <HOST>` | Bind address (default `127.0.0.1`; IP or `localhost`) |
| `--cert <PEM>` / `--key <PEM>` | TLS material — both or neither; enables `https://`/`wss://` |
| `--audit-path <PATH\|none>` | JSONL audit trail (default `~/.xencode/audit.jsonl`; `none` disables) |
| `--allow-insecure-public` | Escape hatch: bind a non-loopback address over plain ws:// |

Posture rules, enforced at startup: a non-loopback `--host` without TLS
refuses to start unless `--allow-insecure-public` is given (then it
binds with a loud clear-text warning); `--cert` without `--key` (or the
reverse) is an error; the banner prints the real scheme — `ws://` stays
`ws://`, only certificates earn `wss://`.

### `xencode plugin <action>`
`list`, `install <path>`, `remove <name>`. Plugins live in `$XCODE_PLUGIN_DIR`,
else `<data dir>/xencode/plugins` — the same directory the TUI loads from at
startup, so the two never disagree.

A plugin is a directory holding `plugin.json` (or `manifest.json`). This build
loads no executable plugin code: the manifest is the whole plugin, and the two
things it can declare are a `prompt_prefix` (placed ahead of the agent's system
prompt on every turn) and `hooks` — `before`/`after` maps of tool name (or `*`)
to an `sh -c` command, the same shape as `agent_hooks` in config.json. A
plugin's hook only lands where config.json is silent, so your own config always
outranks it.

```json
{
  "name": "guardrails",
  "version": "1.2.0",
  "prompt_prefix": "Run cargo test before answering.",
  "xencode_version": "*",
  "hooks": { "before": { "write_file": "echo pre" }, "after": { "*": "echo post" } }
}
```

`list` runs the load and reports what took hold instead of just listing
directories:

```console
$ xencode plugin list
📦 Plugins in /tmp/j08-probe/plugins (xencode 0.1.0):
  future v9.9.9 — NOT LOADED: needs xencode 0.1.0 (declared 9.9.9)
  guardrails v1.2.0 — loaded: prompt prefix, 1 before hook(s), 1 after hook(s)
  1 of 2 loaded — a loaded plugin's prompt prefix and hooks apply to every agent turn.
```

`install <path>` copies the directory (or a single manifest) under that name and
prints the same one-line verdict, so an install nothing can load is visible
immediately. `remove <name>` deletes only that one directory; a name containing
a path separator or `..` is rejected rather than resolved.

### `xencode fetch <url> [--format text|json]`
Fetch a web page and extract research-ready text (title + body, scripts
and markup stripped). `--format json` returns the full `FetchedPage`.

```bash
xencode fetch https://example.com
xencode fetch https://example.com --format json | jq .title
```
Text output caps at 30k chars with a truncation trailer; `--format json`
returns the full body. Invalid `--json-schema` values are rejected up
front instead of degrading silently.

### `xencode review [--base main] [--format text|json]`
PR-level diff triage: files changed between the base and HEAD with line
counts, plus working-tree analysis per file (code issues, image
inventory). `--base HEAD` reviews uncommitted changes. Unanalyzable files
(deleted, binary) get visible notes, never silence.

```bash
xencode review --base main
xencode review --base HEAD --format json | jq '.files[] | {path, issues: (.issues|length)}'
```

### `xencode advisories <action>`

The security advisories that have been published about Rust crates live in two
databases: the curated [RustSec] advisory repository and Google's OSV service,
which mirrors it and adds records of its own. `xencode advisories sync` downloads
both to `<config dir>/advisories` — a shallow clone of the advisory repository,
plus OSV's `crates.io/all.zip` unpacked into one JSON file per record — and every
command after that reads only those files. Nothing here shells out to `cargo
audit`, and the agent's `lookup_advisory` tool makes no request: the network is
used by this one command, when you choose to run it.

Measured on this machine, re-syncing a corpus that already exists:

```text
$ time xencode advisories sync
corpus at /home/sree/.xencode/advisories — 1251 RustSec advisories, 2856 OSV records, 4857 index lines
RustSec revision e2111519ba6d14a5da59a7b2e5c8083ae8a37c01 (pulled); OSV download 3490826 bytes
lookups are offline from here; run `xencode advisories check` to read Cargo.lock

real	0m3.417s
$ du -sh /home/sree/.xencode/advisories
20M	/home/sree/.xencode/advisories
```

The two corpora are not redundant: 732 of the OSV records have no link to a
RustSec number at all and cover 791 crates the curated database does not name,
and the mirrors that do overlap carry a one-word severity the curated record has
no room for. Both are kept in one directory with a tab-separated index
(`package`, `corpus`, file), so a lookup for one crate name reads the index and
then the handful of files it points at.

`xencode advisories check [--path DIR]` judges every package in a lock file:

```text
$ xencode advisories check --path rust
/home/sree/Projects/xencode/rust/Cargo.lock — 419 locked packages; 4 of them are named by 5 advisory record(s):
  lru 0.12.5
    RUSTSEC-2026-0002 [rustsec] 2026-01-07 — affected, 0.16.3 is offered as safe
      https://github.com/jeromefroe/lru-rs/pull/224
    RUSTSEC-2026-0253 [rustsec] 2026-05-12 — affected, 0.18.2 is offered as safe
      https://github.com/jeromefroe/lru-rs/pull/238
  paste 1.0.15
    RUSTSEC-2024-0436 [rustsec] 2024-10-07 — informational (unmaintained)
      https://github.com/dtolnay/paste
  rustls-pemfile 2.2.0
    RUSTSEC-2025-0134 [rustsec] 2025-11-28 — informational (unmaintained)
      https://github.com/rustls/pemfile/issues/61
  ttf-parser 0.25.1
    RUSTSEC-2026-0192 [rustsec] 2026-06-28 — informational (unmaintained)
      https://github.com/harfbuzz/ttf-parser/issues/217
```

That took 0.238 s. Two things in it are deliberate: the header counts packages
and records separately, because one crate can be named by several advisories, and
`informational` rows are not vulnerabilities — an unmaintained crate has nothing
to upgrade to. An empty result says so in the same breath:

```text
  nothing — but that means no advisory matches these versions, not that the dependencies are safe
```

The lock file is looked for from `--path` upwards and the search stops at the
project root (a directory containing `.git`), so in this repository the command
is `xencode advisories check --path rust` — the workspace root has no `Cargo.lock`
of its own, and the error says which flag to use rather than going quiet.

`xencode advisories show CRATE [--version V]` answers for one crate, assessed
against `V` when given (see the `lookup_advisory` section above for the same
text as the model receives). `xencode advisories status` reports what corpus
exists here, how big it is, which RustSec revision it was taken at, and how old
the download is. Every one of them takes `--dir PATH` to read a corpus from
somewhere other than `<config dir>/advisories`.

```bash
xencode advisories sync                       # once; needs network
xencode advisories check --path rust          # offline
xencode advisories show chrono --version 0.4.19
xencode advisories status
```

[RustSec]: https://github.com/RustSec/advisory-db

## 🎯 Usage Examples

### Development Workflow
```bash
# 1. Index the project (inside the TUI)
xencode
# › /init

# 2. Check model health
xencode models list

# 3. Query for code help
xencode query "How do I parse JSON in Rust?"

# 4. Analyze before committing
xencode analyze ./src --format text
```

### Scripting
```bash
#!/bin/bash
# Ask and fail loudly on error
if ! xencode query "$1" --no-cache; then
    echo "❌ Query failed" >&2
    exit 1
fi
```

### JSON output for tooling
```bash
xencode analyze ./assets/logo.png --format json | jq .format
```
