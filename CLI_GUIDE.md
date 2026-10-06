# 🤖 Xencode CLI Guide

The command-line interface for the Xencode AI assistant (Rust binary).
Running `xencode` with no subcommand launches the TUI, which needs a terminal;
every other subcommand works headless in a pipe or a CI step.

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

**It needs a terminal.** The screen draws on standard output and reads keys from
the controlling terminal, so a bare `xencode` in a pipe, a redirect, a cron line
or a CI step says so plainly instead of failing inside the terminal library:

```
$ printf '' | xencode
error: the interactive screen needs a terminal to draw on, and standard output here is not one (a pipe, a redirect, a cron line or a CI step). Without a terminal these work: `xencode query <prompt>` for one answer, `xencode run <task>` for an agent turn, `xencode scan`, `xencode analyze`, `xencode doctor`. `xencode --help` lists the rest.
```

That is the live output of the command above, and it exits non-zero: the screen
was asked for and cannot be shown. Every other subcommand is unaffected — the
whole CLI works headless, and `xencode run --detach` is built for the case where
the terminal may disappear mid-task.

TUI keys — press `?` (or `F1`) in the TUI for the live, panel-aware
keybinding overlay; the authoritative list lives there. Essentials:
`Tab` cycles explorer/editor/chat · `i` edits chat (`Enter` sends,
`Alt+Enter`/`Ctrl+J` newline, `Alt+↑/↓` history, `Tab` completes `/`
commands) · `m` model selector · `s` settings · `e` edit focused file ·
`Ctrl+R` AI review · `Ctrl+Y` PR review · `Ctrl+K` background tasks · `Ctrl+O` worktrees · `Ctrl+L` insights · `Ctrl+B` ByteBot ·
`Ctrl+H` health check · `Ctrl+G` git refresh · `Ctrl+W` close panel ·
`Ctrl+U` cycle the layout · `Ctrl+0` why the screen is arranged as it is
(session layout history; `Enter` shows the pane widths a change moved between) ·
`Ctrl+Space` switch mode (`CODING` ⇄ `ORCHESTRATOR`; both read the same tasks,
agents and git state, so nothing is lost in the switch) ·
`Ctrl+C` or `q` quit. If the terminal has the mouse, a divider between two
side-by-side panes drags to resize them; if you want the terminal's own
drag-select of text back, `xencode config set mouse_capture off` hands the
mouse over (the same row lives on the Settings panel, and it takes effect on
the next frame). Slash commands: `/init`, `/ctx`, `/advise`,
`/impact <file>` (open the blast-radius panel over `xencode impact`'s three
layers),
`/bytebot`, `/plan` (pin or clear the agent's todo list),
`/rewind` (undo the agent's file changes for this session; it refuses to
overwrite a file you edited by hand after the agent wrote it, and
`/rewind [turns] --force` overrides that),
`/lesson` (what a rewind, a run of failing `/verify` checks, or a call you answered
`n` to at the approval prompt drafted as a lesson: `/lesson status` prints it,
`/lesson set <words>` writes your sentence
into it, `/lesson approve` appends that one line to `AGENTS.md` and clears the
draft, `/lesson drop` clears it without writing anything. The draft's lesson line
starts empty and an empty line cannot be approved — the program records what
happened and leaves the reason to you. `AGENTS.md` keeps every byte it had; this
is the only command in the product that writes *into an `AGENTS.md` that already
exists*, and only when you typed it — `xencode bootstrap` creates that file on a
project that has none, and never edits one. Because trust is keyed on content, the
append makes the file untrusted again
until `/trust` covers the new bytes, and the command says so rather than
re-trusting them itself.),
`/gate` (read the reproduction gate's state — phase, the neighbourhood it was
given, the command, and the recorded red and green),
`/gate bugfix [paths…]` (supervise one bug fix: until the agent has run a
reproduction test and had its failure actually observed on code nobody has
changed, writes to production files are refused at execution rather than
asked about, the tools that can only edit production source are taken off the
turn's tool list, and the reproduction is frozen once its failure is on
record), `/gate off` (stop supervising, saying what measurement — if any —
is being thrown away),
`/mcp` (connect every MCP server declared in config; `/mcp status`,
`/mcp stop`, `/mcp read <server> <uri>`, `/mcp prompt <server> <name>`), `/plugin` (report which plugins loaded and what they changed;
`/plugin reload` re-scans the plugin directory), `/skills` (report which
`SKILL.md` skills loaded, which were refused, and how many characters the
prompt pays for them per turn; `/skills reload` re-scans both skill
directories), `/trace [turns]` (what the
recent agent turns did — rounds, tool calls with the arguments they were made
from and their outcome, the files the context put in front of the model, whether
the turn carried the `[d]` decision marker, and any token count a server
reported — read from `.xencode/cache/turns.jsonl` in the project,
so it answers with every model server down), `/cost` (tokens, KV-cache reuse,
p50/p95 speed — pooled, and for each model separately — and spend for the turns
recorded in this project, read from
`.xencode/cache/metrics.jsonl` through its rollup sidecar and priced by
`.xencode/pricing.json` — or, with `price_lookup` on, by a listing
`xencode prices fetch` read off a public catalogue, which every such line names
with its age; a model with no rate in either document is shown as unpriced, never
as free, and it too answers with every model server down), and
`/spawn <task> [#branch]` (run a subagent in a fresh
git worktree next to the project, e.g. `proj-spawn-1` on branch
`xencode/spawn-1`; a `#branch` suffix names the branch). The spawned
agent's live steps stream in the transcript, its final answer is posted
back with `(spawn #<id> · <task>)`, and `/spawn status` lists every
registered run with its worktree location. Your main chat keeps working
while the subagent works. Four more reach the same engines the CLI runs:
`/doctor [env|deps]` probes machine resources, GPUs, memory and environment
facts, `/verify [skip...]` runs the machine-checkable checklist (fmt, lint,
test), `/hotspots [limit]` ranks files by churn, size and bus factor, and
`/agents` inventories the coding-agent CLIs installed on `PATH`, and
`/trust` decides whether an `AGENTS.md` enters the model's
context as instructions or stays marked `[data]` — by content hash, persisted
in `.xencode/cache/agents_trust.json`, so an edit to the file asks again. With
no argument it means this workspace's own file; `/trust src/auth/AGENTS.md`
names a directory's. `/trust status` reports which one it is right now;
`/trust forget` withdraws trust for the current bytes — both take the same
path, and a grant covers only the file it names.
`/egress [text]` shows, without sending anything,
where the next turn would actually go: the provider a model id resolves to,
whether that route stays on this machine or leaves it, whether the egress policy
allows it (or would refuse the turn before sending), how many messages and bytes
a real turn would carry, and how many secrets the redactor would hold back — by
their placeholder tokens, never their values. It rebuilds the same prompt a real
turn arms, so the preview is what would genuinely leave the machine rather than
an estimate; the stable head (system prompt and a trusted `AGENTS.md`) is never
redacted. With no text it previews where the last user turn would have gone.
The `made of:` line names the kinds of text those bytes are built from and how
many tokens each owns, marking the ones that are data rather than instructions —
a file read off disk, a fetched page, an `AGENTS.md` whose bytes have not been
trusted — so a preview cannot read as though your own sentence and a webpage were
the same kind of thing. Files pinned in the Explorer with `Space` are read through
the same intake a real turn uses, so they are counted here too.

#### Secrets in content, not just in file names

The Security auditor panel scans credential *content*, not only files whose name
looks secret. Beyond the name-gated assignment check it locates a bare
`sk-…`/`AKIA…` token, a bearer token, or a pasted private key in ordinary source,
by line, using the one credential pattern list the trace scrubber and the
secrets-taint gate already share. The same list guards the transcript copy: when
`write_file` or `edit_file` writes a credential-shaped value, the file keeps the
bytes you asked for but the summary returned to the model — which is what the
per-turn trace and the session recording keep — is redacted and flagged `[secret]`.
A credential-looking string in a fixture is documentation, so `examples/`,
`testdata/`, `fixtures/`, `samples/` and `*.example`/`*.sample`/`*.template` files
are never flagged, and any repo-relative path you list in
`.xencode/cache/secrets-allowlist` (one per line, `#` starts a comment) is skipped
too; an unreadable allowlist means nothing is skipped, so the scan errs toward
reporting rather than silence.

The same pattern list now also guards the way *in* to the model, not only what a
tool returns. Before a turn's context is sent, any credential-shaped value in the
dynamic tiers — task state, git facts, the repo map, retrieved file bodies, the
prior conversation, the current prompt — is replaced by a placeholder
(`«xencode-secret-1»`, and so on) that carries no secret, and the real value is
kept locally and put back at the one point it is needed: when a tool call whose
arguments name the placeholder is about to run. So the model can be told to run a
command referencing a secret and it still works, while the plaintext never
crossed the provider boundary. The stable head (system prompt plus trusted
`AGENTS.md`) is never rewritten, because those bytes are what a local server
key/value-caches and a per-turn cache-drift check compares. One value seen in two
tiers becomes one placeholder, and the number of secrets held back is reported
without naming any of them. This reduces what leaves the machine; it is pattern
matching, not a proof, so a secret in a shape the list does not know passes
through — the approval gate and the shell sandbox remain the actual wall.

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

#### Reading one page the model asks for: `web_fetch`

`allow_cloud_models` opens a server you chose and `allow_online_docs` dials two
named hosts for a pinned crate. `web_fetch` is the one tool whose address comes
from the model, which is why it is off until you say otherwise:
`xencode config set allow_web_fetch true` adds it to what the agent is offered,
and nothing else.

Offering it is not the same as letting it out. Every call stops at the approval
prompt, in every mode including all-allow, and the prompt shows the address
together with the verdict the fetch itself will reach — the same check, not a
second opinion:

```text
web_fetch http://127.0.0.1:9999/docs
fetch: http://127.0.0.1:9999/docs  (would connect; the page is returned as text, capped; if it turns out to be missing, this site's own /llms.txt index is asked for on the same address)

web_fetch http://169.254.169.254/latest/meta-data/
fetch: http://169.254.169.254/latest/meta-data/  — this one would be refused: 169.254.169.254 is the link-local range, which is where a cloud serves its instance metadata and credentials
```

The prompt says the same thing the fetch does, so the second request is not a
surprise hidden behind a yes.

Answering `a` — allow for the session — is the one answer this tool does not
take. A yes about one page is not a yes about the next host, so the grant is
ignored for `network.request` and the second fetch prompts again. In plan and
autonomous mode the call is refused outright rather than asked: plan must not
reach out, and an unattended run has nobody to answer.

Approval is also not a key into this machine. The address is resolved before the
connection is opened, and every redirect is re-checked at each hop, so a page
cannot point the request inward — RFC1918, carrier-grade NAT, link-local and the
metadata address are unreachable even after a `y`, and a host that resolves only
to internal DNS is refused rather than tried. `127.0.0.1` is allowed on purpose,
so a dev server on this machine stays fetchable. This is the whole tool running
against a server started by the test on loopback:

```text
[http://127.0.0.1:43427/ — Guide — 78 bytes fetched]
Guide fetched body
```

The header names the address the answer actually came from, which is the landed
one after redirects, not the one that was asked for. HTML is reduced to text;
JSON and `text/plain` are returned as they arrived. The text is capped at 30 000
characters, and when anything is cut the answer says how much and repeats the
address so the next call can ask for the rest. A `max_chars` argument can lower
that cap but not raise it — the cap exists to keep a whole page out of the
context window, so a bigger number buys nothing.

A guessed documentation path usually comes back as a missing page, and that is
the one answer worth a second request. On a 404 the tool asks the same address's
root for `/llms.txt`, the index some documentation sites publish for models,
dropping the path, query and fragment because the convention is one file per
site. What comes back is labelled as that index, because a list of pages handed
over as if it were the page asked for is how a model goes on quoting a document
it never read. Both halves are the real tool against a server the test started on
loopback — a site that publishes an index, and one that does not:

```text
[http://127.0.0.1:39335/docs/getting-started-v2.html — 404 not found. What follows is this site's own index for models, at http://127.0.0.1:39335/llms.txt, which is a list of its pages, not the page that was asked for]
# Site

- [Guide](/guide.html): how to start
```

```text
error: server returned status 404, and this site publishes no llms.txt index either — ask for an address you have actually seen, not a path you guessed
```

The second form is the common one. Measured here on 2026-10-04, `docs.rs`,
`tokio.rs`, `actix.rs`, `doc.rust-lang.org` and the cargo book have no such file
at their root — every one answers 404 except `docs.rs`, which answers 400 — so
this is a fallback kept cheap, not a step the fetch takes by default: a page that
arrives is never probed, the miss is reported as a plain miss with no hint that an
index might exist elsewhere, and the index request goes through the same address
guard as the page, so a miss cannot become a second way into this machine.

A call that arrives while the setting is off — from a resumed transcript, say —
is refused in the executor too, and answering the prompt does not change that:
```text
error: web_fetch is not enabled: `xencode config set allow_web_fetch true` offers it, and every call still asks before the request leaves
```

`xencode fetch <url>` is the same reading with you choosing the address, and it
does not go through this gate. It asks for exactly the address you typed: the
`/llms.txt` retry above is what the tool does when a model's guess misses, not
what the command does when you give it a path.

#### Asking an engine you named for addresses: `web_search`

`web_fetch` reads an address the model names. `web_search` is the other half of
that: it finds addresses, by putting the model's question to a search engine
**you** chose in config. It is offered only when `search_provider` names an
engine, and the default is `none` — a machine that never touched the setting
sends the model the same tool list it was sent before.

The five names are `none`, `wikipedia`, `searxng`, `brave` and `tavily`. There is
no default public instance, and that is a measurement rather than a caution:
checked from this machine on 2026-10-04, DuckDuckGo's `lite` endpoint answers with
its *"Unfortunately, bots use DuckDuckGo too"* CAPTCHA (and its developer API is
`410 Gone`), a public SearXNG instance asked for `format=json` replies `200` with
an HTML document, and MDN's JSON search endpoint is `404`. Wikipedia's own API is
the keyless engine that does answer, and it answers about people, places and
concepts and nothing else. `searxng` is an instance you run, addressed by
`search_searxng_url`; `brave` and `tavily` are a paid API behind `brave_api_key`
and `tavily_api_key` (or `API_KEY_BRAVE` / `API_KEY_TAVILY`). A search key is
never a model route: it goes to that engine's host only, and picking `brave`
never sends a Tavily credential anywhere.

```text
xencode config set search_provider wikipedia
xencode config set search_provider searxng
xencode config set search_searxng_url http://127.0.0.1:8888
xencode config set brave_api_key <key>
xencode config set search_provider none      # takes the tool away again
```

A search is a network call whatever else the mode says, so it asks in ask,
edit-allow and all-allow, and is refused outright in plan and autonomous. The
question is shown in full on the prompt, because the question is what leaves:

```text
web_search rust ownership
search: rust ownership
  (the question is sent to the search provider named in the config — `xencode config show` says which one that is — and what comes back is titles, addresses and short snippets. Nothing in that list is read.)
```

Answering does not buy the next one: "allow for the session" does not apply to
this tool, the same rule as `web_fetch`. The answer comes back as a numbered list
of the engine's own titles, links and snippets, headed with the engine and the
question, and closed with the note that those are addresses rather than pages —
reading one is a separate request behind `allow_web_fetch` with its own approval.
Verified live on 2026-10-04 against Wikipedia with no key: the question *"rust
ownership borrow checker"* came back as five titles and five `en.wikipedia.org`
addresses in 0.75s.

Two failures are deliberately not presented as a transport error, because they
are facts about the config:

```text
error: web_search is not enabled: `xencode config set search_provider wikipedia` points it at an engine, `searxng` at one you run yourself, and every call still asks before the question leaves
error: search_provider is `searxng` but search_searxng_url is empty: point it at an instance you run, e.g. `xencode config set search_searxng_url http://127.0.0.1:8888`
```

The first is the executor refusing a call that arrives while no engine is named —
from a resumed transcript, say — which an approval cannot override. The second is
why an engine named halfway stays offered instead of quietly disappearing: the
answer names the half that is missing. An engine that answers with nothing is an
answer, not an error, and says so, because a model told "no results" by a failure
spends the next three calls asking the same question. At most 10 results per
call; a question longer than 400 characters is refused. The address of a
self-hosted instance goes through the same guard as a fetch, so
`search_searxng_url` cannot point the request at a private network or the cloud
metadata service.

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

#### The supply-chain report: `xencode deps`

```bash
xencode deps [--path DIR] [--format text|json]
```

`xencode deps` is one command over the dependency checkers that exist on the
machine. It shells out to `cargo-shear` (unused dependencies) and `cargo-deny`
(advisories, bans, licenses), parses their JSON, and prints every finding in one
stream, alongside two facts that need no external tool at all: any crate pinned
at more than one version in `Cargo.lock`, and the delta of the current lock
against the one committed at `HEAD` — the diff a reviewer wants before merging a
dependency change.

**It is report only, on purpose.** Auto-fixing a dependency — removing an import,
bumping a version, dropping a crate — is exactly how the supply chain becomes the
attack, so this command edits no manifest. Each checker it could not run is
named as unavailable rather than counted as clean:

```text
dependency checkers
  cargo-shear (unused dependencies): 0 error(s), 0 warning(s)
  cargo-deny (advisories, bans, licenses): cargo-deny not installed — advisories and licenses are not checked here; use `xencode advisories check` for the offline RustSec/OSV corpus

findings
  [Medium] duplicate-major getrandom pinned at 3 versions: 0.2.17, 0.3.4, 0.4.3
  ...

report only: this command edits no manifest. Re-run a checker's own fix yourself if you accept a finding.
```

That is real output from this workspace: `cargo-shear` runs and currently finds
nothing unused, and the duplicate-major scan reads its list straight from
`Cargo.lock`. The one finding it used to report here is gone — `cargo-shear` had
flagged an unused `dirs` dependency in the CLI manifest, and that declaration has
since been removed. `cargo-deny` is not installed here, so its column says so
instead of pretending the advisory and license checks ran — the advisory side is
covered offline by
`xencode advisories check` (see above), which reads the synced corpus rather than
shelling out. Install `cargo-deny` and the same command fills that column for
real. `--format json` emits the checkers' status and a `findings` array for a
script to read.

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

# Send images with the prompt. Repeat --image for more than one. Needs a
# vision-capable model — `nvidia:moonshotai/kimi-k3` reads them.
xencode query "What is in this image?" \
  --model nvidia:moonshotai/kimi-k3 \
  --image screenshot.png \
  --image diagram.jpg
```
Sampling flags are read by a model served by llama.cpp, and — except `--grammar`
and `--mirostat`, which Ollama has no field for — by a model served by Ollama as
`options` on the request; see
[What a request to Ollama carries](#what-a-request-to-ollama-carries). The prompt
alone, with no flags, goes to the configured default model.

#### Sending images

`--image <PATH>` rides along with the prompt as an image part on the final user
message — never pasted into the prompt text, which would corrupt it — and the
model sees the picture itself. Repeat the flag for several images; they keep the
order you gave.

Each file goes through the same intake the TUI's attach path uses: it is read,
checked to be a real image, capped in size, and shrunk to the longest side a
vision encoder can use. When a file is re-encoded on the way out, the change is
printed on stderr rather than done quietly:

```
image: screenshot.png (sent as image/jpeg (1568×1080, 210 KiB from 2560×1440, 1.2 MiB))
```

Problems are refused before the request goes out, naming the file — a path that
does not exist, or bytes that are not an image:

```
error: cannot read image /tmp/a.png: No such file or directory (os error 2)
error: /tmp/a.png is not a recognized image
```

A turn that has no user message to attach them to is refused as well
(`the attached images could not be sent with this turn …`), because the
alternative is a request answered from the prompt alone that looks like the
model ignored the picture.

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
  turns, a one-off `xencode query` ask, and the eval judge — rather than
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

#### Instructions for the directories a turn is working in

`AGENTS.md` at the workspace root is the whole of what a project tells the model, and it is
sent on every turn whether the turn is about that corner of the tree or not. A repository
can now say more locally: an `AGENTS.md` inside `src/auth/` is read on a turn that changes
a file under `src/auth/`.

The walk starts at the directory of every file `git status` reports as changed and goes up
to, but not including, the workspace root — the root file is already the stable tier and is
not paid for twice. What comes back is one section below the marker that closes the stable
head:

```text
## Instructions For These Directories

### src/AGENTS.md

…

### src/auth/AGENTS.md

…
```

Least specific first, nearest last, so the rule that applies most narrowly is the one read
closest to the question. Four bounds keep it a feature rather than a leak: at most four
files are read, none larger than 8 KiB, directories under `.git/` or `.xencode/` are never
walked into, and a path that resolves outside the workspace — through a symlink or a `..`
in a target — is refused before it is read.

**Granting one of them by path.** A file nobody has trusted carries the same
`[data]` banner the root `AGENTS.md` uses, so nested bytes cannot slip into the
instruction position unmarked. `/trust src/auth/AGENTS.md` grants a single
directory's file: its argument is a path, resolved against the workspace root and
never against whatever directory xencode happens to be sitting in, and it is
refused unless its last segment is `AGENTS.md` and it resolves inside this
project — a source file, a name that climbs out through `..`, git's own `.git/`
store and xencode's state under `.xencode/` are each rejected before the trust
file is touched. Trust is the exact bytes of the one file named, so a turn can
carry a trusted rule beside an untrusted neighbour, and `/trust status` and
`/trust forget` take the same path and answer for that file alone. The section
header therefore draws the line block by block: a block with no data mark is
project convention and applies below the root file, a marked one is not an
instruction. The decision stays a person's — `/trust` is a command typed at the
prompt, and no tool the model can call reaches it. A clean tree, or a turn
touching only files at the root, loads nothing at all, and the prompt is exactly
what it was before.

Two properties are what make this safe to ship on a local model. The section is a *dynamic*
tier: which directories a turn works in changes turn to turn, and anything that moves
inside the cached head costs a full re-prefill, so `xencode` puts it below the marker and
`/ctx kv` proves the head hash still matches. And it is bounded by a cap of its own — 500
tokens for the whole section (`SCOPED_AGENTS_CAP_TOKENS`), the same kind of ceiling
`state.md` and `notes.md` are given — rather than by whatever the root file left of its 1200.
The shared ceiling was the first design and it broke in practice: the root `AGENTS.md` is the
file a real project writes first and fills up, and a project at its cap would have loaded no
directory rules at all. Files are taken nearest-first, so when 500 tokens do not reach all of
them the budget runs out on the directory *furthest* from the work: the rules next to the
edited file stay whole, and `truncated` is reported the way an over-long root file reports
itself. A turn with no room left gets no nested instructions at all rather than a fragment.

`rust/crates/xencode-context-rs/tests/scoped_agents.rs` is the live proof: it builds a
repository with `git init`, commits two packages that each have an `AGENTS.md`, edits one of
them, and reads the assembled prompt — asserting the edited package's rule is in it, the
other package's is not, an untrusted file arrives behind its banner, and two turns working
in two different packages produce a byte-identical stable prefix. The path check is proven
where it lives, in `xencode-context-rs/src/trust.rs`: one turn dirty in two packages, only
one of them named to `/trust`, and the assertion is per block — the granted package carries
no data mark while its neighbour still does, and the workspace's own file is untouched by a
grant that named a directory. Every refusal is a case of its own: `src/main.rs`, a file
under `.git/` and one under `.xencode/`, a name that climbs out of the workspace, and an
`AGENTS.md` that does not exist yet.

#### The lines a person approved are not the lines a budget drops

The root `AGENTS.md` has a ceiling of its own — 1,200 tokens
(`AGENTS_CAP_TOKENS`) — and the ceiling is applied by keeping the *front* of the
file and stopping. So a project whose instruction file outgrew it lost the tail,
silently, from every prompt. The tail is the worst place to lose: `/lesson
approve` appends the sentence a person typed under `## Lessons` at the end of the
file, and a rewind or a refused tool call drafts it there. The approval queue
therefore kept filling a file the product's own prompts had stopped reading.
Measured live, in a scratch repository with a 9,889-character `AGENTS.md`: the
turn reported 1,198 tokens of `AGENTS.md` — under half the file — and `/ctx kv`
reported a 5,604-byte head that held no lesson.

Two sections are now lifted out of the file before that cut is made, and paid for
on a separate ceiling of 300 tokens (`PREFERENCES_CAP_TOKENS`): `## Lessons`,
which is what this product writes, and `## Preferences`, which is what a person
writes for themselves. Nothing else qualifies. The heading must match whole and
case-insensitively, so `## Lessons from the last release` is somebody's prose
heading and stays where it is in the file. The lifted section runs from its
heading to the next heading of level one or two, or to the end of the file; a
deeper `###` belongs to the section it sits under, and a new `# Chapter` closes
it. Every byte of the file ends up in exactly one of the two halves, so the split
cannot lose text on its own.

Three properties are what make this safe to add to the cached head:

- **A block that will not fit its own ceiling loses nothing that was being
  sent.** Past 300 tokens the remainder falls back into the file's budget rather
  than the bin, so lifting a section out can only ever *add* to a prompt. This
  was not free to get right: markdown reads an unclosed `## Lessons` as
  "everything below it", so on a 17,925-character file whose section never
  closes, one head cut alone reached 4,785 characters of it, and a pin that threw
  its own overflow away would have reached 1,213 — 3,572 characters a model used
  to be given and would then not get. The tests assert that rule by rule, on both
  fixtures, rather than by a hand-picked index.
- **The block rides after the file's bulk, not before it.** That is the order the
  bytes sit in already, and it is the cheaper one for the KV cache: rewording one
  lesson parts the two prompts at that line, and everything ahead of it — the
  whole instruction file — stays inside the prefix a provider can reuse.
- **A project with neither heading sends exactly the bytes it always sent.**
  `/ctx kv` on the same repository without a `## Lessons` block reports the
  identical head under the old and the new binary: 5,604 bytes and sha256
  `7f3d1c7f…` both times. With the block, the head is 5,680 bytes and `/egress`
  reports 1,217 tokens of `AGENTS.md` where it reported 1,198, which is the 74
  bytes of the lesson line and the separator between them.

`/ctx kv` prints the head's size and hash and whether two different turns produce
the same one; `/egress` breaks the turn down by whose words it is made of, and the
lifted block counts as the instruction file it came from — same bytes, same trust
question, different budget, which is the only reason it is listed separately.
`rust/crates/xencode-context-rs/tests/preferences.rs` is the proof: fourteen
checks over two long files — one, 9,889 characters, whose lessons block closes the
file, and one, 17,925, whose `## Lessons` never closes — including that an
approved lesson at the end of a long file arrives, that no byte is sent twice,
that an over-long block is cut at its own ceiling and the remainder still rides
the file's, that both sections share that one ceiling and filling one does not
empty the other, that a `#` chapter heading closes the section, and that a file
with nothing human-owned in it produces byte-identical output and no extra line in the
budget report.

One thing this does not do: it does not change what is *allowed* to be written
into `AGENTS.md`. `/lesson approve` is still the only command that appends a
sentence to a file that already exists, and `xencode bootstrap` still only ever
creates the file where there is none.

#### Carrying the task forward: `/ctx fold`, `/ctx promote`, `/ctx drop`

Tier 4 of the prompt is `state.md` — a few lines about the current task that
re-enter the head of every later turn. Nothing wrote it. `/ctx compact` folded a
transcript into a summary for the chat window and left the file alone (the line it
used to print, "state.md only changes when the model flags it", described a writer
that did not exist), and `/ctx archive` only printed the fold prompt it would have
sent.

`/ctx fold` sends it. What comes back is a summary the model wrote out of whatever
the transcript held — including pages it fetched and files it read — so the fold
does not write `state.md`. It writes `.xencode/state.candidate.md` and says what it
took out of the reply on the way:

```console
[CTX]📚 Canonical transcript synced (+3 new) → 3 entries
[CTX]🧠 Folding 3 entries into state.md's shape — asking llamacpp:/home/sree/.cache/llama.cpp/Qwen3-0.6B-Q4_K_M.gguf.
[CTX]📝 Fold checked — 2 fact lines kept
[CTX]   1 line(s) carried a data banner — quoted from a page, a file or a tool result — and were not written.
[CTX]ℹ️ Nothing is durable yet: /ctx promote writes state.md, /ctx drop discards this.
```

That block is the replay in `rust/crates/xencode-tui-rs/tests/state_fold.rs`, whose
recorded answer is a real Qwen3-0.6B fold: the transcript it was given held a tool
failure under a `[data]` source line, and the model carried that line verbatim into
`## unresolved`. A line that arrived under such a banner is somebody else's bytes,
so it is dropped rather than kept-and-labelled; a line with credential-shaped text in
it is written with that text replaced by `[redacted]`. The two caps are the ones tier
4 is already budgeted for — 15 fact lines and 800 tokens of rendered text — trimmed
across completed, decisions and unresolved in turn so no one section is starved,
keeping the model's own first items.

`/ctx promote` is the act that makes a fold durable. It checks the candidate again,
because an edit made in between can break the shape or paste a fetched block back in,
writes `state.md` atomically and removes the candidate. `/ctx drop` discards a waiting
fold and leaves `state.md` as it was. A fold whose every line was quoted data is
refused outright, and a server that does not answer writes nothing at all.

Writing the file cannot disturb what a running server caches, because `state.md` sits
below the system prompt, `AGENTS.md` and `anchor.md`. `/ctx kv` prints both halves of
that claim:

```console
[CTX]🧱 Stable prefix 813 bytes — sha256 7fd5d5d33424fdbca8d6e63dd9f3285fcfb0f8bd49a99b813c2121fe72f147b1 · cross-request identical: ✅ yes
[CTX]🧾 Tier 4 state.md — 77 tokens in the prompt · 3 fact line(s) on disk
```

Those two lines came from this repository against a `llama-server` on that same model,
after a real fold was promoted. With `state.md` moved out of the way the command
reported the identical sha256 and `0 tokens in the prompt · 0 fact line(s) on disk ·
nothing promoted yet`, which is the point: the durable tier is the only thing that
moved.

##### Where a durable fact came from, and when it stops being believed

A summary outlives the code it was written about, and the file above has no way to say
so. `/ctx promote` therefore stamps each line that names a file in the project:

```text
- the token check runs before the handler in src/auth.rs [src:src/auth.rs@d53614d4]
```

The path is the file the line names — picked out of the sentence, and used only if that
path is actually a file here, so a word that merely looks like one is left alone. The
characters after `@` are this repository's current commit, taken from `git`. A line with
no file in it gets no marker, `## working-on` is never stamped (it is the task, not a
claim about the code), and a line that already carries a marker is left as written.

Then every turn reads the tier through those markers before it enters the prompt. A
stamped line is dropped from that turn when the file it cites has disappeared, has
uncommitted changes, or differs from the commit recorded in the marker — which is what
catches a change that was committed on a clean tree, and a rename, whose old path no
longer exists. Dropping is per-turn: nothing is deleted from `state.md`, and the fact
comes back on its own if the file is reverted. If the repository cannot resolve the
recorded commit at all — a shallow clone, or history that has been pruned — the line is
kept and the same row says `· 1 not checkable here (the check could not run on this
repository)`, because an unreadable past is not a disproven one and a tier going in
unverified should not look like a tier that passed.

`/ctx kv` says which lines it left out:

```console
[CTX]🧾 Tier 4 state.md — 14 tokens in the prompt · 1 fact line(s) on disk · 1 dropped as stale
[CTX]   stale: the token check runs before the handler in src/auth.rs [src:src/auth.rs@d53614d4] — the file it cites has changed since; /ctx fold to re-derive it
```

That pair is real output, from `/ctx kv` in the test
`rust/crates/xencode-tui-rs/tests/state_stale_notice.rs`: a scratch repository with one
commit, a promoted fold citing `src/auth.rs`, then an edit to that file. The commit in
your marker is whatever your head was, so those eight characters will not match. The
count of fact lines is read from the file as written, which is why that row still counts
one line on disk while reporting the prompt lost it.

##### When a fact names the code instead of a file

A fact can also claim something about the symbols themselves, with no path in it, and
that claim is checked the same way. `/ctx promote` reads each line for names this
repository declares — a word shaped like a Rust identifier (`validate_token`, `FoldReport`),
or any word written in backticks — and records what it found:

```text
- validate_token rejects an empty token [chk:validate_token]
- reject_request calls validate_token [chk:reject_request,validate_token,reject_request>validate_token]
```

Every later turn re-runs those names against the tree — one search of the working `.rs`
files per turn, shared by all the lines, and only when a line carries a check. The fact
leaves that prompt when a name is no longer declared anywhere, or when a `caller>callee`
pair no longer appears in the file where the caller is defined. The search reads the
working tree rather than the `.xencode/` index on purpose: an index is a snapshot, and a
snapshot goes on certifying a symbol a rename deleted.

Two rules keep this from dropping notes that are true:

- **Only a name this project declares is ever recorded.** `mpsc::unbounded_channel` is a
  dependency's function, so it gets no check — otherwise a line about it would disappear
  the moment that dependency moved, and there would be no way to tell "this name is gone"
  from "this name was never ours".
- **Ordinary prose gets no check.** `auth` and `parse` are English as often as they are
  identifiers; `Rust-first for new code` and `auth is checked before the handler runs` are
  written to `state.md` exactly as you folded them. A word becomes a name only when it
  looks like one or is in backticks. A line records at most four claims, the first four in
  the sentence, because the marker is bytes out of the same budget as the facts.

The panel says which of the two checks failed:

```console
[CTX]🧾 Tier 4 state.md — 14 tokens in the prompt · 1 fact line(s) on disk · 1 dropped as stale
[CTX]   stale: validate_token rejects an empty token [chk:validate_token] — the code it names is no longer declared here; /ctx fold to re-derive it
```

That is the same test as above, taken on after promoting the line, renaming the function
and committing it. And when there is no repository to search at all — a folder copied out
of a project, a machine without `git` — the line is kept and reported as `not checkable
here`, because a tree that cannot be read is not a tree that said otherwise.

`/ctx fold` and `/ctx archive` read this filtered tier too. The fold rewrites `state.md`,
so handing it a fact the code has disproved would let the model re-derive it as a fresh
line stamped with the current commit — the one way a stale note could survive its own
check. On this repository the search costs about 0.1 seconds for 221 tracked `.rs` files
(7471 declaration lines, measured 2026-10-05), once per turn at most.

The markers are bytes, so both caps are re-checked after stamping: a fold trimmed to

exactly 800 tokens and then stamped would otherwise have its last line's marker cut off
mid-word. Trimming takes lines from the tail of each section, never half a line.

##### Notes the agent keeps for itself: `write_note`

A hard compaction rewrites the conversation, and a thing the model worked out two
hours ago is exactly the thing that does not survive a rewrite. `write_note` is a tool
the agent can call — one string argument, no path — that appends that thing to
`.xencode/notes.md`, a file kept outside the transcript. It is the agent's own scratch
pad, not a message for you and not a place to store code: the tool takes no path, so no
call can address any file but that one.

```text
noted: 1 line(s); 1 note(s) on the pad in .xencode/notes.md
```

Three things are refused or rewritten on the way in, because the pad re-enters the
prompt on every later turn and a bad line there is a bad line in front of every future
request:

- **Text someone else wrote.** A note quoting a fetched page, a file body or a tool
  result carries that source's banner, and the line is not stored — the pad holds what
  the agent concluded, not what it was handed. A call whose every line was refused
  answers `nothing written: it quoted fetched or tool output, which the pad does not
  keep as its own words. 1 note(s) on the pad.` and leaves no file behind, so a refusal
  cannot create a scratchpad every later turn reads.
- **A credential-shaped value** is replaced on the way in, the same pattern list the
  prompt uses, and the reply says how many lines were redacted.
- **A note already on the pad** is not written twice; the reply counts the duplicates it
  skipped.

The pad holds the last 40 notes and the oldest go when it is full, and the reply names
the ones that left. Assembly gives it its own tier, `## Notes To Self`, budgeted at 250
tokens, and when the pad is wider than that the **newest** notes are the ones carried —
the same rule the recent-conversation tier uses, and the same margin requirement, so a
turn with no room left sends no notes rather than half a set.

Compaction cannot eat a note, and that is the point of putting it in a file. `/ctx fold`
and `/ctx archive` hand the model the **whole** pad, not the tier's newest slice, under
its own heading in the fold prompt (`# Notes the agent kept for itself`), because a hard
compaction is a note's only route into `state.md` and the tier that survives it carries
250 tokens. Nothing promotes a note on its own: the fold proposes, `/ctx promote` writes.
Being a file the agent writes, `write_note` is a change to the working tree — plan mode
refuses it, and in ask mode it needs the same approval `write_file` needs.

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

### `xencode interop [--agent NAME]... [--timeout SECS] [--repeat N] [--fan-out] [--out PATH] [--format text|json] [--capture-dir DIR] [--trace CAPTURE]`

The `AR-1` probe: launch every installed coding-agent CLI headless on a read-only task in a
scratch git repository, and record what came back. The scratch directory is created in a temp
location and removed afterwards, so the probe never touches your workspace.

```bash
xencode interop
xencode interop --agent codex --agent cline
xencode interop --out probe.json --format json
```

`--fan-out` runs the selected agents **at the same time** instead of one after another, and
reports what the overlap bought. Measured on five agents: 85.3 s run in series, 29.4 s run
together — 2.9x, with the slowest single worker at 25.9 s, which is the floor any schedule
has to clear. It cannot be combined with `--repeat`, since comparing two runs needs them to
happen one after the other.

Agents you have stood down are skipped, and a report prints who it skipped and why. Naming one
explicitly — `xencode interop --agent claude` — runs it anyway.

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

#### `--repeat N`

Runs each agent `N` times and compares. **One run is a reading; two is a check** — and a
single run has been standing in for a fact it cannot support.

```bash
xencode interop --agent cline --repeat 2
```

What is compared per agent: the event vocabulary, the outcome, whether a session id was
issued at all, whether the stream was machine-readable, and the event count. A fact that
differs is reported with **both values shown**, never averaged, and every varying fact is
repeated under "still unanswered" so a difference cannot scroll past.

A session id's *value* is deliberately not compared: two runs issuing two ids is correct
behaviour, and treating it as a difference would report a working agent as inconsistent.

This is what closed part of the `AR-1` gap. On 2026-09-28 all four captured vocabularies
came back identical across two runs, while **cline's event count did not** — 18 then 20, and
16 then 19 on a second pair. So its vocabulary is reliable and its sequence length is not,
which is exactly the distinction one transcript cannot show.

#### `--check-auth`

Read-only. Reports which agents have a config directory on this machine and, for those that
do not, the command that would fix it. It launches nothing, starts no login, and reads no
credential.

Its limit is worth stating plainly: **a config directory is not a runnable agent.** On the
machine this was written on, all six agents have a config directory and three still refuse to
run. So this answers "is there a config directory", and the refusal itself remains the only
trustworthy signal that an agent can spend.

```bash
xencode interop --check-auth
xencode interop --check-auth --agent gemini --agent crush
```

#### `--capture-dir DIR`

The `AR-4` store. Keeps each agent's **whole run** on disk, not only the lines the report chose
to show, in a directory you name:

```text
<DIR>/<agent>/capture/
    raw.jsonl          every line the vendor printed, verbatim and unredacted
    normalized.jsonl   the common events, each naming the raw line it came from
    metadata.json      what was run, what it said it cost, what was truncated
    envelope.jsonl     AR-5's propagation copy — each event with its session,
                       task, agent, worker id, sequence, timestamp and origin,
                       written REDACTED
```

This costs **no extra run** — it writes the bytes the probe already received. The first three
files are the sealed capture: `raw.jsonl` stays **unredacted** (a redacted raw stream is not a raw
stream), because nothing about the run should be hidden from the record. Two things contain that —
a capture exists only when you ask for one by directory, and all four files are written `0600`
through the same owner-only atomic write the rest of xencode's private state uses.

The fourth file is the `AR-5` **envelope**: the shape an event takes when it leaves the capture and
goes somewhere it can be joined or synced — the ledger, the metrics. Unlike the sealed capture, the
envelope **redacts** every credential-shaped string, because it is the copy that propagates. A token
that appeared in a vendor's stream therefore does not reach `envelope.jsonl`, yet each redacted
event still names its exact `raw.jsonl` line, so recovering the true bytes stays a deliberate act on
the sealed capture rather than a lost fact. Every envelope field is in one of four distinct states —
**observed** (the worker said it), **synthesised** (xencode minted it, like the worker id and the
sequence), **unknown** (nobody said it this run), and **unavailable** (this vendor provably cannot,
with the reason — `agy` sends no correlation id on tool calls) — and nothing defaults to a value
that could be mistaken for a measurement: a stored event whose origin field is missing reads as
*not observed*.

With `--repeat N`, each run lands in its own `<DIR>/run-K/<agent>/capture/` so a second run never
overwrites the first's raw stream.

```bash
xencode interop --agent codex --agent cline --capture-dir ../captures
```

#### `--trace CAPTURE`

Read-only, and it launches nothing: render a capture you already paid for back into the trace
view — one event per line, in stream order, each showing the raw line behind it. It reads a single
capture (either `<agent>` or `<agent>/capture`), or, given the root a probe wrote, **every vendor
under it in one pass**, so two agents that agree on nothing else appear in the same rendering.

```bash
xencode interop --trace ../captures/codex
xencode interop --trace ../captures          # every vendor, one view
```

```text
codex — version not reported
argv: (nothing was launched)
exit: 0, 0 ms, 7 raw line(s), 6 event(s)

   1  session_started          raw#1  session 01a0fad6-8890-7952-a734-03106b589824
   2  message                  raw#3  I’ll read `notes.txt` and return its contents as requested.
   3  tool_started             raw#4  command_execution [item_1]
   ...

run completed: yes
0 of 6 event(s) are xencode's inference, the rest the worker's own words
```

The last line is the point: a normalised stream hides the adapter's judgement, so the count of
events xencode **inferred** versus events the worker **said** is printed right there, and any field
invented to line two vendors up is labelled `xencode`, never blended into the vendor's own words.

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
it by writing your own `model_advice.json` in the settings directory (same shape:
`as_of`, `ollama_preference`, `tiers[]` of `max_file_bytes` and `gguf` entries) —
xencode then reads yours and the `from:` line says which file it answered from. A
user file that fails to parse is refused out loud and the shipped table is used
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
xencode history digest src/main.rs         # last-touch per hunk + 5 recent subjects, ~250 tokens
xencode history digest src/main.rs --json  # the same with char count and cap
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
xencode colab preflight --generate-key # also create colab_ed25519 in the settings dir if missing
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
until `/v1/models` answers, and writes `colab.json` to the state directory — then points
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
xencode colab up --dry-run            # what would start and what config.json would gain
```

`--dry-run` stops before anything happens: no preflight (which would create the
SSH keypair), no session, no forward, no state file, no config write. It prints
the resolved session, runtime, GPU, model, weights source, quant, and the ports,
then the keys a successful bring-up would rewrite and what they would become —
computed by the same code the real bring-up calls, so the preview cannot disagree
with it:

```
$ XCODE_CONFIG_DIR=/tmp/db8 xencode colab up --dry-run
colab up --dry-run — nothing was started and nothing was written.
  session:   xencode-vm
  runtime:   llama.cpp   gpu: T4   model: Qwen/Qwen2.5-7B-Instruct-GGUF
  weights:   hf   quant: Q4_K_M (the default the VM serves)
  forward:   http://127.0.0.1:18000 → the VM's port 18080
  config.json would change: llama_cpp_url: http://localhost:8080 → http://127.0.0.1:18000
  config.json would change: remote_base_url:  → http://127.0.0.1:18000/v1
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

### Where xencode keeps its files
Four kinds of file, in four directories, so that clearing one of them cannot
take the others with it:

| Kind | Directory | What lives there |
| --- | --- | --- |
| settings | `$XDG_CONFIG_HOME/xencode` | `config.json` and its backups, the Colab bridge key pair, `model_advice.json`, `skills/` |
| state | `$XDG_STATE_HOME/xencode` | `audit.jsonl`, `conversation_memory.json`, `colab.json`, `last_panic.log`, `llamaserver.pid` |
| cache | `$XDG_CACHE_HOME/xencode` | cached responses, and the advisory corpora under `advisories/` |
| downloaded models | `$XDG_DATA_HOME/xencode` | GGUF weights — kept out of the cache so a cache cleaner can never delete a multi-gigabyte download |

With the variable unset each one falls back to the usual place: `~/.config`,
`~/.local/state`, `~/.cache`, `~/.local/share`.

An installation that has never been migrated is still one `~/.xencode` directory,
and xencode keeps reading it — a modern directory wins only once it exists,
because a half-finished move is worse than an old layout. `xencode paths` prints
which of the two answers each kind. `XCODE_CONFIG_DIR` points all four at one
directory and takes precedence over everything above; `xencode doctor` then
reports every kind as pinned to it.

A project's own files do not move: `.xencode/` inside the workspace holds the
retrieval index, the recorded sessions and `cache/metrics.jsonl`.

### `xencode paths [--format text|json]`
Print where each kind of file is read from, and name the ones that are still in
`~/.xencode`.

```bash
xencode paths
xencode paths --format json
```

The JSON form gives every kind twice — `in_use` is where the files actually are,
`modern` is where they would go — with `legacy: true` on the ones the migration
has not moved, and `override` set to the `XCODE_CONFIG_DIR` root when one is in
effect.

### `xencode migrate [--dry-run]`
Move the contents of `~/.xencode` into the four directories above.

```bash
xencode migrate --dry-run   # print the whole report, change nothing
xencode migrate
```

Nothing happens on its own; this is the only way files move. The rules the report
holds itself to:

- A destination that already has a file of that name is never overwritten. The
  migration says so, names the file, and reports that xencode now reads the one
  that was already there — which is true, since a directory that exists is what
  makes the old one stop being read.
- `~/.xencode/cache` and `~/.xencode/models` move as directories, onto the cache
  and data directories, so every cached response and downloaded weight keeps the
  name it has. Everything else is sorted by what it is.
- A file that cannot be moved in one step is copied and then removed, and a copy
  that fails partway removes its own half-written destination — an incomplete new
  directory would otherwise be read as the finished one.
- A directory's permissions come with it, so a `0600` `config.json` is still
  `0600`. An existing destination directory is never made more readable than it
  already was.
- `~/.xencode` is deleted only when it is empty. A refusal, or anything you added
  to it yourself, leaves the directory in place.
- Under `XCODE_CONFIG_DIR` the command refuses, because there is nothing to move
  and a person pointing xencode at a directory should not have it emptied.

Both forms print the same report; the dry run only says it would happen.

### `xencode config <action>`
Configuration management. Config is a setting, so it lives in the settings
directory above — `$XDG_CONFIG_HOME/xencode/config.json`, or `~/.xencode` for an
installation that has not been migrated; set `XCODE_CONFIG_DIR` to point xencode
at a different directory.

```bash
xencode config show
xencode config set default_model qwen3:4b
xencode config set mcp_timeout 30
xencode config set llama_cpp_args "--n-gpu-layers all --device Vulkan1"
xencode config set --dry-run default_model qwen3:4b
xencode config reset
```

`--dry-run` recognises the key, validates the value against the configuration on
disk, and prints the change without writing anything:

```
$ XCODE_CONFIG_DIR=/tmp/db8 xencode config set --dry-run default_model qwen3:4b
would set default_model = qwen3:4b — nothing written (--dry-run)
```

A provider credential is set the same way and is never printed back:

```bash
xencode config set openai_api_key <key>
xencode config set qwen_api_key ""          # unset it here
xencode config set --dry-run openai_api_key <key>
would set openai_api_key (value not shown) — nothing written (--dry-run)
xencode config set openai_api_key ""
cleared openai_api_key — the environment variable named for the provider, if any, now supplies it
```

The value may also name a program instead of holding a key:

```
$ xencode config set openrouter_api_key "command:secret-tool lookup service xencode account me"
set openrouter_api_key = a command reference — the secret is read from that command and stays out of config.json
note: the command answered with a key.
```

A credential is then read from three places, in this order:

1. the value in `config.json` — writing it there was a deliberate act, so it wins;
2. a `command:` reference. The stored text is `command:<program> <args>`, that
   program prints the key on its first line, and `config.json` holds no secret, so
   a directory that gets backed up or synced still carries nothing. The program is
   run **directly, not through a shell**: it has to be on `PATH`, and `*` or
   `$HOME` written into the reference will not be expanded. It gets no terminal
   and ten seconds to answer, so a helper waiting on a passphrase nobody can see
   is stopped and named rather than hanging the request, and what it writes on
   standard error is dropped instead of shown. `--dry-run` stores nothing and does
   not run the command; `config set` runs it once immediately, so a reference that
   cannot be read is found at the moment it is written down rather than in the
   middle of a turn.
3. the environment variable named for the provider: `API_KEY_OPENAI`,
   `API_KEY_OPENROUTER`, `API_KEY_GEMINI`, `API_KEY_QWEN`, `API_KEY_REMOTE` (or
   `XENCODE_API_KEY`, which fills only the `remote:` endpoint you run yourself —
   there is deliberately no variable that applies to every provider, because one
   name cannot say which account the key belongs to), and `API_KEY_NVIDIA` (or
   `NVIDIA_NIM_API_KEY`, the name this project shipped first). A blank counts as
   unset on both sides, so an empty `export` cannot hide a configured key.

`config show` names which of the three a credential came from, and never the
value:

```
$ xencode config show
  "api_keys": {
    "google_gemini_api_key": null,
    "nvidia_api_key": null,
    "openai_api_key": "set in config.json (value not shown)",
    "openrouter_api_key": "command reference — command:secret-tool lookup service xencode account me",
    "qwen_api_key": "set in the environment as API_KEY_QWEN",
    "qwen_client_id": null,
    "remote_api_key": null
  },
```

The Settings panel's key rows read the same way: a stored key shows as dots, a
reference shows in full because it names a program rather than a secret, and a
row with nothing stored says which environment variable is answering for it.
Typing `command:…` into one of those rows stores a reference.

A Linux desktop keyring (`org.freedesktop.secrets`) is reachable through tier 2 —
`command:secret-tool lookup …`. Worth being exact about what that buys: it keeps
the secret out of a file that is backed up, synced or shared, and it does nothing
against a process running as you, because the keyring answers anything in your own
session and is not available over SSH or headless at all. It is a different place
to keep the same secret, not a stronger lock on it.

Every save that replaces the file first copies what was there to a timestamped
`config.json.bak.<UTC time>` beside it, owner-only like the config itself, and
keeps the newest five. A save that would write exactly the bytes already on disk
adds no copy — the settings panel saves on every change, and unchanged copies
would push the useful ones out. So the file you just overwrote is one `cp` away:

```
$ ls -l /tmp/db8 | grep config.json.bak
-rw------- 1 sree sree   49 Oct  3 10:40 config.json.bak.20261003T051028.493226852Z
-rw------- 1 sree sree 1510 Oct  3 10:40 config.json.bak.20261003T051028.499388198Z
```

A value that begins with a dash is taken as the value rather than as an option to
`config set`, because the line `xencode hw probe` hands back to paste starts with
`--n-gpu-layers` and quoting it was refused until this was fixed.

The file carries its own format version as `config_version`, which `xencode`
writes and reads; it is not a `config set` key. A file naming an *older* version
is read, converted, and stamped with the current number on the next save, so a
`config.json` written before the key existed still works. A file naming a
*newer* version than this binary is refused — reading and writing both — because
an older binary cannot see fields it was not built for and would drop them:

```
$ XCODE_CONFIG_DIR=/tmp/db2-newer xencode config set default_model qwen3:4b
error: /tmp/db2-newer/config.json declares config version 9, and this xencode only knows versions up to 1. It was not read, and nothing was written to it — an older binary cannot see the fields a newer one added and would drop them on save. Run the xencode that wrote this file, or point XCODE_CONFIG_DIR at a config this one can read.
```

The same refusal applies to `config show`, and `xencode doctor` reports it as two
rows: `config` (the settings are unread, so that run is on defaults) and
`config:version` (the two numbers). A JSON value that is not an object — a list,
a bare string — is refused the same way rather than read as "all defaults", which
is what used to happen.

A file that is not JSON at all — the trailing comma a hand edit leaves — is refused
too, and refused **at the write** as well as the read. That second half is what
matters: plenty of commands load the config, fall back to defaults when the load
fails, and save. Before this, one of those succeeded and replaced the file with the
default block, keys included:

```
$ XCODE_CONFIG_DIR=/tmp/df1 xencode llamacpp set-path /tmp/some.gguf
error: /tmp/df1/config.json is not readable JSON: trailing comma at line 1 column 83. Nothing was read from it and nothing was written to it, so whatever the file held is still there. Repair it by hand or restore a `config.json.bak.<time>` copy from beside it.
```

`xencode doctor` says the same in its `config` row, and its `config:version` row
reads `absent` — there is no readable object there to ask. Repair the file by hand
when you can read what broke. When you cannot, `xencode config reset` is the one
command allowed to write over it, because discarding the file is what it is for —
and it copies the unreadable bytes to a `config.json.bak.<time>` on the way, so the
settings are still there to read out of the copy:

```
$ XCODE_CONFIG_DIR=/tmp/df1 xencode config reset
configuration reset to defaults
$ ls /tmp/df1
config.json
config.json.bak.20261005T061357.025100159Z
```

The first of those two is the readable default block; the second is what was in the
file before, byte for byte.

An empty or whitespace-only `config.json` is *not* treated as damage: it is what
`touch` leaves, there are no bytes in it to protect, and refusing to save over it
would be a dead end.

The interactive screen refuses nothing, because a session on defaults is still a
usable session — it says out loud that this is what happened. On the first frame a
toast names the situation, and the full refusal is written into the chat, where it
stays after the toast has faded:

```text
 ⚠ settings not read — this session starts on defaults

settings not read: /tmp/df6-live/config.json is not readable JSON: trailing comma
at line 1 column 59. Nothing was read from it and nothing was written to it, so
whatever the file held is still there. Repair it by hand or restore a
`config.json.bak.<time>` copy from beside it.
```

That is a frame captured from the running interface, not a sketch: the file named
in it was unchanged by the session that opened on it.

`config set` keys (values are validated; `config show` prints the JSON):
`mcp_servers`, `agent_hooks` and `model_profiles` are nested structures, so they are edited directly in the JSON instead, or managed in the TUI where a panel exists for them.

| Key | Type | Notes |
|-----|------|-------|
| `default_model` | string | e.g. `qwen3:4b` |
| `ollama_url`, `llama_cpp_url` | string | provider endpoints |
| `remote_url`, `remote_key` | string | Remote / Colab endpoint (any OpenAI-compatible server, e.g. `http://127.0.0.1:18000/v1` + its bearer token); empty `remote_key` clears it. Its bearer token also comes from `XENCODE_API_KEY` / `API_KEY_REMOTE` when nothing is stored here |
| `openai_api_key`, `openrouter_api_key`, `google_gemini_api_key`, `qwen_api_key`, `nvidia_api_key` | string | Provider credentials, `remote_key` included. Each takes a key, or `command:<program> <args>` to keep the key out of the file, and each has an environment variable of the same name if the file says nothing — see the three tiers above. `config show` reports where one lives and never its value |
| `nvidia_api_key` | string | NVIDIA NIM token for `nvidia:<vendor/model>` (e.g. `nvidia:deepseek-ai/deepseek-v4.1-flash`); empty clears it. Also read from `NVIDIA_NIM_API_KEY` when unset here — the config value wins. **Generate it from the model's own page** (`build.nvidia.com/<vendor/model>` → Get API Key), not from the account page: an account-wide key lists models but every call returns `404 … Not found for account`. Expect 250–300s on a cold start, raise `response_timeout` (default 30s) accordingly, and give `max_tokens` room for the model's reasoning tokens |
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
| `cost_budget_usd_micros` | number | Warning threshold for one conversation's spend, in millionths of a dollar ($5.00 = `5000000`). Unset by default; it warns in the status bar and never refuses a request. Spend is priced the way every cost report prices it — from `.xencode/pricing.json` in the project, and from the fetched listing for a model that file does not name when `price_lookup` is on — see [Where a price comes from](#where-a-price-comes-from). **Not a `config set` key** — edit it in the JSON. |
| `price_lookup` | bool | Whether a cost report is allowed to read a price off `.xencode/cache/price-lookup.json`, the copy of a public catalogue that `xencode prices fetch` wrote. `true`/`false`; empty is refused (`invalid boolean: `), because there is no sensible reading of "clear it" for a switch that already defaults off. Off by default, and turning it on authorises nothing to dial out: it decides only whether the copy already on disk is consulted, and a rate you wrote in `pricing.json` still outranks anything in it — see [Where a price comes from](#where-a-price-comes-from) |
| `power_cents_per_kwh` | number | What a kilowatt-hour costs where this machine runs, in cents — `12.5` for 12½¢. Used to price the electricity a **local** generation drew, from the kernel's own power counter: the `⚡` line a finished turn prints and the `energy_uj` field on its metrics row. Unset, the line still shows the watt-hours and says `no $/kWh set`, because the machine drew the power either way and only the price is unknown. `config set power_cents_per_kwh ""` clears it again; a negative tariff, or one above 1000 cents, is refused where it is written — see [What a local turn cost in power](#what-a-local-turn-cost-in-power) |
| `budget_tokens_per_day` | number | What one calendar day may spend, in prompt plus completion tokens. One of four daily caps — see [What a day's caps do when they are passed](#what-a-days-caps-do-when-they-are-passed). Anywhere from 1 to a billion tokens; an empty value clears it again |
| `budget_energy_wh_per_day` | number | What one calendar day may draw at the wall, in watt-hours, from the kernel's own package counter. On a machine that publishes no counter nothing can ever be weighed against it, and `config set` says so as it is written. From 1 to 24000 watt-hours; empty clears — see [What a day's caps do when they are passed](#what-a-days-caps-do-when-they-are-passed) |
| `budget_usd_micros_per_day` | number | What one calendar day's tokens may cost at the rates in `.xencode/pricing.json`, in millionths of a dollar ($5.00 = `5000000`). Provider spend only, and a model the table does not know counts as nothing against it, so an unpriced day cannot pass this cap. From 1 micro-dollar up to $10,000 a day; empty clears — see [What a day's caps do when they are passed](#what-a-days-caps-do-when-they-are-passed) |
| `budget_minutes_per_day` | number | What one calendar day's turns may take in the aggregate, in minutes — the seconds the turns themselves ran, not the time the interface sat open. From 1 to 1440 minutes, which is a whole day; empty clears. Crossing any of the four buys the **next** turn down one rung of `hardware_profile` and never refuses anything — see [What a day's caps do when they are passed](#what-a-days-caps-do-when-they-are-passed) |
| `llama_cpp_temperature`, `llama_cpp_top_k`, `llama_cpp_min_p`, `llama_cpp_max_tokens`, `llama_cpp_seed` | number | llama.cpp sampling defaults, read from the JSON; `config set` does not accept them, and the TUI's Settings panel covers the same fields. An unset one sends nothing and the server decides — see "Repeatable answers" above. |
| `cache_enabled`, `memory_enabled` | bool | `true`/`false` |
| `layout` | string | TUI body arrangement: `classic`, `chat-first`, `zen`, or a name declared in `layout_templates` (unknown → classic at render, with the reason printed by `config set` and toasted in the TUI) |
| `layout_templates` | object | body layouts you wrote, `"name" → tree`, chosen by setting `layout` to that name; `Ctrl+U` and the Settings Layout row cycle the presets and these names together. A tree is a leaf `{"leaf": {"slot": "editor", "focus": "editor"}}` or a split `{"split": {"horizontal": true, "parts": [[child, share], …]}}`; slots are `explorer`, `editor`, `chat`, `input`, `terminal`, foci are `explorer`, `editor`, `chat`, and a share is `{"percent": n}`, `{"min": n}` or `{"length": n}`. Example — editor and chat, 70/30: `"editor-first": {"split": {"horizontal": true, "parts": [[{"leaf": {"slot": "editor", "focus": "editor"}}, {"percent": 70}], [{"leaf": {"slot": "chat", "focus": "chat"}}, {"percent": 30}]]}}`. Not settable with `config set` (as with `mcp_servers`, it is a nested structure) — edit the JSON, or `config show` to read it back. A template that cannot be built (mistyped slot, a split of one child, a zero share, a shape this build does not know) is refused by name and renders classic; it never fails the load, so the rest of your config survives it |
| `layout_views` | object | saved arrangements, `"slot" → tree`, for the `Ctrl+1`…`Ctrl+9` view chords — the same tree shape `layout_templates` takes, under one of the nine slot names (`Code`, `Chat`, `Terminal`, `Focus`, `Review`, `Split` for the first six, then `7`, `8`, `9`). Six of the nine are seeded in the program, so an empty map still switches screens; `Ctrl+Shift+<digit>` writes a slot into this map, and a stored entry replaces its seed. Not settable with `config set` (as with `layout_templates`, it is a nested structure) — edit the JSON, or `config show` to read it back. An entry that cannot be read is refused by name with the reason and the rest of the config still loads |
| `active_theme` | string | UI theme: `ocean`, `midnight`, `forest`, `terminal`, `dracula`, `solarized`, `nord`, `light` (default `ocean`); cycled live in the TUI. **Not a `config set` key** — `xencode config set` has no arm for it, so edit it in the JSON. An unrecognised name is *not* rejected: `ThemeColors::get` falls through to the `ocean` palette, so a typo renders as ocean and reads back as the typo. Check the spelling against the eight above. |
| `hardware_profile` | string | How much project context a run may spend: `auto` (default — chosen from this machine's memory), `low`, `balanced`, `high`; anything else is rejected by `config set` and reported by a run if it is already in the file. See [Which hardware profile the budget spends against](#which-hardware-profile-the-budget-spends-against) |
| `rounded_borders` | bool | rounded panel corners |
| `show_scrollbars` | bool | scrollbars on chat & explorer panes |
| `show_line_numbers` | bool | editor line-number gutter + current-line highlight |
| `mouse_capture` | bool | whether xencode asks the terminal for the mouse at all: on (default) reads the wheel, pane clicks and divider drags; off hands them back, which is how a terminal's own drag-select of text comes to work again. Takes effect mid-session from the TUI's `Mouse Capture` row, and next start from here |
| `agent_approval` | string | agent tool-approval mode: `ask` (edits and shell prompt, reads free), `edit-allow` (edits free, shell prompts), `all-allow` (all local tools free, but a stranger's MCP server and the network still prompt), `plan` (read-only **and enforced**: an edit, shell call or external tool is denied, not merely prompted, so a plan turn cannot write even on a stale "always allow" grant — a grant only replaces a prompt, never a denial; and in this mode the write and shell tools are not offered to the model at all, only the read-only ones), or `autonomous` (reads, edits and shell run free with no human, but anything reaching an external MCP server or the network is denied rather than asked, because there is nobody to answer a prompt; this is what separates it from `all-allow`, which still lets those two prompt). Unknown → `ask`. Once the session has touched secrets (a key file read, an env dump, secret-shaped output), every later shell call asks in every mode — including past an `all-allow` setting or a session grant given before the secrets were read. A one-shot approval at the prompt still runs the call; headless, the prompt's absence denies it as before. All five are cycled live in the TUI's `Agent Approval` row. |
| `agent_max_rounds` | integer | assistant→tool rounds allowed per chat turn before the model must answer in prose (`1`–`64`, default `16`) |
| `agent_command_timeout` | integer | seconds the agent's foreground `run_command` may take before it is killed (`1`–`600`, default `30`); slow work belongs in `background_start` |
| `agent_repair_max_iters` | integer | how many times one chat turn may send a failing project check (`cargo test`, `cargo clippy`, discovered from a `Cargo.toml` in the workspace root) back to the model for another repair round before the turn ends reporting the task incomplete (`0` turns the exit-code "done" gate off entirely; default `3`). The checks run through the same approval gate as any shell command, and only a real exit `0` finishes the turn |
| `agent_fallback_models` | list | comma-separated ordered alternates for the agent's turns (I4-01), e.g. `xencode config set agent_fallback_models "qwen2.5:14b,google_gemini:gemini-2.0-flash"`. The configured default model is always tried first, so this list holds only fallbacks (duplicates of it are dropped). A candidate is abandoned — and the chain moves on — only when it failed **before emitting any token** and the error is not our own response-decode failure; a token already on screen, or a `Parse` error, fixes the model in place. Each candidate gets one attempt per step and the transcript records a `[FALLBACK]` line when the chain moves. A candidate that would send the conversation somewhere the primary would not — a cloud API as the alternate for a local model, or the reverse — is never tried, and the transcript names it as skipped instead; a `remote:` endpoint counts as local only when its configured URL points at this machine (`localhost`, `127.x`, `::1`, `.local`). `xencode query` is single-shot and does not use this chain. An empty list (the default) disables fallback. |
| `session_recording` | bool | Write down every model call of an agent turn — the request, the response bytes as they arrived, and what each tool returned — to `.xencode/cache/sessions/<run-id>.jsonl`, so `xencode replay` can run that turn again. Off by default. Only the routes whose bytes this program reads itself are recordable: Ollama, llama.cpp, a `remote:` endpoint and OpenRouter. Asking for a recording of an Anthropic, Gemini or Qwen model is refused with the reason, because those have their own readers and a "recording" of them would be a paraphrase. |
| `allow_cloud_models` | bool | Whether a prompt may reach an internet service at all. Off by default — and off for a config written before the key existed — so `qwen:…`, `google_gemini:…`, an OpenRouter-style `vendor/model` when an OpenRouter key is set, and a `remote:` endpoint whose URL is not this machine are refused before a connection is opened, with the refusal naming this key. A key in `api_keys` is not permission for the trip; it identifies you to the provider. The TUI status bar prints the rule in force (`🔒 local only` / `🌐 cloud allowed`) and Settings → Providers has a **Cloud Models** row that toggles it. |
| `allow_online_docs` | bool | Whether the agent's `read_docs` tool may fetch a crate's documentation when cargo has not unpacked it on this machine. Off by default, and independent of `allow_cloud_models` — turning one on does not turn on the other, because one is a prompt leaving and the other is a text file arriving. With it off, `read_docs` answers from cargo's own copy and says what else would be needed to get more. Open it with `xencode config set allow_online_docs true`. |
| `allow_web_fetch` | bool | Whether the agent is offered the `web_fetch` tool, which reads one page or API response at an address **the model names**. Off by default, and independent of both switches above. Turning it on only offers the tool: every call stops at the approval prompt showing the exact address and whether that address could land, and "allow for the session" does not apply to this tool — a yes about one page is not a yes about the next host. Approval is not a route into this machine either: the address is resolved and refused before the connection, and again at every redirect, so RFC1918, carrier-grade NAT, link-local and cloud-metadata addresses are unreachable even after a `y`, while `127.0.0.1` is allowed so a local dev server stays fetchable. The text handed back is capped at 30 000 characters and says how much of the page was left out. A page the server reports as missing buys one more request on the same address — its root `/llms.txt`, the index some documentation sites publish for models — returned labelled as that index, and reported as a plain miss when there is none. Open it with `xencode config set allow_web_fetch true`. |
| `search_provider` | string | Which search engine, if any, the agent's `web_search` tool asks. `"none"` (the default) leaves the tool unoffered; the other names are `wikipedia`, `searxng`, `brave` and `tavily`. There is no default public instance on purpose: DuckDuckGo's free endpoints answer this machine with a bot CAPTCHA or `410 Gone`, a public SearXNG instance refuses to serve JSON, and MDN's JSON search endpoint is `404` (measured 2026-10-04), so a tool built on one would break weekly. Wikipedia is the keyless engine that does answer and covers people, places and concepts; `searxng` needs `search_searxng_url`; `brave` and `tavily` need their own key. The name is checked against that list as you type it, so a typo is told at the keyboard rather than at the first search of the next session. Every call still asks: a search sends the model's question off this machine, and a yes about one question is not a yes about the next. |
| `search_searxng_url` | string | The address of the SearXNG instance to ask, used only when `search_provider` is `searxng`. Must be an `http://` or `https://` URL, or empty to clear it (`config set search_searxng_url ""`), and any trailing `/` is trimmed as it is stored. It has to be an instance you run, because that is the only kind that serves JSON. Whatever it holds is resolved and refused before the connection, by the same address guard `web_fetch` uses, so it cannot point the search at a private network or the cloud's metadata service. Setting `search_provider searxng` without this is answered by naming the missing half rather than by failing a request. |
| `run_command_sandbox` | bool | Run each `run_command`, `background_start` and shell hook inside a `bubblewrap` (`bwrap`) mount namespace. Off by default; open it with `xencode config set run_command_sandbox true`, and it needs `bwrap` installed. With it on: the workspace and `~/.cargo` are bind-mounted writable so a build still works, the rest of the home directory is replaced by an empty tmpfs so `~/.ssh` and the files under it are *absent* rather than merely unreadable, and the network namespace is dropped. A single call that must reach the network passes `net: true` (an argument on `run_command` and `background_start`); shell hooks get no such grant and always run with the net off under the sandbox. There is **no silent fallback**: if the setting is on and `bwrap` is not present, the command is refused with the reason instead of being run unsandboxed. It bounds what a command can read outside the project — it does not contain the compiler: `build.rs` scripts and anything reachable from the writable workspace run free inside, so this is a fence on the home and the network, not a full jail. |
| `mcp_timeout` | integer | seconds a server may take to handshake and answer before it is reported failed (`1`–`300`, default `30`) |
| `model_profiles` | list of objects | saved profiles the TUI's Custom Models panel (J-05) shows: `{ "name": "...", "model": "ollama:qwen2.5:7b", "temperature": 0.2, "max_tokens": 2048, "for_task": "bugfix" }`. `temperature` and `max_tokens` are optional — omit them and the panel renders "unset — the server decides" and sends nothing. `model` takes exactly the form `default_model` does. `Enter` applies a profile to the next turn; `s` in the panel writes the whole list back here; `f` cycles `for_task` through `bugfix`, `general` and none. There is no `top_p`: no provider path in this workspace sends it, and only llama.cpp receives these two knobs in the request body |
| `model_routing` | bool | Whether a profile's `for_task` mark is acted on by itself. Off by default, so a marked profile still only applies by hand. On, the first profile whose mark matches the turn runs that turn on its model: `bugfix` for a prompt that says something is broken (`fix`, `fails`, `crash` and similar words), `general` for every other prompt, and a mark naming a reading this version does not have (or no mark at all) matches nothing. A profile that would move a llama.cpp model is refused instead — a running `llama-server` holds one model at a time — and the chat prints why. See [Turn routing](#turn-routing) |
| `mcp_servers` | object | MCP servers to offer as tools: `"name" → { "command": "...", "args": [...], "env": {...} }` to spawn, or `"name" → { "url": "https://…", "headers": {"authorization": "Bearer …"} }` for a hosted endpoint (exactly one of `command`/`url`; a token in the URL is shown masked, header values never). Besides tools, a server's listed resources and prompts are readable via `/mcp read <server> <uri>` and `/mcp prompt <server> <name> [key=value …]`; nothing is started until you run `/mcp` |
| `agent_hooks` | object | shell hooks around **approved** agent tool calls: `"before"` and `"after"` maps from an exact tool name (or `"*"` for every tool) to a command run via `sh -c` in the workspace root. A failing `before` hook vetoes the call (nothing runs, no rewind point, output shown as `error: pre-hook vetoed this call`); a passing one has its output prepended to the result. The `after` hook always runs and its output is appended. Hook output is capped like `run_command` (stderr merged, tail kept). Each hook is handed its event as JSON on **stdin** — `{"hook_event_name": "PreToolUse"\|"PostToolUse", "tool_name", "tool_input", "cwd", "session_id"}` — so a script can read the target out of `tool_input` and decide per call (e.g. veto one `write_file` by its path); the event is never passed in the command line, where `/proc` would expose it |

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

Because the event arrives on stdin, a hook can act on the specific call rather than
on the tool name alone. This `before` hook vetoes only a `write_file` aimed at a
protected path and lets every other write proceed (the payload is compact JSON, so
`"path":"` sits directly against the value):

```json
{
  "agent_hooks": {
    "before": {
      "write_file": "grep -q '\"path\":\"\\.env\"' && { echo '.env is protected' >&2; exit 2; }; exit 0"
    }
  }
}
```

A non-zero exit still vetoes (`error: pre-hook vetoed this call`), so here a write
whose `tool_input.path` is `.env` is stopped before anything is written, and nothing
about the event is passed in the command line itself.

### `xencode cache <action>`
Response cache management: `stats`, `clear`, `gc`.

```bash
xencode cache stats
xencode cache clear
xencode cache gc --max-mb 500
```

`gc` drops the oldest cached responses until the ones left fit under the size you
asked for (`--max-mb` counts 1 048 576 bytes to the megabyte), and prints how much
went. Oldest means last written: a read refreshes an entry in the running process
but is not recorded in the file, so the file cannot claim to know which answers
were read recently. The advisory corpora under `advisories/` are not counted and
cannot be removed — they are a download, not an answer — so the cap is a cap on
the responses. Only the cache directory is touched: the settings, the session
records and a downloaded model are in other directories, which is what makes a
command like this safe to run.

### What a local turn cost in power

A local model is not free because no invoice arrives for it: the machine drew
power while it answered, and somebody paid for that at the meter. When a chat
turn on a local model finishes, xencode prints one line saying what it drew:

```
⚡ ≈ 0.03 Wh · ≈ $0.000004 · 15 s — CPU package only; no graphics power was
reported · estimated, this machine only
```

That is real output from a 15-second answer off a local `llama-server`, with the
tariff below set to `12.5`. The figure is a **reading**, not a model of what the
turn should have cost. It
comes from the kernel's own energy counter (`/sys/class/powercap/intel-rapl:*`),
read when the turn started and again when it ended, so the number is the joules
between those two readings. Three things follow from that, and each of them is
said out loud in the line rather than left for you to discover:

- **The counter is the whole CPU package.** A compile, a browser, or your
  neighbour's container all sit in the same total. Nothing in userspace can take
  them back out, so the figure is an estimate of the turn's share, and it is
  labelled as one.
- **A discrete GPU is polled, not metered.** Where `nvidia-smi` will answer
  `power.draw`, the two polls at the ends of the window are averaged and added to
  the total. Where it answers `[N/A]` — which is what a laptop whose card is
  switched off at the connector does — the line ends with `CPU package only; no
  graphics power was reported`, and the card is missing from the total rather
  than counted as zero.
- **A machine with no counter has no number.** An AMD or Arm box that publishes
  no package domain reads `energy unknown · 46 s — this machine reports no energy
  counter to read` — the seconds are the only thing such a machine can be
  measured against — with the reason on the same line, and no price on it even
  when a tariff is set, because the setting is not what is missing here. Nothing
  is drawn as a turn that cost nothing, and a three-digit watt-hour figure is
  never rounded down to `0.00 Wh`, because that rendering is the one that reads
  as free.

The price comes from `power_cents_per_kwh` in your settings (see the table
above). Without it you get the watt-hours and the words `no $/kWh set` — the
electricity was bought, the tariff just was not written down. Set it from your
own bill:

```bash
xencode config set power_cents_per_kwh 12.5
```

What is measured is also recorded, in the same per-project metrics log `/cost`
reads: `energy_uj`, `elapsed_ms`, `power_w` (mean watts over the window) and
`est_cost_micros`. Only a turn whose prompt stayed on this machine is priced
that way; a cloud turn's electricity went into a provider's meter, so putting
this reading on its row would bill the same seconds twice — once for the work
and once for a CPU that was only waiting on a socket. Rows written before this
existed have none of the four fields and still read, which is why the log stays
append-only. `/cost` prices the same turns by tokens from `.xencode/pricing.json`
— that is the provider's bill, and this is yours; the two are never added
together.

### Where a price comes from

A cost report reads its rates out of two documents, and they are not the same
kind of document:

| Document | Written by | What it is trusted for |
| --- | --- | --- |
| `.xencode/pricing.json` | you, by hand | every model it names, forever, until you edit it |
| `.xencode/cache/price-lookup.json` | `xencode prices fetch` | a model `pricing.json` does not name, for 7 days |

Nothing is fetched on its own. `xencode prices fetch` is the only thing in
xencode that dials out for a price, and it is asked for; the request carries no
key and sends nothing about this project, because the catalogue it reads
(OpenRouter's public model listing) is published for anybody. What it writes is a
copy of what a third party charged on that day, so two rules hold it in check:

- **A rate you wrote by hand always wins.** The fetched list is only consulted for
  a model `pricing.json` does not name, and the report says which of the two every
  figure came from.
- **The copy expires after 7 days** (`PRICE_TTL_DAYS`). Past that it is a document
  about what something used to cost, so those models come back as *no price* —
  reported as unknown, never as free — until the list is fetched again. Nothing is
  re-fetched behind your back to fix that, and the report says what to do.

Whether the fetched copy is read at all is the `price_lookup` setting, which is
off by default: a project that never sets it behaves exactly as it did before the
listing could be fetched.

A model whose records name it as a local tag — `llamacpp:qwen3-0.6b`, `qwen2.5:7b`
— matches nothing in a catalogue and is never priced from one. Guessing that a
local model is some distant model with a similar name is how a wrong price gets
believed.

Only the dimensions the records actually count are read: input, output, and cached
input where the catalogue publishes a cache-read rate of its own. What it also
lists and xencode deliberately does not read — a price for *writing* to the cache,
higher rates for long-context tiers, audio, images and web searches — has no
counter in the metrics, so a figure built from it would be invented rather than
looked up.

```bash
$ xencode prices fetch
459 prices read off https://openrouter.ai/api/v1/models and written to /tmp/cx4-live/.xencode/cache/price-lookup.json
  7 entries the listing gave in a shape no price could be read out of, counted and left out
note: a report reads that file for 7 days, then stops pricing from it until it is fetched again.

$ xencode prices
pricing.json — /tmp/cx4-live/.xencode/pricing.json
  nothing there yet, so no model is priced by hand
fetched listing — /tmp/cx4-live/.xencode/cache/price-lookup.json
  459 prices off openrouter, read on 2026-10-03, 0 days ago
  7 entries carried no readable price
models this project has run: 1; priced from the listing: 1; with no price in either document: 0
  • llamacpp:qwen/qwen3-8b — the listing's qwen/qwen3-8b: $0.117 in / $0.455 out per million tokens, no cache rate — reads billed as input, which is an upper bound

$ xencode prices
pricing.json — /tmp/cx4-live/.xencode/pricing.json
  nothing there yet, so no model is priced by hand
fetched listing — /tmp/cx4-live/.xencode/cache/price-lookup.json
  459 prices off openrouter, read on 2026-09-24, 9 days ago
  7 entries carried no readable price
  • the listing on disk is 9 days old, past the 7 days a looked-up rate is taken for — nothing is priced from it, and `xencode prices fetch` reads them again
models this project has run: 1; priced from the listing: 0; with no price in either document: 1
  • llamacpp:qwen/qwen3-8b — no price. A cost is reported as unknown, never as nothing.
```

The second run is the same project nine days later with the list untouched, and
the last line is the point of the whole design: the spend for that model went from
a figure to *unknown*, not to zero.

`/cost` in the TUI says the same thing where the money is. Under `Per model:` a
looked-up rate carries its own line, and so does the day's dollar cap. (Both
transcripts above were driven by a local `llama-server` publishing the
catalogue's own model name, so the price path could be checked end to end without
sending a prompt off the machine — the token counts are real, the answers were
never OpenRouter's.)

```text
  Per model:
    llamacpp:qwen/qwen3-8b — 6830 prompted · 348 generated · $0.000957 · in $0.117/M · out $0.455/M · cache reads at the input price
    • 1 price read off the openrouter catalogue on 2026-10-03, 0 days ago
Everything recorded: $0.000957
Today (2026-10-03), against the caps set in the config:
  • dollar cap $5 · today $0.000957 · room left
  • 1 price read off the openrouter catalogue on 2026-10-03, 0 days ago
```

That block is the report as it was recorded on 2026-10-03, with one model in the
project. A `Per model:` line now carries two more things, and the format is the
point rather than the numbers, so it is written as a shape:

```text
  <model> — <n> records · <prompted> prompted · <generated> generated · <the price, or why there is none> · p50 <speed> tok/s (<m> records that reported one)
```

The count is there because a rate over three records is not a property of a
model, and the speed is there because the pooled line above it cannot be one
either. `cache/metrics-rollup.json` keeps the newest 512 rate samples for the
project and the newest 64 for each model it has seen, so a slow model and a fast
one used in the same week stop producing a single median that belongs to neither:
six turns at 2, 4, 6, 10, 20 and 30 tok/s report `p50 10.0` for the project, `p50
4.0` for the model that ran at 2, 4 and 6, and `p50 20.0` for the one that ran at
10, 20 and 30. A server that never reported a rate leaves the clause out rather
than writing `0.0 tok/s`.

None of it decides anything. Model selection reads the config and the hardware
profile; these numbers are printed and nothing branches on them, because a turn
that ran slowly while the machine was busy is not evidence about the model.

### What a day's caps do when they are passed

Four settings put a limit on one calendar day's usage, in the four units xencode
can actually measure it in:

| Setting | Unit |
| --- | --- |
| `budget_tokens_per_day` | prompt plus completion tokens |
| `budget_energy_wh_per_day` | watt-hours off the kernel's package counter |
| `budget_usd_micros_per_day` | what the day's tokens cost at the rates a cost report uses — `pricing.json`, then the fetched listing — in millionths of a dollar |
| `budget_minutes_per_day` | the seconds the day's turns ran, added up |

**A passed cap buys the next turn down; it never refuses one.** The day's records
are read at the boundary before a turn is built — never in the middle of one,
because a refusal that lands between an edit and the check that was supposed to
catch it is how these tools lose people's work. If a cap has been reached, the
hardware profile steps one rung down (`HIGH → BALANCED → LOW`) and the turn runs
with that smaller room: a shorter context window, fewer retrieved files, less of
each one. At `LOW` there is nothing further to give up, xencode says so once, and
the day keeps being spent. Every turn still answers.

```bash
$ xencode config set budget_tokens_per_day 100
set budget_tokens_per_day = 100

$ xencode config set budget_minutes_per_day 0
error: budget_minutes_per_day cannot be 0 — a cap of nothing is crossed by the first turn, which is not a budget

$ xencode config set budget_minutes_per_day 2000
error: budget_minutes_per_day must be between 1 and 1440 minutes

$ xencode config set budget_minutes_per_day ""
cleared budget_minutes_per_day — nothing is set for it, and the behaviour is what it was before it was ever named
```

A cap of 0 is refused where it is written: a limit nothing can respect is a way to
switch the product off that reads as a budget, and each bound above is where the
unit stops being an amount anyone spends in a day.

What a cap did is printed in the transcript when it does it, and `/cost` shows
the numbers behind it — today's figures beside every cap that is set:

```
📉 Today has spent 3562 tokens against the 100 tokens you set on its token cap,
so this turn takes the smaller context profile: HIGH → BALANCED. Nothing is
refused. /ctx shows what it means in tokens, and /cost shows the day's figures.

Today (2026-10-03), against the caps set in the config:
  • token cap 100 tokens · today 7238 tokens · passed
```

Two caps can only ever be weighed against what the machine and the price table
actually report, and xencode says so instead of guessing:

- **Energy** is read from `/sys/class/powercap`, so on a box that publishes no
  package counter the line reads `nothing this machine reported to weigh against
  it today`. `config set` warns you of that as you write the cap. A silent
  counter is not a day that used no power, and the cap never passes on a guess.
- **Dollars** come from `.xencode/pricing.json`, which knows cloud rates. Models
  it does not know count as nothing, so a day of unpriced local models cannot
  pass this cap either — the line names how many models have no price.

The day is the *local* date the records fall on, and only the last 400 days are
kept in the rollup. The session-level warning is a separate thing:
`cost_budget_usd_micros` warns about one conversation and is not one of these
four.

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

### `xencode run "the task" [--detach] [--resume ID]`
Run an agent turn from the command line. Foreground by default; `--detach`
forks a child that survives the terminal, persisting every completed round
under `.xencode/cache/detached/<run-id>/` — the spec, one `rounds.jsonl`
line per round, an `exit.json` the child writes only when it finishes, a pid
hint, and the log. Status is derived from the exit file and `/proc`, never
stored, so a kill reads as `crashed` rather than as whatever it said before
it died. `--resume` continues a crashed run from its last completed round:
prior rounds' turns become the fresh loop's history, and their tool results
are replayed, never re-executed.

```bash
xencode run "fix the typo in README" --max-rounds 8
xencode run "migrate the schema" --detach --max-minutes 30 --max-cost 2.00
xencode run --list
xencode run --show 1790240197
xencode run --log 1790240197 --tail 20
xencode run --resume 1790240197
xencode run --stop 1790240197
```

Three caps end a run besides the model finishing: `--max-rounds`,
`--max-minutes` (wall-clock, system time — a suspended laptop counts), and
`--max-cost` in dollars. Caps are checked between rounds, and a spent cap
ends the loop with `[STOPPED]` and an exit naming which one. `--max-cost`
needs a model `pricing.json` (or the fetched listing) names and a route that
reports token counts; without both the run is refused up front, because a
cap that cannot count cannot stop. A resume spends the same caps minus what
is already used — resuming into a spent budget is refused rather than
started so its first round can end it.

Nobody is listening on a detached run, so approvals run `edit-allow`: file
edits are pre-approved and anything else is refused where it stands, exactly
as the eval harness runs headless. `--allow-shell` opts into `all-allow`.
`--stop` asks the child to die with SIGTERM and leaves a stop request behind,
so the run reads as `stopped`, not `crashed`. Resuming a finished, stopped
or still-running run is refused with the reason.

### `xencode runs [list|show|trailer]`
Which runs happened, what each asked a person, and the commit trailer naming
it. When an agent turn ends in the TUI, one row is appended to
`.xencode/cache/runs.jsonl`: the run's own id, the model that answered it,
every approval question a person answered while it went, and where its
recording is (when one was made). It joins rather than copies: `show` reads
the run's session verification rows out of the evidence ledger instead of
storing a second copy of them. A recording exists only with
`session_recording` on; the run row exists either way. Reads files only, so
it works with every model server down.

```bash
xencode runs                       # the newest twenty runs, oldest first
xencode runs list --limit 5        # fewer of them
xencode runs show 1700000000-aaaa  # one run: model, decisions, checks
xencode runs trailer 1700000000-aaaa  # the Assisted-by block for a commit message
```

An id is the full run id or enough of its start to name one run and no other
— the same rule `xencode replay` uses, so an id that works there works here.
Every line `trailer` prints is a `Token: value` trailer, so `git
interpret-trailers` reads it as trailers when it sits at the end of a commit
message.

### `xencode audit verify [PATH]`Check the session server's audit log for records that were changed after they
were written. Each record carries a digest of its own contents and the digest of
the record before it, so editing, removing or moving a line is reported on a
specific line. Defaults to `audit.jsonl` in the state directory. Exits non-zero when
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
Conversation memory (kept as `conversation_memory.json` in the state directory):
`list`, `show <session>`. `gc` and `evidence` work on a different file — the
project's own `.xencode/state.md` — and are described below.

```bash
xencode memory list
xencode memory show <session-id>
xencode memory gc [--apply]
xencode memory evidence [--format text|json]
```

#### `xencode memory gc`
Every durable fact is checked against the repository on the way into a prompt. A
fact whose cited file has been deleted, or changed since the revision the fact was
written at, or that names a symbol the code no longer declares, or that describes a
call that no longer happens, is left out of that turn. The model simply stops being
told it. The line stays in `state.md`, where whoever promoted it can still read it.

This command is the record of that exclusion, and the only way to act on it. The
queue is `.xencode/facts.tombstones.jsonl`, written by the ordinary turn path — a
fact is stamped the first time the code contradicts it, and leaves the queue the
moment it no longer does, including on a turn where the check could not run at all
(no git, an unresolvable revision). So the clock measures *unbroken* contradiction.

```text
$ xencode memory gc
Durable facts: 1 contradicted, 1 past 12 months, 0 removed
  contradicted for 13 months — the file it cites has changed since: the login entry point is src/auth.rs
  `--apply` retires the 1 above; a fact that stops being contradicted leaves the queue instead of ageing toward removal.
```

| Invocation | Behavior |
|---|---|
| `xencode memory gc` | Report: what is queued, how long each has been contradicted, and how many of those are past twelve months. Writes only the queue, never `state.md`. |
| `xencode memory gc --apply` | Additionally remove the twelve-month-or-older facts from `.xencode/state.md`, and keep them in the queue marked retired so there is a list of what earlier runs removed. |

Three boundaries hold in every mode:

- Nothing is retirable before twelve months of unbroken contradiction, and a
  report is the default. A year of records is what makes removal a decision rather
  than a hunch about a diff.
- `state.md` is edited line by line, not parsed and re-written, so the facts that
  stay, the `## working-on` text and any section a person typed in by hand come
  out with the same bytes they went in with.
- `AGENTS.md` is never in scope. Those are a human's instructions; the product
  writes there at one step only, `/lesson approve`. A fact the code places in a
  different file than the one it cites is disagreement, not staleness — `xencode
  doctor` names those, and removing them is never this command's call.

A fact retired by `--apply` and later promoted again starts a new clock; it does
not inherit the retired one's.

### `xencode memory evidence [--format text|json]`

Every durable fact carries the file and revision it was written against, and every
turn re-checks it. Neither of those answered the question a cleanup starts from:
*how much of this code has this fact been re-checked against, and what is that
worth?* The turn that assembles a marked fact into a prompt now files its answer
against the revision it was checked at, in `.xencode/facts.evidence.jsonl` — one
row per fact, one verdict per revision. This command reads that ledger out loud.

The unit is a **revision, not a turn**. The check is deterministic, so a fact looked
at forty times at one commit was looked at once; counting turns would let a busy
afternoon read like a fact that survived forty changes. A turn where the check could
not reach a conclusion — git would not answer, or the revision the line names cannot
be resolved — is filed so the gap is visible, and is never counted: not-an-answer is
not an answer. Rows come out weakest evidence first, because the line a person should
read is the one with the least behind it.

```text
$ xencode memory evidence
Durable facts, weakest evidence first — the interval is over the checks this repository ran, and is not a chance that the fact is true
  session folding lives in src/sessions.rs
    checked against 3 revisions; 3 of them found nothing to contradict it — the 95% interval over a future check agreeing runs 43.9% to 100.0%
    verified by the file-and-name re-check, at revision 5da8a584 since 2026-10-06
  the login entry point is src/auth.rs
    checked against 3 revisions; 2 of them found nothing to contradict it — the 95% interval over a future check agreeing runs 20.8% to 93.9%
    verified by the file-and-name re-check, at revision 5da8a584 since 2026-10-06
    contradicted now: the file it cites has changed since
```

Three things the report is careful not to say.

- **Not a probability that the fact is true.** The interval is over the checks this
  repository ran, and what those checks can answer is narrow: the cited file is still
  there, still matches the revision it was written at, and still declares the name the
  line talks about. A fact about *why* a decision was made passes that forever and
  proves nothing about the reasoning.
- **Never a single number.** A range is printed, not a score. `checked against 1
  revision` running 20.7% to 100.0% is the honest sentence about one observation; the
  same evidence written as `0.62` is a lie with two decimals. A footer counts the rows
  that reach fewer than two revisions, where no verdict is available yet.
- **No model's name.** `verified by` names the search that answered — this binary
  looking for the cited file and the cited name at one revision, at one moment — not
  the model that happened to be driving the turn. Attributing a mechanical verdict to a
  model would be exactly the cross-model transfer this project refuses.

| Invocation | Behavior |
|---|---|
| `xencode memory evidence` | The ledger as text: each fact's own sentence, its interval, the revision and day of its last check, and, for a fact currently contradicted, the reason the code gave. |
| `xencode memory evidence --format json` | The same rows as an array: `fact`, `revisions_checked`, `survived`, `unchecked`, `wilson_95_of_next_check_agreeing`, `verified_by`, `last_revision`, `contradicted_by`. |

A fact removed from `state.md` takes its tally with it — the evidence belongs to the
line, and deleting the line is the decision the tally would have argued about. A project
with nothing marked never creates the file at all: the turn asks one metadata question
and moves on, so a repository whose facts all hold costs nothing extra.

### `xencode tasks <action>`
File-backed background tasks. State lives in `.xencode/tasks/` under the
current directory, so tasks started here are visible to later `xencode
tasks` runs in the same project (the TUI's `Ctrl+K` panel keeps its own
in-process registry). Status is derived on read from the task's exit file,
killed flag, and process liveness.

The TUI's in-process task manager applies a 30-minute wall-clock limit by
default. On Unix, stopping a task or reaching its limit kills its process group,
including ordinary child processes started by the command. A task started
through the manager API can use a shorter per-task limit.

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
| `--audit-path <PATH\|none>` | JSONL audit trail (default `audit.jsonl` in the state directory; `none` disables) |
| `--allow-insecure-public` | Escape hatch: bind a non-loopback address over plain ws:// |

Posture rules, enforced at startup: a non-loopback `--host` without TLS
refuses to start unless `--allow-insecure-public` is given (then it
binds with a loud clear-text warning); `--cert` without `--key` (or the
reverse) is an error; the banner prints the real scheme — `ws://` stays
`ws://`, only certificates earn `wss://`.

### `xencode plugin <action>`
`list`, `install <git-url> | <path>`, `update <name>`, `remove <name>`. Plugins
live in `$XCODE_PLUGIN_DIR`, else `<data dir>/xencode/plugins` — the same
directory the TUI loads from at startup, so the two never disagree.

A plugin is a directory holding `plugin.json` (or `manifest.json`). This build
loads no executable plugin code: the manifest is the whole plugin, and the two
things it can declare are a `prompt_prefix` (placed ahead of the agent's system
prompt on every turn) and `hooks` — `before`/`after` maps of tool name (or `*`)
to an `sh -c` command, the same shape as `agent_hooks` in config.json. A
plugin's hook only lands where config.json is silent, so your own config always
outranks it. Because they run through the same path, a plugin's hooks are handed
the same event JSON on stdin that `agent_hooks` commands receive.

The `permissions` list is checked at load, not decoration: a plugin may add a
prompt prefix only if it declares `prompt`, and may register hooks — which run a
shell command in the workspace — only if it declares `hooks`. A plugin that uses
a capability it did not declare, or names one xencode does not recognise (only
`prompt` and `hooks` mean anything to a manifest-only plugin), is refused
outright: it registers nothing and contributes nothing to the agent loop, and
`list` / `/plugin` say why.

```json
{
  "name": "guardrails",
  "version": "1.2.0",
  "permissions": ["prompt", "hooks"],
  "prompt_prefix": "Run cargo test before answering.",
  "xencode_version": "*",
  "hooks": { "before": { "write_file": "echo pre" }, "after": { "*": "echo post" } }
}
```

`list` runs the load and reports what took hold instead of just listing
directories — including the exact prompt lines each loaded plugin inserts and,
for a git install, the commit it is pinned to:

```console
$ xencode plugin list
📦 Plugins in /tmp/j08-probe/plugins (xencode 0.1.0):
  future v9.9.9 — NOT LOADED.
      it contributes nothing: needs xencode 0.1.0 (declared 9.9.9)
  guardrails v1.2.0 — loaded.
      it puts 1 line(s) ahead of the agent's system prompt on every turn:
      | Run cargo test before answering.
      it declares 1 before hook(s) and 1 after hook(s), each of which runs a shell command in the workspace.
  loose v0.1.0 — NOT LOADED.
      it contributes nothing: declares hooks that run a shell command but did not declare the "hooks" permission in its manifest
  1 of 3 loaded — a loaded plugin's prompt prefix and hooks apply to every agent turn.
```

`install` takes either a git URL (`https://…`, `git@…:…`, `file:///…`) or a
local path to a plugin directory or manifest. With a git URL it clones, verifies
the manifest against the same rules the loader uses, and **prints what the plugin
declares — its permissions and the full prompt text — before anything is copied**
into the plugin directory, so you see what will reach the model before it can.
The install is pinned to one commit and says which, so what is installed can be
named exactly later; `--rev <branch|tag|commit>` installs at that ref instead of
the repository's default branch. A repository with no readable manifest, or a
second copy of a plugin that is already installed, is refused — the latter points
you at `update` or `remove` instead of overwriting. The permission check runs at
install too, not only at load, so a manifest that would be denied the moment it
was read is never copied into the plugin directory in the first place:

```console
$ xencode plugin install /srv/plugins/guardrails
error: 'guardrails' adds a prompt prefix and declares hooks that run a shell command but did not declare the "prompt", "hooks" permissions in its manifest, so this build would not load it
```

That exits 1 and leaves the plugin directory without a `guardrails` entry.

```console
$ xencode plugin install file:///srv/plugins/guardrails
What it declares:
  permissions: prompt
  prompt prefix: 1 line(s), put ahead of the agent's system prompt on every turn:
    | Run cargo test before answering.
  hooks: none
Nothing else happens: this build loads no plugin code.

✅ Installed guardrails v1.2.0 into ~/.local/share/xencode/plugins/guardrails.
   Pinned to commit c45e3c2…. Nothing more is fetched until `xencode plugin update guardrails` is run.
```

`update <name>` fetches that plugin's own repository again (a plugin installed
from a local path has no record of where it came from and is refused) and shows
what changed. A bare version bump applies quietly, but an update that changes the
prompt prefix, its hooks, or its permissions is shown as a unified manifest diff
and marked `NOT APPLIED` — it changes what reaches the agent on every turn, so
it is only installed when acknowledged with `--yes`. `--rev <ref>` moves the
plugin to that branch, tag or commit; without it, a plugin pinned to a specific
commit stays there even when its branch moves on.

`remove <name>` lists the prompt lines the plugin was contributing, then deletes
only that one directory; a name containing a path separator or `..` is rejected
rather than resolved.

### `xencode mcp serve [--workspace <dir>] [--allow <tool>]`

The other direction: instead of xencode calling somebody else's MCP server,
xencode *is* the server, on standard input and output, for an editor, a script or
another agent to drive.

```console
$ xencode mcp serve --workspace ~/code/api
xencode serving /home/sree/code/api read-only over stdio; a call that would write or run a command is refused. Restart with --allow <tool> to permit one.
```

It publishes six tools — `read_file`, `list_dir`, `search_files`, `write_file`,
`edit_file`, `run_command` — each with the same JSON Schema the model is given, so
a client that has never seen xencode can call them.

A caller on a pipe has no prompt to answer, so nobody can approve a write.
xencode does not resolve that by auto-approving or by hanging: it starts
**read-only**. The three reads run; the other three are refused, and the refusal
names the flag that would have permitted that one tool:

```console
`run_command` is a shell tool and this `xencode mcp serve` was started in read-only
mode, so nothing that changes files or runs a command is executed. Start it with
`--allow run_command` to permit this one tool.
```

`--allow <tool>` repeats, one tool at a time — permitting `write_file` does not
open `edit_file` or the shell. Whatever it permits, a `path` or `cwd` argument
that resolves outside `--workspace`, or into `.git` or the xencode config
directory, is refused even for an allowed tool, and that refusal has no flag that
would change it. A `--allow` naming something xencode does not publish stops at
startup rather than being quietly ignored.

The one thing the workspace argument does *not* confine is a command the caller
was allowed to run: the check reads the arguments a call carries, not the text of
a shell command, so `--allow run_command` lets that command touch files anywhere
your own shell can. Starting the server with it prints a warning to standard
error saying so, because there is no approval prompt on a pipe and the grant is
the whole approval. A refusal also says what the launch really permits rather than
calling it read-only after the fact:

```console
$ xencode mcp serve --workspace /tmp/m5-ws2 --allow run_command
xencode serving /tmp/m5-ws2 over stdio, permitting run_command in addition to reads. A `path` or `cwd` that leaves that directory stays refused.
warning: `run_command` is permitted. A permitted command runs exactly as the caller wrote it, so it can touch files outside /tmp/m5-ws2; that is a shell, not a jailed one, and there is no approval prompt on a pipe.

`write_file` is a file-changing tool and this `xencode mcp serve` was started permitting only `run_command` beyond the three reads, so it is not executed. Start it with `--allow write_file` to permit this one tool.
```

Everything meant for the person starting the server goes to standard error;
standard output carries only protocol messages. `xencode mcp serve --help` prints
the description of the mode and its limits.

The same six names reach a client as `mcp__xencode__<tool>` once that client
imports them, which is why published names go through the same sanitize-and-fit-
in-64 rule xencode's own MCP client applies to a tool it imports.

### Recipe: driving a real browser with Playwright MCP (no product code)

Browser verification is a documented recipe, not a feature: Playwright MCP is a
Node subprocess you declare like any other server, and the agent drives it
through the same approval gate as every external tool. Say plainly that it is a
Node subprocess — there is no embedded browser in xencode.

Declare it under `mcp_servers` (a `command` to spawn; `npx` must be on `PATH`):

```json
"mcp_servers": {
    "playwright": {
        "command": "npx",
        "args": ["-y", "@playwright/mcp", "--headless", "--image-responses", "omit"],
        "env": {
            "PLAYWRIGHT_BROWSERS_PATH": "/home/sree/.cache/ms-playwright"
        }
    }
}
```

`/mcp` starts it; a working handshake lists its tools (25 with
`@playwright/mcp` 0.0.83):

```console
◈ ✓ playwright · 25 tool(s)
```

The agent reaches them as `mcp__playwright__<tool>` — `browser_navigate`,
`browser_take_screenshot`, and the rest — each behind the `External` approval,
which always asks: answer `a` once to allow the whole class for the session.
Verified live against a local dev server: the agent ran
`browser_navigate({"url": "http://127.0.0.1:8099/"})` (the server log showed a
real Chromium `GET /` plus `/favicon.ico`), then
`browser_take_screenshot({"filename": "shot.png"})`, which wrote a real
1280×720 PNG (18,330 bytes) into the workspace. Attach it with `Space` in the
file explorer (a `📌` marks it); on send it travels as a data URL in the final
user turn.

Three traps, all met live. First, xencode sends `{}` — not `null` — as the
params of `tools/list`, `resources/list` and `prompts/list`, because a strict
server answers the handshake and then drops a `null`-params list call without a
word, which reads as a 30-second timeout. Second, twenty-five tool definitions
are roughly nine thousand tokens of context: a local `llama-server` started with
`-c 8192` refuses the turn (`request (9232 tokens) exceeds the available
context size`), so start it bigger (verified with `-c 32768 -np 1`) or the turn
fails before the model reads a word. Third, the attach half ends at the model:
a text-only local model answers the image part with an error (`image input is
not supported — hint: … you may need to provide the mmproj`), so the
screenshot lands on disk and is sent, but having the model *read* it needs a
vision-capable model, not a bolder prompt.

### Skills (`SKILL.md`)

A skill is one directory holding one `SKILL.md`: a name and a one-line
description of when to use it at the top, the instructions themselves below.

```markdown
---
name: release-notes
description: Use when the user asks what changed between two tags.
---

List the commits oldest first. Quote a commit's subject, never its hash.
```

Xencode scans two directories when the TUI starts —
`skills/` in the settings directory (set `$XCODE_SKILLS_DIR` to move it) and
`.xencode/skills`
inside the workspace — and a project skill replaces a user skill of the same
name rather than sitting beside it. What goes into the system prompt is a list:
one heading, then one line per skill (`name — description`, a description longer
than 220 characters cut with an ellipsis). The instructions stay on disk.

The model reaches a body through `load_skill`, a read-only tool that takes a
`name` and nothing else — never a path, so it opens no way to read outside the
skill directories, and it asks for no approval. It is offered only when at least
one skill is installed: with none, the tool list and the system prompt are
byte-for-byte what they were before skills existed. Asking for a name that is
not installed is answered with the names that are.

`/skills` in the TUI reports the outcome of the scan, and `/skills reload` runs
the scan again without restarting:

```console
Skills in /home/sree/.xencode/skills and /tmp/m3-ws/.xencode/skills: 1 loaded.
answer-with-marker [user] — Use this whenever the user asks what this project is…
Menu for 1 skill: 337 characters on every turn. Their instructions are 230
characters in all, and reach the model one skill at a time through load_skill.
```

A document that cannot be used is named with its reason instead of being loaded
silently: a file with no instructions behind the frontmatter is refused, frontmatter
that never closes keeps the fields it did declare and says so, and a file with no
frontmatter at all loads under its directory name with a description taken from
its first line — flagged as inferred, because nobody wrote one.

The list is why a large skill directory stays cheap. Measured against a local
llama.cpp model with the same prompt every time: no skills installed, 3391
prompt tokens; 30 skills installed, 4127 — and `/trace` confirms the turn made
no tool call, so none of those 30 bodies (94,948 characters, 22,380 tokens by
the server's own tokenizer) entered the prompt.

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

### `xencode anchor [path] [--timeout 900] [--dry-run] [--format text|json]`

Find this repository's build and test commands, **run them**, and record only the
ones that actually worked.

Reads CI workflows, `justfile`/`Makefile`/`mise.toml`, `package.json`/`Cargo.toml`
and the README, then records each command with where it came from and whether it
exited zero. Output lands in `.xencode/anchor.md`, which is read into the stable
prompt head — so every model request already knows how to build and test this
project instead of guessing.

**A command that was not run is never called working.** Each entry states
`verified — ran it, it exited 0` or admits the failure with its exit code, and a
timeout is recorded as *unverified* rather than as a pass, because a command
nobody saw finish has proved nothing. When nothing is verified, the file says so
at the top. `--dry-run` lists what was found without executing anything.

The render is **deterministic**: no timestamp, no absolute path, no duration,
sorted input. Two runs over the same repository produce identical bytes, so the
stable head keeps its cache instead of churning every session. Both halves are
enforced by tests.

Real output on this repository:

```
  4 candidate(s), 4 verified

    build     verified   cd rust && cargo build --release -p xencode-cli
    lint      verified   cd rust && cargo clippy --workspace --all-targets -- -D warnings -A clippy::format-in-format-args
    format    verified   cd rust && cargo fmt --check
    test      verified   cd rust && cargo test --workspace --verbose
```

The `cd rust &&` prefix is the interesting part. CI runs these from `rust/`, and
that key is written *after* the `run:` it applies to. Dropping it would produce a
command that passes in CI and fails from the repository root — a green tick on a
broken recipe, which is the one outcome this command exists to prevent.

### `xencode bootstrap [path] [--check] [--format text|json]`

Write the three files a project xencode has never seen is missing, from what is on
its disk and nothing else.

A fresh clone has no `AGENTS.md`, no `.xencode/anchor.md` and no example of the
settings file. This product reads all three — the first two go into the head of
every prompt — so on an unseen project the model is handed an empty instruction
sheet and no idea what the repository contains. `xencode bootstrap` fills that in.

It is **declarative, and that is the whole design.** Every byte it writes is a
name, a number, or a blank question. No build ran, no test ran, no model was asked,
so the file cannot claim a command this project does not use. That is deliberate: a
model inventing a project's CI config is how 9,371 lines of plausible fiction ended
up in this repository once already, and were deleted.

```
Project: /tmp/demo
  git branch main at 2b2724bc, 5 files
  write        AGENTS.md               questions only: nothing ran, so no command is guessed
  write        .xencode/anchor.md      what was read off this disk, with no build or model in it
  write        .xencode.example.json   every key this binary reads, at its default, credentials absent

3 files written. Nothing that already existed was touched.
```

**A file that exists is never written, and there is no flag to make it happen.**
There is no `--force`, because `AGENTS.md` is a person's file: the second run reports

```
  keep         AGENTS.md               already here, and this command does not replace a person's file
```

and the bytes are identical before and after (`md5sum` of all three files, taken
after the second run and again after a third, prints the same three hashes both
times). `--check` reports and creates nothing — not
even the `.xencode/` directory — so you can see what it would say about your own
repository first. Here is what it says about xencode's own, which is the case worth
seeing because all three files are already answered here:

```
$ xencode bootstrap . --check
Project: /home/sree/Projects/xencode
  git branch main at e62b2beb, 309 files
  keep         AGENTS.md               already here, and this command does not replace a person's file
  keep         .xencode/anchor.md      already here, and this command does not replace a person's file
  keep         .xencode.example.json   already here, and this command does not replace a person's file

Report only: 0 would be written, 3 already in place. Nothing created.
```

What each file is:

- `AGENTS.md` — the headings a project's instructions need, each followed by an
  HTML comment asking the one question that can only be answered by someone who
  knows. It names no command. `/lesson approve` remains the only thing that writes
  a *sentence* into a file that is already there; this one only creates the file.
- `.xencode/anchor.md` — what git reports: branch, revision, the file count, the
  names at the repository root, and extensions by count. It carries no clock and no
  absolute path, because the anchor is inside the byte-stable prompt head and a
  timestamp there makes every request re-send the whole thing. It is written by
  `xencode anchor`'s own writer, so the prompt reads it from the one path it knows.
- `.xencode.example.json` — the settings template, generated from the same struct
  this binary loads and saves rather than typed out, so a key listed there exists
  and a key missing was added after this binary was built. All nine credential
  fields are `null` and both hook maps are empty: nothing that authenticates as
  anything is written into somebody's repository.

Two things this command deliberately does **not** seed:

- **Skills.** A `SKILL.md` with only frontmatter is rejected by the loader with
  *"has no instructions after its frontmatter"*, so a stub skill is not a skill —
  it is a parse error in the project's way.
- **Hooks.** There is no project-local config loader; `agent_hooks` is read from
  `~/.config/xencode/config.json`, so seeding hooks would edit settings that apply
  to every project on this machine and make them run `sh -c` on every approved tool
  call. The generated template shows the two empty hook maps and leaves them empty.

A name read off disk (a file called `prod-openai-api-key.txt`, say) is written into
the anchor only after `redact_secrets` has had it, because that text is copied into
a file that goes into a prompt and may go to a model. The settings template is
excluded from that scrub on purpose: `redact_secrets` replaces the *value* beside
anything that looks like a credential key, and running it over the template turned
`"openai_api_key": null` into `"openai_api_key": "[redacted]"` — watched happening,
and the reason the scrub is applied per file rather than to everything.

The relation to `xencode anchor` is one direction only: `bootstrap` creates an
anchor that says *nothing was verified*, and `xencode anchor` is the command that
replaces it with commands it actually ran. On a new project, run `bootstrap`, then
`anchor`.

### `xencode doctor [--format text|json]`

The bug report. One command, one list of rows, everything a person would have to
type by hand to answer "what is wrong with my machine": does the configuration
parse, is it a version this binary can read, can anyone else read the secrets in
it, is there room on the volume the
state lives on, how much disk the response cache has taken, the project's own
index/git/metrics/cache rows, every endpoint the config would dial, whether the
server behind the default model actually knows that model by name, each declared
MCP server, the Colab bridge, and whether the project's stored facts still agree
with the code beside them.

The rows are the same structure in both formats. `--format json` serialises
`SelfCheck { name, state, detail, fix }` as it is, and the text listing prints
that same list with a mark in front — so the file you attach to an issue and the
screen you read cannot disagree. A `fix` key is present only where there is
something to run: a size, or a finding whose remedy belongs to a provider rather
than to this machine, carries no action.

```
$ xencode doctor
  PASS   config                 /home/sree/.xencode/config.json
  PASS   config:version         no version key — written before versions existed, so it is migrated on read and stamped 1 on the next save
  PASS   permissions:config     600 — owner only
  PASS   permissions:colab-key  600 — owner only
  PASS   disk:state             /home/sree/.xencode has 113 GiB free
  PASS   size:cache             /home/sree/.xencode/cache holds 1 file(s), 364 B
  ABSENT index                  no index manifest; run /init for project-aware answers
                                fix: run /init in the TUI to build the project index
  PASS   git                    /home/sree/Projects/xencode
  ABSENT metrics                no metrics recorded yet
  PASS   cache                  /home/sree/Projects/xencode/.xencode/cache
  FAIL   provider:ollama        localhost:11434 refused: nothing is listening; start it with `ollama serve` or point the config elsewhere
                                fix: ollama serve
  FAIL   provider:llamacpp      127.0.0.1:18000 refused: nothing is listening; start it with `llama-server --model <path>` or point the config elsewhere
                                fix: llama-server --model <path>
  FAIL   provider:remote        127.0.0.1:18000 refused: the remote endpoint does not answer from here
                                fix: check the network; the address dialed is the one the config names
  FAIL   model                  Ollama at http://localhost:11434: Ollama not running: error sending request for url (http://localhost:11434/api/show)
                                fix: ollama serve
  PASS   colab:CLI              /home/sree/.local/bin/colab
  PASS   colab:version          0.7.2 >= 0.7.0
  PASS   colab:ssh bridge       `colab ssh` accepted (proxy-mode bridge available)
  PASS   colab:API              `colab sessions` succeeded (backend reachable)
  PASS   colab:OpenSSH          ssh at /usr/bin/ssh
  PASS   colab:OpenSSH          ssh-keygen at /usr/bin/ssh-keygen
  PASS   colab:SSH ed25519 key  /home/sree/.xencode/colab_ed25519 (+ .pub)

  failing: provider:ollama, provider:llamacpp, provider:remote, model
```

Every one of those sentences came from the machine: the free space is a real
`statvfs` of the volume holding the state directory, the mode is the file's own
permission bits, the sizes are a walk of the directory, and the Colab rows are
the answers of the installed `colab` 0.7.2 — which is asked through the same
preflight `xencode colab up` runs through, so the report and the gate cannot
drift apart on what version is acceptable. The bridge is only probed where it
exists: on a machine that has never had `colab` installed, never written a state
file and never made a keypair, its network probes would be traffic about
somebody else's setup, and the report says one `ABSENT colab` row instead. It
never generates a key — a report reads the machine, it does not create secret
material on it.

The failing branches were watched failing, not inferred. Running the local
`llama-server` on `127.0.0.1:18000` and pointing `default_model` at
`llamacpp:qwen3-0.6b` turned three of those rows over:

```
  PASS   provider:llamacpp      127.0.0.1:18000 accepts TCP
  PASS   provider:remote        127.0.0.1:18000 accepts TCP
  PASS   model                  llamacpp:qwen3-0.6b: llama-server at http://127.0.0.1:18000 has a model loaded
```

and making `config.json` world-readable turned the secret row over:

```
  FAIL   permissions:config     644 — readable beyond the owner, and it holds secrets
                                fix: chmod 600 /home/sree/.xencode/config.json
```

The `knowledge:stale` row is the last one, and it was watched in three states in a
throwaway repository — a stored fact citing `src/auth.rs`, then that file moved
out from under it, then a fact whose citation points at a file that never held the
name it was written against:

```
  PASS   knowledge:stale        2 durable facts, every one agreed with by the code
  FAIL   knowledge:stale        0 believed, 1 dropped, 0 could not be checked — login lives in src/gone.rs: the file it cites is gone
                                fix: re-read the file each dropped line cites and promote a corrected fact; the lines stay in state.md until you say otherwise
  FAIL   knowledge:stale        1 believed, 0 dropped, 0 could not be checked, 1 places a name in a file that does not declare it (validate_token rejects an empty token before src/session.rs… names validate_token but cites src/session.rs)
                                fix: re-read the cited file and the files that do declare the name, then correct the citation in state.md; the fact itself was not dropped for this
```

The third row is not a contradiction. The file the fact cites has not moved and
`validate_token` is still declared — the tree just declares it in `src/auth.rs`,
which the fact never mentions. Nothing is taken out of the turn for it: which of
the two files the fact meant is a question for whoever wrote it, so the row says so
and leaves the line where it is.

A project that has never stored a fact reports `ABSENT`, not a pass over nothing.
The row reads `.xencode/state.md` and writes nothing back; the lines it names stay
where they are, because whether a contradicted fact is wrong or the code is was a
person's call. The same audit with each finding spelled out is
`xencode doctor --env`, and `--format json` carries both.

`--format json` prints the same rows plus three summary keys:

```
$ xencode doctor --format json
{"checks":[{"detail":"/home/sree/.xencode/config.json","name":"config","state":"pass"},…],
 "doctor":"report","failing":["provider:ollama","provider:llamacpp","provider:remote","model"],
 "ok":false,"version":"0.1.0"}
```

The exit code stays zero, because a laptop with no local model server running is
a normal laptop. The FAIL rows are the signal.

### `xencode doctor --selfcheck [--format text|json]`

The slice of that report a person runs when xencode itself looks broken: index,
git, providers, MCP servers, the default model, metrics, cache. Same rows, same
`fix` strings, none of the state-file or bridge checks. Absent is not failed.

Each check goes through the code path the feature uses, so a row says something
about the real route rather than about a guess. Providers are dialled at the
addresses in the settings directory's `config.json` — `ollama_url`, `llama_cpp_url`, and
`remote_base_url` when it is set — not at a remembered port, and a cloud
provider is dialled only when a key is configured for it. An MCP server is
started by the same client `/mcp` uses, asked to introduce itself, and killed;
a row is a completed handshake, and a failure is the client's own sentence.

```
$ xencode doctor --selfcheck
  ABSENT index                  no index manifest; run /init for project-aware answers
                                fix: run /init in the TUI to build the project index
  PASS   git                    /home/sree/Projects/xencode
  ABSENT metrics                no metrics recorded yet
  PASS   cache                  /home/sree/Projects/xencode/.xencode/cache
  FAIL   provider:ollama        localhost:11434 refused: nothing is listening; start it with `ollama serve` or point the config elsewhere
                                fix: ollama serve
  FAIL   provider:llamacpp      127.0.0.1:18000 refused: nothing is listening; start it with `llama-server --model <path>` or point the config elsewhere
                                fix: llama-server --model <path>
  FAIL   provider:remote        127.0.0.1:18000 refused: the remote endpoint does not answer from here
                                fix: check the network; the address dialed is the one the config names
  FAIL   model                  Ollama at http://localhost:11434: Ollama not running: error sending request for url (http://localhost:11434/api/show)
                                fix: ollama serve

  failing: provider:ollama, provider:llamacpp, provider:remote, model
```

That run points at `127.0.0.1:18000` because that is where this machine's
config sends `llamacpp` and `remote` — a check hardcoded to the usual port would
have reported on a server nobody here addresses. Pointing `ollama_url` at a port
that is serving, and declaring three servers — one that runs, one whose command
is not there, one that says nothing about how it is reached — gives the other
four shapes, all from one live run:

```
  PASS   provider:ollama    127.0.0.1:631 accepts TCP
  PASS   mcp:talking        /home/sree/mcp-doctor-probe.sh started and answered the handshake; it offers tools
  FAIL   mcp:ghost          cannot start MCP server `ghost`: No such file or directory (os error 2)
  FAIL   mcp:undecided      the declaration names neither a "command" to spawn nor a "url" to reach
```

`--format json` prints the same `checks` list and the same `doctor`, `failing`,
`ok` and `version` keys, with `"doctor": "selfcheck"`.


### `xencode doctor --deps [--format text|json]`

Dependency health, composed: direct deps with locked versions, pending updates
from a dry run, advisory state from the local corpus. Offline reads as
"unknown", never "clean"; a dry run that never ran says so. Exits non-zero on
any vulnerable dependency.

### `xencode agents [--format text|json] [--contract]`

List installed roster agents with versions and how each was installed
(`mise:<tool>`, cargo, npm, system, user-local, unknown — only what the path
shows). Discovery only: nothing is installed, upgraded, or written.

`--contract` re-reads every roster claim from the agents' live `--help` and
reports confirmed or contradicted per claim with evidence. A contradiction is
a stale roster cell, and a firewall test fails the build on any of them.

### `xencode impact <file> [--limit 15] [--format text|json]`

Who has to be re-checked if this file changes, in three layers kept apart because
they are three strengths of evidence. The **crate** layer is exact: it reads
`cargo metadata --no-deps` and names the workspace member the file lives in, the
members that depend on it directly (with the edge kind), and the transitive set
behind it. The **file** layer is a prediction, capped at three hops: it builds the
symbol graph straight from the git-tracked sources — no pre-built `.xencode` index
needed — and lists the files that link this one through a `use` path, a `mod`
declaration or an `impl`. A link means a resolved name, not a type-checked call
site. The **coupling** layer reads one `git log` and lists the files whose history
moves with this one, plus the file's own commit count. Runs headless; a workspace
nested below the git root is reconciled so history still reads.

```
xencode impact crates/xencode-core-rs/src/lib.rs
xencode impact crates/xencode-core-rs/src/lib.rs --format json
```

### TUI slash: `/impact <file>`

Opens a dedicated fan-out panel over the same three layers. One row per target,
per crate header, per consumer file: each crate row names its hop and the
dependency kind of the direct edge; each file row names its own hop, the
`use`/`mod`/`impl` payloads that resolved, and its co-change count with the
target. Never co-changed, co-changed n times, and no readable git here stay
three different claims on the row, exactly as on the CLI. ↑/↓ walk the rows,
`Enter` opens one row's evidence, `→` descends onto a file row only (crate
headers and the target itself are inert), `←` pops the descend stack, `r`
re-runs the query in place, `o` opens the file in the editor, `Esc` unwinds
detail → descend stack → chat one stage at a time. The panel never recomputes
on cursor motion.

```
/impact crates/xencode-core-rs/src/lib.rs
```

### `xencode removal <file> [--limit 15] [--format text|json]`

What deleting this file would cost — the dependency graph with one node removed,
the two directions of an edge. **Broken links** are the files whose `use`, `mod`
or `impl` resolved to the target; they dangle the instant it is gone. **Newly dead
files** are the modules only this file pulled into the build — found by comparing
what a crate root reaches today against what it reaches once the node is deleted,
so a grandchild stranded at any depth is caught, not just a direct `mod` child.
An unreferenced root is a crate, not dead code, so `lib.rs`/`main.rs`/`mod.rs` are
never reported orphaned. Runs headless, no `.xencode` index needed. Deletion is
stronger evidence than an edit: a consumer can absorb a change, never a missing
file.

```
xencode removal crates/xencode-core-rs/src/tasks.rs
xencode removal crates/xencode-colab-rs/src/lib.rs --format json
```

### `xencode hotspots [--limit 10] [--format text|json]`

Rank files by commits × bytes with bus factor and CODEOWNERS owners, as
advice rows that each carry an action. Build outputs and other generated dirs
are skipped — a panel led by a `.rlib` is decorative.

### `xencode doctor --env [--format text|json]`

Probe and display this machine: cores, memory, PSI, cgroup limit, GPUs,
journalctl and dmesg readability, colab route presence — plus the
configuration-drift row. Best-effort throughout: what cannot be read is
reported absent, never an error.

It also prints what the project's durable knowledge looks like from here, which is
the part that answers a question the screen cannot otherwise:

```
  durable facts: 1 reaching the model, 1 dropped, 0 that could not be checked, 1 that the code places in another file
    dropped — the file it cites is gone: login lives in src/gone.rs
    disagrees — validate_token rejects an empty token before src/session.rs runs names validate_token, declared in src/auth.rs rather than the cited src/session.rs
```

The four numbers are not a sum of one thing. *Reaching the model* is what a turn
will actually be told; *dropped* is what was left out of that, with the line and
the reason under it; *could not be checked* is a fact the code has neither
confirmed nor contradicted, which is not the same as either; and *places in another
file* is a fact the code has not disproven either — its file has not moved and its
name is still declared, just somewhere the fact does not mention. `--format json`
puts the same under a `durable_facts` key as `believed`, `unverifiable`, a `dropped`
list of `{line, reason}` and a `disagreeing` list of `{line, name, cited,
declared_in}`. Restoring the file changed the reason rather than clearing the row —
`the file it cites has changed since` — because the comparison is against what the
project looks like now, not against the commit the fact cites.

Read-only, on purpose. The same audit runs when a prompt is built, and what it
concludes there is written into the turn and not into `state.md`: the dropped facts
are withheld for that turn, the misplaced ones arrive with a `## Sources disagree`
note beside them, and the file a person promoted keeps every byte of them. A
diagnostic that edited memory would be deciding what a person may keep. `xencode
memory` is where a fact is changed.

### `xencode session <name|resolve|export>`

Name a run so it survives without its id (`session name <run> <name>`; existing
names are never repointed), resolve a name, id prefix, or `latest` to the full
id, and print a transcript (`session export <target> [--redacted]`). Resume
restores the model, server, tool root, call count, and opening messages across
processes. `--redacted` scrubs secrets with the trace module's patterns; an
ambiguous prefix is refused with the candidates named rather than guessed.

### `xencode envcheck [--format text|json]`

Report environment keys read in code against the templates that document them.
Read-but-undocumented comes with file:line references and the sources searched;
documented-but-unreferenced is never called unnecessary; OS-provided keys
(`HOME`, `PATH`, `USERPROFILE`, …) are listed separately, never reported; and
`.unwrap()`/`.expect()` reads outside tests are flagged. No template found is
stated outright rather than implied.

### `xencode generate <completions|man> [--shell <bash|fish|zsh|powershell|elvish>]`

Print shell completions or the `xencode(1)` man page, generated from the clap
definition — never written by hand. The committed copies under
`docs/completions/` and `docs/man/` are drift-checked twice: a unit test
asserts byte-equality with fresh output, and a CI job regenerates them and
fails on `git diff`. If they drift, regenerate them; do not edit them.

### `xencode toolchain <lint|fix|fmt|shear> [--allow-dirty] [--format text|json]`

The Rust toolchain kit as gated tools — structured evidence for an agent repair
loop instead of terminal prose.

```bash
xencode toolchain lint               # clippy diagnostics, grouped by lint
xencode toolchain lint --format json # the same as a machine-readable document
xencode toolchain fix                # apply clippy fixes (refuses a dirty tree)
xencode toolchain fix --allow-dirty  # accept the overwrite risk explicitly
xencode toolchain fmt                # fail unless formatting is clean
xencode toolchain shear              # unused or misplaced dependencies
```

**`fix` refuses a dirty tree.** `cargo clippy --fix` rewrites files, and an
overwrite of uncommitted work looks exactly like your own edit afterwards. The
refusal names the dirty files so you know what to commit or stash; `--allow-dirty`
accepts the risk out loud. Every run reports the diffstat of what changed.

Real session on a seeded lint:

```
  BEFORE: 2 clippy::needless_return
  fix changed files:

  src/lib.rs | 2 +-
  AFTER: no clippy diagnostics
```

`lint` JSON carries the count, the per-lint groups, and every diagnostic with
file, line, and whether rustc offered a machine-applicable suggestion — the
shape a loop needs to pick the next fix and to prove the count reached zero.

### `xencode mutants [--diff <ref>] [--timeout 60] [--check-repair <file.json>] [--format text|json]`

Find the code whose wrongness no test would notice.

```bash
xencode mutants                       # mutants in the working-tree diff
xencode mutants --diff main           # mutants in the diff against main
xencode mutants --check-repair fix.json  # judge a proposed repair
```

Needs `cargo install cargo-mutants`.

A mutant is a small sabotage — an operator flipped, `true` returned instead of a
computation. The suite runs against it; one that still passes is **missed**, and
a missed mutant is a test that cannot tell right from wrong. The run is scoped
to the diff, because without that the whole suite runs once per mutant in the
workspace, which is minutes to hours here. A missing report is an error, never
an empty success: "all caught" inferred from a file that was never written is a
false green.

**Repairing a missed mutant is gated, because the obvious repair is to weaken
the test.** Measured here: deleting the assertion that caught a mutant turns
8 caught into 1 missed, and adding a tautology `assert!(x || !x)` on top keeps
the suite green while the mutant still survives. Neither a test count nor a
passing suite is therefore a defence. `--check-repair` judges a JSON file with
the patch, the assertion counts before and after, the mutants targeted, and the
re-run of the *same* mutant set. All four conditions must hold: only test code,
no fewer assertions, never the file under mutation, and the same set re-run and
now caught.

Two behaviours worth knowing. `--in-diff` takes a file the command writes
itself, because cargo-mutants does not accept a ref and reports the wrong
argument as a missing file. And with pinned `a/`/`b/` prefixes: git's default
mnemonic prefixes (`i/` for the index, `w/` for the worktree) make the same
diff yield "No mutants to filter" — a clean summary meaning no work was done.

**The score is rolled up per function.** Each mutant is attributed to the
function it sits in (read from cargo-mutants' own `function.function_name`), and
the report gives one score per `(file, function)` sorted worst-first, so the
weakest function is the first line rather than one to hunt for:

```
per function (worst first):
  mutants of `is_even` (src/lib.rs): 1/2 caught — 1 survived
  mutants of `classify` (src/lib.rs): 2/2 caught
```

A mutation the tool places outside any function is named under its own bucket,
never blurred into the whole file. Unviable and timed-out mutants are kept out of
the denominator: a mutant that cannot compile, or never finished, says nothing
about the tests, so it is neither counted as caught nor against the score. A
function with no viable mutant gets no score, and a run where nothing was both
generated and viable prints one honest note instead of a table of zeros or an
unbacked 100%. `--format json` carries the same rollup as a structured `symbols`
array.

On this repository's own change it reports 2 caught, 8 missed, each one naming
the function to strengthen first.

### `xencode verify [--skip test|lint|fmt] [--timeout 1800] [--format text|json]`

Run the machine-checkable checklist: the full test suite, clippy with zero
tolerance, and `cargo fmt --check`. Each slot is verified by its exit code —
nothing here is graded by a model — and each leaves a ledger row plus an
artifact the verdict points at. Skipped slots are reported alongside, never
counted as passed.

### `xencode test --isolate <substring> [--base HEAD] [--repeat 3]`

Classify one failing test instead of running the suite: the same filtered
test runs against the clean base tree in a throwaway worktree and against
your tree, and the verdict is one word — `PRE_EXISTING_FAILURE` (fails on
base too, so do not fix it here), `INTRODUCED`, `FLAKY`, or `INCONCLUSIVE`
(the base could not run, so nothing is claimed). The worktree is removed
afterwards; your tree is never touched.

### `xencode cov [--base <ref>] [--test <cmd>] [--show-missing-lines] [--format text|json]`

Report which lines **this diff** added were never executed.

```bash
xencode cov                      # working tree against HEAD
xencode cov --base main          # against a ref
xencode cov --show-missing-lines # just file → line numbers
```

Needs `cargo install cargo-llvm-cov` and `rustup component add llvm-tools-preview`.

A green suite answers "did nothing break", not "did the lines I just write run at
all". This answers the second one. Output is line numbers, not a percentage,
because a percentage is a number nobody can act on and a line number is a place
to go and read.

**A line with no coverage data is reported as `no data`, never as uncovered.**
A diff touching `Cargo.lock` would otherwise look catastrophically untested,
which blames your code for the tool's blind spot.

```
  0 of 160 measurable added line(s) executed; 3 added line(s) had no coverage data
  build: warm — reused the instrumented target directory (212s)

  crates/xencode-cli/src/main.rs
    0% of measurable added lines ran
    never run: 390-408, 962-967, 3842-3976
```

**The first run is slow and the output says so.** Coverage rebuilds every crate
with instrumentation in its own target directory — 272 s cold and 7.2 GB here —
then reuses it, so later runs are much cheaper. The `build:` line always reports
which kind of run happened.

### `xencode perf [record|check|show] [--filter <name>] [--alert-pct N] [--alpha A] [--format text|json]`

Measure the workspace's hot paths against a stored baseline, and say so when the
machine is too noisy to answer the question at all.

```bash
xencode perf record            # measure all seven paths, store them as the baseline
xencode perf check             # measure them again and compare
xencode perf check --filter retrieve   # compare only the paths whose name contains this
xencode perf show              # the stored baseline, measuring nothing
```

`xencode perf record` deliberately takes no `--filter`: a baseline covering three
of seven paths is not a baseline, and every one of them has to be measured on a
quiet machine for the numbers stored together to mean anything.

Seven criterion benchmarks run over **this repository on disk** — the index scan,
symbol extraction over every Rust file, the dependency-graph build, BM25 scoring,
hybrid retrieval, conversation compaction and the token trimmer — ten samples
each, in `rust/crates/xencode-context-rs/benches/hot_paths.rs`. Each path then
gets a Mann-Whitney test of its current samples against the stored ones. Below
1,000,000 possible splits of the two groups the p-value is computed exactly, by
enumerating every way the pooled ranks could have been dealt; beyond that the
tie-corrected normal approximation stands in, and the report prints which of the
two produced the number you are reading.

The comparison is only made when both runs measured the same tree: the baseline
remembers how many files it was recorded over, and a different count refuses
every path rather than comparing a 200-file index against a 340-file one.

```
  cargo bench -p xencode-context-rs --bench hot_paths -- extract_symbols
  200 file(s) measured, baseline recorded over 200, in 12.3 s
  alert at 5% of the baseline, judged at α = 0.05, verdicts refused past a 5% spread

  index_build/extract_symbols
    REGRESSION   delta +16.55%   p = 0.0000 (exact permutation)   spread 4.9%
    the samples separate at the 5% level (p = 0.0000) and the path runs +16.55% slower than the baseline

  truncation/truncate_to_tokens
    not measured  delta      —   p =   —   (not tested)   spread   —
    in the baseline but not measured by this run

  7 path(s) compared: 1 regression(s), 0 refusal(s)
```

That is the exit-1 case: `error: 1 hot path(s) measurably slower than the
baseline`. The run above measured one path with `--filter`, which is why the
other six report `not measured` — a path left out of the run is named as left
out, and never reuses whatever sample file the previous run happened to leave
behind. With no filter, the same seven paths against the same baseline and no
code change read as `7 path(s) compared: 0 regression(s), 0 refusal(s)` and
`exit 0`, every line `no change` and every delta inside ±2.2%.

**A verdict needs a quiet machine, and this refuses to invent one.** Ten timing
samples on an ordinary laptop sit about 1.6% apart at the best of times, so the
spread of a run is checked before its number is used: if either the new samples
or the stored baseline vary by more than 5% of their own level, that path reports
`NO VERDICT` and says which side was too wide, instead of calling noise a
regression.

```
  index_build/scan_tree
    NO VERDICT   delta +80.70%   p = 0.0000 (exact permutation)   spread 6.1%
    this run's spread is 6.1% of its own level, past the 5% a verdict is allowed to rest on

  note: at least one path was measured above the 5% spread a verdict is allowed to rest on — this machine was doing something else

  7 path(s) compared: 0 regression(s), 1 refusal(s)
```

That is a run made while four other processes were burning cores. `+80.70%` is
what contention did to the wall-clock of a directory walk, and the harness
declines to call it a regression — exit 0, with the refusal printed and counted,
because "the machine was busy" is a finding, not a failure. `perf record` applies
the same rule before storing anything: `error: refused to record a baseline from
a run wider than 5% spread on: index_build/extract_symbols at 6.6%; wait for the
machine to go quiet, or pass --force to record it anyway`.

The other branch that is not a failure is a difference too small to flag:

```
  retrieve/bm25_build_and_score
    no change    delta +2.18%   p = 0.0355 (exact permutation)   spread 2.0%
    separated at p = 0.0355 but only +2.18% away, under the 5% this harness reports as a change
```

The samples did separate, but by less than the alert level, so it is not reported
as a change — a harness that flagged every statistically-visible 1% drift would be
switched off within a week.

The baseline lives at `.xencode/perf/baseline.json` at the repository root and is
not committed: it is a record of one machine on one day, and comparing it across
those is the mistake this command exists to prevent.

`--format json` emits the machine-readable form: for `check`, the medians in
nanoseconds, the delta, the p-value with its method, the spread and the outcome
per path; for `show`, the stored samples per path, unscaled.

### `xencode prices [show|fetch] [--url <URL>] [--format text|json]`

Which two documents a cost report reads its rates out of, what each of them
says, and which of the models this project has actually run have no rate in
either. Nothing is computed here — no spend, no totals. The command exists for
provenance: a number on a cost report is only as good as the paper it came from,
and one of those two papers is somebody else's catalogue read on a day that has
already passed. See [Where a price comes from](#where-a-price-comes-from) for
what the two documents are and the two rules that hold the fetched one in check.

```bash
xencode prices                       # the same as `prices show`
xencode prices show --format json    # the same answer, machine-readable
xencode prices fetch                 # read the public catalogue again
xencode prices fetch --url <URL>     # …from somewhere that publishes the same document
```

`show` reads the disk and nothing else; it never dials out, and it prints the
listing's age in days rather than asking anybody whether it is still right.

```
$ xencode prices
pricing.json — /tmp/cx4-live/.xencode/pricing.json
  nothing there yet, so no model is priced by hand
fetched listing — /tmp/cx4-live/.xencode/cache/price-lookup.json
  459 prices off openrouter, read on 2026-10-03, 0 days ago
  7 entries carried no readable price
models this project has run: 1; priced from the listing: 1; with no price in either document: 0
  • llamacpp:qwen/qwen3-8b — the listing's qwen/qwen3-8b: $0.117 in / $0.455 out per million tokens, no cache rate — reads billed as input, which is an upper bound
```

The models named at the end are the ones read out of this project's own records,
so `show` answers "what would my next `/cost` say" for the models that were
actually run — not the 459 the catalogue happens to list. A run model with no
rate in either document is named as unpriced rather than counted as free.

`--format json` prints `pricing_json` (path, whether it is present, the models it
names, and any entry it refused to read with the reason), `listing` (source,
`fetched_at_unix_ms`, `fetched_on`, `age_days`, `expired`, `priced_models`,
`unreadable` — or `null` when nothing has been fetched), `price_lookup`,
`listing_is_read` (whether a report is consulting the listing *right now*, which
needs the setting on and the copy inside its 7 days), and then `models_run`,
`priced_from_listing` and `unpriced`.

`fetch` is the only thing in xencode that dials out for a price, and it is asked
for. The request carries no key and sends nothing about this project — the
listing is published for anybody to read — just a `User-Agent` of
`Xencode/<version> (price lookup)`. It waits 20 seconds for an answer and refuses
one larger than 8 MiB: the document it fetched here is 764,719 bytes covering 466
models, so eight megabytes is the same listing several times over, which is a
server that has started answering a different question. A fetch that fails leaves
the previous copy exactly where it is, ages and all; the three ways it can fail,
and what each says. The first two are from a live run against an address nothing
is listening on; the third is checked by a test that serves a real 503 over a
socket, because it would need somebody else's server to be broken to show it
here:

```
$ xencode prices fetch --url ftp://openrouter.ai/models
error: only http/https can be fetched: ftp://openrouter.ai/models

$ xencode prices fetch --url http://127.0.0.1:1/api/v1/models
error: the price listing could not be reached: error sending request for url (http://127.0.0.1:1/api/v1/models)

error: the price listing answered 503 instead of a body
```

Each exits 1. The `--url` is there for a gateway that publishes the same
document, and for pointing the command at an address you control; it is not a
way to make a cost report read a price from somewhere you have not checked, since
what arrives is parsed for the same three fields and cached with the same 7-day
life.

### `xencode release-notes [--from <ref>] [--to <ref>] [--release <version>] [--out <path>] [--force] [--format text|json]`

Put the release notes together from the two places this project already writes
about what shipped: the commits since the last release, and the `## [Unreleased]`
block of `CHANGELOG.md`. The output is a draft, and the two lists of where those
sources disagree are the reason to generate it.

```bash
xencode release-notes                            # the draft on standard output
xencode release-notes --from HEAD~6              # only the work after that commit
xencode release-notes --release 2.2.0            # label the heading with a version
xencode release-notes --out draft/notes.md       # write it where a person will edit it
```

Nothing is parsed out of the commit subjects, and there is no `feat:` / `fix:`
vocabulary to learn: this repository's 900-odd messages are already sentences, and
the categories come from the changelog's own `### Added` / `### Changed` /
`### Fixed` headings, in the order the file keeps them. What ties an entry to a
commit is the plan id the heading names — `QO-4`, `M-6` — matched against the ids
the commit subjects name.

The range starts at the newest tag. Where there is none, as here, the draft says
so and covers every reachable commit rather than inventing a boundary:

```
$ xencode release-notes --release 2.2.0 --out /tmp/qo6_draft/notes.md
  draft written to /tmp/qo6_draft/notes.md
  this repository has no tags, so the draft covers every commit reachable from HEAD
  903 commits, 33 of them named by the changelog's unreleased block (131 entries)
  870 commits no entry accounts for, listed in the draft
  3 entries name no commit in the range, listed in the draft
```

Both lists are in the draft, newest commit first. The first is work that would go
out in a release no reader was told about:

```
### 870 commits with no changelog entry

- `afb9326` — Rewrite one crate-graph assertion into the form the current lint asks for
- `59bf825` — Add the task evidence graph (EVd-8) after checking ten outside projects
- `45a6c2a` — Add /impact <file>: a fan-out panel over a file's blast radius
```

and the second is an entry whose id no commit in the range carries — usually
because the commit that shipped it describes the work without naming the id:

```
### 3 entries naming no commit in this range

- **`QD-2`: `/impact <file>` — the blast-radius panel in the TUI** — named QD-2, not in the range
- **`QD-5`: `xencode removal <file>` — what deleting a file would cost** — named QD-5, not in the range
- **`QD-1`: change-impact analysis with `xencode impact <file>`** — named QD-1, not in the range
```

An unexplained list that long is not something anyone can read, so the draft
names the first 50 and counts the rest:

```
- … and 820 more, oldest not printed; pass `--from <ref>` to put the range around this release alone
```

`--out` refuses a file that is already there, because the whole point is that a
person edits the draft afterwards and regenerating it over their wording is how
an afternoon of it is lost. `--force` says you meant it:

```
$ xencode release-notes --out /tmp/qo6_draft/notes.md
error: /tmp/qo6_draft/notes.md already exists and may hold edits. Pass --force to replace it.
```

Nothing here rewrites `CHANGELOG.md` — read from it, written beside it.
`--format json` returns the same draft as data: the entries with their categories
and ids, the commit count with how many of them an entry names, the ids matched
on both sides, and the two gap lists with each commit's seven-character hash.

### `xencode test [--package <name>] [--retries N] [--stress-count N] [--timeout 1800] [--format text|json]`

Run the test suite through `cargo nextest`, and **never call a test that only
passed on retry a pass**.

```bash
xencode test                            # whole workspace
xencode test --package xencode-core-rs  # one crate
xencode test --retries 3                # allow retries, still refuses to pass a flake
xencode test --stress-count 5           # surface order dependence
```

**Why this exists rather than plain `cargo test`.** nextest runs each test in its
own process. Its default, though, accepts a broken test as a success — measured
here on a test that fails once and passes on retry:

| `--flaky-result` | exit code | summary |
|---|---|---|
| `pass` (default) | **0** | `1 passed (1 flaky)` |
| `fail` | 100 | `1 failed` |

A green exit code then stops meaning anything. So `xencode test` always passes
`--retries` and `--flaky-result fail` itself, which means a repository's own
`nextest.toml` or a `NEXTEST_FLAKY_RESULT` in the environment cannot change the
result, and it calls a run a pass only when the exit code is zero **and** nothing
was flaky.

Flaky tests are listed by name, because a quarantine list has to say *which*:

```
  FLAKY — passed only on a retry, so not a pass:

    mycrate some::module::test
```

**Two things it will not hide.** With `--flaky-result fail`, nextest cancels at
the first flake, so one pass does not reach every test — the output says so and
points at `--retries 0` for the full list. And a non-zero exit that names no test
is not a test failure; the command reports that the build or workspace failed
instead of pointing you after tests that are not broken.

The workspace is located, not assumed: this repository's manifest is under
`rust/`, and `xencode test` from the root finds it. If several subdirectories have
a `Cargo.toml`, the candidates are named rather than one being guessed.

Without nextest installed, it falls back to the repository's own test command
(proved at that moment, not borrowed from an earlier verdict) and says so —
including that a fallback run cannot check for retries, so its result cannot be
compared to a nextest run.

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
