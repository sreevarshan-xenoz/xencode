# Changelog

All notable changes to the Xencode project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — the instructions a model is given, named and versioned
Five prompts went over the wire from strings written in the middle of Rust code:
the agent system prompt, the tool vocabulary that rides on the end of it, the
transcript-folding prompt, and the two subagent briefs. They are files now, under
`rust/crates/xencode-context-rs/prompts/`, pulled in at build time. As files a
reworded instruction is a readable diff instead of a line inside a `format!`, and
as compiled-in text they cannot change underneath a running session — which is
what llama.cpp's prompt-cache reuse needs from the front of every request (§13).
Each prompt's version is a hash of its own text and each set's digest a hash of
those, so there is no number to forget to bump and an edit cannot be filed as
"nothing changed". A stray trailing newline is the one thing that does *not* move
a version: it is trimmed before hashing, because it is trimmed before sending too.

The set digest now rides on every metrics line in `.xencode/cache/metrics.jsonl`
and every turn line in `.xencode/cache/turns.jsonl`, so a slow or good answer can
be traced back to the instructions that produced it. A line written by an older
build reads back as "no prompt set recorded", which is the truth rather than a
guess. `/ctx prompts` lists each prompt with its version and the file it came
from, and prints the digest the rows carry.

`/ctx eval` now appends every arm's score, its depth and that digest to
`.xencode/cache/eval.jsonl`, and only prints a change against an earlier run of
the same arm at the same depth taken under the same instructions — otherwise it
says which digest the earlier run used instead. Measured end to end on this
repo's own gold set: three arms at MRR 0.349 / 0.769 / 0.787 over 18 queries, a
second run reporting `+0.000` for all three, and appending one sentence to
`prompts/agent-system.md` moving the digest from `8abca0eb4098` to `c908e9589468`
and turning all three comparisons into the refusal. Reverting the file brought the
digest and the comparisons back. Two things to keep honest about that: retrieval
scoring does not read any of these prompts, so a prompt edit cannot change a
score — what the grouping buys is that a score change is not *blamed* on the
retriever when something else moved; and the log starts empty, so no eval number
from before this change can be compared at all. 10 tests added (980 → 990).

### Added — answers that can be produced again
A llama.cpp request carried a temperature but no seed, so the server drew a fresh
one every time and nothing about a turn's output could be repeated. Two settings
now exist for that: `llama_cpp_seed` in `config.json` (also a "Llama Seed" row in
the TUI's Settings panel) and `xencode query --seed` for a single run, either
alongside the existing `--temperature`. A negative seed means "keep choosing", the
same as `-1` on llama.cpp's own command line, and is sent as such.

What a turn was *asked* to use is recorded with it. Each turn's line in
`.xencode/cache/metrics.jsonl` now carries `temperature` and `seed`, and
`.xencode/cache/metrics-rollup.json` keeps a count of how many generated turns were
pinned and what the newest one used. Only turns that actually produced tokens are
counted — the log is mostly context-assembly lines that never asked a model
anything, and calling them "not repeatable" would have been wrong. The rollup's
version went from 1 to 2, and an old sidecar is rebuilt rather than read: its new
fields would have come back as zeros and looked like a measurement of nothing.

`/cost` closes the loop by saying what the recorded turns can be reproduced from:
which share ran repeatably, what the newest of them used, and — where nothing was
pinned — that `llama_cpp_seed` is what to set. A project with no generated turns
says nothing about repeatability at all.

The seed was checked by running it, against a `llama-server 0.4.0-dev` started
locally from a `Dolphin3.0-Qwen2.5-1.5B` GGUF: at `temperature 1.5` the same prompt
with one seed gave the same answer every time and no seed gave different answers,
and through the binary `xencode query --temperature 1.5 --seed 42` printed the same
line on three runs. Two things break that even when the seed is pinned, and both
are written into the guide where the flag is documented: `xencode query` answers a
different question on each run because it pulls recent turns from shared
conversation memory (run it with `memory_enabled: false`, or a fresh config
directory), and llama.cpp's prefix cache can flip a sampled token between a cold
and a warm request. Note also that this recording covers TUI turns — `xencode query`
still writes no metrics line, and on a GPU, floating-point ordering means a pinned
seed does not make output reproducible on its own. 8 tests added (972 → 980).

### Added — `/cost` says what this project's turns add up to
The TUI has been writing one line to `.xencode/cache/metrics.jsonl` for every
turn that assembles a context, and a performance panel that re-read that whole
file each time it was opened. Both halves are now finished: the lines are folded
into a small sidecar, and the fold is what the cost answer is built from.

`.xencode/cache/metrics-rollup.json` holds the running totals — rows seen, tokens
prompted, served from the KV cache and generated, the same split per session and
per model, the newest line each hardware profile produced, and p50/p95
generation and prompt-evaluation speed over the last 512 turns that reported a
rate. Folding is incremental: a refresh reads only what was appended since the
last one. If the log is replaced or truncated the totals are rebuilt from
scratch rather than quietly double-counted, and a sidecar from another version or
half-written is discarded and recomputed. Percentiles are exact ranks over the
samples kept, each printed with how many samples it covers, and a turn whose
server reported no rate is left out of the window instead of counting as zero.
Measured against the 164 lines a real session had already left in this
repository's log (42,199 bytes, 21 sessions): the rollup's totals matched an
independent hand sum of the raw JSON exactly — 18,395 tokens prompted, none
cached, none generated — and at that size reading the log cost 135 µs against
46 µs for the sidecar (optimized build, this laptop). Repeating the same lines to
16,400 rows, 4.1 MiB, is where the difference is the point: 12.3 ms for the full
read, while the sidecar stays 8 KiB and 36 µs. Those rows carry no server speeds,
which is why the speed windows are empty for them; the report says "No server
reported a speed for these records" rather than showing a rate.

`/cost` prints all of it in the chat pane — records, sessions and the span they
cover, prompted versus generated tokens and the KV-cache share, the two speed
figures, then the breakdown per session (the current one marked `→`) and per
model. It reads local files and asks no model anything, so it answers with every
server down, like `/trace`.

Money is only known if you say what things cost, so prices live in
`.xencode/pricing.json` in the project as dollars per million tokens per model,
with an optional separate rate for cache reads. Editing that file changes the
next answer; nothing is compiled in. A model with no entry is reported as `price
unknown`, a partly priced session as `at least $x (no price for N model)`, a
missing file as missing, and a line that would not parse is named in the report
instead of being dropped. No figure is ever invented, and `$0` means a price of
zero was written down, not that a price is absent.

`cost_budget_usd_micros` in `config.json` turns the session's spend into a status
bar row — `💸 $0.42/$5.00` when every model used has a price, `💸 13120 tok` when
one does not — updated as each turn finishes, with one warning line when the
budget is crossed. It warns and does nothing else: no request is refused over it.

The performance panel and `/ctx kv` read through the same sidecar now, and the
recent-turn rows there come from a bounded 256 KiB read of the end of the log
instead of the whole file. 27 tests added (945 → 972), covering the fold, its
rebuild cases, the price rules and the report wording.

### Added — `xencode query` can write its answer as one JSON event per line
`--format ndjson` on `xencode query` prints a `start` line naming the model, the
client that was dialed, whether the prompt stayed on this machine and the
conversation id, then a `token` line per piece of the answer as it arrives, then
exactly one closing line — `done` with the whole answer and how long it took, or
`error` with why there is no answer. The plain words go to stdout as before when
the flag is left off, and a failure still prints its readable message on stderr,
so stdout of a `--format ndjson` run is parseable from first line to last. Every
line carries `"v": 1`, and the rule a reader follows is written down: a version
it does not know stops, an event type it does not know is skipped.

The property a script is built on is that the token lines, concatenated, are
exactly the answer the `done` line reports. A cached reply holds that too — it
arrives as one token line, not only as a `done` line. Checked against a local
llama.cpp server: a 49-line stream (one `start`, 47 `token`, one `done`) whose
answer contains blank lines and a list reassembled byte for byte through `jq`,
and a second run of the same prompt answered from cache in 40 ms with
`"cached": true`.

Token counts are `null` unless the route reported them. A llama.cpp server
publishes them only when its stream ends with a usage chunk, and the build on
this machine (0.4.0-dev, 10809) does not, so the measured runs above carry
elapsed time and no rate. There is no `--stream` flag and no `tool` event:
`--format ndjson` streams already, and `xencode query` sends a single request
without running the agent loop, so it has no tool calls to report.

### Added — the audit log can be checked for edits made afterwards
The session server's `audit.jsonl` recorded who did what, and nothing could say
whether the file still held what had been written to it. Each record now carries
a digest of its own contents and the digest of the record before it, so
`xencode audit verify` reports an edited, deleted, moved or appended record on a
specific line and exits non-zero:

```
line 2: the contents do not match the digest recorded on this line
/home/sree/.xencode/audit.jsonl: 2 records, chain broken — …
```

A log written before this change is carried along rather than discarded: its
older records are counted and named as proving nothing about themselves, and the
first new record links to them, so deleting one of the old lines still breaks
the chain. A file that stops half-way through a record — what a crash or a full
disk leaves behind — is reported as an interrupted write rather than as
tampering.

The limits are the limits a hash chain without a key has. Whoever can rewrite
the whole file can recompute every digest, and cutting the end off a log leaves a
shorter chain that verifies cleanly, because nothing outside the file says how
long it should be. This detects an edit; it does not prove the log is complete.

### Fixed — a streamed answer is no longer shortened by where the network cut it
A model's answer arrives as a series of small reads, and the boundaries between
them are chosen by the network stack rather than by the protocol. Every one of
the eight streaming readers here used to decode a single read on its own and
split it on newlines. That loses text in two ways at once: a data line that
straddles two reads fails to parse as JSON in both halves, so the answer comes
back with a piece missing from the middle; and a read that ends inside a
multi-byte character — any Japanese, Cyrillic or accented text — fails to decode
as UTF-8, so the whole read is thrown away and the answer can come back empty.
Neither was reported. Nothing said "the server wrote more than this".

Reads now pass through one buffer that keeps whatever is not yet a complete line
and hands it to the next read, and a line that is genuinely invalid UTF-8 is
passed through with replacement characters instead of discarded. Applies to
Ollama, llama.cpp, OpenRouter, the OpenAI-compatible path, Anthropic, Gemini and
Qwen.

What makes this trustworthy rather than plausible: the streams it is checked
against were recorded from a real server, not written by hand. Four recordings
of `llama-server` output are committed under
`rust/crates/xencode-providers-rs/tests/fixtures/cassettes/` with the machine,
build, model and command that produced them, and a local server replays them on
a real port, deliberately cutting each line in two mid-character. The Japanese
recording is 2,967 bytes, and all 2,968 ways of cutting it into two pieces are
tried: the answer reassembles whole at every one of them. The previous behaviour
is kept in the test and compared against the same stream at its 2,966 interior
cut points; it lost text at 2,949 of them, every single one except the 17 that
happened to land exactly on a line break. One recording covers a two-turn agent
run whose tool call arrives in twelve pieces — the test executes the real command
and checks that the result the model then saw was the one the shell produced.

That last recording also shows a gap rather than fixing one: a thinking model can
spend its whole answer on reasoning that this product does not read, and finish
on the token limit without producing a visible character. Replayed, it answers
blank. The test asserts the blank, so the change that fills it — surfacing
reasoning output — has to be a deliberate one.

### Added — every agent turn leaves a record, and `/trace` reads it back
Until now a finished turn left nothing behind you could look at. Each one now
appends a line to `.xencode/cache/turns.jsonl` in the project — how long it ran,
how many rounds the loop took, which model and server answered, whether it
stopped on a provider error, and each tool call with how it ended — and the new
`/trace [turns]` command prints the newest fifty of them in the TUI, with a
per-task total on the first line. It reads a local file, so it works with every
model server down.

The omissions are the point as much as the contents. Tool output is where secrets
turn up — a `read_file` of a `.env`, a `curl -v`, a failing test printing an
environment — so a trace stores no prompt text (only a short digest of it), no
tool arguments, and no complete output. Each output contributes a short tail that
is scrubbed of anything shaped like a key, token, password or private key before
it reaches the file. Scrubbing matches known shapes, so a secret in a form none
of them covers would still be written; that is the honest limit of this.

Token counts are reported only when a server actually reported one — llama.cpp
does, Ollama does not — and cost is never estimated, so those fields are `null`
and `/trace` says which is the case rather than printing a number it made up.

### Changed — a recorded request says which model it went to
Each row in `.xencode/cache/metrics.jsonl` carried token counts and speeds and
nothing about itself: no conversation, no model, no server. Anything read out of
that file was therefore one average over every model and session ever used,
which is the wrong denominator for a question like "is retrieval finding the
right files". A row now also records `session_id`, `model`, `provider` (`ollama`,
`llamacpp`, `remote`, `openrouter`, `qwen`, `anthropic`, `google_gemini`) and
`source` (`local` or `cloud`). `est_cost_micros` and `power_w` are part of the
shape as well and are `null` on every row written today, because nothing
measures a price or a wattage yet and a number invented here would be
indistinguishable from a measurement later.

Old rows are untouched: those names are simply absent from them and read back as
empty, so the profiler panel keeps working across a file holding both kinds of
line. The provider and destination are answered by the same code that decides
where a model id goes, which is what keeps a row from recording a local server
for a request that was about to leave the machine. One limit worth stating: rows
are written where a context is assembled and where llama.cpp reports timings, so
`xencode query` writes none and a cloud request has no row yet.

### Changed — a model on the internet is off until you say yes
The routing decision added in the previous entry only stopped a *fallback* from
changing where your conversation goes; choosing a cloud model still worked
without asking. Now there is a setting for that: `allow_cloud_models` in
config.json, off by default. While it is off, a `qwen:…`, `google_gemini:…`,
OpenRouter-style `vendor/model` or `remote:…`-at-a-remote-host model is refused
before any connection is opened, and the refusal names the model and the command
that would allow it. Turn it on with
`xencode config set allow_cloud_models true`, or with the new **Cloud Models**
row in Settings → Providers.

**If you use a cloud provider today, this is the one thing to know:** a config
written by an older version has no such key, and no key means off. Set it once
and it stays set. A key in `api_keys` is deliberately not treated as permission
— filling in a Qwen or OpenRouter key says who you are to that service, not that
your code may go there.

The TUI now says which rule is running: the status bar reads `🔒 local only` or
`🌐 cloud allowed`, and the model list's `[cloud]` label is computed by the same
rules the request routers use. That label used to guess from the name, which got
two things wrong in the same list — an Ollama model called `qwen-72b-chat` was
marked cloud, and `vendor/model` ids were marked cloud even with no OpenRouter
key configured, which is the one case where such a model runs locally.

One boundary is stated rather than glossed: the rule looks at the server this
program talks to, so `remote:` is judged by the address in `remote_base_url`. A
Colab GPU reached through `xencode colab up` arrives at `127.0.0.1`, so it is
allowed while `allow_cloud_models` is off; the virtual machine on the far end of
that tunnel is still Google's, and `xencode colab down` is what ends it.

### Fixed — a provider going down no longer moves your conversation elsewhere
A local model that failed — Ollama restarting, a rate limit, a bad key — handed
the whole exchange to the next entry in `agent_fallback_models`, whatever that
entry was. The chain was built from strings, with no idea which entries leave
the machine, and every error except our own response-decode failure advanced it.
So the README's "your code never leaves your machine" posture lasted only until
the first transient failure, at which point a conversation held with
`qwen2.5:7b` could be continued by an internet API. The same shape waited in
reverse for a cloud primary with a local alternate.

Where a prompt is going is now decided in one place,
`xencode-providers-rs/src/egress.rs`, by reading the same prefix rules the three
request routers already used: `anthropic:`, `qwen:` and `google_gemini:` are
off-machine; `llamacpp:` / `llama:` and a bare Ollama name are local; a model
containing `/` is off-machine only when an OpenRouter key is configured, because
without one it falls through to Ollama; and `remote:` is judged by the host its
configured URL actually names, not by its prefix. A fallback candidate is now
tried only if it sends the conversation where the primary would have sent it,
and a candidate that was skipped for that reason is named in the transcript as
`[FALLBACK]not tried: …` rather than vanishing silently — "no fallback ran" and
"the only fallback you configured would have leaked" are different situations
and now look different. A request refused by policy ends the turn: no retry, and
no next candidate, because a refusal is a decision rather than a failure.

Nothing was refused by that change on its own. The setting that decides whether
an internet-connected route may be used at all shipped allowing every route,
which is today's behaviour, and the classifier plus the fallback rule are what
this entry delivers; making the local-first choice the default is the entry
above.

### Fixed — an attached screenshot no longer goes out at full size
Attaching an image sent the file's bytes exactly as they sat on disk. A
1901 × 1061 screenshot left this workspace as 2.3 million base64 characters in
the request, and a 2880-pixel wallpaper preview as 1.16 million, on every turn
that kept the message in history — while vision models resample to roughly a
1568-pixel long edge anyway, so the extra detail never reached the model. The
20 MiB image ceiling that the inventory command enforces was also never applied
to the attach path at all.

Attached PNG and JPEG images are now decoded, capped at 1568 px on the longest
edge, and recompressed as JPEG at quality 80 — measured on real files from this
machine: the screenshot above went to 258 KiB (353 339 characters), the wallpaper
to 276 KiB (378 163), a 2560 × 1700 photograph to 679 KiB (927 295). Counted at
the same four-characters-per-token the context budgeter uses, that first
screenshot is worth about 88 000 tokens of request where it used to cost about
577 000. Every one of those payloads now fits under 1 MiB, and the file a real
user would notice — an image with transparency — keeps its alpha channel and
stays PNG, since flattening it would throw away information the model may need.
GIF, WebP, BMP, ICO and SVG go out untouched, because only the PNG and JPEG
codecs are linked; anything that fails to decode, or that would come out bigger
than it went in, is sent as it arrived rather than refused. When an image is
changed on the way out, the prompt says so:
`(image changed before sending: sent as image/jpeg …)`.

### Changed — the file finder reads what files say about themselves
The lexical scoring pass used to run *after* the candidate list had been cut to
its top few, so it could only reshuffle files the filename-and-symbol pass had
already surfaced — and the documents it scored were built entirely from
identifiers, never a sentence. A question phrased in ordinary words could
therefore only be answered by a file whose name happens to contain one of them.
Every indexed file now also stores its documentation prose (capped at 1.2 KB
per file, with code samples inside those comments left out), and the lexical
pass runs over the whole index *before* the cut, so a file with no matching name
and no matching symbol can still reach the prompt on the strength of what its
own docs say.

Measured against this workspace's 18-question gold set over an index of 155
files: filename, symbol and dependency ranking alone gets the right file first
28% of the time and into the top five 50% of the time, a mean reciprocal rank of
0.338. Adding the lexical pass over names and symbols lifts those to 67% / 94% /
0.782. Adding the documentation prose on top leaves the first two where they are
and lifts the rank to 0.796 — one question that landed fourth now lands second.
Cost per query in a release build: 1.1 ms without the lexical pass, 12.5 ms over
names and symbols, 16.5 ms including prose, against a generation that takes
seconds. One question — "refreshing a path that is not rust does nothing" — is
still missed by every pass.

The hybrid pass is on in chat by default; `XCODE_HYBRID=0` puts a turn back on
filename-and-symbol ranking, and `/ctx eval` prints all three numbers side by
side so the switch can be tested rather than trusted. The eval scores the
without-prose run on purpose, because most of the gain came from moving the pass
earlier, not from the prose.

### Fixed — the repository map now sees modules, traits and what a crate exports
The structural layer underneath `/ctx`, `/advise` and `xencode advise` was a set
of regular expressions over Rust text, and four of its holes were large enough to
make the results wrong rather than merely coarse. A `mod x;` line — the only thing
that connects `lib.rs` to the files of its crate — produced no dependency at all,
so a module tree was invisible: this workspace's index goes from 127 recorded
dependencies to 211, of which 82 are module declarations, every one of which
resolves to a real file. `xencode advise` reported 18 files here as orphans; the
eight it stops reporting are ordinary source files that a crate root declares with
`mod` (`anthropic.rs`, `gemini.rs`, `qwen.rs`, `mcp.rs`, `panic.rs`, `voice.rs`,
`collab_client.rs`, `widgets/spinner.rs`), and the ten it still reports are
integration tests under `tests/`, which no module declares — the honest ones. An `impl MyTrait for MyType` block is a
dependency on the file that declares the trait, and none of those edges existed
because no `use` statement has to mention it — two appear in this workspace, and
the rest of the `impl` blocks here implement traits from outside the indexed tree,
which are skipped rather than guessed at. `enum`, `trait` and `type` declarations
were not collected at all, a `struct` without `pub` was invisible, and `const fn`
and `extern "C" fn` were not counted as functions: 3,029 indexed symbol names are
now 3,540. Finally, the names a crate re-exported were recorded as the *module*
the re-export came from, so `pub use database::Pool` was filed under `database`
and the export inventory of this workspace held 57 directory names and not one
thing anyone can import — it now holds the 274 names actually exported, aliases and
brace groups included, and a glob (`pub use foo::*`) contributes none rather than
one wrong name.

What that buys is measurable and mixed, and both halves are reported. On the
eighteen-question retrieval scorecard for a real index of this workspace, the
hybrid pass improves — first-answer rate 0.39 → 0.44, mean reciprocal rank
0.444 → 0.472 — while the plain structural pass loses a little ordering precision
(mean reciprocal rank 0.366 → 0.338) without losing any reach: the rank-1 and
top-five rates are unchanged at 0.28 and 0.50, because three answers moved one
place down as the newly visible module edges pulled other files up beside them.
More accurate structure is better material for a text-matching second pass than
for a raw edge count. `self::` paths are also resolved relative to the module's
own directory now, which is what Rust means; this workspace contains no such
import, so nothing here moved.

### Fixed — the file watcher now reports what it could not do
Two ways this feature could fail without saying anything. If the workspace could
not be watched at all — on a repository this size, usually because the operating
system's limit on watched paths is already used up — the watcher quietly gave up,
and "nothing changed" looked exactly like "watching is off". It now says so once
in chat: `⚠ file watching is off: <reason>`. Second, the watcher waits for a pause
in file activity before reporting a batch of changes, and a large pause never
came: a `git checkout` or a build that keeps touching files held the batch
indefinitely. The pause it will wait for is now capped at one second, and a batch
that has been growing for two seconds is reported even while activity continues.
Asking for a longer pause than the cap no longer delays a warning by that much,
because changes from one save arrive within milliseconds anyway.

### Changed — the retrieval scorecard measures something harder
`/ctx eval` grades the file finder against a set of questions whose answers are
known-good files in this repository. One of those questions pointed at
`cmd_output.rs`, a file that has never existed, so it could never be answered and
quietly pulled every score down; and all ten questions were answerable from a file
name alone, which is not a question worth asking a finder. The corpus is now
eighteen questions. The unreachable one was retargeted to the file that really
does trim command output, and the additions are phrased the way people actually
type — including negations and conditions: "secret files are flagged without their
contents being read", "refreshing a path that is not rust does nothing". Expect
lower numbers, and read them as the honest size of the gap: on a real index of
this workspace the deterministic pass scores 0.28 at rank 1 and 0.50 within the
top five, the hybrid pass 0.39 and 0.50. Every one of the eighteen answers still
comes first when it is put against three unrelated files, so the misses are about
telling files apart across a whole repository rather than about the questions
being wrong. A test now opens each gold answer on disk, so a question pointing at
a missing file fails the suite by name.

### Fixed — the security scan no longer flags ordinary code for using common words
`xencode analyze` reported High-severity path traversal (CWE-22) on lines that
merely contained the word `input`, and Medium-severity SSRF (CWE-918) on any line
containing `url` or `user`. A function declaration like `fn parse_input(raw:
&str) -> String {` was enough to produce a finding, so the scan's output could not
be believed. In both patterns the words were listed outside the group they belong
to, so the pattern language read them as separate, independent searches rather
than as arguments that must appear *inside* a file or request call. Both are
grouped correctly now. On a sample of five single-line files, six findings came back
before the change and two after — and the two that remain are the real ones, an
`open(user_path)` and a `fetch(url)`. Four tests pin both directions, each
re-checked against the old pattern to confirm it would have caught this.

### Fixed — a crash in the interface gives your terminal back and leaves a trace
When the TUI hit a bug it panicked with raw mode and the alternate screen still
switched on, so its own error message went onto a screen you could no longer
scroll or type into, your shell came back with keypresses misbehaving, and
nothing recorded that any of it had happened. A crash now puts the terminal back
first — normal input, no mouse capture, cursor visible — and writes the message,
the source location, and a backtrace if `RUST_BACKTRACE` was set to
`~/.xencode/last_panic.log`, readable only by you. The file is there for the
next release's health check to report; until then it is the one place a crash
can be described after the fact.

### Fixed — a log left half-written by a crash is still readable
`.xencode/cache/metrics.jsonl`, the file the TUI profiler reads, is appended one
record per line. A process killed mid-append — or a disk that filled — used to
leave a final line that was only part of a record, and a reader had no way to
tell that interrupted line from a record that was genuinely wrong. Reading a
log now drops the partial last line and reports that it did, while a corrupt
line anywhere else in the file is counted rather than passed over, because that
says something is broken in the writer. The profiler keeps the turns it had
already recorded.

### Fixed — your configuration file is no longer readable by everyone
`~/.xencode/config.json` holds provider API keys as plain text, and every
version so far wrote it with the operating system's default file permissions —
`644`, readable by any other user on the machine and by anything that can read
a backup or synced copy of your home directory. The manuals told you to
`chmod 600` it by hand; nothing stopped the next save from leaving whatever
mode the editor happened to create. Saving now goes through one helper that
writes to a fresh private file in the same directory, flushes it to disk, and
renames it into place, so a crash part-way through leaves the previous config
intact rather than a truncated one, and the file that appears is readable only
by you. A config that was already group- or world-readable is tightened the
next time a setting is saved. The same helper now backs every other piece of
state Xencode keeps on disk — the response cache, the project index and
transcript under `.xencode/`, conversation memory, Colab session state, and the
background-task registry that other `xencode tasks` processes read at any moment.

### Docs — what the orchestrator proposal was still missing (Milestone S revision)
The response to the recorded appendix added four items the original thirty-eight did not
contain, and they are the parts the idea needed to be an architecture rather than a
feature list. A common event protocol is now its own piece of work and the centre of the
runtime: six vendors can only be coordinated by something they all normalise into, and the
adapters are thin precisely because that target exists. A task contract fixes what "done"
means before a worker is launched — allowed and forbidden files, expected output, the
verification commands, the completion condition — because the everyday failure of this whole
feature is an agent that reports itself finished when nobody ever stated the condition. A
result envelope keeps a worker's claims in a separate field from its evidence, so a
reviewing agent receives a machine-readable engineering record instead of its predecessor's
prose. And a veto that the blocked party cannot lift, clearable only by a named reviewer, a
human, or a policy that says what it clears: that is the one kind of authority no vendor has
ever had over another, and therefore the honest answer to why anything belongs in the middle
at all. Parallelism stopped being a configured maximum and became a computation over whether
tasks are independent, whether their file sets collide, and what they cost.

Two rules arrived with them. "Do not scrape terminal output" is now a fallback order rather
than a preference: read a worker's structured stream, otherwise derive from repository state
ourselves, never scrape a terminal for text. And the rule the feature will be judged on —
live control over a worker's permissions is real for exactly one vendor today, while every
other vendor only accepts policy fixed before it starts, so the interface must display which
of the two it is looking at. Claiming the first when holding the second is how a safety
feature becomes a false assurance, and it is the same standard every other panel in this
project is already held to. What comes next is deliberately not a user interface: the
measurement, then the protocol derived from it, then the state model, then the task graph.
Plan items in the wave order: 284.

### Docs — the multi-agent orchestrator proposal, measured (Milestone S)
A third external proposal list arrived: thirty-eight items describing an
`/orchestrator` mode that would run Codex, Claude Code, Gemini CLI, OpenCode and Agy as
workers under one plan, one permission broker, one shared memory and one merge decision.
Unlike the earlier lists, this one could be checked against reality rather than argued
about, because all five of those programs are installed on the machine this project is
developed on — so the appendix records a table of what each one's own help text actually
advertises, read today. Three results reshaped the idea. The five have already converged
on the same headless contract (single-prompt mode, a machine-readable event stream, the
same three-rung approval ladder, resumable sessions), which makes the proposed adapter
layer a normaliser over command-line flags rather than five separate protocols, and drops
three of its twelve proposed methods because no vendor implements them. Roughly a third of
what was proposed is already shipped by the vendors being orchestrated, including their
own session daemons, subagent definitions, non-interactive review commands and health
checks — and Claude Code can import another agent's configuration wholesale. And the
permission broker, which the proposal treats as the centrepiece, has exactly one working
seam today: one vendor will route its approval requests to a model-context-protocol tool,
which is the same surface task M-5 already plans to expose, so that task moves earlier in
the order and every other vendor gets policy granted before launch — which is not control
and is now described as such. The proposal does reverse one standing rejection, on grounds
the plan accepts: the agent task graph was parked because a single local model saturates
this laptop, and workers calling a remote service do not. What replaces that ceiling is
written down instead of glossed over: one machine verifying everyone's output, a small
local model doing the planning, and real money spent per fan-out. Thirteen proposals were
already in the plan under another name, thirteen survive in a narrowed form, ten are new,
two are rejected, and twenty-two new task IDs join the wave order — the first of which is
a measurement: run each installed agent headless on a tiny read-only task and record what
its event stream actually carries. Nothing else in the appendix may be treated as a
commitment until that matrix exists. Nothing here is built; the wave count in Milestone R
grows from fifteen to eighteen accordingly.

### Docs — the plan finally has an order (Milestone R)
Four research appendices had declined to rank their candidates, which was honest
about value but left no way to start anything. The owner supplied an ordering
instead of a ranking: fifteen dependency waves over all 258 recorded plan items,
from "fix what makes today's output untrustworthy" through observability, the
model layer, code intelligence, verification, trust and durable knowledge out to
the product surface. Each item is placed exactly once and keeps the identifier the
appendix that produced it gave it, so provenance survives ordering; the placement
was checked by script rather than by eye. Seven of the proposed placements moved,
because the plan already fixes some dependencies in ink — most importantly, the
project-notes writer cannot be a first-day item, since it must not exist before
the knowledge source-class model that stops it turning a bad summary into durable
truth. One item is parked as genuinely contested rather than resolved: speculative
decoding, which an earlier milestone declined for this laptop's hardware and a
later one re-proposed for a rented GPU. What the waves deliberately do not decide
is which item in a wave is worth doing first.

### Docs — the commit rule now names what a commit completes
The standing rule said "commit each change before the next". With an ordered plan
that was too loose to audit, so `AGENTS.md` now defines the unit as a single plan
item rather than a wave, requires every commit message to name the item or items it
completes, requires both identifiers when two items turn out to be one change, and
makes wave completion traceable — a wave is done only when each non-deferred item
in it can be matched to a commit naming it. It also states plainly that commit
messages are user-facing product history and must be written in plain, fully named
English, with a list of the internal shorthand that is not acceptable in them.
Nothing about the code changed.

### Docs — Milestone Q: the second hundred, dispositioned
A second external proposal list (100 items around project DNA, time, system,
trust and experimentation) was checked item by item against the Rust tree rather
than adopted: 25 were already planned under existing IDs, 35 narrowed, 10 new,
30 rejected, recorded as 25 do-not-build register rows, and taking the research
pool from 176 to 208 options — **unranked**, as with N, O and P. Four of the
list's premises did not survive contact with this machine: it assumes no GPU (the
box has an MX250, 2 GiB), the plaintext-key `config.json` at mode 644 is
described as a missing capability when the OS keyring is present and simply
unused, the eval fixture `gold.json` names a `cmd_output.rs` that is not in the
tree, and two `xencode-analysis-rs` security regexes group their alternation
wrong, so any line containing the bare word `input` — or `url` — reports High
severity. Nothing was built; recording the defects is not fixing them.

### Docs — two promises the code never made
The research passes behind Milestones N–P checked manual claims against the
implementation, and two did not hold. `README.md` advertised "error
classification and targeted fix suggestions": there is no classifier — tool and
shell failures reach the model as `error:`/`exit <code>` results and the agent's
instructions tell it to name the failure — so the bullet now says that, and says
the classifier is planned rather than present. The Roadmap section claimed there
was "no open milestone" and "no known gap between promise and code"; five
planning tracks (L, M, N, O, P) are recorded, and the same passes listed live
gaps as defects. The stale crate count (14 → 15) is corrected with it.

### Docs — the product description catches up with the product
Milestone K made the model endpoint something you point at, so "offline-first"
in the tagline, the highlights list and the privacy row no longer described the
build. Replaced with "local-first, bring your own model" in `README.md` and
`docs/USER_MANUAL.md`, and the GitHub repo description with it. The absolute
"works with no network" claim is deliberately not made — that needs the offline
conformance test (plan item LF-8) before it can be asserted.

### Milestone K — remote providers + Google Colab — complete ✅
A GPU you do not own as an inference backend: `remote:` routes any
OpenAI-compatible endpoint (dedicated to the Colab SSH-forward case in the
docs), Settings gains real provider-key editing with masked `Secret` rows,
and the Colab bridge gets its preflight gate. The connectivity decision that
shapes the milestone: single supported transport is the official
`google-colab-cli` `colab ssh --proxy-mode` WebSocket SSH bridge — no public
tunnel, which Colab's free tier forbids (account suspension). Version trap
caught in code: `google-colab-cli` 0.6.0 shipped without the `ssh`
subcommand (upstream issue #102); preflight requires >= 0.7.0.

### Added
- **Colab bridge proven on a live VM, and fixed by what the run showed**
  (K-2c/K-3 follow-up). A free-tier T4 was brought up, served a Q4_K_M GGUF
  through llama.cpp, and answered real `xencode query -m 'remote:…'` calls over
  the forward; Provider Health went green, `up --reconnect` restored a killed
  forward, and `down` left no session and no orphan process. Six defects came
  out of that run: the bridge now logs in as **root** (Colab injects the key
  for root only, so the old default could never authenticate), the VM-side port
  moved to **18080** (Colab's own proxy permanently holds `8080`), the
  bootstrap reports `READY` only once `/v1/models` serves (a 0.5B model took
  ~39 s to load, so a spawn-time `READY` raced the probe), `--quant` /
  `config colab_quant` choose the GGUF quant, bridge-slot errors
  (`Already-active SSH session`, `banner exchange` timeouts) are retried rather
  than failing the bring-up, `--reconnect` tries the forward before re-running
  the bootstrap (9 s vs a full re-download), and the detached forward no longer
  inherits stderr — a piped `xencode colab up | grep` used to hang until the
  tunnel died.
- **Colab survivability: 12-hour-reap detection + one-key reconnect** (K-3).
  `xencode colab status` now compares the recorded `started_at` age against
  the endpoint: older than 12 hours (Colab's per-VM runtime limit) with a
  dead forward is reported as a reaped VM, with the exact recovery command.
  `xencode colab up --reconnect` rebuilds the bridge from `colab.json`
  alone: a live endpoint short-circuits to its URL (zero `colab`/`ssh`
  calls); otherwise the session is re-created if Colab reaped it, the
  bootstrap re-runs, and the forward re-spawns until `/v1/models` answers —
  no re-typing of the original flags. Missing `colab.json` fails fast with
  a fix suggestion. The TUI Provider Health panel gains a Remote/Colab
  forward row: seeded like the other providers (unconfigured remote URLs
  read "Remote URL not configured (Settings → Remote URL)"), the health
  check probes `{remote_base_url}/models` with a 5s cap, and Connection
  Details lists the configured Remote URI.
- **Colab VM lifecycle** (K-2c). `xencode colab up` brings a Colab VM up
  end-to-end: creates the session (`colab new --gpu <gpu> -s <name>`) when
  absent, pushes an ssh bootstrap that installs and starts the chosen runtime
  bound to `127.0.0.1` only (llama.cpp — pinned prebuilt llama.cpp release +
  one HF GGUF on 18080 — or ollama — `ollama serve` on 11434, its tags flow into
  the model picker), holds the `-N -l root -L` forward, waits for `/v1/models`,
  writes `~/.xencode/colab.json`, and points
  `llama_cpp_url`/`ollama_url`/`remote_base_url` at the forward.
  `xencode colab status` reports forward
  pid / `colab sessions` listing / endpoint probe and never fails hard;
  `xencode colab down` is idempotent (kill forward, `colab stop`, clear
  state). Flag > `config colab_*` fallbacks; session names validated before
  any shell use.
- **Colab preflight gate** (K-2a). `xencode colab preflight` verifies in one
  pass that the bridge is usable before any VM is brought up: the `colab` CLI
  on PATH, version >= 0.7.0 (version string *and* a functional `colab ssh
  --help` probe catch the 0.6.0 trap), backend auth via `colab sessions`,
  ssh/ssh-keygen on PATH, and an ed25519 key pair under the xencode config dir
  (`xencode colab preflight --generate-key` creates it). Each failing check
  prints a runnable fix; the command exits non-zero when any check fails.
  Lives in the new `xencode-colab-rs` crate.
- **`xencode config set remote_url` / `remote_key`** (Milestone K ¶). Any
  OpenAI-compatible endpoint can now be pointed at from config — the Remote /
  Colab provider kind routes through `OpenAICompatibleProvider` in every
  `ProviderManager` path (`remote_base_url` + `api_keys.remote_api_key`).
- **Settings edits provider endpoints and keys** (K-1b). The Providers section
  edits real values instead of reporting key presence: Remote URL as a text
  row, Remote/Gemini/Qwen/OpenRouter keys as masked `Secret` rows with a
  last-four tail on display. Commit saves through `App::save_config()`;
  empties clear the key; `Esc` discards the editing buffer so a plaintext key
  never lingers.

### Milestone J — complete ✅
Seven TUI panels rendered hardcoded phrase lists (voice, terminal assistant,
security auditor, profiler, custom models, learning mode, multi-language) and
the plugin registry loaded no runtime. All eight items landed — the
security auditor and the profiler are real, the terminal assistant delegates to
a model through the agent's approval gate, the multi-language panel tabulates
a real `scan_tree` walk and translates through a real model call, the custom
models panel edits `model_profiles` in `config.json`, the learning mode
panel teaches files the project index actually found, the voice panel
records from the microphone and keeps a WAV, transcribing only when a whisper
CLI is installed, and a manifest now loads for real (J-08). No TUI panel ships
scripted content and no plugin manifest claims a capability this build lacks.
`NEXT_PLAN_TASKS.md` §
Milestone J tracks the work item by item, with a done-when rule per panel.

### Added
- **Plugin runtime** (J-08). `xencode_plugin_rs::PluginRuntime::load(dir,
  xencode_version)` discovers `plugin.json` / `manifest.json`, skips a manifest
  whose `xencode_version` does not accept this build (reported, not silently
  dropped), registers each remaining one with the `Host`, and flattens the
  session into the two effects a plugin may have: a `prompt_prefix` and
  `before`/`after` hooks. Plugin hooks only land where `agent_hooks` in
  config.json is silent, so the config always wins and the approval gate is
  untouched. There is no dynamic linking and no plugin code — `entry_point` left
  with the Python runtime it named.
- `xencode plugin list` / `install` print what actually took hold (`guardrails
  v1.2.0 — loaded: prompt prefix, 1 before hook(s), 1 after hook(s)`, or
  `NOT LOADED: needs xencode 0.1.0 (declared 9.9.9)`), and the TUI gained
  `/plugin [reload]`, which reports the same verdicts plus the hook and prompt
  totals in effect for the session.
- New `default_plugin_dir()`: `$XCODE_PLUGIN_DIR`, else
  `<data dir>/xencode/plugins` — the directory the CLI installs into is the one
  the TUI loads from at startup.

### Removed
- **Unused deployment surface and stale design docs.** Deleted `k8s/`
  (`deployment.yaml` pinned the Python API's port 8000, `postgres.yaml` and
  `secrets.template.yaml` fed `DATABASE_URL`/`REDIS_URL`/`JWT_SECRET_KEY` to
  nothing) and `monitoring/prometheus.yml` (the Rust server exposes no metrics
  route). `.github/workflows/ci-cd.yml` lost its `deploy-staging` and
  `deploy-production` jobs and its `k8s/**` path trigger — it is now gate →
  image build → push → Trivy; `ci.yml` remains the plain test gate. Also ten
  Python-era documents under `docs/` (analytics, security-scanning,
  performance dashboard, provider-health/phase-3/validation/terminal-test
  summaries, `features/benchmark-wizard.md`, `phase5/fallback-engine.md`), all
  of them describing modules and `localhost:8000` routes that are not in this
  tree and linked from no current doc.
- **Dead Rust, deleted with no replacement.** `xencode-context-rs`
  `cmd_output` (a command-output blob index that nothing ever wrote to or read
  from), the never-implemented `Embedder` extension trait with its `cosine`
  helper and `Bm25::max_score_per_doc`, `expand_dependencies`, and the unused
  `CancelFlag` alias. `xencode-analysis-rs` lost `embeddings`
  (`EmbeddingClient`), `vector_store` (`VectorStore`, `cosine_similarity`) and
  `indexer` (`ChunkIndexer`, `DocumentChunk`) — the retrieval path in
  `xencode-context-rs` never called them. Dropped the `serde_json` and `uuid`
  dependencies those modules were the only users of; suite is 684 tests.
- **Python-era tooling configs.** `.pre-commit-config.yaml` (bandit, ruff,
  mypy, safety, detect-secrets against a baseline file that does not exist) and
  `.bandit` (a findings baseline for code that is no longer in the tree). The
  gates are `cargo fmt --check`, `cargo clippy --workspace --all-targets` and
  `cargo test --workspace`, which CI already runs. Also untracked
  `.xencode/technical_debt.db`, a 274 KiB SQLite artifact of the Python-era
  `technical_debt_manager.py` (`debt_scans` / `debt_items` tables) that no Rust
  code opens — `.gitignore` already covers `.xencode/`.
- **The Python-era documentation archive, deleted rather than kept as history.**
  `DOCUMENTATION.md` (737 lines: a "dual-stack architecture — Python feature
  system + Rust core runtime", a credential vault, ensemble reasoning, 12
  crates/65 tests), `PRD.md`, `project details.md`, `docs/FEATURES.md` (a
  "wild ideas" backlog whose migration status line still said 12 crates/65
  tests), `docs/ARCHITECTURE_DIAGRAMS.md` (mermaid diagrams of an API Gateway, a
  Connection Pool module and a Distributed Cache — the Python package shape, not
  the 14-crate workspace) and `BROWSER_LOGIN_PLAN.md`. Also
  `docs/superpowers/` (4,282 lines of executed migration plans, their specs and
  agent-worker instructions, linked from nothing) and three unused screenshots
  (`images/4-6.jpg`; the README shows 1-3). The README archive table that pointed
  at them is gone, `docs/INSTALL_MANUAL.md` and the README's architecture pointer
  now point at real files, and `docs/ROADMAP.md` was rewritten from the tree:
  its `xencode --git-commit` / `--git-review` / `--git-branch suggest` flags and
  `/analyze` / `/smart` chat commands never existed in the Rust CLI (the real
  entry points are `xencode review` and the TUI's `Ctrl+R` / `Ctrl+Y` /
  `Ctrl+S`), and `docs/api_documentation.md` lost its second half — the
  `xencode.core.*` module reference and Python usage examples (307 → 48 lines,
  the server's auth matrix kept). 9,371 lines of removed docs in total.

### Fixed
- **`xencode plugin remove <name>` accepted a path.** The name was interpolated
  straight into the plugin directory, so `../../etc` resolved outside it. Removal
  now goes through the same `PluginRegistry::plugin_path` guard the installer
  uses and rejects an invalid name (`fix(plugin)`).
- **Provider fallback chain (I4-01)** now honours its own eligibility rule:
  `retry::is_fallback_eligible` shipped with the feature but was never called,
  so a response xencode could not decode walked every configured alternate
  even though each reproduces the failure. The chain advances only on a clean,
  eligible error (`fix(agent)`).
- **TUI tests are hermetic.** `App::new()` read *and* wrote the developer's
  real `~/.xencode/conversation_memory.json` and restored its last messages,
  so a full-workspace run could fail a chat test with another test's transcript
  (`left: "/init abort", right: "/mcp status"`). Added `App::for_tests()`
  (non-persistent memory), made `ConversationMemory::memory_dir()` respect
  `XCODE_CONFIG_DIR`, and stopped persisting slash commands to conversation
  memory — a local TUI verb used to be replayed to the model forever. Tests no
  longer write the developer's real `~/.xencode/config.json` either:
  `App::for_tests()` turns config persistence off (`fix(tui)`).

### Documentation
- **README re-verified against the code**, not just re-worded. Corrected: the
  chat-input command list said "eight" while `SLASH_COMMANDS` has nine, and the
  profiler's metrics file was given as `.xencode/metrics.jsonl` instead of
  `.xencode/cache/metrics.jsonl` (`context-rs::metrics_path`); the prerequisites
  table called Ollama "required" and pinned an "Rust 1.80+" MSRV that nothing in
  the tree declares (CI builds `stable`). Added: the 16 real CLI subcommands,
  `xencode query`'s llama.cpp sampling flags, what `install.sh` / `install.ps1`
  actually do and where each puts the binary, the `release.yml` tag →
  smoke-test → GitHub Release path, the seven Milestone J panels described
  per-panel, and a "nothing scripted" highlight. The Roadmap section stopped
  listing shipped work (retry budgets, provider health UX, multimodal, team
  workflows) as future and now names what is genuinely unbuilt — TTS output,
  arena mode, coordinating several agent runs, AI commit messages, session
  replay — with the Anthropic key and CRDT wiring called out as parked decisions.
- Milestone I close-out honesty sweep (I4-02): every manual was re-checked
  against the tree. Removed fiction: the credential vault and `xencode vault
  init|migrate|status` (no such commands), ensemble/multi-model "reasoning"
  (the fallback chain is sequential), "language-aware AST analysis"
  (per-language heuristics), in-chat `/help /models /model /project /status
  /clear /exit` (eight real slash commands), a first-run setup wizard (there is
  none), Python-era setup and linting in `CONTRIBUTING.md` (`venv`,
  `requirements.txt`, `pytest`, `ruff`, `mypy`, `bandit`, a `dev` branch), and
  `.xencode.example.json`, which described a Python config shape and is now the
  real flat `config.json`, validated through the loader.
- Stated plainly what is not real: seven TUI panels (voice interface, terminal
  assistant, security auditor, performance profiler, custom models, learning
  mode, multi-language) render scripted content; the plugin commands manage
  manifests with no runtime that loads `XencodePlugin`; `GET /api/models`
  appends hardcoded remote entries; `.xencode.example.json`, which described a
  Python config shape and is now the real flat `config.json`. Each of those
  gaps was closed by Milestone J (J-01…J-08) in this same unreleased cycle.
- Recorded a routing gap: `xencode-providers-rs` has an Anthropic client but
  `ApiKeys` has no `anthropic_api_key` and both entry points pass `None`, so an
  `anthropic:…` model cannot authenticate — Claude ids work through
  OpenRouter.
- Counts and flags corrected: 14 crates, 692 tests; `xencode query --session`
  (not `--session-id`); `xencode llamacpp list|set-path`, `cache clear`,
  `memory show <id>`; `models health` requires a model name; README version
  badge back to the real `0.1.0`.

### Added
- Voice Interface is real (Milestone J, J-07). `Enter` used to play a scripted
  session: four hardcoded phrases with invented results (`run tests` →
  "✅ 142 tests passed, 0 failed"), an audio level cycling through
  `0.3, 0.6, 0.8, 0.9, 0.7, 0.4` regardless of any microphone, and a "speaking"
  state for a product with no text-to-speech. Now `Enter` spawns the first
  recorder found on `PATH` (`arecord`, then `pw-record`, then `parec`) with a
  fixed argv that streams raw 16-bit mono 16 kHz — no shell string, so this
  never touches the agent's `run_command` approval path — and every 100 ms
  chunk of what it actually sent yields one RMS reading, which drives the level
  bar, the session peak and the clip length. `Enter` again, or `Esc` while
  recording, ends the capture early and keeps the clip; `m`/`Space` mutes by
  discarding audio rather than saving a silent file. The PCM is written to
  `.xencode/voice/clip-<unix>.wav` through a hand-built RIFF header. Speech text
  comes only from a whisper CLI's stdout (`whisper`, `whisper-cpp`,
  `whisper-cli`); with none installed the panel names the clip it kept and says
  there is no speech engine, and an engine that fails, prints nothing or dies
  contributes its own error or nothing at all. Deleted with this: the canned
  phrase list, the fake level cycle, and the dead `voice_commands`,
  `voice_confidence` and `voice_language` state. Verified against the real
  microphone path — `arecord` on this machine produced a valid 1.85 s
  16-bit mono WAV and the no-engine note — and covered without a microphone by
  nine `voice` module tests that substitute `cat` on a prepared PCM file, six
  app tests for the level/clip/error/mute reductions, and three panel renders.
- Learning Mode panel is real (Milestone J, J-06). It opened with one invented
  lesson — "Rust Ownership Basics", a `calculate_length` snippet, an "exercise"
  with nothing to submit it to — and a quiz it marked correct whenever option 1
  was picked, because `learn_quiz_correct` compared the selection to `0`. Now
  `Enter` reads `.xencode/index/symbols.json` and queues the files that declare
  something (most declarations first, ties by path, five at a time), so the
  lesson list belongs to this workspace: a file with no declarations is not a
  lesson, and a workspace with no index gets "No project index — run /init
  first" and no request. Each lesson shows that file's own text — capped at a
  line boundary, with a note saying how much of it was sent — next to the
  declarations the index recorded, then makes one model call asking for
  `{explain, question, options, answer, why}` about that exact text. The
  model's sentences are drawn under "The model says:"; the answer key is the
  model's, the panel names which option was the key and prints its reason, and a
  reply with no usable key in it is shown as the reply instead of becoming a
  canned question. `p`/`n` walk the queue, `r` re-asks. Dropped `learn_exercise`
  and the `learn_progress_pct` score (progress now means position in the queue);
  fixed the header box, which fitted 1 of its 2 lines so the count never
  showed.
- Custom Models panel is real (Milestone J, J-05). Pressing `Enter` used to seed
  four invented profiles (`Code Assistant`, `Creative Writer`, `Bug Hunter`,
  `Code Reviewer`) with sliders and a "Profile saved!" line that wrote nothing.
  The panel now lists `model_profiles` from `config.json` — a new
  `ModelProfile { name, model, temperature, max_tokens }` in `xencode-config-rs`,
  `#[serde(default)]` so an older config still loads — and an empty list renders
  "None yet — press n" instead of samples. `n` adds a profile from the current
  session's settings, `-`/`+` moves temperature over 0.0–2.0, `←`/`→` steps the
  token budget along 64…8192, and an unset knob starts from the value the session
  would send anyway. `Enter` applies to the next turn in memory only (model id,
  the llama.cpp knobs, the selector's position, plus a `switch` when the id
  points at the llama.cpp server); `s` is the only key that writes `config.json`,
  and it reports what happened — `wrote N profile(s)`, `config.json unchanged:
  {error}`, or that persistence is off in this session. `t` sends one request
  with exactly that profile's settings through the shared `SingleShot` path, so
  the status line is the provider's own reply or its own error. `top_p` was
  **deliberately left out**: `merge_llamacpp_options` is the only place sampling
  knobs reach a request, and it carries temperature and max tokens to llama.cpp —
  nothing in this workspace sends top_p, so the panel states that llama.cpp is
  where these numbers land rather than implying Ollama and cloud endpoints obey
  them. Dead state removed with the seeds: `models_editing`, `models_saving`,
  `models_test_output`, `models_profiles` (the 5-tuple list) and
  `start_custom_models`.
- Multi-language panel is real (Milestone J, J-04). Its "language detection" was
  a fixed table (`python 34.2%`, `javascript 27.1%`, …) for files that are not
  in this workspace, and its "Supported Languages" list was hand-written. `Enter`
  or `d` now runs `scanner::scan_tree` on the workspace root in
  `spawn_blocking` and streams the languages that are *actually present* — file
  count, line count from the scanner's own `count_loc` (blanks and
  comment-leading lines excluded) and share of those lines — sorted by lines,
  then name. The notes under the table carry the walk's fine print: total
  files/lines, what ignore rules skipped, and the fact that secret and binary
  files are listed but never read, so they add files and zero lines rather than
  a made-up size. The language legend is
  `scanner::Language::ALL` (a new const, so the panel reads the enum instead of
  copying it), with `▸` marking what the walk found; two new scanner tests pin
  that `ALL` and `language_for_extension` agree in both directions and that
  every `as_str()` is unique and lowercase. Translation makes one real model
  call through the same single-shot path as the terminal assistant: `Tab` selects
  From / To / Text, typing edits that field, `Enter` asks the configured model
  and prints its reply — or prints the provider's error, drawn as an error.
  Empty text is refused without spending a request. Dead state went with the
  scripted table: `lang_active`, `lang_supported`, `term_asst_active`,
  `term_asst_cursor`, `open_terminal_assistant` and the sender-less
  `[TERM]output:` protocol arm.
- Terminal assistant panel is real (Milestone J, J-03). It used to open with
  five hardcoded "suggested commands" — including `docker system prune -af` and
  `rm -rf node_modules && npm install` — each with a risk badge, and nothing
  behind any of it. Now the panel opens as a question field: `Enter` makes
  **one** call to the configured model, whose prompt carries the workspace path,
  its top-level entries from the context index and the current git branch, and
  asks for a JSON array of `{command, risk, why}` capped at 8. Fences, prose and
  a bare object are all tolerated; a reply with no commands in it is shown as
  the reply rather than converted into invented suggestions. Risk labels only
  escalate — `DESTRUCTIVE_PATTERNS` overrides a command the model called safe,
  and a label the model raised is never lowered. `f` filters by risk, `j`/`k`
  select, `y`/`Enter` runs the selection **through `execute_tool_call_approved`
  with the session's `ApprovalCtx`** — the same policy, modal, hooks and
  checkpoint group as a model-issued `run_command`, with `approval_ctx()` now
  the single place that context is built. Outcomes, denials included, land in
  the panel's history; provider errors are printed, flattened to one line. A
  `Ctrl+T`-style shell path was deliberately not added: the panel has no route
  to a binary that the gate is not on.
- Performance profiler panel is real (Milestone J, J-02): the table used to list
  six invented functions (`process_data`, `query_database`, …) with random CPU,
  memory and latency numbers. `Enter` now measures: process CPU from two
  `/proc/self/stat` reads 250 ms apart, resident memory from `/proc/self/statm`
  against `/proc/meminfo`, and the numbers the session already has — average
  turn latency, llama.cpp tokens/s and token counts, per-provider health
  latency (or the error that stopped it), and the last 6 rows of
  `.xencode/metrics.jsonl` with KV reuse, prompt tokens, tok/s and retrieved
  files. A gauge with no measurement renders `n/a` rather than `0`, and the
  panel says when there has been no turn, no health check or no metrics file.
  `fastrand` left the dependency list with the random numbers. Fixed the bug
  that hid this: the Gauges box laid out 3 of its 5 lines, so the Memory and
  Latency gauges had never been on screen.
- Security auditor panel is real (Milestone J, J-01): `Enter` in the panel used
  to print seven hardcoded findings about a `config.py` that is not in this
  tree. It now walks the workspace with the same file list the context engine
  uses (`scanner::scan_tree`, off the UI thread) and runs
  `VulnerabilityScanner::scan_file` — the scanner behind `xencode analyze` — on
  every readable file, streaming each hit as
  `[SECURITY]finding:SEVERITY|type|file:line|message — recommendation`. Secret
  files the walk flags come in as Medium findings, the progress bar is files
  scanned over files found, and the log lines carry the real totals plus every
  file it could not read. If the walk itself fails the panel says
  `scan failed: <reason>` and goes idle instead of showing an empty scan as a
  clean one. Findings are capped at 200 on screen; the reported counts stay the
  true totals. `CodeAnalyzer` is *not* included — it reports style and
  maintainability issues, not security ones.
- `/spawn` subagent in a git worktree (Milestone I, I3-03): `/spawn
  <task> [#branch]` runs the delegated agent loop (`/bytebot`'s engine)
  inside a fresh sibling worktree next to the project — `<dir>-spawn-<id>`
  on the generated branch `xencode/spawn-<id>`, or `<dir>-spawn-<id>-<branch>`
  when the task ends with a `#branch`. The subagent gets its own tool root
  and a fresh checkpoint store (so `/rewind` in the main chat never touches
  its files) while your own chat keeps working. Live tool calls stream in
  the transcript; on completion the agent's final answer is posted as
  `(spawn #<id> · <task>)` and a `⏺ spawn #<id> done/failed — branch … at …,
  N/M call(s) completed` line names the worktree. `/spawn status` lists every
  registered run with its worktree location
- Agent pre/post tool-use hooks (Milestone I, I3-02): the `agent_hooks` config
  key declares shell commands that run around *approved* agent tool calls — a
  `before` map and an `after` map, each keyed by exact tool name or `*` for
  every tool. Commands run via `sh -c` in the workspace root with the same
  budget and output cap as `run_command` (stderr merged, tail kept). A
  `before` hook that exits non-zero **vetoes the call**: nothing runs, no
  rewind point is taken, and the model sees `error: pre-hook vetoed this
  call:` plus the hook's output. A passing `before` hook has its output
  prepended to the result; the `after` hook always runs and its output is
  appended. Hooks are config-edited in the JSON like `mcp_servers`
- MCP tool servers (Milestone I, I3-01): a new `xencode-mcp-rs` crate speaks
  the Model Context Protocol over stdio. Servers are declared in config under
  `mcp_servers` (`command`, `args`, `env` — credentials never in `args`) and
  are started only when you ask with `/mcp`; a broken or missing server reports
  its own failure in a `[MCP]✗ …` line instead of stalling the TUI. Each live
  tool is offered to the model as `mcp__<server>__<tool>` and routed through
  the same approval gate as every other agent tool (`External` class — always
  `y`/`n`, never waved through by autonomy), and `/mcp stop` withdraws the
  tools of everything it stops. `/mcp status` lists what is running; the
  per-handshake budget is `mcp_timeout` config (default 30 s, `config set
  mcp_timeout`, `1`–`300`)
- ByteBot is the real agent (Milestone I, I2-04): `/bytebot <task>` and the
  panel's Enter used to play a recording — six fixed steps, `sleep`s, and
  invented output like "🧪 Running test suite (142 tests)" and "✅ All tests
  pass". Both now run the *same* tool loop as a chat turn, with the task as its
  only user turn (project guidelines, git state and retrieved context, no chat
  history), so the panel's **Execution Steps** are the calls the model actually
  made: `⏳` while one runs, then `✅ done`, `✗ denied`, `✗ refused` or `❌
  failed`. The progress bar is finished calls over calls made — it can move
  backwards, which is honest. The approval gate is exactly your
  `agent_approval` setting (`ask` prompts every edit and every command), the
  writes are checkpointed so one `/rewind` undoes the whole run, a plan the run
  posts shows in the same strip, and a provider failure is printed as the error
  it was. `is_generating` no longer claims a delegated run, `/rewind` refuses
  while ByteBot is working, and the panel keeps the task visible while it runs
  instead of a stock "Executing…" line
- The agent's plan is now on screen (Milestone I, I2-03): `update_plan(items)`
  lets the model post its todo list — one `{text, status}` object per step, at
  most 12 — and it renders above the chat transcript as a `☰ Plan 2/5` strip
  with `✓` done (struck through), `▶` in progress and `·` pending. The compact
  strip shows the first 6 steps and says how many are hidden; `/plan` pins the
  whole list, `/plan clear` drops it, and neither opens a chat turn. Posting a
  plan is a read-only call, so it never costs an approval even in `ask` mode.
  Parsing is deliberately tolerant of what local models actually write (bare
  strings, `- [x]` markdown, invented keys like `title`/`state`, a JSON string
  instead of an array); a rejected update is an error the model can act on and
  leaves the visible plan untouched. On a pane too short for both the strip and
  six lines of transcript the strip is dropped rather than squeezing the chat
- `run_command` for the agent (Milestone I, I2-02): the missing `t` in
  edit→test→fix. The model runs one foreground `sh -c` line in the workspace
  root and gets back `$ <command>`, `exit <code>`, then the combined
  stdout+stderr — the last 8 KiB of it, because a failing build ends with the
  reason. It is a `shell command` class call, so the approval prompt shows the
  literal command line (there is no diff to show) and `edit-allow` still asks.
  `agent_command_timeout` (default 30 s; new `Command Timeout` Settings row in
  5-second steps, `xencode config set` accepts 1–600) kills anything that
  overruns and tells the model nothing was captured, pointing it at
  `background_start` for slow work
- Agent checkpoints and `/rewind` (Milestone I, I2-01): every write or edit the
  user approves is snapshotted first — the exact prior bytes, or "this file did
  not exist" — grouped by chat turn in memory. `/rewind` undoes the last turn
  that changed files, `/rewind 3` the last three; created files are deleted,
  changed files come back byte-for-byte, turns that wrote nothing are skipped
  rather than counted. Nothing touches git, nothing is persisted, and a file
  larger than 4 MiB says out loud that `/rewind` cannot undo it. It refuses to
  run mid-generation, and if the rewound file is open with unsaved edits the
  editor warns instead of discarding them
- The agent tool loop is now file-capable and gated (Milestone I, I1-04): the
  chat turn offers `file_tools()` next to the background and advice tools,
  and every requested call goes through the permission policy first —
  `Allow` runs it, `Deny` answers `error: refused by the permission policy`
  without ever prompting, and `Ask` raises the approval overlay and waits. A
  yes executes the call, an "always allow" answer also grants that tool class
  for the rest of the session (shared state, so the next message inherits
  it), and a no — or a prompt whose UI disappeared mid-wait — returns
  `DENIED_RESULT`, worded so the model explains instead of retrying the same
  call. The model is taught the vocabulary, the workspace boundary and the
  no-retry rule in a short tools block appended to the system turn. New
  `agent_max_rounds` config key (default 16, clamped 1..=64, settable with
  `xencode config set`) replaces the old hard-coded 8-round valve
- File tools for the agent (Milestone I, I1-02): `read_file` (paged, numbered
  lines), `list_dir`, `search_files` (regex walk that skips `target/`,
  `node_modules/` and dot-dirs, 100-hit cap), `write_file` and `edit_file`
  (exact old→new replace, `all` for multi-match) are defined in
  `xencode-providers-rs::file_tools` and executed in
  `xencode-tui-rs::agent_tools`, each returning a unified diff built with
  `similar`. Every executor refuses paths outside the workspace on its own,
  and `background_start` now resolves a relative `cwd` against the workspace
  root instead of the process directory
- Agent approval prompt (Milestone I, I1-03): when the policy says `Ask`, the
  TUI shows a topmost modal with the call, its class and the exact bytes at
  stake — colored unified diff for writes/edits, the literal command line for
  shell calls. `y` allows, `a` allows and remembers the class for this
  session, `n`/`Esc` deny, `k`/`j` scroll; every other key (quit chords
  included) is swallowed and Enter is deliberately *not* an answer. Stacked
  calls queue FIFO, denials come back to the model as a message telling it
  not to retry unchanged, and each answer is logged in the chat as
  `⚙ write_file src/lib.rs · approved`. The keys also appear in the `?` help
  overlay and the status bar
- Agent permission policy core (Milestone I, I1-01): every tool call is now
  classified against an `agent_approval` mode — `ask` (default),
  `edit-allow`, `all-allow` — via a single `classify` in
  `xencode-tui-rs::agent_tools`; read-only tools always run, file edits and
  shell calls prompt or auto-allow per mode, and paths outside the workspace,
  inside `.git/` or the config dir are hard-denied in every mode. New
  `Agent Approval` Settings row (Cycle) and `xencode config set
  agent_approval`; session "always allow" grants live in `App.agent_grants`
  until quit. Approval prompts and file tools land in I1-02…I1-04
- Selectable TUI body layouts and display preferences (Milestone H): three
  layout presets — `classic` (20/50/30 as before), `chat-first` (explorer
  hidden, editor 25 %, chat 75 %), `zen` (one pane fills the body, following
  focus) — cycled live with `Ctrl+U` or picked on the Settings panel, plus
  `rounded_borders`, `show_scrollbars` (chat & explorer) and
  `show_line_numbers` (editor gutter + current-line highlight). All four
  persist to `~/.xencode/config.json` and are settable via
  `xencode config set`; unknown layout names fall back to classic. The
  config dir honors `XCODE_CONFIG_DIR`
- The Collaboration Hub gains a real client (Milestone G, G3-01):
  `xencode-tui-rs::collab_client` logs in over HTTP, opens the WebSocket,
  authenticates with the first-frame `auth` token (creating a session via
  `POST /sessions/create` when none is chosen), and translates every server
  frame through pure `frame_to_tokens` into the app's `[COLLAB]` grammar —
  a `members:<json>` snapshot replaces the old per-member simulation, errors
  arrive as `error:`. 30 s ping keepalive with a 60 s liveness deadline;
  no auto-reconnect by design. The simulated session (fake alice/bob/carol
  sync pulses) and its dead state fields are deleted; hub keys land with
  G3-02
- Persistent audit trail for collaboration servers (Milestone G, G1-04):
  `xencode-server-rs::audit::AuditSink` mirrors every `WorkspaceManager`
  event — creations, joins, role changes, and denials (including
  authenticated join refusals: unknown session, session full) — as one
  JSONL line each, appending across restarts. A failed open or write warns
  once and disables the sink; the session plane never notices. Defaults to
  disabled so tests touch no disk; the `--audit-path` flag arrives with the
  CLI server work (G2-01)
- `repo_advise` agentic tool (F3-02): the chat model can now call repository
  insights directly — broken imports, cycles, hubs, orphans — with an optional
  path `filter`, a 40-finding cap and `error:` strings instead of panics; both
  surfaces share a new `advise_from_snapshot` loader in `xencode-context-rs`
  that the insights panel and the CLI now read through as well
- `xencode advise [FILTER] [--json] [--limit 40]` — repository insights
  (broken imports, cycles, hubs, orphans) read straight from the `.xencode`
  snapshot in the current directory; positional filter matches finding file
  paths, `--limit 0` shows all, and a missing index exits 1 with a
  "run /init first" error instead of an empty report
- Insights panel (`Ctrl+L` or Feature Navigator #17, F2-01): deterministic refactor findings from the live symbol graph — broken imports, cycles, hubs, orphans — as a color-coded selectable list; Enter shows the full message, `o` opens the advised file in the editor, `r` recomputes; `/advise` now renders through the same snapshot path, so chat and panel always agree
- The TUI file watcher now keeps the symbol/dep snapshot live (F1-02): modified/removed `.rs` files trigger `refresh_rust_file` before dependent-file warnings are computed, so toasts and `/advise` reflect current imports without another `/init`
- Live symbol-graph refresh (`xencode-context-rs::refresh`, F1-01): `refresh_rust_file` re-extracts one already-indexed `.rs` file from disk (or drops it when deleted), rebuilds `deps.json` and keeps `files.json`/`manifest.json` (sizes, loc, mtimes) consistent — a refreshed snapshot makes the next `/init` report fresh; the watcher now also excludes `.xencode` so snapshot rewrites never loop back into themselves
- `xencode worktree list | add <path> [<branch>] | remove <path>` (D3-04) over the D3-01 helpers; omitted branch lets git name the new branch after the directory, dirty removals stay refused by git and the main checkout is never removable
- Background tasks can run in another directory (D3-03): `TaskManager::start_with_cwd`, an optional `cwd` on the `background_start` tool (echoed back as `in <dir>` in the result), and the context bundle's git summary now lists extra git worktrees with branch and dirty markers
- Worktree panel (`Ctrl+O` or Feature Navigator, D3-02): lists git worktrees with branch, short HEAD and dirty marker (★ main / ⚡ dirty / ◯ clean), `a` walks a two-stage add prompt (path → branch), `d` removes via y/N confirm — the main worktree is refused before any git call and dirty worktrees stay protected by git's own guard; `r` refreshes
- Git worktree helpers (`xencode-context-rs::worktree`, D3-01): pure `parse_worktree_list` for `git worktree list --porcelain` (spaces in paths, detached/locked/prunable/bare) plus explicit-arg `worktree_list`/`worktree_add`/`worktree_remove` shells sharing `gitinfo`'s error reporting
- `xencode tasks` CLI (D2-02): file-backed background-task registry (`xencode-core-rs::tasks_file`) shared through `.xencode/tasks/` — `list [--json]`, `start <cmd> [--name]`, `poll <id> [--lines]`, `stop <id>`, `rm <id>`; tasks survive the starting process (EXIT-trap wrapper records exit codes, output captured to `<id>.out`), status is derived on read from exit file + killed flag + `/proc` liveness, and `rm` refuses still-running tasks
- Background Tasks panel (`Ctrl+K` or Feature Navigator, D2-01): live registry view with per-status colored rows (running / exited / killed), Enter → scrollable output detail for the selected task, `x` stop and `d` remove dispatched through the event channel so keys never block, wheel + resize clamping included
- Agentic background-task tool loop in chat (D1-02): the model can call `background_start` / `background_poll` / `background_stop` (schemas in `xencode-providers-rs::background_tools`), executed client-side against the shared task registry with up to 8 tool rounds per turn; each call and result echoes into chat as a `⚙` system line
- Background task registry (`xencode-core-rs::tasks`, D1-01): pure `TaskStore`/`TaskRecord` (status Running/Exited(code)/Killed, 500-line capped output tail) plus a tokio-backed `TaskManager` that spawns commands through `sh -c`, drains stdout+stderr without blocking polls, and refuses to remove or drop a still-running task's child process
- Chat input history & autocomplete: Alt+Up/Down recalls sent prompts (draft stashed and restored, adjacent duplicates skipped), Tab completes slash commands — `/init`, `/ctx`, `/advise`, `/bytebot` — with the command list surfaced in the help overlay and as a toast on a lone `/` (Milestone E5)
- Multiline chat input: the input box is now a real text area — Enter sends, Alt+Enter (or Ctrl+J) inserts a newline, arrows move inside multi-line drafts (Milestone E5)
- `light` TUI theme (8th palette), selectable via Settings ←/→ or `active_theme = "light"` in config; theme cycling now runs through one shared `THEME_NAMES` list instead of three duplicated arrays, and the settings Theme row shows dots for all themes (Milestone E4)

### Changed
- TUI header reads ` ✦ xencode [layout] ⎇branch model` with the focused
  panel name on the right; parts drop out at 72/60/40 columns. Panel
  borders all route through one `panel_block` (rounded-corner aware), body
  pane titles drop their emoji under 30 columns, popups that would collapse
  below 16×6 grow to 80 % of the screen, and toasts no longer overlay the
  chat input on short terminals (Milestone H)
- Manuals tell the truth about team mode (Milestone G, G4-01): the user
  manual gained a Collaboration Hub key table and lost its fabricated
  claims (`0.0.0.0` default, "credential vault… email verification");
  `docs/api_documentation.md`'s Python-era JWT auth matrix was replaced by
  the real route table (public vs bearer vs WS close codes) with the legacy
  module sections marked as such; README's CRDT-sync bullet (the module is
  unwired) now describes token auth, RBAC and the audit trail.
- The Collaboration Hub is real now, not a demo (Milestone G, G3-02): `c`
  creates and connects a session, `j` edits a session id to join, `Enter`
  connects, `r` retries, `Tab` cycles the server/user/session fields, and
  `Esc` stops editing → disconnects (aborting the worker) → closes; Ctrl+W
  hangs up on its way out too. Fabricated telemetry — the timestamp-derived
  port, "Protocol: WebSocket (TLS)" on a plain connection, `<15ms` latency,
  hardcoded alice/bob/carol members — is gone: the panel shows the real
  server URL, an honest `ws (no TLS)`/`wss (TLS)` transport line, the live
  member list with role badges, connected-for time, and any worker error
  verbatim. The help overlay lists exactly the keys the hub handles.
- `xencode server` is local-first by default (Milestone G, G2-01): it binds
  `127.0.0.1` unless told otherwise (`--host` accepts an IP or `localhost` —
  no DNS resolution behind the user's back), serves TLS only when given
  `--cert` + `--key` (rustls; half a pair is an error), and refuses any
  non-loopback plain bind unless `--allow-insecure-public` is set — with a
  clear-text warning even then. The startup banner prints the honest scheme
  (`ws://` never claims `wss://`) and the audit target; `--audit-path` picks
  the JSONL file (default `~/.xencode/audit.jsonl`, `none` disables), so the
  G1-04 sink is now actually wired. The banner's stale `/ws/{session_id}/{username}`
  route text was corrected to the real `/ws/{session_id}`.
- The WebSocket is no longer an identity claim (Milestone G, G1-03): the route
  is `/ws/{session_id}` — the username is gone from the URL — and the first
  frame must be `auth` with a token the server issued (5 s timeout). Joins run
  through `WorkspaceManager::join` (self-join lands as Editor; the session
  creator keeps Admin), `activity` relay requires Editor+ — a Viewer gets an
  `rbac_denied` error frame and a `Denied` audit entry — and the relayed
  `user` is always the server's authenticated identity, never whatever the
  client's JSON claims. `MAX_SESSION_MEMBERS` (10) is enforced with a 4409
  close (existing members may still reconnect); rejections arrive as an error
  frame plus a 44xx close (4401 bad token, 4404 no session). The wire format
  lives once in `xencode-collaboration-rs::wire` (serde `type`-tagged enums
  shared by server and future client), the server's parallel in-memory
  session map is deleted, and presence (who is connected) is now distinct
  from membership (who belongs). Covered by 12 end-to-end handshake tests
  over in-process duplex pipes — no ports.
- Server auth is real (Milestone G, G1-02): `xencode-server-rs` gained a
  `TokenStore` (random `xencode_<uuid v4 hex>` tokens, 24 h TTL, pruned on
  issue and lookup, constant-time compare) and an `Authed` bearer extractor.
  `/auth/login` now rejects the previously-silently-ignored `api_key` with a
  400 instead of minting a token anyway, `/auth/verify` resolves a token it
  actually issued — the old prefix-and-length check that accepted any >20-char
  `xencode_` string is gone. `POST /sessions/create`, `GET /sessions/{id}`
  (which now 404s for unknown ids instead of an empty 200) and the llama.cpp
  load/unload endpoints require a valid token; permissive CORS was deleted;
  `/api/llamacpp/status` and the public `/api/config` no longer leak the
  model path, executable or launch args
- Collaboration RBAC groundwork (Milestone G, G1-01): the sole admin can no longer
  be *demoted* by an admin role-change (only removal was guarded), both guards now
  write `Denied` audit events, and `WorkspaceManager` gained `join` (idempotent
  self-join as Editor), `create_workspace_with_id` and `log_denied` — the API the
  hardened server will consume
- Docs close-out (Milestone F4): NEXT_PLAN.md marks Milestone F complete and its
  backlog now carries only team-mode hardening; the TUI panel count (24),
  Feature Navigator entries (17), key tables, `xencode advise` docs and the
  CLI subcommand list were re-verified against the tree — 13 crates, 508 tests,
  zero warnings at this commit
- Docs close-out (Milestone D4): NEXT_PLAN.md marks Milestone D complete and drops stale backlog claims (the real-time watcher, multimodal inputs and PR review have all shipped); NEXT_PLAN_TASKS.md subcommand checklist refreshed from `xencode --help`; counts re-verified against a full run — 13 crates, 494 tests, zero warnings
- Docs close-out (Milestone E7): QUICK_START/USER_MANUAL TUI sections rewritten against the real keymap — removed the never-existing `/help` `/models` `/model` `/clear` `/exit` slash commands, documented the help overlay, multiline input, history recall and Tab completion; README panel count corrected to 21; the unbound `h:refresh` Provider Health status hint now shows the working `Ctrl+H`
- TUI key handling restructured into a new `keymap.rs`: global Ctrl chord table + per-focus handlers replace the ~900-line `match` in the event loop. Panels now own their letter keys — SecurityAuditor `s` (sort toggle) and CustomModels `s` (save profile) work again instead of opening Settings — and `j`/`k` are typable in every text field (commit message, ByteBot command, settings URL edit); ByteBot history recall is ↑ only (E6-01)
- Every panel color renders through new semantic `ThemeColors` slots (`success`/`warning`/`danger`/`info`/`accent_secondary`): 63 hardcoded ANSI constants in the renderer are gone, so themes can actually restyle severities, diffs and status badges (Milestone E4)
- Body panel layout has one source of truth: `ui::body_chunks` feeds both the renderer and mouse focus hit-testing (`ui::body_hit_test`), replacing the duplicated 20%/70% magic numbers in the event loop (Milestone E4)
- Settings panel navigation is bound to a real row list: new `focus::SETTINGS_ROWS` feeds the panel labels, section ranges, and the ↑↓/wheel bounds, replacing the hardcoded `14` (Milestone E4)
- Removed legacy Node wrapper (`package.json`, `package-lock.json`, `bin/xencode.js`) and completed-migration docs (`docs/RUST_MIGRATION_PLAN.md`, `docs/RUST_MIGRATION_STATUS.md`); the product is Rust-only under `rust/`.

### Added
- TUI toast notification layer: file-watch warnings now surface as transient top-right overlays (6s TTL, deduped) instead of polluting chat history as fake system lines (Milestone E3)
- TUI keybinding help overlay (`?`/`F1`): modal panel-aware keymap generated from the real key handler, with scroll and status-bar discoverability (Milestone E3)
- TUI markdown rendering for assistant chat: fenced code blocks with language labels (correct even mid-stream), headings, bullets/ordered lists, block quotes, rules, inline `code`/**bold**/*italic* (Milestone E3)
- Image input pipeline: magic-byte format detect, header dimensions, data URLs (`analysis-rs` images module), `analyze` CLI inventory, `MessageContent` parts with per-backend rendering (Ollama/Anthropic/Gemini/OpenAI-compatible), TUI attach sends images as message parts
- Web extraction for research: timeout/capped fetch with content-type gate, pure HTML→text (scripts/styles stripped, entities decoded), `FetchedPage` record, `fetch` CLI subcommand
- PR-level diff triage: rename-aware numstat parsing and capped per-file diffs (`gitinfo`), `review` CLI command with working-tree analysis and documented JSON
- Document parsing into context: PDF (pdf-extract) and DOCX (zip + w:t runs) text extraction with size/char caps, TUI attach inlines extracted text with explicit skip notes
- Workspace RBAC + audit log: Admin-gated membership, self-leave, last-admin guard, sequenced audit events including denials
- TUI PR review dashboard: per-file diff browsing (`Ctrl+Y`, Feature Navigator entry) over rename-aware numstat + capped diffs, base toggle HEAD<->main, scrollable diff pane

### Fixed
- TUI: every scroll offset (chat, code review, provider health, security findings, help overlay) is re-clamped when the terminal resizes, so shrinking then growing back no longer re-exposes stale scroll-past-the-end blank space; render and clamp now share one set of line builders (E6-02)
- TUI: text fields (git commit message, ByteBot command, settings URL edit) are now fully typable — space and the letters `i / m s q ?` no longer trigger shortcuts while typing; Left arrow moves the cursor instead of deleting a character; VoiceInterface `m` mute reachable again (E2-06)
- TUI (Milestone E2): ByteBot Enter executes the typed command instead of clobbering it with history; `git commit` runs off the UI thread with the result reported as a chat line; dead `FocusArea::Terminal` removed; unused `file_scroll_offset` deleted; provider-health/security/code-review scroll clamped (no more blank render past the end) and CodeReview output made scrollable; mouse wheel wired for Settings/ModelSelector/CustomModels/CodeReview
- Docs: corrected stale crate/test counts (`NEXT_PLAN.md`, `docs/ROADMAP.md` said 331 tests; real count verified at 421/13 crates via `cargo test --workspace`), removed Python-era fiction from `docs/USER_MANUAL.md` and `docs/INSTALL_MANUAL.md` (dead `xencode.sh`/pip instructions, missing HTML reference)
- Context budget: state/git tiers emitted only when margin-admitted; tail truncation hard-cuts single-line overflow; preview compaction flag only when content dropped
- Symbol graph: brace-group import expansion and crate-root fallback for 2018 paths; broken-import checks per anchored pair
- TUI: no silent attachment drops, exact watch-dedup, kind|path watch tokens, honest /advise counts, visible save errors
- CLI: full-tree analyze with skip counts and documented dir JSON schema, scan --format json, fetch text cap, strict --json-schema
- Providers: request timeouts (total for single-shot, time-to-first-token for streams), Gemini key via header, NXDOMAIN-only DNS retry direction, tool-call demux for index-less fragments
- Cache: deterministic LRU sequence counter (fixes intermittent eviction flakes), float-tolerant legacy seeding test
- Real-time workspace watcher with debounced proactive warnings for tracked/attached/open files
- Repository insights: dependency cycles, hub files, orphans, broken imports, affected dependents (`advise` module + `/advise` TUI command with dependents in watch warnings)
- Rust CI gates (fmt, clippy `-D warnings`, full workspace suite), Rust binary GitHub Releases, Rust Docker/runtime and installers
- Enhanced Security Scanning with Bandit Integration
- Comprehensive vulnerability database with CVE patterns
- OWASP Top 10 vulnerability detection
- Multi-language security analysis (Python, JavaScript, Java)
- Security report generation (summary, detailed, executive)
- Automated risk assessment and compliance scoring
- 50+ Bandit security rules with CWE mappings
- Async security scanning for performance
- Demo application for security scanning capabilities
- Comprehensive test suite (21 tests) for security features

### Removed
- Legacy Python stack: `xencode/` package, pytest suite, examples, benchmarks, deployment helpers, packaging configs, and the local venv (Rust workspace under `rust/` is the product)
- Python install paths (pip/venv/PyInstaller) from `install.sh`/`install.ps1`, the Python Dockerfile stages, and pytest/ruff/bandit CI jobs

### Enhanced
- Security analyzer with multiple scanning methods
- Pattern-based vulnerability detection
- Dependency vulnerability checking
- Security metrics calculation and reporting
- Integration with existing code analysis workflow

### Security
- Code injection detection (eval, exec, dynamic execution)
- SQL injection pattern matching
- Cross-site scripting (XSS) vulnerability detection
- Path traversal security checks
- Weak cryptography identification
- Hardcoded secrets detection
- Unsafe deserialization checks
- Command injection vulnerability scanning

## [2.1.0] - 2026-03-30

### Added
- Dynamic feature API route mounting with auth protection for runtime feature endpoints
- Missing feature hook implementations for CLI, TUI, and API integration across core feature modules
- API auth regression tests for protected routes and dynamically mounted feature routes
- Integration smoke tests for health and info endpoints
- Release and deployment validation scripts for version consistency and deployment secrets

### Changed
- Hardened collaborative CLI async execution and TUI panel update safety in unmounted contexts
- Enforced JWT auth on code analysis and document routers
- Improved CI/CD workflows with marker-matrix test jobs, collect-only precheck, and deploy secret preflight
- Updated Kubernetes secret handling to use templated secrets manifest

## [3.0.0] - 2024-10-15

### Added
- Phase 2 Core Infrastructure
- Intelligent Model Selection Engine
- Advanced Caching System with hybrid storage
- Smart Configuration Management
- Advanced Error Handling Framework
- System Health Monitoring and Coordination
- Comprehensive test suite (24 tests)
- Interactive demo system
- Production-ready deployment scripts

### Performance
- 99.9% performance improvement with advanced caching
- Sub-millisecond response times for cached operations
- Automated hardware optimization
- Real-time system health monitoring

### Reliability
- 95%+ automatic error recovery rate
- Enterprise-grade resilience framework
- Intelligent fallback strategies
- Context-aware diagnostic messaging

### Developer Experience
- Zero-configuration setup
- Interactive deployment wizard
- Hot-reload configuration updates
- Comprehensive documentation

## [2.0.0] - 2024-09-15

### Added
- Core AI assistant functionality
- Basic model management
- Simple caching system
- Configuration management
- CLI interface

### Changed
- Improved response handling
- Enhanced model selection
- Better error messages

## [1.0.0] - 2024-08-15

### Added
- Initial release
- Basic AI chat functionality
- Ollama integration
- Simple CLI interface