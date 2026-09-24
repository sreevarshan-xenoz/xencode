# Recorded provider traffic

These four files are recordings of traffic that actually happened on a
developer machine, taken from a `llama-server` process answering a local GGUF
model over a loopback port. No provider account, no network, no hand-written
expectations. `src/playback.rs` serves them again on a real socket so the
streaming readers can be tested below the HTTP boundary — the place where an
in-process fake cannot catch a stream arriving in pieces.

Each file carries a `captured` block naming the server build, the model file,
how the bytes were taken and when, and a note saying what the recording shows.
Treat that block as part of the test: a fixture whose provenance is missing is
worth nothing, because the next reader cannot tell a capture from a guess.

| File | What it holds |
| --- | --- |
| `answer-one-eighty.json` | A short factual answer arriving as eleven content deltas. |
| `answer-japanese.json` | An answer in Japanese, so the frames split across three-byte characters. |
| `calculator-two-turns.json` | One agent exchange: a tool call whose arguments arrive in twelve fragments, then the answer that follows the tool result. |
| `reasoning-only.json` | A thinking model whose every delta is `reasoning_content` and none is `content`. Nothing in the product reads that field, so this recording replays as an empty answer; see `tests/cassette_replay.rs`. |

To add a recording: run a real server, capture both the request body and the
response stream to files, then write the cassette with the request under
`interactions[].request.body` and the stream unmodified under
`interactions[].response.body` — including the `data:` lines and the blank
lines between them. Put something distinctive from the question in
`body_contains`, so the cassette cannot answer a question it was not asked;
`Cassette::validate` refuses a fixture whose own recorded request would not
satisfy its matcher.
