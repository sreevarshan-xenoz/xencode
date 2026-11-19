# Small-model autonomous editing: measured results (2026-11-19)

This file records real runs. It is not a claim that small models can edit code.
The 0/9 and 0/8 results are legitimate research outcomes and are kept verbatim;
they are not hidden behind an easier benchmark.

## Machine and pinned settings

- GPU: NVIDIA GeForce MX250, 2048 MiB (driver 580, CUDA 13.0). llama.cpp CUDA
  12.8 build reports `CUDA0` with 1958 MiB free.
- Model: Qwen3-1.7B Q4_K_M GGUF, SHA-256 verified against the pinned advice
  table (`unsloth/Qwen3-4B-GGUF` is a different model — see the 4B limit below).
- Server flags from `hw probe`: 7168-token window after the preflight shortened
  8192, flash attention on, q8_0 keys, q8_0 values, stable alias.
- Eval: `xencode eval run --model llamacpp:Qwen3-1.7B-Q4_K_M`, all ten seeded
  shapes, `--max-rounds 8 --max-tokens 1024 --timeout 180`, temperature 0,
  seed 42, prompts `449d6f4fd09a`, permissions `edit-allow`. The same settings
  are used in every run below.

## Baseline (before any change)

Command output: **pass rate 0/9 (0%)**, one case never graded.

| Shape | Rounds | Tool calls | Changed | Notable |
|---|---:|---:|---|---|
| off-by-one | 7 | 5 | none | `task_no_edit` in round 0, no edit followed; escalation (3 failed edits) |
| missing-value | 4 | 8 | `config.toml`, `src/config.toml` | wrong files |
| early-return | 9 | 7 | none | `task_no_edit` in round 6, edit followed |
| swallowed-error | 2 | 0 | none | prose only; `task_no_edit` in round 0 |
| inverted-condition | 5 | 5 | none | never graded: llama.cpp 500, failed to parse tool-call JSON |
| ignored-result | 5 | 3 | none | `task_no_edit` in round 2; escalation (2 failed edits) |
| lost-update | 2 | 0 | none | prose only; `task_no_edit` in round 0 |
| stale-cache | 9 | 14 | `src/lib.rs` | changed the file, grader still failed |
| negative-floor | 3 | 1 | none | `task_no_edit` in round 1 |
| utf8-boundary | 2 | 0 | none | prose only; `task_no_edit` in round 0 |

Failure clusters from the recorded traces:

- **No tool call at all** (3): swallowed-error, lost-update, utf8-boundary.
- **Tool calls but no correct edit** (6): off-by-one, missing-value (wrong
  files), early-return, ignored-result, stale-cache (wrong edit), negative-floor.
- **Tool-call JSON rejected by the server** (1): inverted-condition.

The failed `edit_file` calls were inspected at the argument level. Two
representative errors:

- `edit_file src/main.rs error: cannot read src/main.rs: No such file or
  directory` — the model named a file that does not exist in the seeded repo.
- `edit_file src/lib.rs error: "old" not found in src/lib.rs (0 matches, even
  ignoring each line's surrounding whitespace) — nothing in the file
  resembles it` — the model invented the text it asked to replace.
  (`src/lib.rs` is the only source file in these fixtures.)

## A/B: failed-edit error shows a real file excerpt

Change under test: when `old` matches nothing, the `edit_file` error now prints
a bounded, numbered excerpt of the real file, centred on the lines sharing the
most words with the requested `old`, so a small model has literal text to copy.
177 agent-tool tests passed. No other harness setting changed.

Result: **pass rate 0/8 (0%)**, two cases never graded. This is **not an
improvement** over the 0/9 baseline and is not claimed as one.

| Shape | Rounds | Tool calls | Notable |
|---|---:|---:|---|
| off-by-one | 6 | 6 | `task_no_edit` in round 4 |
| missing-value | 5 | 3 | `task_no_edit` in round 0, edit followed |
| early-return | 8 | 7 | no correct edit |
| swallowed-error | 2 | 0 | prose only |
| inverted-condition | 9 | 7 | `task_no_edit` in round 5, edit followed |
| ignored-result | — | — | never graded: tool-call JSON parse error |
| lost-update | — | — | never graded: tool-call JSON parse error |
| stale-cache | 3 | 3 | changed its own test, not `src/lib.rs` |
| negative-floor | 10 | 9 | no correct edit |
| utf8-boundary | 2 | 0 | prose only |

One A/B trace showed a third failure shape: `edit_file was not carried out: the
arguments were not readable as text: EOF while parsing a string at line 1
column 31 — it began '{"path": "src/lib.rs", "old": "'`. The tool call was cut
off mid-argument. The two never-graded cases also failed on tool-call JSON
parse errors. These are **tool-call reliability** failures, not evidence about
whether the model can reason about the fix.

## Important limitations of these comparisons

- llama.cpp on this CUDA stack is **not deterministic** at temperature 0 and
  seed 42: the malformed-JSON failures landed on different shapes in the two
  runs. A single run cannot support a claim about a small delta.
- The 4096-token experiment tests only whether the model can **finish emitting
  an edit operation** without truncation. It does not establish that the edit
  is correct. Grader results decide that, not completion of the call.
- The benchmark is preserved: ten seeded shapes, pinned prompts, seed, rounds,
  graders and per-case traces. No case was replaced with an easier one.

## 4096-token experiment (bounded, three affected cases)

The three shapes whose traces showed truncated or rejected tool calls were
re-run with `--max-tokens 4096 --timeout 240`. Result: **pass rate 0/1 (0%)**,
two cases never graded.

| Shape | Outcome |
|---|---|
| off-by-one | never graded: no response within 240 s |
| ignored-result | never graded: no response within 240 s |
| inverted-condition | fail: 9 rounds, 8 tool calls, `src/lib.rs` untouched, also changed `src/LICENSE` |

Raising the cap did not recover a single case; two of three timed out instead.
This is evidence against "the answer cap is the binding limit". The failed-edit
excerpt change from the A/B was reverted — it was an inconclusive tweak and is
not kept as a fix. The next investigation is the edit protocol itself.

## Proposed direction (design only, not implemented)

Make small-model editing a structured operation instead of asking a small
model to reproduce a whole-file `old`/`new` payload:

```json
{
  "operation": "replace",
  "file": "src/example.rs",
  "target": { "start_line": 42, "end_line": 45 },
  "expected_old": "short expression",
  "replacement": "new expression"
}
```

The harness would validate the target and expected text against the real file,
apply the edit only on a unique match, then run the case's own grader and
independently verify the changed file. This is an architectural hypothesis, not
a proven solution; it needs its own benchmark result before any claim.

### Implemented as an opt-in experiment (not yet a result)

A `patch_lines` tool was added to the real tool menu. It takes a 1-based line
range, a short `expected_old` and a `replacement`; a single planner validates
the range and checks that `expected_old` matches exactly one line in it before
writing. Refusals print the real numbered lines of the range. The same planner
produces the approval preview, so the diff a person approves is the bytes the
executor writes. Four tests cover a valid edit, a wrong fragment, an ambiguous
or out-of-range target, and preview/execution equality.

A probe with the tool merely *offered* showed the 1.7B model still chose
`edit_file`/`write_file` — offering a new tool does not change behaviour. So an
opt-in `--structured-edits` eval flag was added: when set, the whole-block edit
tools are removed and `patch_lines` is the only edit path offered. It is off by
default; every existing eval and install keeps its current menu.

## Structured-edits experiment — result

With `--structured-edits` and the original pinned settings: **pass rate 0/10
(0%)**, all ten cases graded. The protocol did not improve task completion. It
did change the failure profile in a way that matters, and the two must not be
confused:

- **Tool-call reliability improved.** There were no fabricated `old` strings,
  no truncated tool arguments and no tool-call JSON parse errors — all ten
  cases reached a verdict (the earlier runs left one or two ungraded).
- **Reasoning capability did not change.** `patch_lines` was used in only two
  cases. Both calls were mechanically successful (`outcome: done`, the target
  file changed, `failed_edit_count: 0`), but both patches were semantically
  wrong: in `off-by-one` the model added an empty-case guard while leaving the
  buggy `0..len()-1` loop in place, and in `missing-value` it replaced the
  function's signature line while leaving the old body. Four cases called no
  tool at all. So a correctly applied patch is still not a correct fix.

This is the key separation: the structured protocol fixed *how the model emits
an edit*, not *what the model decides to edit*. Autonomous small-model bug
fixing remains 0/10, and that number is kept as-is. No token budget was raised
again and no prompt was stacked on top of this result.
