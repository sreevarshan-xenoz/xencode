---
name: commit-message
description: Suggest a commit message from the real staged diff. Use when changes are staged and you need a message that follows this repository's plain-English commit rules.
allowed-tools:
  - read_file
  - run_command
  - ask_user
---
# Suggest a commit message from the staged diff

The message must describe real changes in words a developer who has never seen
this session can understand. Do not guess what was staged and do not invent a
summary: read the diff.

## Steps

1. Check what is staged with `git status --short` and `git diff --cached`. If
   nothing is staged, say so and stop — there is no message to write.
2. Use the repository's own command to get a suggestion grounded in the diff:
   `xencode git commit-message`. It reads the real `git diff --cached` and asks
   the configured local model through the existing provider stack. It only
   prints a suggestion; it never commits unless `--commit` is passed. Pass
   `--model <name>` only when the user asks for a different model.
3. If the model cannot be reached, say that plainly and write the message from
   the staged diff yourself. Never present a made-up message as the model's
   answer.
4. Check the result against `AGENTS.md`. The subject must name the plan ID (for
   example `SK-1`) and be fully plain English. Expand any shorthand the reader
   could not know, state what was wrong and what now happens, and keep the body
   in the same register. A correctly quoted code symbol or file name is fine.
5. Confirm with the user before committing. Run `git commit -m "<message>"` only
   when they asked for a commit; otherwise print the suggested message and stop.
   Never push unless the user explicitly asks in the current session.
