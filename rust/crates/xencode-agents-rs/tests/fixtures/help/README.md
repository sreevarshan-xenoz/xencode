# Recorded agent help screens

These files are the real `--help` output of the agents listed below, captured
on 2026-11-17 on the developer machine. They exist so the contract test can
actually check the roster on a CI runner where no coding agent is installed.
Live help always takes precedence when an agent is present.

| File prefix | CLI | Version when captured |
|---|---|---|
| `opencode` | opencode | 2.3.0 |
| `cline` | cline | 3.0.74 |
| `claude` | claude (Claude Code) | 2.1.311 |
| `agy` | agy | 1.5.10 |
| `kilo` | kilo | 7.8.27 |
| `kiro-cli` | kiro-cli | 3.2.1 |

A `<name>--<subcommand>.txt` file, when present, is that subcommand's help.
Agents with no recorded screen are simply not tested on a machine where they
are absent — no output is invented for them.

Refresh a screen with `<cli> --help > <prefix>.txt 2>&1` after reading the new
version's help, and update the version in this table in the same change.
