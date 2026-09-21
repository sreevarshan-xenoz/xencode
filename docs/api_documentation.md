# Xencode API Documentation

> Scope: the HTTP/WebSocket surface of the Rust collaboration server. The
> Python module reference that used to fill the second half of this file
> (`xencode.core.files`, `xencode.core.models`, `xencode.security.validation`,
> a benchmarking suite, Python usage examples) is gone — no code in this tree
> implements those names. For internals, read the crate: `rust/crates/*` is the
> only codebase, and its tests describe real behavior.

## Overview

Xencode is an AI-powered development assistant (Rust CLI/TUI with local
language models). This document describes its hosted HTTP/WebSocket API.

## Auth Matrix

The Rust collaboration server (`xencode server`, axum) issues real bearer
tokens (`xencode_<uuid>`, 24 h TTL). `POST /auth/login` takes `{"username"}`
and returns a token; WebSocket identity is the first `auth` frame, never
the URL.

- Public (no auth required):
  - `GET /` — health check
  - `GET /api/config` · `GET /api/models` · `GET /api/status` ·
    `GET /api/llamacpp/status` (never leak host paths)
  - `POST /auth/login` — identity claim; the bind surface is the perimeter
- Bearer token required (`Authorization: Bearer xencode_...`):
  - `POST /sessions/create`
  - `GET /sessions/{id}` — 200 for known, 404 for unknown
  - `POST /auth/verify` — returns `{username, expires_at}` of the presented token
  - `POST /api/llamacpp/load` · `POST /api/llamacpp/unload`
- WebSocket, first-frame auth:
  - `GET /ws/{session_id}` — frame #1 must be `{"type":"auth","token":…}`;
    close `4401` bad token, `4403` RBAC denied, `4404` unknown session,
    `4409` session full (10 members max)
- No CORS layer: the only clients are the TUI (reqwest — unaffected by
  CORS) and curl-style tooling; permissive CORS would only widen the
  browser attack surface for no consumer.

Session lifecycle is in-memory only (a restart drops peers); every create,
join, relay, denial and removal is appended as one JSONL line to
`~/.xencode/audit.jsonl` by default.

## Elsewhere

- CLI verbs, flags and every `config.json` key (`model_profiles`,
  `agent_hooks`, `mcp_servers`, …): [`../CLI_GUIDE.md`](../CLI_GUIDE.md)
- TUI keys, panels and slash commands: [`USER_MANUAL.md`](USER_MANUAL.md)
- Crate layout and build/test commands: [`../README.md`](../README.md)
