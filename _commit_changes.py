#!/usr/bin/env python3
import subprocess
import os

os.chdir("E:/xencode")

# Stage all changes
subprocess.run(["git", "add", "-A"], check=True)
print("Staged all files.")

# Commit
msg = (
    "Add JWT auth enforcement to vault API + expand test coverage\n"
    "\n"
    "- xencode/api/auth.py: Fix verify_jwt_token to gracefully handle\n"
    "  missing xencode.auth.vault module (vault -> env var -> dev default)\n"
    "- xencode/api/routers/vault.py: Add JWT auth (Depends verify_jwt_token)\n"
    "  to health/init/migrate endpoints + token-based WebSocket auth\n"
    "- tests/api/test_vault_api.py: 34 tests with auth token fixtures +\n"
    "  auth-failure coverage (missing/invalid/expired tokens)\n"
    "- tests/integration/test_vault_integration.py: 24 live-server tests\n"
    "  with auth headers + 4 auth-failure scenarios\n"
    "- DOCUMENTATION.md: Add vault migration tutorial (6 steps)\n"
    "- xencode-codebase-reference.html, xencode-codebase-agent-knowledge.json:\n"
    "  Document vault API router and WebSocket endpoints"
)
subprocess.run(["git", "commit", "-m", msg], check=True)
print("Committed.")

# Push
subprocess.run(["git", "push", "origin", "main"], check=True)
print("Pushed to origin/main.")
