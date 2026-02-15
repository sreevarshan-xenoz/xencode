#!/usr/bin/env node

const { spawnSync } = require("node:child_process");
const path = require("node:path");

const packageRoot = path.resolve(__dirname, "..");

function pythonCandidates() {
  if (process.env.PYTHON && process.env.PYTHON.trim()) {
    return [{ cmd: process.env.PYTHON.trim(), args: [] }];
  }

  if (process.platform === "win32") {
    return [
      { cmd: "py", args: ["-3"] },
      { cmd: "python", args: [] },
      { cmd: "python3", args: [] }
    ];
  }

  return [
    { cmd: "python3", args: [] },
    { cmd: "python", args: [] }
  ];
}

function run(candidate, extraArgs) {
  return spawnSync(candidate.cmd, [...candidate.args, ...extraArgs], {
    stdio: "inherit",
    env: {
      ...process.env,
      PYTHONPATH: process.env.PYTHONPATH
        ? `${packageRoot}${path.delimiter}${process.env.PYTHONPATH}`
        : packageRoot
    }
  });
}

function hasPython(candidate) {
  const check = spawnSync(candidate.cmd, [...candidate.args, "--version"], {
    stdio: "ignore"
  });
  return check.status === 0;
}

const candidates = pythonCandidates();
const candidate = candidates.find(hasPython);

if (!candidate) {
  console.error("Xencode wrapper could not find Python 3.");
  console.error("Install Python 3.8+ and try again.");
  process.exit(1);
}

const args = process.argv.slice(2);
const result = run(candidate, ["-m", "xencode.cli", ...args]);

if (result.error) {
  console.error(`Failed to start Xencode: ${result.error.message}`);
  process.exit(1);
}

process.exit(result.status ?? 1);
