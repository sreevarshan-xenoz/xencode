#!/usr/bin/env bash
# Offline test for release asset and URL derivation. No network is used: the
# functions under test are pure, so fake OS/architecture pairs are enough.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$ROOT/scripts/release-assets.sh"
FIXTURE="$ROOT/rust/crates/xencode-cli/tests/fixtures/dist-plan.json"

# shellcheck source=scripts/release-assets.sh
source "$LIB"

BASE='https://github.com/sreevarshan-xenoz/xencode/releases/download/v0.1.0'
FAILED=0

pass() { printf '  \u2705 %s\n' "$1"; }
fail() {
    printf '  \u274c %s\n' "$1"
    FAILED=$((FAILED + 1))
}

assert_eq() {
    local label="$1" expected="$2" actual="$3"
    if [ "$actual" = "$expected" ]; then
        pass "$label"
    else
        fail "$label"
        printf '       expected: %s\n       actual:   %s\n' "$expected" "$actual"
    fi
}

assert_succeeds() {
    local label="$1" expected="$2" actual
    if actual="$("${@:3}" 2>/dev/null)"; then
        assert_eq "$label" "$expected" "$actual"
    else
        fail "$label (derivation returned failure)"
    fi
}

assert_refused() {
    local label="$1"
    if xencode_release_asset "$2" "$3" >/dev/null 2>&1; then
        fail "$label (unsupported platform was accepted)"
    else
        pass "$label"
    fi
}

printf 'Release asset derivation test\n'
printf 'Library: %s\n' "$LIB"
printf '\n'

printf 'Asset names from uname-style pairs:\n'
assert_succeeds 'Linux x86_64' \
    'xencode-cli-x86_64-unknown-linux-gnu.tar.xz' \
    xencode_release_asset Linux x86_64
assert_succeeds 'Linux aarch64' \
    'xencode-cli-aarch64-unknown-linux-gnu.tar.xz' \
    xencode_release_asset Linux aarch64
assert_succeeds 'Darwin x86_64' \
    'xencode-cli-x86_64-apple-darwin.tar.xz' \
    xencode_release_asset Darwin x86_64
assert_succeeds 'Darwin arm64' \
    'xencode-cli-aarch64-apple-darwin.tar.xz' \
    xencode_release_asset Darwin arm64
assert_succeeds 'Windows x86_64' \
    'xencode-cli-x86_64-pc-windows-msvc.zip' \
    xencode_release_asset MINGW64_NT-10.0 x86_64
assert_refused 'unsupported Linux/riscv64' Linux riscv64
assert_refused 'unsupported Windows/ARM64' MINGW64_NT-10.0 ARM64

printf '\nDownload URLs:\n'
assert_succeeds 'Linux x86_64 asset URL' \
    "$BASE/xencode-cli-x86_64-unknown-linux-gnu.tar.xz" \
    xencode_release_asset_url Linux x86_64
assert_succeeds 'Linux x86_64 checksum URL' \
    "$BASE/xencode-cli-x86_64-unknown-linux-gnu.tar.xz.sha256" \
    xencode_release_checksum_url Linux x86_64
assert_succeeds 'Windows x86_64 asset URL' \
    "$BASE/xencode-cli-x86_64-pc-windows-msvc.zip" \
    xencode_release_asset_url MINGW64_NT-10.0 x86_64
assert_succeeds 'Windows x86_64 checksum URL' \
    "$BASE/xencode-cli-x86_64-pc-windows-msvc.zip.sha256" \
    xencode_release_checksum_url MINGW64_NT-10.0 x86_64

override="$(XENCODE_RELEASE_VERSION=2.3.4 bash -c \
    'source "$1"; xencode_release_asset_url Linux x86_64' _ "$LIB")"
assert_eq 'version override via XENCODE_RELEASE_VERSION' \
    'https://github.com/sreevarshan-xenoz/xencode/releases/download/v2.3.4/xencode-cli-x86_64-unknown-linux-gnu.tar.xz' \
    "$override"

printf '\nChecksum verification (real files, no network):\n'
CHECKSUM_DIR="$(mktemp -d "${TMPDIR:-/tmp}/xencode-checksum-test.XXXXXX")"
trap 'rm -rf "$CHECKSUM_DIR"' EXIT
printf 'a real archive payload\n' > "$CHECKSUM_DIR/archive.tar.xz"
if command -v sha256sum >/dev/null 2>&1; then
    GOOD_HASH="$(sha256sum "$CHECKSUM_DIR/archive.tar.xz" | awk '{print $1}')"
else
    GOOD_HASH="$(shasum -a 256 "$CHECKSUM_DIR/archive.tar.xz" | awk '{print $1}')"
fi
printf '%s  %s\n' "$GOOD_HASH" 'archive.tar.xz' > "$CHECKSUM_DIR/archive.tar.xz.sha256sum"
if xencode_verify_checksum "$CHECKSUM_DIR" 'archive.tar.xz' >/dev/null 2>&1; then
    pass 'matching checksum is accepted'
else
    fail 'matching checksum was rejected'
fi
printf '%064d  %s\n' 0 'archive.tar.xz' > "$CHECKSUM_DIR/archive.tar.xz.sha256sum"
if xencode_verify_checksum "$CHECKSUM_DIR" 'archive.tar.xz' >/dev/null 2>&1; then
    fail 'wrong checksum was accepted'
else
    pass 'wrong checksum is refused'
fi

printf '\nComparison with the captured dist plan:\n'
if [ -f "$FIXTURE" ]; then
    printf '  Fixture: %s\n' "$FIXTURE"
    for url in \
        "$BASE/xencode-cli-x86_64-unknown-linux-gnu.tar.xz" \
        "$BASE/xencode-cli-x86_64-unknown-linux-gnu.tar.xz.sha256" \
        "$BASE/xencode-cli-x86_64-pc-windows-msvc.zip" \
        "$BASE/xencode-cli-x86_64-pc-windows-msvc.zip.sha256"; do
        if grep -Fq "$url" "$FIXTURE"; then
            pass "fixture contains $url"
        else
            fail "fixture is missing $url"
        fi
    done
else
    printf '  \u2139  Fixture not present on this branch; the expected URLs above are\n'
    printf '     the exact strings captured from the dist plan by DI-2.\n'
fi

printf '\n'
if [ "$FAILED" -eq 0 ]; then
    printf 'All release asset derivation checks passed.\n'
    exit 0
fi
printf '%s release asset derivation check(s) failed.\n' "$FAILED"
exit 1
