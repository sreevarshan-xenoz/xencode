#!/usr/bin/env bash
# Derive cargo-dist release asset names and download URLs from a host OS/arch,
# and verify a downloaded archive against its checksum. No network calls are
# made here. install.sh sources this file, and
# scripts/tests/test-release-assets.sh sources it to check the derivation and
# the checksum verification without downloading a release.

# The owner, repository and version below match the release shape captured by
# cargo-dist in rust/crates/xencode-cli/tests/fixtures/dist-plan.json.
XENCODE_RELEASE_OWNER="${XENCODE_RELEASE_OWNER:-sreevarshan-xenoz}"
XENCODE_RELEASE_REPO="${XENCODE_RELEASE_REPO:-xencode}"
XENCODE_RELEASE_VERSION="${XENCODE_RELEASE_VERSION:-0.1.0}"

# xencode_release_target OS ARCH
# Prints one of the five target triples from the dist plan, or fails for an
# unsupported pair. OS and ARCH are uname-style values (e.g. "Linux x86_64",
# "Darwin arm64", "MINGW64_NT-10.0 x86_64"); they may also be plain lowercase
# names, which is how the derivation test calls this function.
xencode_release_target() {
    local os arch
    os="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
    arch="$(printf '%s' "${2:-}" | tr '[:upper:]' '[:lower:]')"

    case "$os" in
        linux)
            case "$arch" in
                x86_64|amd64)
                    printf '%s' 'x86_64-unknown-linux-gnu'
                    return 0
                    ;;
                aarch64|arm64)
                    printf '%s' 'aarch64-unknown-linux-gnu'
                    return 0
                    ;;
            esac
            ;;
        darwin|macos|osx)
            case "$arch" in
                x86_64|amd64)
                    printf '%s' 'x86_64-apple-darwin'
                    return 0
                    ;;
                aarch64|arm64)
                    printf '%s' 'aarch64-apple-darwin'
                    return 0
                    ;;
            esac
            ;;
        mingw*|msys*|cygwin*|windows_nt)
            case "$arch" in
                x86_64|amd64)
                    printf '%s' 'x86_64-pc-windows-msvc'
                    return 0
                    ;;
            esac
            ;;
    esac

    printf 'unsupported release platform: OS=%s ARCH=%s\n' "${1:-}" "${2:-}" >&2
    return 1
}

# xencode_release_asset OS ARCH
# Prints the exact archive filename from the dist plan.
xencode_release_asset() {
    local target
    target="$(xencode_release_target "${1:-}" "${2:-}")" || return 1
    case "$target" in
        x86_64-pc-windows-msvc)
            printf 'xencode-cli-%s.zip' "$target"
            ;;
        *)
            printf 'xencode-cli-%s.tar.xz' "$target"
            ;;
    esac
}

# xencode_release_base_url [VERSION]
# Prints the GitHub Releases download prefix for a version (default: plan version).
xencode_release_base_url() {
    local version="${1:-$XENCODE_RELEASE_VERSION}"
    printf 'https://github.com/%s/%s/releases/download/v%s' \
        "$XENCODE_RELEASE_OWNER" "$XENCODE_RELEASE_REPO" "$version"
}

# xencode_release_asset_url OS ARCH [VERSION]
# Prints the full download URL of the platform archive.
xencode_release_asset_url() {
    local asset
    asset="$(xencode_release_asset "${1:-}" "${2:-}")" || return 1
    printf '%s/%s' "$(xencode_release_base_url "${3:-}")" "$asset"
}

# xencode_release_checksum_url OS ARCH [VERSION]
# Prints the full download URL of the per-archive .sha256 file that dist emits.
xencode_release_checksum_url() {
    local asset
    asset="$(xencode_release_asset "${1:-}" "${2:-}")" || return 1
    printf '%s/%s.sha256' "$(xencode_release_base_url "${3:-}")" "$asset"
}

# xencode_verify_checksum WORK_DIR ASSET
# Verify ASSET in WORK_DIR against a sha256sum-format check file named
# "<ASSET>.sha256sum" that was written by the installer. Uses sha256sum -c on
# Linux and shasum -a 256 -c where that is the available tool.
xencode_verify_checksum() {
    local work="$1" asset="$2"
    if command -v sha256sum &> /dev/null; then
        (cd "$work" && sha256sum -c "$asset.sha256sum")
    else
        (cd "$work" && shasum -a 256 -c "$asset.sha256sum")
    fi
}

# xencode_host_os / xencode_host_arch
# Report the current host's uname values, used when no explicit pair is given.
xencode_host_os() {
    uname -s 2>/dev/null || printf 'unknown'
}

xencode_host_arch() {
    uname -m 2>/dev/null || printf 'unknown'
}
