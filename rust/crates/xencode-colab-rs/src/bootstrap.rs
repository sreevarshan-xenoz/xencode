//! The VM-side bootstrap script `xencode colab up` pushes over the ssh bridge
//! and runs as `bash -s`. It installs and starts the chosen inference runtime
//! bound to `127.0.0.1` only — nothing the VM exposes to the public internet,
//! which is the Colab term-of-service line this feature stays on.
//!
//! The script is a string built from the user's runtime / model / weights
//! choices; everything not user-typed is a lone command of that runtime. The
//! generator is unit-tested at the string level (the boot is real Colab, so
//! the hermetic suite never runs it).
#![forbid(unsafe_code)]

use crate::orchestrate::shell_quote;

/// llama.cpp runtime: pinned `llama-cpp-python[server]` (its `llama-server`
/// executable and Python module bundle one another; installing the wheel also
/// fetches the bundled `llama-server` binary) + a GGUF pulled from Hugging
/// Face. The `hf` weights source is the bootstrap path; `drive`/`gcs` are
/// refused here (mounting those from a bare VM needs extra plumbing) — the
/// error tells the user how to proceed.
///
/// The model id is a Hugging Face repo id (e.g. `Qwen/Qwen2.5-7B-Instruct
/// -GGUF`). llama.cpp no longer ships prebuilt linux binaries on the release
/// channel (the v0.4.1 release carries only a nightly-tag marker), so the
/// server comes from the PyPI wheel instead — verified against live PyPI
/// (`llama-cpp-python` latest is 0.3.35 at the time of writing).
pub fn llama_cpp_bootstrap(model: &str, weights_source: &str, port: u16) -> Result<String, String> {
    if weights_source != "hf" {
        return Err(format!(
            "llama.cpp bootstrap supports weights_source=\"hf\" only \
             (the \"{weights_source}\" source needs Drive/GCS mounting \
             plumbing not wired up yet) — use hf, or run the runtime inside \
             the VM yourself and point Settings → Remote URL at it"
        ));
    }
    Ok(format!(
        r#"set -euo pipefail
# xencode bootstrap — llama.cpp on 127.0.0.1:{port}
pip install --quiet --upgrade pip
pip install --quiet "llama-cpp-python[server]==0.3.35"
pip install --quiet huggingface-cli
HF_REPO={repo}
mkdir -p "${{HOME}}/xencode-gguf"
huggingface-cli download "$HF_REPO" --local-dir "${{HOME}}/xencode-gguf"
GGUF=$(find "${{HOME}}/xencode-gguf" -name '*.gguf' | head -n1)
test -n "$GGUF" || {{ echo "no .gguf found in ${{HF_REPO}}"; exit 1; }}
nohup python3 -m llama_cpp.server --model "$GGUF" --host 127.0.0.1 --port {port} > "${{HOME}}/xencode-llama.log" 2>&1 &
echo "READY {port}"
"#,
        repo = shell_quote(model),
    ))
}

/// ollama runtime: the official install script + `ollama serve` bound to the
/// loopback, then the model tag pulled from the ollama registry. The tag comes
/// through `--model` on the CLI and must match ollama tag syntax
/// (namespaces, `:tag`, `/`), so it is passed through as-is rather than
/// shell-quoted — ollama validates tags itself and fails the pull cleanly.
pub fn ollama_bootstrap(model: &str, port: u16) -> String {
    [
        "set -euo pipefail".to_string(),
        format!("# xencode bootstrap — ollama on 127.0.0.1:{port}"),
        "curl -fsSL https://ollama.com/install.sh | sh".to_string(),
        format!("export OLLAMA_HOST=127.0.0.1:{port}"),
        "nohup ollama serve > \"$HOME/xencode-ollama.log\" 2>&1 &".to_string(),
        format!("ollama pull {model}"),
        format!("echo READY {port}"),
    ]
    .join("\n")
}

/// The bootstrap script for `runtime` on `port`. Returns a script string or a
/// "report itself unpowered"-style error (unsupported source / unknown
/// runtime). The two supported runtimes cover the plan's "pinned
/// llama-server, or ollama" — each gets its own module doc for trade-offs that
/// the `--runtime` help surfaces to the user.
pub fn bootstrap_script(
    runtime: &str,
    model: &str,
    weights_source: &str,
    port: u16,
) -> Result<String, String> {
    match runtime {
        "llama.cpp" => llama_cpp_bootstrap(model, weights_source, port),
        "ollama" => Ok(ollama_bootstrap(model, port)),
        other => Err(format!(
            "unknown colab runtime {other:?} (expected \"llama.cpp\" or \"ollama\")"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_cpp_script_pins_the_server_binds_loopback_and_quotes_model() {
        let script =
            llama_cpp_bootstrap("Qwen/Qwen2.5-7B-Instruct-GGUF", "hf", 8080).expect("hf supported");
        // The pinned wheel that ships the llama-server binary.
        assert!(script.contains("llama-cpp-python[server]==0.3.35"));
        // Model id single-quoted into the script, never raw.
        assert!(script.contains("'Qwen/Qwen2.5-7B-Instruct-GGUF'"));
        assert!(script.contains("--host 127.0.0.1 --port 8080"));
        assert!(script.contains("READY 8080"));
        assert!(script.contains("nohup python3 -m llama_cpp.server"));
    }

    #[test]
    fn llama_cpp_refuses_non_hf_weights_with_a_fix() {
        let err = llama_cpp_bootstrap("repo/id", "gcs", 8080).expect_err("gcs refused");
        assert!(err.contains("weights_source"), "names the field: {err}");
        assert!(err.contains("hf"), "points at the supported source: {err}");
    }

    #[test]
    fn ollama_script_pulls_the_tag_and_binds_loopback() {
        let script = ollama_bootstrap("qwen2.5:7b", 11434);
        assert!(script.contains("curl -fsSL https://ollama.com/install.sh | sh"));
        assert!(script.contains("OLLAMA_HOST=127.0.0.1:11434"));
        assert!(script.contains("ollama pull qwen2.5:7b"));
        assert!(script.contains("READY 11434"));
    }

    #[test]
    fn bootstrap_script_rejects_unknown_runtime() {
        let err = bootstrap_script("kobold", "m", "hf", 8080).expect_err("unknown");
        assert!(err.contains("kobold"));
        assert!(err.contains("llama.cpp"));
        assert!(err.contains("ollama"));
    }
}
