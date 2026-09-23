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

/// Pinned llama.cpp nightly build whose release carries prebuilt Ubuntu
/// binaries. `releases/latest` is the empty `v0.4.1` stub (a `nightly-tag.txt`
/// and nothing else), so the build number must be explicit — verified against
/// the GitHub API on 2026-09-23, where `b11120` ships
/// `llama-b11120-bin-ubuntu-cuda-12.8-x64.tar.gz` and friends.
pub const LLAMA_CPP_BUILD: &str = "b11120";

/// Where Colab's free-tier runtime keeps its NVIDIA driver libraries. A bare
/// `bash -s` login does not get them on the loader path, so `nvidia-smi`
/// reports "couldn't find libnvidia-ml.so" and a GPU build silently falls back
/// to CPU work — measured live on a T4 runtime on 2026-09-23.
const NVIDIA_LIB_DIR: &str = "/usr/lib64-nvidia";

/// llama.cpp runtime: the pinned prebuilt `llama-server` (CUDA build when the
/// runtime really has a GPU, plain x64 when it does not) plus one GGUF file
/// resolved straight from the Hugging Face API. Nothing is compiled on the VM,
/// which is the difference between a bring-up that takes a minute and one that
/// spends twenty building wheels.
///
/// The model id is a Hugging Face repo id (`Qwen/Qwen2.5-7B-Instruct-GGUF`) and
/// `quant` the file-name fragment to pick within it (`Q4_K_M`); a repo with no
/// matching file falls back to its first GGUF rather than to a failure.
/// The `hf` weights source is the bootstrap path; `drive`/`gcs` are refused
/// here (mounting those from a bare VM needs extra plumbing) — the error tells
/// the user how to proceed.
pub fn llama_cpp_bootstrap(
    model: &str,
    weights_source: &str,
    quant: &str,
    port: u16,
) -> Result<String, String> {
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
# xencode bootstrap — llama-server {build} on 127.0.0.1:{port}
export LD_LIBRARY_PATH="{nvidia_lib}:/usr/local/cuda-12.8/lib64:${{LD_LIBRARY_PATH:-}}"
DIR="${{HOME}}/xencode-llama"
rm -rf "$DIR"; mkdir -p "$DIR"
if nvidia-smi -L >/dev/null 2>&1; then
  ASSET='llama-{build}-bin-ubuntu-cuda-12.8-x64.tar.gz'; OFFLOAD='--n-gpu-layers 99'
else
  ASSET='llama-{build}-bin-ubuntu-x64.tar.gz'; OFFLOAD=''
  echo "no GPU visible in this runtime — serving on CPU (colab new --gpu T4 gets a real one)"
fi
curl -fsSL -o "$DIR/llama.tar.gz" "https://github.com/ggml-org/llama.cpp/releases/download/{build}/${{ASSET}}"
tar -xzf "$DIR/llama.tar.gz" -C "$DIR"
SERVER=$(find "$DIR" -name llama-server -type f | head -n1)
test -n "$SERVER" || {{ echo "llama-server missing from ${{ASSET}}"; exit 1; }}
LIBDIR=$(dirname "$(find "$DIR" -name 'libggml*.so' | head -n1)")
export LD_LIBRARY_PATH="${{LIBDIR}}:${{LD_LIBRARY_PATH:-}}"
HF_REPO={repo}
FILES=$(curl -fsSL "https://huggingface.co/api/models/$HF_REPO" \
  | python3 -c 'import json,sys;[print(f["rfilename"]) for f in json.load(sys.stdin)["siblings"]]')
GGUF_NAME=$(printf '%s\n' "$FILES" | grep -i '\.gguf$' | grep -i {quant} | head -n1)
if [ -z "$GGUF_NAME" ]; then GGUF_NAME=$(printf '%s\n' "$FILES" | grep -i '\.gguf$' | head -n1); fi
test -n "$GGUF_NAME" || {{ echo "no .gguf in $HF_REPO"; exit 1; }}
echo "weights: $GGUF_NAME"
curl -fL --retry 3 -o "$DIR/model.gguf" "https://huggingface.co/$HF_REPO/resolve/main/${{GGUF_NAME}}"
nohup "$SERVER" -m "$DIR/model.gguf" --host 127.0.0.1 --port {port} -c 4096 $OFFLOAD \
  > "${{HOME}}/xencode-llama.log" 2>&1 &
echo "READY {port}"
"#,
        build = LLAMA_CPP_BUILD,
        nvidia_lib = NVIDIA_LIB_DIR,
        port = port,
        repo = shell_quote(model),
        quant = shell_quote(quant),
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
    quant: &str,
    port: u16,
) -> Result<String, String> {
    match runtime {
        "llama.cpp" => llama_cpp_bootstrap(model, weights_source, quant, port),
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
        let script = llama_cpp_bootstrap("Qwen/Qwen2.5-7B-Instruct-GGUF", "hf", "Q4_K_M", 18080)
            .expect("hf supported");
        // The pinned prebuilt release, no on-VM compilation.
        assert!(script.contains(&format!("releases/download/{LLAMA_CPP_BUILD}/")));
        assert!(script.contains(&format!(
            "llama-{LLAMA_CPP_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz"
        )));
        // Model id and quant single-quoted into the script, never raw.
        assert!(script.contains("'Qwen/Qwen2.5-7B-Instruct-GGUF'"));
        assert!(script.contains("'Q4_K_M'"));
        assert!(script.contains("--host 127.0.0.1 --port 18080"));
        assert!(script.contains("READY 18080"));
        // The GPU layer-offload flag only reaches the server when the runtime
        // really has a GPU; the driver dir has to be on the loader path first.
        assert!(script.contains("/usr/lib64-nvidia"));
        assert!(script.contains("nvidia-smi -L"));
        assert!(script.contains("--n-gpu-layers 99"));
        assert!(script.contains("nohup \"$SERVER\""));
    }

    #[test]
    fn llama_cpp_refuses_non_hf_weights_with_a_fix() {
        let err = llama_cpp_bootstrap("repo/id", "gcs", "Q4_K_M", 18080).expect_err("gcs refused");
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
        let err = bootstrap_script("kobold", "m", "hf", "Q4_K_M", 18080).expect_err("unknown");
        assert!(err.contains("kobold"));
        assert!(err.contains("llama.cpp"));
        assert!(err.contains("ollama"));
    }
}
