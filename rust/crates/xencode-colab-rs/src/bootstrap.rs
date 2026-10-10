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
export LD_LIBRARY_PATH="{nvidia_lib}:${{LD_LIBRARY_PATH:-}}"
DIR="${{HOME}}/xencode-llama"
RELEASE="https://github.com/ggml-org/llama.cpp/releases/download/{build}"
# A server from an earlier bring-up still holds the port, and would answer the
# readiness check for the new one; stop it and give the port time to free.
if pkill -x llama-server; then sleep 3; fi
rm -rf "$DIR"; mkdir -p "$DIR"
# The CUDA 12 build needs libcudart.so.12 and libcublas.so.12. Colab's runtime
# moved to CUDA 13 (seen 2026-10-10) and has neither, and without them
# llama.cpp quietly serves on the CPU, so the runtime libraries the same
# release ships beside the build are fetched with it. The newer driver runs
# CUDA 12 programs.
if nvidia-smi -L >/dev/null 2>&1; then
  ASSET='llama-{build}-bin-ubuntu-cuda-12.8-x64.tar.gz'; OFFLOAD='--n-gpu-layers 99'
  CUDART='cudart-llama-{build}-bin-ubuntu-cuda-12.8-x64.tar.gz'
else
  ASSET='llama-{build}-bin-ubuntu-x64.tar.gz'; OFFLOAD=''; CUDART=''
  echo "no GPU visible in this runtime — serving on CPU (colab new --gpu T4 gets a real one)"
fi
curl -fsSL -o "$DIR/llama.tar.gz" "$RELEASE/${{ASSET}}"
tar -xzf "$DIR/llama.tar.gz" -C "$DIR"
if [ -n "$CUDART" ]; then
  mkdir -p "$DIR/cudart"
  curl -fsSL -o "$DIR/cudart.tar.gz" "$RELEASE/${{CUDART}}"
  tar -xzf "$DIR/cudart.tar.gz" -C "$DIR/cudart"
  CUDART_DIR=$(dirname "$(find "$DIR/cudart" -name 'libcudart.so.12*' | head -n1)")
  export LD_LIBRARY_PATH="${{CUDART_DIR}}:${{LD_LIBRARY_PATH}}"
fi
SERVER=$(find "$DIR" -name llama-server -type f | head -n1)
test -n "$SERVER" || {{ echo "llama-server missing from ${{ASSET}}"; exit 1; }}
LIBDIR=$(dirname "$(find "$DIR" -name 'libggml*.so' | head -n1)")
export LD_LIBRARY_PATH="${{LIBDIR}}:${{LD_LIBRARY_PATH:-}}"
HF_REPO={repo}
QUANT={quant}
FILES=$(curl -fsSL "https://huggingface.co/api/models/$HF_REPO" \
  | python3 -c 'import json,sys;[print(f["rfilename"]) for f in json.load(sys.stdin)["siblings"]]')
{pick}
echo "weights: $(printf '%s ' $PARTS)"
for PART in $PARTS; do
  curl -fsSL --retry 3 -o "$DIR/$(basename "$PART")" "https://huggingface.co/$HF_REPO/resolve/main/$PART"
done
nohup "$SERVER" -m "$DIR/$FIRST_PART" --host 127.0.0.1 --port {port} -c 4096 $OFFLOAD \
  > "${{HOME}}/xencode-llama.log" 2>&1 &
SERVER_PID=$!
# READY must mean "serving", not "spawned": llama.cpp binds its socket only
# after the weights are loaded (39 s for a 0.5B on a T4, measured live), so a
# forward that probes too early hits an empty port. A server that died while
# loading is reported at once rather than waited on.
WAITED=0
until curl -fsS "http://127.0.0.1:{port}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "llama-server exited before serving; last log lines:"
    tail -n 20 "${{HOME}}/xencode-llama.log"
    exit 1
  fi
  sleep 5; WAITED=$((WAITED + 5))
  if [ "$WAITED" -ge 900 ]; then
    echo "server did not start serving within 900s; last log lines:"
    tail -n 20 "${{HOME}}/xencode-llama.log"
    exit 1
  fi
done
# The first request on a fresh server pays a one-time start-up cost (36 s for
# a 7B on a T4, measured 2026-10-10, against 34 tokens a second afterwards);
# pay it here so the person's first question does not.
curl -fsS "http://127.0.0.1:{port}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{{"messages":[{{"role":"user","content":"hi"}}],"max_tokens":1}}' >/dev/null 2>&1 || true
# llama.cpp falls back to the CPU without a word, and its log names no device
# at the default detail level, so ask the GPU which processes it holds.
if [ -n "$OFFLOAD" ] && ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -qx "$SERVER_PID"; then
  echo "warning: llama-server is not using the GPU and is serving on the CPU, which is many times slower; see ~/xencode-llama.log on the VM"
fi
echo "READY {port}"
"#,
        build = LLAMA_CPP_BUILD,
        nvidia_lib = NVIDIA_LIB_DIR,
        port = port,
        repo = shell_quote(model),
        quant = shell_quote(quant),
        pick = pick_gguf_parts(),
    ))
}

/// The step of the llama.cpp bootstrap that chooses what to download, from
/// `$FILES` (the repo's file list), `$QUANT` and `$HF_REPO`. It sets `PARTS`
/// (every file to fetch, in order) and `FIRST_PART` (the file the server is
/// given).
///
/// Most quants of larger models are split into parts named
/// `<name>-00001-of-00002.gguf`. llama.cpp loads the first part and finds the
/// rest beside it by name, so every part is fetched under its own name. A
/// quant that matches nothing falls back to the repo's first GGUF; `|| true`
/// keeps `pipefail` from ending the script on that empty match.
fn pick_gguf_parts() -> &'static str {
    r#"GGUF_NAME=$(printf '%s\n' "$FILES" | grep -i '\.gguf$' | grep -i -F -- "$QUANT" | head -n1 || true)
if [ -z "$GGUF_NAME" ]; then GGUF_NAME=$(printf '%s\n' "$FILES" | grep -i '\.gguf$' | head -n1 || true); fi
test -n "$GGUF_NAME" || { echo "no .gguf in $HF_REPO"; exit 1; }
STEM=$(printf '%s' "$GGUF_NAME" | sed -E 's/-[0-9]{5}-of-[0-9]{5}\.gguf$//')
if [ "$STEM" != "$GGUF_NAME" ]; then
  PARTS=$(printf '%s\n' "$FILES" | grep -F -- "$STEM-" | grep -E -- '-[0-9]{5}-of-[0-9]{5}\.gguf$' | sort)
else
  PARTS=$GGUF_NAME
fi
FIRST_PART=$(basename "$(printf '%s\n' "$PARTS" | head -n1)")"#
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
        assert!(script.contains(&format!("releases/download/{LLAMA_CPP_BUILD}\"")));
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

    #[cfg(unix)]
    /// The file list of `Qwen/Qwen2.5-7B-Instruct-GGUF` as the Hugging Face
    /// API returned it on 2026-10-10: most quants there are split in parts.
    const QWEN_7B_FILES: &str = ".gitattributes
LICENSE
README.md
qwen2.5-7b-instruct-fp16-00001-of-00004.gguf
qwen2.5-7b-instruct-fp16-00002-of-00004.gguf
qwen2.5-7b-instruct-fp16-00003-of-00004.gguf
qwen2.5-7b-instruct-fp16-00004-of-00004.gguf
qwen2.5-7b-instruct-q2_k.gguf
qwen2.5-7b-instruct-q3_k_m.gguf
qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf
qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
qwen2.5-7b-instruct-q5_0-00001-of-00002.gguf
qwen2.5-7b-instruct-q5_0-00002-of-00002.gguf
qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf
qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf
qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf
qwen2.5-7b-instruct-q6_k-00002-of-00002.gguf
qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf
qwen2.5-7b-instruct-q8_0-00002-of-00003.gguf
qwen2.5-7b-instruct-q8_0-00003-of-00003.gguf";

    /// Run the script's file-picking step in a real bash and return the
    /// parts it would download, one per line.
    #[cfg(unix)]
    fn picked_parts(files: &str, quant: &str) -> Vec<String> {
        // Other tests point `$PATH` at an empty folder while they run.
        let _env = crate::testutil::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let script = format!(
            "set -euo pipefail\n{}\nprintf '%s\\n' $PARTS\n",
            pick_gguf_parts()
        );
        let out = std::process::Command::new("bash")
            .arg("-c")
            .arg(script)
            .env("FILES", files)
            .env("QUANT", quant)
            .env("HF_REPO", "Qwen/Qwen2.5-7B-Instruct-GGUF")
            .output()
            .expect("bash runs");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[cfg(unix)]
    #[test]
    fn a_split_model_is_fetched_whole_in_part_order() {
        assert_eq!(
            picked_parts(QWEN_7B_FILES, "Q4_K_M"),
            vec![
                "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
            ]
        );
        assert_eq!(
            picked_parts(QWEN_7B_FILES, "Q8_0").len(),
            3,
            "every part of a three-part model"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_single_file_model_is_fetched_alone() {
        assert_eq!(
            picked_parts(QWEN_7B_FILES, "Q2_K"),
            vec!["qwen2.5-7b-instruct-q2_k.gguf"]
        );
    }

    #[test]
    fn the_server_keeps_its_file_names_and_a_dead_server_is_not_waited_for() {
        let script = llama_cpp_bootstrap("Qwen/Qwen2.5-7B-Instruct-GGUF", "hf", "Q4_K_M", 18080)
            .expect("hf supported");
        // Parts keep their own names, or llama.cpp cannot find part two.
        assert!(!script.contains("model.gguf"), "{script}");
        assert!(script.contains("-m \"$DIR/$FIRST_PART\""), "{script}");
        // A download shows its errors but not a progress meter, which would
        // otherwise be what a failure report quotes.
        assert!(script.contains("curl -fsSL --retry 3 -o"), "{script}");
        // The wait stops as soon as the server process is gone.
        assert!(script.contains("SERVER_PID=$!"), "{script}");
        assert!(script.contains("kill -0 \"$SERVER_PID\""), "{script}");
    }

    #[test]
    fn the_gpu_build_brings_its_own_cuda_runtime_and_says_when_it_fell_back() {
        let script = llama_cpp_bootstrap("Qwen/Qwen2.5-7B-Instruct-GGUF", "hf", "Q4_K_M", 18080)
            .expect("hf supported");
        // Colab moved from CUDA 12 to 13; the CUDA 12 build needs the runtime
        // libraries that the same release ships beside it.
        assert!(
            script.contains(&format!(
                "cudart-llama-{LLAMA_CPP_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz"
            )),
            "{script}"
        );
        assert!(script.contains("libcudart.so.12"), "{script}");
        // llama.cpp falls back to the CPU silently; the bootstrap does not.
        // Its log names no device at the default detail level, so the GPU's
        // own list of processes is what is asked.
        assert!(script.contains("serving on the CPU"), "{script}");
        assert!(
            script.contains("nvidia-smi --query-compute-apps=pid"),
            "{script}"
        );
        assert!(!script.contains("grep -q 'CUDA0'"), "{script}");
        // The first request on a fresh server pays a one-time start-up cost
        // (36 s on a T4, 2026-10-10); the bootstrap pays it before READY.
        let warm = script.find("max_tokens\":1").expect("one warm-up request");
        assert!(warm < script.find("echo \"READY").unwrap(), "{script}");
        // A server left by an earlier bring-up holds the port and would
        // answer for the new one, so it is stopped before anything starts.
        let stop = script
            .find("pkill -x llama-server")
            .expect("stops an earlier server");
        assert!(stop < script.find("nohup").unwrap(), "{script}");
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
