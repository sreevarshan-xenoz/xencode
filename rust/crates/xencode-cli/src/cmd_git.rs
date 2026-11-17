//! `xencode git commit-message` — turn the real staged diff into a commit
//! message suggestion, using the same local model stack as `xencode query`.
//!
//! The staged diff comes straight from `git diff --cached`; it is never read
//! from a snapshot or invented here. The model is reached through the existing
//! [`ProviderManager`], so no new HTTP or provider code lives in this module.
//! Nothing is committed unless the caller explicitly passes `--commit`.

use super::*;

/// Instructions that go ahead of the diff in the model's single user turn.
/// Keeping them in one message means the whole prompt — instructions and diff —
/// is one string that can be checked against what the server received.
const PROMPT_HEAD: &str = "\
You write git commit messages. Given the staged diff below, reply with one \
commit message and nothing else. Use a short imperative subject line (under \
72 characters). If the change needs explanation, follow the subject with a \
blank line and a short body saying what changed and why. Do not use markdown \
fences, headings, or commentary.\n\nStaged diff (from `git diff --cached`):";

/// The staged diff from git, or `None` when the index holds no changes.
///
/// An empty result is reported as "nothing staged" by the caller rather than
/// being sent to the model, which would otherwise invent a message about
/// nothing.
fn staged_diff(dir: &std::path::Path) -> Result<Option<String>, String> {
    let output = std::process::Command::new("git")
        .current_dir(dir)
        .args(["diff", "--cached", "--no-color"])
        .output()
        .map_err(|e| format!("could not run git diff --cached: {e}"))?;
    if !output.status.success() {
        let why = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "git diff --cached failed (is this a git repository?): {why}"
        ));
    }
    let diff = String::from_utf8_lossy(&output.stdout).into_owned();
    if diff.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(diff))
    }
}

/// The prompt sent to the model: instructions followed by the verbatim diff.
fn build_prompt(diff: &str) -> String {
    format!("{PROMPT_HEAD}\n{diff}")
}

pub(crate) async fn run_git_commit_message(
    model_override: Option<String>,
    commit: bool,
) -> Result<(), String> {
    let dir =
        std::env::current_dir().map_err(|e| format!("cannot read the current folder: {e}"))?;
    let diff = match staged_diff(&dir)? {
        Some(diff) => diff,
        None => {
            return Err(
                "nothing is staged — stage changes with `git add` first, then run \
                 `xencode git commit-message` again"
                    .to_string(),
            )
        }
    };

    let config = XencodeConfig::load().unwrap_or_default();
    let model = model_override.unwrap_or_else(|| config.default_model.clone());
    let prompt = build_prompt(&diff);

    // The same provider stack `xencode query` uses, minus the cache, memory and
    // project-context assembly: a commit message only needs the staged diff.
    let client = OllamaClient::new(&config.ollama_url, config.response_timeout);
    let llama_client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);
    let provider = ProviderManager::new(
        client,
        api_key(&config, SecretProvider::OpenRouter),
        api_key(&config, SecretProvider::Qwen),
        api_key(&config, SecretProvider::Gemini),
        None,
    )
    .with_llama_cpp(llama_client)
    .with_remote(
        &config.remote_base_url,
        api_key(&config, SecretProvider::Remote),
    )
    .with_nvidia(api_key(&config, SecretProvider::Nvidia))
    .with_request_timeout(config.response_timeout)
    .with_egress_policy(EgressPolicy::new(config.allow_cloud_models));

    let answer = provider
        .generate(&model, &[ChatMessage::text("user", prompt)])
        .await
        .map_err(|e| format!("could not get a commit message from {model}: {e}"))?;
    let message = answer.trim().to_string();
    if message.is_empty() {
        return Err(format!(
            "the model at {model} returned an empty commit message"
        ));
    }
    println!("{message}");

    if commit {
        // Commit only what is staged (no `-a`), reusing the signing environment
        // the TUI's git commit panel passes so a signed-commit setup keeps
        // working here too.
        let mut command = std::process::Command::new("git");
        command.current_dir(&dir).args(["commit", "-m", &message]);
        for (name, value) in xencode_tui_rs::gitsign::signing_env() {
            command.env(name, value);
        }
        let status = command
            .status()
            .map_err(|e| format!("could not run git commit: {e}"))?;
        if !status.success() {
            return Err("git commit failed — nothing was committed".to_string());
        }
        println!("committed the staged changes");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_carries_the_verbatim_staged_diff() {
        let diff =
            "diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-old\n+new\n";
        let prompt = build_prompt(diff);
        assert!(prompt.contains("Staged diff (from `git diff --cached`):"));
        assert!(prompt.contains(diff));
        assert!(prompt.contains("+new"));
    }

    #[test]
    fn an_empty_diff_is_reported_as_nothing_staged() {
        // An index with no changes against HEAD yields no diff output.
        let dir = tempfile::tempdir().unwrap();
        let git = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .current_dir(dir.path())
                .args(args)
                .output()
                .expect("git runs");
            assert!(out.status.success(), "git {args:?}: {:?}", out);
        };
        git(&["init", "-q"]);
        git(&["config", "user.name", "tester"]);
        git(&["config", "user.email", "tester@example.invalid"]);
        assert_eq!(staged_diff(dir.path()).unwrap(), None);
    }
}
