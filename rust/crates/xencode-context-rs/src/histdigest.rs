//! History digest per edited file (`GH-1`).
//!
//! The "why does this exist" signal at tier-5 scale: the last-touch subject
//! for each hunk the working tree changed, plus the five most recent subjects
//! touching the path. Capped at roughly 250 tokens (~1000 characters),
//! because raw `blame`/`log` output runs 10–80× that budget — summarize, never
//! paste `-p`. A digest that exceeds the cap is cut with the cut marked, not
//! silently.
//!
//! Everything here is read-only against git. A file with no history (untracked,
//! or outside a repository) yields an empty digest with the reason, never an
//! error: "no history" is a fact about the file, not a failure of the tool.

/// Maximum characters in a digest. ~250 tokens at four characters each.
pub const DIGEST_CHAR_CAP: usize = 1000;

/// Subjects touching a path, most recent first, capped at five.
pub fn recent_subjects(root: &std::path::Path, path: &str, count: usize) -> Vec<String> {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args([
            "log",
            "--format=%h %s",
            "-n",
            &count.to_string(),
            "--",
            path,
        ])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// Hunk start lines (new side) of the working-tree diff for one file.
fn changed_lines(root: &std::path::Path, path: &str) -> Vec<u32> {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args(["diff", "--unified=0", "--", path])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some(rest) = line.strip_prefix("@@") {
            if let Some(plus) = rest.split('+').nth(1) {
                let start: u32 = plus
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.split(',').next())
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0);
                if start > 0 {
                    lines.push(start);
                }
            }
        }
    }
    lines
}

/// Last-touch subject for one line, from `blame` on HEAD.
fn blame_subject(root: &std::path::Path, path: &str, line: u32) -> Option<String> {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args([
            "blame",
            "--porcelain",
            "-L",
            &format!("{line},{line}"),
            "HEAD",
            "--",
            path,
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    let mut hash: Option<&str> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("summary ") {
            let short = hash.map(|h| h.get(..8).unwrap_or(h)).unwrap_or("?");
            return Some(format!("{short} {}", rest.trim()));
        }
        if hash.is_none() {
            if let Some(token) = line.split_whitespace().next() {
                if token.len() >= 40 && token.chars().all(|c| c.is_ascii_hexdigit()) {
                    hash = Some(token);
                }
            }
        }
    }
    None
}

/// The digest: last-touch per changed hunk, then recent subjects, capped.
pub fn history_digest(root: &std::path::Path, path: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for line in changed_lines(root, path) {
        if let Some(subject) = blame_subject(root, path, line) {
            if !seen.contains(&subject) {
                seen.push(subject.clone());
                parts.push(format!("touched line {line}: {subject}"));
            }
        }
    }
    for subject in recent_subjects(root, path, 5) {
        if !seen.contains(&subject) {
            seen.push(subject);
        }
    }
    let mut out = if parts.is_empty() && seen.is_empty() {
        "no history: untracked, uncommitted, or outside a repository".to_string()
    } else {
        let mut out = parts.join("\n");
        let fresh: Vec<&String> = seen
            .iter()
            .filter(|s| !parts.iter().any(|p| p.contains(*s)))
            .collect();
        if !fresh.is_empty() {
            if !out.is_empty() {
                out.push_str("\nrecent:\n");
            } else {
                out.push_str("recent:\n");
            }
            out.push_str(
                &fresh
                    .iter()
                    .map(|s| format!("- {s}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        out
    };
    if out.len() > DIGEST_CHAR_CAP {
        let mut end = DIGEST_CHAR_CAP.min(out.len());
        while !out.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        out = format!("{}… [cut at ~250 tokens]", &out[..end]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_repo(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("xe-hist-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn git(repo: &std::path::Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .current_dir(repo)
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "git {args:?}");
    }

    fn seed(label: &str) -> std::path::PathBuf {
        // One repo per test: parallel tests sharing a directory init, commit,
        // and remove over each other, and every failure then blames git.
        let repo = temp_repo(label);
        git(&repo, &["init", "-q", "-b", "main"]);
        git(&repo, &["config", "user.email", "t@t"]);
        git(&repo, &["config", "user.name", "t"]);
        std::fs::write(repo.join("f.txt"), "one\ntwo\nthree\n").unwrap();
        git(&repo, &["add", "-A"]);
        git(&repo, &["commit", "-qm", "first"]);
        std::fs::write(repo.join("f.txt"), "one\nTWO\nthree\n").unwrap();
        git(&repo, &["commit", "-qam", "second"]);
        repo
    }

    #[test]
    fn recent_subjects_come_newest_first_and_stop_at_five() {
        let repo = seed("recent");
        for n in 3..=8 {
            std::fs::write(repo.join("f.txt"), format!("one\nV{n}\nthree\n")).unwrap();
            git(&repo, &["commit", "-qam", &format!("change {n}")]);
        }
        let subjects = recent_subjects(&repo, "f.txt", 5);
        assert_eq!(subjects.len(), 5);
        assert!(subjects[0].contains("change 8"), "{subjects:?}");
        assert!(subjects[4].contains("change 4"), "{subjects:?}");
    }

    #[test]
    fn a_changed_hunk_names_its_last_touch() {
        let repo = seed("hunk");
        std::fs::write(repo.join("f.txt"), "one\nTWO!\nthree\n").unwrap();
        let digest = history_digest(&repo, "f.txt");
        assert!(digest.contains("touched line 2"), "{digest}");
        assert!(digest.contains("second"), "{digest}");
        assert!(digest.contains("recent:"), "{digest}");
    }

    #[test]
    fn the_digest_fits_the_tier() {
        // Overflow comes from many changed hunks touching different commits,
        // not many commits on one line: identical subjects dedup by design, the
        // recent list is capped at five, and consecutive edits merge into one
        // hunk. So thirty lines, each committed separately, then all edited.
        let repo = seed("cap");
        let mut lines: Vec<String> = (0..30).map(|n| format!("line {n}\n")).collect();
        std::fs::write(repo.join("f.txt"), lines.concat()).unwrap();
        git(&repo, &["commit", "-qam", "thirty lines"]);
        // Each commit touches exactly one line, so every line blames a
        // different commit. Rewriting the whole file each time would blame
        // everything on the last commit instead.
        for n in 0..30 {
            lines[n] = format!("line {n} from commit {n} stretching the message a little\n");
            std::fs::write(repo.join("f.txt"), lines.concat()).unwrap();
            git(
                &repo,
                &[
                    "commit",
                    "-qam",
                    &format!("commit number {n} with a fairly long message"),
                ],
            );
        }
        let edited: String = (0..30)
            .map(|n| {
                if n % 2 == 0 {
                    format!("EDITED line {n} with replacement text making it longer\n")
                } else {
                    format!("line {n} from commit {n} stretching the message a little\n")
                }
            })
            .collect();
        std::fs::write(repo.join("f.txt"), &edited).unwrap();
        let digest = history_digest(&repo, "f.txt");
        assert!(
            digest.len() <= DIGEST_CHAR_CAP + 30,
            "cut is marked: {}",
            digest.len()
        );
        assert!(digest.contains("cut at ~250 tokens"), "{digest}");
    }

    #[test]
    fn no_history_is_a_fact_not_an_error() {
        let repo = temp_repo("empty");
        assert!(history_digest(&repo, "f.txt").contains("no history"));
        let repo = seed("untracked");
        std::fs::write(repo.join("new.txt"), "hi\n").unwrap();
        assert!(history_digest(&repo, "new.txt").contains("no history"));
    }
}
