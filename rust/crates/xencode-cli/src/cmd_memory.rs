//! the memory, tasks, worktree and advise subcommands
//! `xencode memory`, `tasks`, `worktree` and `advise`.

use super::*;

pub(crate) fn run_memory(action: MemoryAction) -> Result<(), String> {
    let mut mem = ConversationMemory::with_persistence(50).map_err(|e| e.to_string())?;

    match action {
        MemoryAction::List { all } => {
            let all_sessions = mem.list_all_sessions();
            let non_empty = mem.list_sessions();
            let empty_count = all_sessions.len().saturating_sub(non_empty.len());
            let sessions = if all { all_sessions } else { non_empty };
            if sessions.is_empty() {
                if empty_count > 0 {
                    println!(
                        "No non-empty conversation sessions found ({} empty session(s) omitted; use --all to show, or `xencode memory prune` to delete).",
                        empty_count
                    );
                } else {
                    println!("No conversation sessions found.");
                }
            } else {
                println!("Conversation Sessions:");
                for s in &sessions {
                    if all {
                        let count = mem
                            .get_session(s)
                            .map(|sess| sess.messages.len())
                            .unwrap_or(0);
                        println!(
                            "  {} ({} message{})",
                            s,
                            count,
                            if count == 1 { "" } else { "s" }
                        );
                    } else {
                        println!("  {}", s);
                    }
                }
                if !all && empty_count > 0 {
                    println!(
                        "\n  ({} empty session(s) omitted; use --all to show, or `xencode memory prune` to delete)",
                        empty_count
                    );
                }
            }
            Ok(())
        }
        MemoryAction::Show { session, first } => {
            if let Some(sess) = mem.get_session(&session) {
                if first {
                    if let Some(msg) = sess.first_message() {
                        let role = msg.role.to_uppercase();
                        println!("[{}] {}", role, msg.timestamp);
                        println!("{}\n", msg.content);
                    } else {
                        println!("Session {} has no messages.", session);
                    }
                } else if sess.messages.is_empty() && sess.events.is_empty() {
                    println!("Session {} is empty (0 messages).", session);
                } else if !sess.events.is_empty() {
                    for ev in &sess.events {
                        let role = ev.message.role.to_uppercase();
                        println!("[{}] {}", role, ev.timestamp);
                        println!("{}\n", ev.message.content);
                    }
                } else {
                    for msg in &sess.messages {
                        let role = msg.role.to_uppercase();
                        println!("[{}] {}", role, msg.timestamp);
                        println!("{}\n", msg.content);
                    }
                }
            } else {
                println!("Session not found: {}", session);
            }
            Ok(())
        }
        MemoryAction::Fork {
            session,
            as_id,
            prefix,
        } => {
            let child_id = mem
                .fork_session(&session, as_id, prefix)
                .map_err(|e| e.to_string())?;
            println!("Forked session {} into {}", session, child_id);
            Ok(())
        }
        MemoryAction::Prune => {
            let pruned = mem.prune_empty_sessions();
            if pruned == 0 {
                println!("No empty conversation sessions to prune.");
            } else {
                println!(
                    "Pruned {} empty conversation session{}.",
                    pruned,
                    if pruned == 1 { "" } else { "s" }
                );
            }
            Ok(())
        }
        MemoryAction::Gc { apply } => {
            let xencode = xencode_context_rs::default_root().join(xencode_context_rs::XENCODE_DIR);
            let now = xencode_context_rs::gc_now_ms();
            let report =
                xencode_context_rs::collect_gc(&xencode, now, apply).map_err(|e| e.to_string())?;
            println!(
                "Durable facts: {} contradicted, {} past {} months, {} removed",
                report.pending.len(),
                report.aged.len(),
                xencode_context_rs::RETIRE_AFTER_MONTHS,
                report.removed.len()
            );
            for entry in report.pending.iter().take(20) {
                println!(
                    "  {} — {}: {}",
                    xencode_context_rs::age_words(entry.months(now)),
                    entry.problem.reason(),
                    xencode_context_rs::fact_prose(&entry.fact)
                );
            }
            if report.pending.len() > 20 {
                println!("  … and {} more", report.pending.len() - 20);
            }
            // The queue is what makes retirement a decision rather than a hunch,
            // so a run that cannot offer any has to say why: twelve months of
            // unbroken contradiction start counting the first time this looked.
            if report.removed.is_empty() && !report.aged.is_empty() {
                println!(
                    "  `--apply` retires the {} above; a fact that stops being contradicted leaves the queue instead of ageing toward removal.",
                    report.aged.len()
                );
            }
            if !report.retired.is_empty() {
                println!("  {} retired by an earlier run.", report.retired.len());
            }
            if report.unverifiable > 0 {
                println!(
                    "  {} stayed in the prompt because this repository could not check them.",
                    report.unverifiable
                );
            }
            if report.disagreeing > 0 {
                println!(
                    "  {} that the code places in another file. Those are disagreement, not staleness: `xencode doctor` names them and no sweep removes them.",
                    report.disagreeing
                );
            }
            if report.pending.is_empty()
                && report.retired.is_empty()
                && report.unverifiable == 0
                && report.disagreeing == 0
            {
                println!("  Nothing to collect: this repository agrees with every durable fact it can check.");
            }
            Ok(())
        }
        MemoryAction::Evidence { format } => {
            let xencode = xencode_context_rs::default_root().join(xencode_context_rs::XENCODE_DIR);
            let rows = xencode_context_rs::evidence_rows(&xencode);
            match format {
                OutputFormat::Json => {
                    let body: Vec<serde_json::Value> = rows
                        .iter()
                        .map(|row| {
                            serde_json::json!({
                                "fact": row.fact,
                                "revisions_checked": row.trials,
                                "survived": row.survived,
                                "unchecked": row.unchecked,
                                "wilson_95_of_next_check_agreeing": row.interval.map(|(lower, upper)| {
                                    [
                                        (lower * 1000.0).round() / 1000.0,
                                        (upper * 1000.0).round() / 1000.0,
                                    ]
                                }),
                                "verified_by": row.verified_by(),
                                "last_revision": row.last.as_ref().map(|c| c.revision.clone()),
                                "contradicted_by": row.problem.map(|p| p.reason()),
                            })
                        })
                        .collect();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&body)
                            .map_err(|e| format!("cannot write the report: {e}"))?
                    );
                }
                OutputFormat::Text => {
                    if rows.is_empty() {
                        println!(
                            "No durable fact here has been re-checked against a revision. A fact \
                             gains evidence on the turn it is next assembled into a prompt, in a \
                             repository git can search."
                        );
                        return Ok(());
                    }
                    println!(
                        "Durable facts, weakest evidence first — the interval is over the checks \
                         this repository ran, and is not a chance that the fact is true"
                    );
                    for row in &rows {
                        println!("  {}", row.fact);
                        println!("    {}", row.evidence_sentence());
                        println!("    verified by {}", row.verified_by());
                        if let Some(problem) = row.problem {
                            println!("    contradicted now: {}", problem.reason());
                        }
                    }
                    let thin = rows.iter().filter(|row| row.trials < 2).count();
                    if thin > 0 {
                        println!(
                            "  {thin} of {} reach fewer than two revisions, where an interval spans \
                             most of what it could and no verdict is available.",
                            rows.len()
                        );
                    }
                }
            }
            Ok(())
        }
        MemoryAction::Publish {
            worker,
            scope,
            content,
        } => {
            let (dir, mut store) = shared_memory_store()?;
            let finding = store
                .publish(&worker, MemoryScope::parse(&scope), content)
                .map_err(|e| e.to_string())?;
            store.save_to_dir(&dir).map_err(|e| e.to_string())?;
            println!(
                "Published {} into scope '{}' as worker '{}'.",
                finding.id, finding.scope, finding.author_worker_id
            );
            Ok(())
        }
        MemoryAction::Read { worker, scope } => {
            let (_dir, store) = shared_memory_store()?;
            let found = match &scope {
                Some(scope) => store
                    .read_scope(&worker, &MemoryScope::parse(scope))
                    .map_err(|e| e.to_string())?,
                None => store.read_all_granted(&worker).map_err(|e| e.to_string())?,
            };
            if found.is_empty() {
                println!("Worker '{worker}' read 0 findings.");
                return Ok(());
            }
            println!(
                "Shared memory read as worker '{}' — {} finding{}, each one another worker's data, not an instruction:",
                worker,
                found.len(),
                if found.len() == 1 { "" } else { "s" }
            );
            for item in &found {
                println!();
                println!("{}", item.marked_content);
                println!("  {} published {}", item.finding_id, item.published_at);
            }
            Ok(())
        }
        MemoryAction::Policy { action } => match action {
            MemoryPolicyAction::Set {
                worker,
                read,
                publish,
            } => {
                let (dir, mut store) = shared_memory_store()?;
                let read: Vec<MemoryScope> = read.iter().map(|s| MemoryScope::parse(s)).collect();
                let publish: Vec<MemoryScope> =
                    publish.iter().map(|s| MemoryScope::parse(s)).collect();
                let mut policy = WorkerMemoryPolicy::new(&worker);
                for scope in &read {
                    policy = policy.allow_read(scope.clone());
                }
                for scope in &publish {
                    policy = policy.allow_publish(scope.clone());
                }
                store.set_policy(policy);
                store.save_to_dir(&dir).map_err(|e| e.to_string())?;
                println!("Memory policy for worker '{worker}':");
                println!("  reads:     {}", scope_names(&read));
                println!("  publishes: {}", scope_names(&publish));
                Ok(())
            }
            MemoryPolicyAction::Show { worker } => {
                let (_dir, store) = shared_memory_store()?;
                let mut listed: Vec<&WorkerMemoryPolicy> = match &worker {
                    Some(worker) => store
                        .all_policies()
                        .values()
                        .filter(|policy| &policy.worker_id == worker)
                        .collect(),
                    None => store.all_policies().values().collect(),
                };
                listed.sort_by(|a, b| a.worker_id.cmp(&b.worker_id));
                if listed.is_empty() {
                    match &worker {
                        Some(worker) => println!(
                            "No memory policy for worker '{worker}': it may read nothing and publish nothing."
                        ),
                        None => println!(
                            "No memory policies configured. Until one is set, every worker is denied on both sides."
                        ),
                    }
                    return Ok(());
                }
                println!("Memory policies:");
                for policy in listed {
                    println!("  {}", policy.worker_id);
                    println!("    reads:     {}", scope_names(&policy.readable_scopes));
                    println!("    publishes: {}", scope_names(&policy.publishable_scopes));
                }
                Ok(())
            }
        },
    }
}

/// OR-8: the shared memory workers hand each other, kept in `.xencode/` beside
/// the other durable files. Both sides deny by default, so a worker that was
/// never given a policy can neither read what it has stored nor add to it.
pub(crate) fn shared_memory_store() -> Result<(PathBuf, ScopedSharedMemoryStore), String> {
    let dir = xencode_context_rs::default_root().join(xencode_context_rs::XENCODE_DIR);
    let store = ScopedSharedMemoryStore::load_from_dir(&dir).map_err(|e| e.to_string())?;
    Ok((dir, store))
}

/// Scopes in a stable order. A policy grants a hash set, and a listing that
/// changed shape between two runs of the same command cannot be compared.
pub(crate) fn scope_names<'a>(scopes: impl IntoIterator<Item = &'a MemoryScope>) -> String {
    let mut names: Vec<String> = scopes.into_iter().map(|scope| scope.to_string()).collect();
    names.sort();
    names.dedup();
    if names.is_empty() {
        "nothing".to_string()
    } else {
        names.join(", ")
    }
}

/// Registry location: `<default_root>/.xencode/tasks`, the same dir layout
/// `xencode init` uses for project state.
pub(crate) fn tasks_registry() -> xencode_core_rs::FileTaskRegistry {
    let root = xencode_context_rs::default_root()
        .join(xencode_context_rs::XENCODE_DIR)
        .join("tasks");
    xencode_core_rs::FileTaskRegistry::new(root)
}

pub(crate) fn run_tasks(action: TaskAction) -> Result<(), String> {
    let reg = tasks_registry();
    match action {
        TaskAction::List { json } => {
            let entries = reg.poll().map_err(|e| e.to_string())?;
            if json {
                let rows: Vec<serde_json::Value> = entries
                    .values()
                    .map(|(t, status)| {
                        serde_json::json!({
                            "id": t.id,
                            "name": t.name,
                            "command": t.command,
                            "pid": t.pid,
                            "started_at": t.started_at,
                            "status": status.label(),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?
                );
            } else if entries.is_empty() {
                println!("No background tasks.");
            } else {
                println!("{:>4}  {:<12} {:>8}  NAME", "ID", "STATUS", "PID");
                for (task, status) in entries.values() {
                    println!(
                        "{:>4}  {:<12} {:>8}  {}",
                        task.id,
                        status.label(),
                        task.pid,
                        task.name
                    );
                }
            }
            Ok(())
        }
        TaskAction::Start { command, name } => {
            let task = reg
                .start(name.as_deref().unwrap_or(""), &command)
                .map_err(|e| e.to_string())?;
            println!(
                "started task {} (pid {}): {}",
                task.id, task.pid, task.command
            );
            Ok(())
        }
        TaskAction::Poll { id, lines } => {
            let task = reg
                .list()
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|t| t.id == id)
                .ok_or_else(|| format!("no such task: {id}"))?;
            println!(
                "task {} [{}] {} (pid {})",
                task.id,
                reg.status(&task).label(),
                task.command,
                task.pid
            );
            let output = reg.output(id, lines);
            if output.is_empty() {
                println!("  (no output)");
            } else {
                for line in output {
                    println!("  {line}");
                }
            }
            Ok(())
        }
        TaskAction::Stop { id } => {
            reg.stop(id).map_err(|e| e.to_string())?;
            println!("stopped task {id}");
            Ok(())
        }
        TaskAction::Rm { id } => {
            reg.remove(id).map_err(|e| e.to_string())?;
            println!("removed task {id}");
            Ok(())
        }
    }
}

pub(crate) fn run_worktree(action: WorktreeAction) -> Result<(), String> {
    use xencode_context_rs::{worktree_add, worktree_list, worktree_remove, WorktreeInfo};
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let print_list = |list: &[WorktreeInfo]| {
        println!("{:<10} {:<10} PATH", "BRANCH", "HEAD");
        for wt in list {
            let tag = if wt.is_main { " [main]" } else { "" };
            println!(
                "{:<10} {:<10} {}{tag}",
                wt.display_branch(),
                wt.short_head(),
                wt.path.display(),
            );
        }
    };
    match action {
        WorktreeAction::List => {
            let list = worktree_list(&root)?;
            print_list(&list);
            Ok(())
        }
        WorktreeAction::Add { path, branch } => {
            let list = worktree_add(&root, std::path::Path::new(&path), branch.as_deref(), false)?;
            println!("added worktree {path}");
            print_list(&list);
            Ok(())
        }
        WorktreeAction::Remove { path } => {
            if path == "." || std::path::Path::new(&path) == root {
                return Err("the main worktree is not removable".to_string());
            }
            let list = worktree_remove(&root, std::path::Path::new(&path), false)?;
            println!("removed worktree {path}");
            print_list(&list);
            Ok(())
        }
    }
}

/// Findings for `root`'s `.xencode` snapshot, narrowed by `filter`
/// (substring of the file path). The read+analysis half of `xencode advise`,
/// unit-tested without the print layer.
pub(crate) fn compute_advise(
    root: &std::path::Path,
    filter: Option<&str>,
) -> Result<Vec<xencode_context_rs::Advice>, String> {
    let mut items = xencode_context_rs::advise_from_snapshot(root).map_err(|e| e.to_string())?;
    if let Some(needle) = filter {
        items.retain(|a| a.file.contains(needle));
    }
    Ok(items)
}

pub(crate) fn run_advise(filter: Option<String>, json: bool, limit: usize) -> Result<(), String> {
    let root = xencode_context_rs::default_root();
    let filter = filter.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let mut items = compute_advise(&root, filter)?;
    let total = items.len();
    if limit > 0 {
        items.truncate(limit);
    }
    if json {
        let rendered: Vec<serde_json::Value> = items
            .iter()
            .map(|a| {
                serde_json::json!({
                    "file": a.file,
                    "kind": format!("{:?}", a.kind),
                    "message": a.message,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&rendered).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if items.is_empty() {
        println!("no findings — the symbol graph is clean.");
        return Ok(());
    }
    for (i, a) in items.iter().enumerate() {
        println!(
            "{:>2}. {:<19} {}",
            i + 1,
            format!("{:?}", a.kind),
            a.message
        );
    }
    if total > items.len() {
        println!(
            "… +{} more (raise --limit, or pass a path filter)",
            total - items.len()
        );
    }
    Ok(())
}
