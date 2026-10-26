//! the mcp and plugin subcommands
//! `xencode mcp` and `xencode plugin`, with the plugin declaration helpers.

use super::*;

/// `xencode mcp serve` (M-5): xencode as the server, on a pipe.
pub(crate) async fn run_mcp(action: McpAction) -> Result<(), String> {
    use xencode_tui_rs::agent_tools::{new_task_runtime, HeadlessPolicy};
    use xencode_tui_rs::mcp_serve;

    match action {
        McpAction::Serve {
            workspace,
            allow,
            team,
        } => {
            let root = std::fs::canonicalize(&workspace).map_err(|error| {
                format!(
                    "workspace {} cannot be opened: {error}",
                    workspace.display()
                )
            })?;
            // A flag naming nothing is a typo, and a typo accepted quietly leaves
            // the operator believing a tool is permitted when it is not.
            let published = mcp_serve::exposed_tool_names();
            for name in &allow {
                if !published.contains(name) {
                    return Err(format!(
                        "`--allow {name}` names no tool xencode publishes; the six are {}",
                        published.join(", ")
                    ));
                }
            }
            let policy = HeadlessPolicy::new(allow.clone());
            // stdout carries the protocol, so anything meant for the person who
            // started the server has to go to stderr.
            if policy.is_read_only() {
                eprintln!(
                    "xencode serving {} read-only over stdio; a call that would write or \
                     run a command is refused. Restart with --allow <tool> to permit one.",
                    root.display()
                );
            } else {
                eprintln!(
                    "xencode serving {} over stdio, permitting {} in addition to reads. \
                     A `path` or `cwd` that leaves that directory stays refused.",
                    root.display(),
                    allow.join(", ")
                );
                // The boundary is checked on the arguments, not inside a shell
                // command, so an allowed `run_command` reaches wherever this
                // user's shell can — and nobody is on the pipe to approve it.
                if allow.iter().any(|name| name == "run_command") {
                    eprintln!(
                        "warning: `run_command` is permitted. A permitted command runs \
                         exactly as the caller wrote it, so it can touch files outside \
                         {}; that is a shell, not a jailed one, and there is no approval \
                         prompt on a pipe.",
                        root.display()
                    );
                }
            }
            mcp_serve::serve(
                root,
                env!("CARGO_PKG_VERSION"),
                policy,
                new_task_runtime(),
                team,
            )
            .await
        }
    }
}

pub(crate) fn run_plugin_action(action: PluginAction) -> Result<(), String> {
    let plugin_dir = default_plugin_dir();
    let version = env!("CARGO_PKG_VERSION");

    match action {
        PluginAction::List => {
            // Not a listing of the plugin directory: this is the same load the
            // TUI performs at startup, so "installed" and "took hold" cannot
            // drift apart.
            let runtime = PluginRuntime::load(&plugin_dir, version);
            if runtime.reports().is_empty() {
                println!("No plugins installed in: {}", plugin_dir.display());
                println!("Use 'xencode plugin install <git-url|path>' to install a plugin.");
            } else {
                println!(
                    "📦 Plugins in {} (xencode {version}):",
                    plugin_dir.display()
                );
                for report in runtime.reports() {
                    print!("{}", render_declaration(report));
                }
                println!(
                    "  {} of {} loaded — a loaded plugin's prompt prefix and hooks apply to every agent turn.",
                    runtime.loaded_count(),
                    runtime.reports().len()
                );
            }
        }
        PluginAction::Install { source, rev } => {
            // Clone or read, verify the manifest and show what it declares —
            // before a single file lands where the loader will scan it.
            let pending = if xencode_plugin_rs::is_git_source(&source) {
                xencode_plugin_rs::prepare_git(&plugin_dir, &source, rev.as_deref(), version)
            } else {
                xencode_plugin_rs::prepare_path(&plugin_dir, std::path::Path::new(&source), version)
            }
            .map_err(|e| e.to_string())?;
            println!("{}", pending.declaration());
            let outcome = pending.apply().map_err(|e| e.to_string())?;
            println!(
                "\n✅ Installed {} v{} into {}.",
                outcome.name,
                outcome.version,
                outcome.destination.display()
            );
            match (
                outcome.source.url.as_deref(),
                outcome.source.commit.as_deref(),
            ) {
                // An install that named no commit still pinned one: say which,
                // so what is installed can be named exactly later.
                (Some(_), Some(commit)) => println!(
                    "   Pinned to commit {shown}{rest}. Nothing more is fetched until \
                     `xencode plugin update {name}` is run.",
                    shown = format_commit(commit),
                    rest = match outcome.source.git_ref.as_deref() {
                        Some(rev) if rev != commit => format!(" (installed from {rev})"),
                        _ => String::new(),
                    },
                    name = outcome.name
                ),
                _ => println!(
                    "   Copied from a local path, so no commit is pinned. A later install of the \
                     same name asks you to remove it first."
                ),
            }
        }
        PluginAction::Update { name, rev, yes } => {
            let plan =
                xencode_plugin_rs::prepare_update(&plugin_dir, &name, rev.as_deref(), version)
                    .map_err(|e| e.to_string())?;
            println!(
                "{}: {} from {} at {}",
                if plan.current {
                    "Already up to date"
                } else {
                    "Compared"
                },
                name,
                plan.url,
                format_commit(&plan.to_commit)
            );
            if plan.current {
                println!(
                    "   Installed copy is {} at {} and declares the same thing.",
                    plan.from_version,
                    format_commit(&plan.to_commit)
                );
                return Ok(());
            }
            for change in &plan.changes {
                println!("   · {change}");
            }
            if !plan.diff.is_empty() {
                println!("   Its manifest differs:");
                for line in plan.diff.lines() {
                    println!("   {line}");
                }
            }
            if plan.reaches_agent && !yes {
                println!(
                    "   NOT APPLIED: {} still v{} at {}. The version above changes what this \
                     plugin puts in front of the agent on every turn, so read the diff and \
                     re-run with --yes to install it.",
                    name,
                    plan.from_version,
                    format_commit(plan.from.commit.as_deref().unwrap_or("no recorded commit"))
                );
                return Ok(());
            }
            let (from, to) = (plan.from_version.clone(), plan.to_version.clone());
            let source = plan.apply().map_err(|e| e.to_string())?;
            println!(
                "   ✅ Applied: v{from} → v{to}, now pinned to {}. Run `/plugin reload` in a \
                 running xencode, or start a new session, for it to take hold.",
                format_commit(source.commit.as_deref().unwrap_or("no recorded commit")),
            );
        }
        PluginAction::Remove { name } => {
            // The name comes from the command line, so it passes the same guard
            // a manifest name does: one normal path component, nothing that can
            // walk out of the plugin directory.
            let path = PluginRegistry::new(plugin_dir.clone())
                .plugin_path(&name)
                .ok_or_else(|| format!("Invalid plugin name: {name}"))?;
            if !path.exists() {
                return Err(format!("Plugin '{name}' not found"));
            }
            // Say what is being taken away, so a removal can be checked against
            // the intention — including which commit the copy came from.
            let runtime = PluginRuntime::load(&plugin_dir, version);
            let contributions: String = runtime
                .reports()
                .iter()
                .filter(|report| report.name == name)
                .map(render_declaration)
                .collect();
            std::fs::remove_dir_all(&path).map_err(|e| format!("Failed to remove: {}", e))?;
            println!("✅ Plugin '{name}' removed.");
            let lines: Vec<&str> = contributions
                .lines()
                .map(str::trim_start)
                .filter(|line| !line.is_empty())
                .collect();
            if !lines.is_empty() {
                println!("   It was contributing:");
                for line in lines {
                    println!("     {line}");
                }
                println!(
                    "   A session already running still has the old prompt until /plugin reload."
                );
            }
        }
    }
    Ok(())
}
