//! A smaller tool menu for a turn that reads as a bugfix (TS-1).
//!
//! Prompt nudges were tried first and did not move the eval pass rate: a small
//! model still had to pick its call from the whole menu. This module acts on the
//! menu itself instead. It is deliberately pure — one prompt, one tool list, one
//! flag — so the decision can be read and tested without an agent loop, a model
//! or a workspace.
//!
//! The reading is [`xencode_context_rs::shape_of`], the same whole-word rule the
//! retrieval weights and the model profiles use. Nothing is classified twice
//! and nothing is guessed: a prompt that says something is broken is a bugfix,
//! and everything else — a rename, a question, a large edit — is `general`.
//!
//! The filter is a denylist, not an allowlist. Off-task tools are named by the
//! families that take a small model away from a local edit (background jobs,
//! the web, off-machine search and skills); every other tool is left exactly as
//! the caller offered it. That matters for composition: the caller has already
//! applied the index, approval-mode and reproduction-gate filters, and a tool
//! the current mode needs — `update_plan` in a plan, `ask_user` in a task that
//! pauses for a person — is never one of the named families, so it survives.

use xencode_context_rs::{shape_of, TaskShape};
use xencode_providers_rs::ToolDefinition;

/// Tools a local edit needs, and which must survive a focused bugfix menu.
/// `search_files` is the one `search_*` name that is kept: it finds the file to
/// edit, while a question sent to a configured search engine does not.
pub const ESSENTIAL_TOOLS: &[&str] = &[
    "read_file",
    "list_dir",
    "search_files",
    "write_file",
    "edit_file",
    "run_command",
];

/// Tools a mode or the turn itself may require. The filter is a denylist, so
/// these are preserved without a special case; they are named here so a reader
/// can see that the keep set is not limited to file editing.
pub const MODE_TOOLS: &[&str] = &["update_plan", "ask_user"];

/// Whether a tool belongs to a family that is off-task for a bugfix.
///
/// `load_skill` is the real name of the skill tool (`skill_tools()` in
/// xencode-providers-rs); the `skill_` prefix is kept as well so an MCP server
/// that publishes its own skill-named tools is treated the same way.
fn off_task(name: &str) -> bool {
    name.starts_with("background_")
        || name.starts_with("web_")
        || (name.starts_with("search_") && name != "search_files")
        || name.starts_with("skill_")
        || name == "load_skill"
}

/// Tools that change an existing file by reproducing its text. In structured
/// edit mode these are replaced by `patch_lines`, which validates a short
/// expected fragment against a real line range before writing.
pub const WHOLE_BLOCK_EDIT_TOOLS: &[&str] = &[
    "edit_file",
    "edit_symbol",
    "ast_edit",
    "codemod",
    "rename",
    "write_file",
];

/// Offer only the validated line-range patch as the edit path when `enabled`.
/// With the flag off the menu is returned unchanged, so no existing install
/// loses a tool. With it on, whole-block edit tools are removed and
/// `patch_lines` is appended, making a short, validated patch the one way to
/// change an existing file. Read-only and command tools are left alone.
pub fn require_structured_edits(tools: Vec<ToolDefinition>, enabled: bool) -> Vec<ToolDefinition> {
    if !enabled {
        return tools;
    }
    let mut kept: Vec<ToolDefinition> = tools
        .into_iter()
        .filter(|def| !WHOLE_BLOCK_EDIT_TOOLS.contains(&def.name.as_str()))
        .collect();
    if !kept.iter().any(|def| def.name == "patch_lines") {
        let patch = xencode_providers_rs::file_tools()
            .into_iter()
            .find(|def| def.name == "patch_lines")
            .expect("file_tools publishes patch_lines");
        kept.push(patch);
    }
    kept
}

/// Narrow `tools` for a bugfix prompt when `enabled` is on.
///
/// With the flag off, or for a `general` prompt, the menu is returned unchanged:
/// an existing install sees exactly the tools it saw before, and no call the
/// executor can dispatch becomes unreachable from the normal menu. With the flag
/// on and the prompt reading as a bugfix, only the off-task families above are
/// removed; the file and command tools and every mode tool stay.
pub fn focus_tools(prompt: &str, tools: Vec<ToolDefinition>, enabled: bool) -> Vec<ToolDefinition> {
    if !enabled || shape_of(prompt).shape != TaskShape::Bugfix {
        return tools;
    }
    tools
        .into_iter()
        .filter(|def| !off_task(&def.name))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real tool constructors, in the same order the agent loop builds them.
    /// No stand-in definitions: if a constructor gains a tool, these tests see it.
    fn full_menu() -> Vec<ToolDefinition> {
        let mut tools = xencode_providers_rs::background_tools();
        tools.extend(xencode_providers_rs::advise_tools());
        tools.extend(xencode_providers_rs::file_tools());
        tools.extend(xencode_providers_rs::command_tools());
        tools.extend(xencode_providers_rs::debug_tools());
        tools.extend(xencode_providers_rs::plan_tools());
        tools.extend(xencode_providers_rs::repro_tools());
        tools.extend(xencode_providers_rs::web_tools());
        tools.extend(xencode_providers_rs::search_tools());
        tools.extend(xencode_providers_rs::skill_tools());
        tools.push(xencode_providers_rs::ask_user_tool());
        tools
    }

    fn names(tools: &[ToolDefinition]) -> Vec<String> {
        tools.iter().map(|def| def.name.clone()).collect()
    }

    fn has(names: &[String], name: &str) -> bool {
        names.iter().any(|n| n == name)
    }

    #[test]
    fn a_bugfix_drops_only_the_off_task_families() {
        let before = full_menu();
        let before_names = names(&before);
        let after = focus_tools("the count is wrong, please fix it", before, true);
        let after_names = names(&after);

        let removed: Vec<String> = before_names
            .iter()
            .filter(|name| !has(&after_names, name.as_str()))
            .cloned()
            .collect();
        let expected_removed: Vec<String> = [
            "background_start",
            "background_poll",
            "background_stop",
            "web_fetch",
            "web_search",
            "load_skill",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(removed, expected_removed, "only off-task tools drop");
        // No tool is invented, and the order of the surviving menu is unchanged.
        let kept: Vec<String> = before_names
            .iter()
            .filter(|name| has(&after_names, name.as_str()))
            .cloned()
            .collect();
        assert_eq!(after_names, kept);
        for essential in ESSENTIAL_TOOLS {
            assert!(
                has(&after_names, essential),
                "{essential} is needed to find and edit the file: {after_names:?}"
            );
        }
        for mode in MODE_TOOLS {
            assert!(
                has(&after_names, mode),
                "{mode} is needed by the current mode: {after_names:?}"
            );
        }
    }

    #[test]
    fn search_files_survives_the_search_filter() {
        let before = full_menu();
        assert!(
            has(&names(&before), "search_files"),
            "the real file tools must offer search_files"
        );
        let after = focus_tools("why does the cache break on a cold start", before, true);
        assert!(has(&names(&after), "search_files"));
        assert!(!has(&names(&after), "web_search"));
    }

    #[test]
    fn a_general_prompt_leaves_the_menu_untouched() {
        for prompt in [
            "rename the caps struct and its callers",
            "what does this file do",
            "add a flag that asks the server for a count",
        ] {
            assert_eq!(shape_of(prompt).shape, TaskShape::General, "{prompt:?}");
            let before = full_menu();
            let expected = names(&before);
            let after = focus_tools(prompt, before, true);
            assert_eq!(names(&after), expected, "{prompt:?}");
        }
    }

    #[test]
    fn focus_off_leaves_a_bugfix_menu_untouched() {
        let before = full_menu();
        let expected = names(&before);
        let after = focus_tools("the build fails on a cold start", before, false);
        assert_eq!(names(&after), expected);
    }

    #[test]
    fn every_dropped_tool_is_still_reachable_with_focus_off() {
        // A tool the executor dispatches must never become unreachable: the
        // default path keeps the whole menu, so the focus filter is the only
        // thing that can hide a name and it is reversible by turning focus off.
        let focused = focus_tools("fix the crash", full_menu(), true);
        let unfocused = focus_tools("fix the crash", full_menu(), false);
        let focused_names = names(&focused);
        for def in &unfocused {
            assert!(
                has(&focused_names, &def.name) || off_task(&def.name),
                "{} was neither kept nor a known off-task tool",
                def.name
            );
        }
        assert!(
            focused_names.len() < names(&unfocused).len(),
            "a focused bugfix menu should actually be smaller"
        );
    }

    #[test]
    fn structured_edits_leaves_only_the_validated_patch_as_an_edit_path() {
        let normal = require_structured_edits(full_menu(), false);
        assert_eq!(names(&normal), names(&full_menu()));

        let structured = require_structured_edits(full_menu(), true);
        let names = names(&structured);
        for tool in WHOLE_BLOCK_EDIT_TOOLS {
            assert!(!has(&names, tool), "{tool} was still offered");
        }
        assert!(has(&names, "patch_lines"), "the patch tool is offered");
        assert!(has(&names, "read_file"), "reading is still offered");
        assert!(has(&names, "run_command"), "commands are still offered");
    }
}
