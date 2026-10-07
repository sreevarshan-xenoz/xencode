# Print an optspec for argparse to handle cmd's options that are independent of any subcommand.
function __fish_xencode_global_optspecs
    string join \n dump-config h/help V/version
end

function __fish_xencode_needs_command
    # Figure out if the current invocation already has a command.
    set -l cmd (commandline -opc)
    set -e cmd[1]
    argparse -s (__fish_xencode_global_optspecs) -- $cmd 2>/dev/null
    or return
    if set -q argv[1]
        # Also print the command, so this can be used to figure out what it is.
        echo $argv[1]
        return 1
    end
    return 0
end

function __fish_xencode_using_subcommand
    set -l cmd (__fish_xencode_needs_command)
    test -z "$cmd"
    and return 1
    contains -- $cmd[1] $argv
end

complete -c xencode -n "__fish_xencode_needs_command" -l dump-config -d 'Dump the engine composition and resolved configuration as JSON (AF-3)'
complete -c xencode -n "__fish_xencode_needs_command" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_needs_command" -s V -l version -d 'Print version'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "scan" -d 'Scan a workspace and list all entries'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "config" -d 'Configuration management'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "models" -d 'Local model management (Ollama & llama.cpp)'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "cache" -d 'Response cache management'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "audit" -d 'The session server\'s audit log'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "advisories" -d 'Known security advisories for the crates this project depends on'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "deps" -d 'Supply-chain report: shell to the installed dependency checkers (cargo-shear, cargo-deny) and stream their findings. Report only — it never edits a manifest or auto-fixes a dependency'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "query" -d 'Send a query to a model'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "memory" -d 'Manage conversation memory'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "tasks" -d 'Manage background tasks (file-backed, survives this process)'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "worktree" -d 'Manage git worktrees of the current repository'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "colab" -d 'Google Colab bridge: preflight, then up / status / down for a model server running on a Colab VM'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "remote" -d 'Manage remote inference hosts reached over SSH: add, list, use, forget'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "computers" -d 'Manage and inspect registered computer backends (AF-4)'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "compete" -d 'Compete candidate implementations on isolated branches, verify each, and let a person pick one (AF-5)'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "merge" -d 'Evaluate merge conflicts with git merge-tree and land branches under a human approval gate (OR-5)'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "advise" -d 'Repository insights from the .xencode snapshot: broken imports, import cycles, hub files and orphans'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "server" -d 'Start the collaboration server'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "analyze" -d 'Analyze code for issues and vulnerabilities'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "fetch" -d 'Fetch a web page and extract research-ready text'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "interop" -d 'Measure what the coding-agent CLIs on this machine actually do'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "anchor" -d 'Find this repository\'s build and test commands, run them, and record only the ones that actually worked'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "toolchain" -d 'Run the project\'s own toolchain checks, and report structured evidence'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "doctor" -d 'Write one bug report: configuration, secrets, disk, providers, models, MCP servers and the Colab bridge. Flags narrow it to one part'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "session" -d 'Name sessions, resolve them, and export redacted transcripts'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "paths" -d 'Where xencode keeps its own files: settings, session records, cache and downloaded models, and whether they are still in `~/.xencode`'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "migrate" -d 'Move the files in `~/.xencode` to the four directories they belong in. Nothing is overwritten and the old directory is only removed once empty'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "verify" -d 'Run the machine-checkable checklist: test, lint, fmt — each verified, none graded'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "envcheck" -d 'Report environment keys read in code against the templates that document them'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "agents" -d 'List installed agents with versions and install provenance, inspect worker health (AR-8), or manage continuation packages (AR-7)'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "hotspots" -d 'Rank files by churn times size with bus factor and owners'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "impact" -d 'What a change to one file affects: the crates that depend on its crate, the files that link it, and the files its history is coupled to'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "removal" -d 'What deleting one file would cost: the links it holds up, and the modules that become dead code the moment it is taken out'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "generate" -d 'Print shell completions or the man page; both are generated from the clap definition, never written by hand'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "mutants" -d 'Find code whose tests cannot tell right from wrong'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "cov" -d 'Report which lines this diff added were never executed'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "perf" -d 'Measure the hot paths against a stored baseline, and refuse the verdict when the run is too noisy to support one'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "prices" -d 'Where the prices a cost report uses come from, and reading them again'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "test" -d 'Run the tests, and never call a test that only passed on retry a pass'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "release-notes" -d 'Draft the release notes from the commits since the last release and the changelog block this project keeps, and report where the two disagree'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "review" -d 'Review the diff between a base branch and HEAD, file by file'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "replay" -d 'Run a recorded session again from the bytes it was made of'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "runs" -d 'Which runs happened, what each asked a person, and the commit trailer naming it. Reads `.xencode/cache/runs.jsonl` only, so it works with every model server down'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "run" -d 'Run an agent turn from the command line, in the foreground or detached so it survives the terminal. A detached run persists every completed round under `.xencode/cache/detached/<run-id>/`, so a kill is resumed with `--resume` instead of restarted, and stops on round, wall-clock and cost caps as well as the model finishing'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "eval" -d 'Score the agent on defects that were seeded on purpose'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "plugin" -d 'Manage plugins'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "mcp" -d 'Let another program drive xencode\'s tools over Model Context Protocol'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "llamacpp" -d 'llama.cpp server management (status/start/stop/load/unload)'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "hw" -d 'What this machine can serve, read from the machine'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "history" -d 'How fast this repository\'s history is to ask about, and how to speed it up'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "bootstrap" -d 'Write the files a project xencode has never seen is missing'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "tui" -d 'Launch the Terminal User Interface'
complete -c xencode -n "__fish_xencode_needs_command" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand scan" -l max-depth -d 'Maximum directory depth to traverse' -r
complete -c xencode -n "__fish_xencode_using_subcommand scan" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand scan" -l hidden -d 'Include hidden files and directories'
complete -c xencode -n "__fish_xencode_using_subcommand scan" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand config; and not __fish_seen_subcommand_from show dump set reset help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand config; and not __fish_seen_subcommand_from show dump set reset help" -f -a "show" -d 'Display current configuration'
complete -c xencode -n "__fish_xencode_using_subcommand config; and not __fish_seen_subcommand_from show dump set reset help" -f -a "dump" -d 'Dump the engine composition and resolved configuration as JSON (AF-3)'
complete -c xencode -n "__fish_xencode_using_subcommand config; and not __fish_seen_subcommand_from show dump set reset help" -f -a "set" -d 'Set a configuration value'
complete -c xencode -n "__fish_xencode_using_subcommand config; and not __fish_seen_subcommand_from show dump set reset help" -f -a "reset" -d 'Reset configuration to defaults'
complete -c xencode -n "__fish_xencode_using_subcommand config; and not __fish_seen_subcommand_from show dump set reset help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from show" -l composition -d 'Include full engine composition summary'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from show" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from dump" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from set" -l dry-run -d 'Validate and report the change without writing config.json'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from set" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from reset" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from help" -f -a "show" -d 'Display current configuration'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from help" -f -a "dump" -d 'Dump the engine composition and resolved configuration as JSON (AF-3)'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from help" -f -a "set" -d 'Set a configuration value'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from help" -f -a "reset" -d 'Reset configuration to defaults'
complete -c xencode -n "__fish_xencode_using_subcommand config; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand models; and not __fish_seen_subcommand_from list health default advice help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand models; and not __fish_seen_subcommand_from list health default advice help" -f -a "list" -d 'List all installed Ollama models'
complete -c xencode -n "__fish_xencode_using_subcommand models; and not __fish_seen_subcommand_from list health default advice help" -f -a "health" -d 'Check health of a specific model'
complete -c xencode -n "__fish_xencode_using_subcommand models; and not __fish_seen_subcommand_from list health default advice help" -f -a "default" -d 'Show the smart-selected default model'
complete -c xencode -n "__fish_xencode_using_subcommand models; and not __fish_seen_subcommand_from list health default advice help" -f -a "advice" -d 'Say which GGUF this machine can serve, from the dated advice table, with the pinned address and checksum to fetch it by'
complete -c xencode -n "__fish_xencode_using_subcommand models; and not __fish_seen_subcommand_from list health default advice help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand models; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand models; and __fish_seen_subcommand_from health" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand models; and __fish_seen_subcommand_from default" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand models; and __fish_seen_subcommand_from advice" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand models; and __fish_seen_subcommand_from help" -f -a "list" -d 'List all installed Ollama models'
complete -c xencode -n "__fish_xencode_using_subcommand models; and __fish_seen_subcommand_from help" -f -a "health" -d 'Check health of a specific model'
complete -c xencode -n "__fish_xencode_using_subcommand models; and __fish_seen_subcommand_from help" -f -a "default" -d 'Show the smart-selected default model'
complete -c xencode -n "__fish_xencode_using_subcommand models; and __fish_seen_subcommand_from help" -f -a "advice" -d 'Say which GGUF this machine can serve, from the dated advice table, with the pinned address and checksum to fetch it by'
complete -c xencode -n "__fish_xencode_using_subcommand models; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and not __fish_seen_subcommand_from stats clear gc help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and not __fish_seen_subcommand_from stats clear gc help" -f -a "stats" -d 'Show cache statistics'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and not __fish_seen_subcommand_from stats clear gc help" -f -a "clear" -d 'Clear all cached responses'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and not __fish_seen_subcommand_from stats clear gc help" -f -a "gc" -d 'Drop the oldest cached responses until the cache directory fits under a size. The downloaded advisory corpora are not counted and cannot be removed by this command'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and not __fish_seen_subcommand_from stats clear gc help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and __fish_seen_subcommand_from stats" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and __fish_seen_subcommand_from clear" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and __fish_seen_subcommand_from gc" -l max-mb -d 'Largest the cached responses may be, in megabytes' -r
complete -c xencode -n "__fish_xencode_using_subcommand cache; and __fish_seen_subcommand_from gc" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and __fish_seen_subcommand_from help" -f -a "stats" -d 'Show cache statistics'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and __fish_seen_subcommand_from help" -f -a "clear" -d 'Clear all cached responses'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and __fish_seen_subcommand_from help" -f -a "gc" -d 'Drop the oldest cached responses until the cache directory fits under a size. The downloaded advisory corpora are not counted and cannot be removed by this command'
complete -c xencode -n "__fish_xencode_using_subcommand cache; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand audit; and not __fish_seen_subcommand_from verify help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand audit; and not __fish_seen_subcommand_from verify help" -f -a "verify" -d 'Check an audit log for records that were changed after they were written'
complete -c xencode -n "__fish_xencode_using_subcommand audit; and not __fish_seen_subcommand_from verify help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand audit; and __fish_seen_subcommand_from verify" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand audit; and __fish_seen_subcommand_from help" -f -a "verify" -d 'Check an audit log for records that were changed after they were written'
complete -c xencode -n "__fish_xencode_using_subcommand audit; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and not __fish_seen_subcommand_from sync show check status help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and not __fish_seen_subcommand_from sync show check status help" -f -a "sync" -d 'Download both corpora: 6.3 MB of RustSec text plus a 3.5 MB OSV archive, about 20 MB unpacked'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and not __fish_seen_subcommand_from sync show check status help" -f -a "show" -d 'What the corpus says about one crate, judged against a version when given'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and not __fish_seen_subcommand_from sync show check status help" -f -a "check" -d 'Judge every package in this project\'s Cargo.lock against the corpus'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and not __fish_seen_subcommand_from sync show check status help" -f -a "status" -d 'Whether a corpus exists here, how big it is, and when it was taken'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and not __fish_seen_subcommand_from sync show check status help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from sync" -l dir -d 'Where to keep them (default: <config dir>/advisories)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from sync" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from show" -l version -d 'Version to judge; without it the records are listed but not assessed' -r
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from show" -l dir -d 'Corpus location (default: <config dir>/advisories)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from show" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from check" -l path -d 'Project to read Cargo.lock from (default: the current directory)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from check" -l dir -d 'Corpus location (default: <config dir>/advisories)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from check" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from status" -l dir -d 'Corpus location (default: <config dir>/advisories)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from status" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from help" -f -a "sync" -d 'Download both corpora: 6.3 MB of RustSec text plus a 3.5 MB OSV archive, about 20 MB unpacked'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from help" -f -a "show" -d 'What the corpus says about one crate, judged against a version when given'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from help" -f -a "check" -d 'Judge every package in this project\'s Cargo.lock against the corpus'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from help" -f -a "status" -d 'Whether a corpus exists here, how big it is, and when it was taken'
complete -c xencode -n "__fish_xencode_using_subcommand advisories; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand deps" -l path -d 'Project to check (default: the current directory)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand deps" -l format -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand deps" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand query" -s m -l model -d 'Model to use (overrides config default)' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l session -d 'Session ID to attach to' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l temperature -d 'llama.cpp sampling: temperature (e.g. 0.7)' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l top-k -d 'llama.cpp sampling: top-k' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l min-p -d 'llama.cpp sampling: min-p (e.g. 0.05)' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l mirostat -d 'llama.cpp sampling: mirostat mode (0, 1, or 2)' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l seed -d 'llama.cpp sampling: seed, for output that can be produced again' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l max-tokens -d 'llama.cpp sampling: max generated tokens' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l grammar -d 'llama.cpp sampling: GBNF grammar file/string' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l json-schema -d 'llama.cpp sampling: JSON schema for structured output' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l image -d 'Image to send with the prompt (repeatable); needs a vision-capable model' -r
complete -c xencode -n "__fish_xencode_using_subcommand query" -l format -d 'How to write the answer: plain words, or one JSON event per line' -r -f -a "text\t''
ndjson\t''"
complete -c xencode -n "__fish_xencode_using_subcommand query" -l no-cache -d 'Do not use cached responses'
complete -c xencode -n "__fish_xencode_using_subcommand query" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and not __fish_seen_subcommand_from list show fork prune gc evidence help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and not __fish_seen_subcommand_from list show fork prune gc evidence help" -f -a "list" -d 'List all conversation sessions'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and not __fish_seen_subcommand_from list show fork prune gc evidence help" -f -a "show" -d 'Show transcript of a session'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and not __fish_seen_subcommand_from list show fork prune gc evidence help" -f -a "fork" -d 'Fork a conversation session into a child holding an exact event prefix'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and not __fish_seen_subcommand_from list show fork prune gc evidence help" -f -a "prune" -d 'Delete conversation sessions that have no messages'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and not __fish_seen_subcommand_from list show fork prune gc evidence help" -f -a "gc" -d 'List the durable facts this repository contradicts, with how long each has been contradicted for. `--apply` retires the ones past a year'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and not __fish_seen_subcommand_from list show fork prune gc evidence help" -f -a "evidence" -d 'How many revisions each durable fact has been re-checked against, and what that evidence supports saying. Prints an interval, never a confidence'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and not __fish_seen_subcommand_from list show fork prune gc evidence help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from list" -l all -d 'Show all sessions, including empty ones (0 messages)'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from show" -l first -d 'Print only the first message recorded in the session\'s event log'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from show" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from fork" -l as-id -d 'New session ID (defaults to <session>_fork_<timestamp>)' -r
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from fork" -l prefix -d 'Prefix length: number of parent events to inherit (defaults to all)' -r
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from fork" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from prune" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from gc" -l apply -d 'Remove the facts contradicted for twelve months or longer from `.xencode/state.md`'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from gc" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from evidence" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from evidence" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "list" -d 'List all conversation sessions'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "show" -d 'Show transcript of a session'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "fork" -d 'Fork a conversation session into a child holding an exact event prefix'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "prune" -d 'Delete conversation sessions that have no messages'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "gc" -d 'List the durable facts this repository contradicts, with how long each has been contradicted for. `--apply` retires the ones past a year'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "evidence" -d 'How many revisions each durable fact has been re-checked against, and what that evidence supports saying. Prints an interval, never a confidence'
complete -c xencode -n "__fish_xencode_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and not __fish_seen_subcommand_from list start poll stop rm help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and not __fish_seen_subcommand_from list start poll stop rm help" -f -a "list" -d 'List known tasks with their derived status'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and not __fish_seen_subcommand_from list start poll stop rm help" -f -a "start" -d 'Start a background task (survives this CLI process)'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and not __fish_seen_subcommand_from list start poll stop rm help" -f -a "poll" -d 'Show a task\'s status and captured output'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and not __fish_seen_subcommand_from list start poll stop rm help" -f -a "stop" -d 'Ask a running task to stop'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and not __fish_seen_subcommand_from list start poll stop rm help" -f -a "rm" -d 'Forget a finished task and delete its files'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and not __fish_seen_subcommand_from list start poll stop rm help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from list" -l json -d 'Emit JSON instead of a table'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from start" -s n -l name -d 'Human-friendly label (defaults to the command)' -r
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from start" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from poll" -l lines -d 'Trailing output lines to print' -r
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from poll" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from stop" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from rm" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from help" -f -a "list" -d 'List known tasks with their derived status'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from help" -f -a "start" -d 'Start a background task (survives this CLI process)'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from help" -f -a "poll" -d 'Show a task\'s status and captured output'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from help" -f -a "stop" -d 'Ask a running task to stop'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from help" -f -a "rm" -d 'Forget a finished task and delete its files'
complete -c xencode -n "__fish_xencode_using_subcommand tasks; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and not __fish_seen_subcommand_from list add remove help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and not __fish_seen_subcommand_from list add remove help" -f -a "list" -d 'List git worktrees of the current repository'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and not __fish_seen_subcommand_from list add remove help" -f -a "add" -d 'Create a worktree at <path>, checking out <branch> (or a new branch named after the directory when omitted)'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and not __fish_seen_subcommand_from list add remove help" -f -a "remove" -d 'Remove a worktree (git refuses dirty worktrees; main never removable)'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and not __fish_seen_subcommand_from list add remove help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and __fish_seen_subcommand_from add" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and __fish_seen_subcommand_from remove" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and __fish_seen_subcommand_from help" -f -a "list" -d 'List git worktrees of the current repository'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and __fish_seen_subcommand_from help" -f -a "add" -d 'Create a worktree at <path>, checking out <branch> (or a new branch named after the directory when omitted)'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and __fish_seen_subcommand_from help" -f -a "remove" -d 'Remove a worktree (git refuses dirty worktrees; main never removable)'
complete -c xencode -n "__fish_xencode_using_subcommand worktree; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and not __fish_seen_subcommand_from preflight up status down help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and not __fish_seen_subcommand_from preflight up status down help" -f -a "preflight" -d 'Verify the google-colab-cli bridge is usable before bringing a VM up'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and not __fish_seen_subcommand_from preflight up status down help" -f -a "up" -d 'Bring a Colab VM up: create the session, install the runtime, and hold an SSH forward so the VM\'s OpenAI endpoint appears on the laptop'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and not __fish_seen_subcommand_from preflight up status down help" -f -a "status" -d 'Report the Colab bridge state: forward pid, `colab sessions`, and a /v1/models probe on the forward'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and not __fish_seen_subcommand_from preflight up status down help" -f -a "down" -d 'Tear the Colab bridge down: kill the forward, `colab stop`, clear state'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and not __fish_seen_subcommand_from preflight up status down help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from preflight" -l generate-key -d 'Generate the ed25519 key pair into the xencode config dir if absent'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from preflight" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -l session -d 'Session name (defaults to config colab.session)' -r
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -l gpu -d 'GPU accelerator (T4, L4, G4, H100, A100) for colab new' -r
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -l runtime -d 'Inference runtime on the VM: llama.cpp (pinned llama-server + GGUF, heavier install, one-shot) or ollama (tags flow into the model picker)' -r
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -l model -d 'Model repo/tag installed on the VM (defaults to config colab.model)' -r
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -l weights -d 'Weights source: hf (llama.cpp only; drive/gcs are refused with a fix)' -r
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -l quant -d 'GGUF quantization fragment to serve (llama.cpp only, e.g. Q4_K_M; defaults to config colab.quant)' -r
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -l local-port -d 'Local port the SSH forward exposes (defaults to config / 18000)' -r
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -l remote-port -d 'VM-side port the runtime binds (0 = runtime-native: llama.cpp 18080, ollama 11434; defaults to config)' -r
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -l reconnect -d 'Rebuild a broken bridge from the recorded colab.json (re-spawn the forward, or re-create the VM if it was reaped) instead of a full up'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -l dry-run -d 'Report what a successful bring-up would start and would write to config.json, without touching the VM, the bridge or the config'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from up" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from status" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from down" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from help" -f -a "preflight" -d 'Verify the google-colab-cli bridge is usable before bringing a VM up'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from help" -f -a "up" -d 'Bring a Colab VM up: create the session, install the runtime, and hold an SSH forward so the VM\'s OpenAI endpoint appears on the laptop'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from help" -f -a "status" -d 'Report the Colab bridge state: forward pid, `colab sessions`, and a /v1/models probe on the forward'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from help" -f -a "down" -d 'Tear the Colab bridge down: kill the forward, `colab stop`, clear state'
complete -c xencode -n "__fish_xencode_using_subcommand colab; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and not __fish_seen_subcommand_from add list use forget show help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and not __fish_seen_subcommand_from add list use forget show help" -f -a "add" -d 'Record a remote host profile'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and not __fish_seen_subcommand_from add list use forget show help" -f -a "list" -d 'List recorded remote host profiles'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and not __fish_seen_subcommand_from add list use forget show help" -f -a "use" -d 'Select the active remote host profile used by default'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and not __fish_seen_subcommand_from add list use forget show help" -f -a "forget" -d 'Remove a recorded remote host profile'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and not __fish_seen_subcommand_from add list use forget show help" -f -a "show" -d 'Show details of a remote host profile (or the active profile if omitted)'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and not __fish_seen_subcommand_from add list use forget show help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from add" -l runtime -d 'Inference runtime: llama.cpp or ollama' -r
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from add" -l model -d 'Model repo/tag to serve on the remote machine' -r
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from add" -l port -d 'SSH port (defaults to 22 or the port parsed from host)' -r
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from add" -l local-port -d 'Local port the SSH forward listens on (defaults to 18100)' -r
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from add" -l remote-port -d 'Remote port the inference runtime binds inside the machine (0 = runtime default)' -r
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from add" -l force -d 'Overwrite an existing profile with this name'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from add" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from use" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from forget" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from show" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from help" -f -a "add" -d 'Record a remote host profile'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from help" -f -a "list" -d 'List recorded remote host profiles'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from help" -f -a "use" -d 'Select the active remote host profile used by default'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from help" -f -a "forget" -d 'Remove a recorded remote host profile'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from help" -f -a "show" -d 'Show details of a remote host profile (or the active profile if omitted)'
complete -c xencode -n "__fish_xencode_using_subcommand remote; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and not __fish_seen_subcommand_from list show use probe help" -l json -d 'Output results as JSON'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and not __fish_seen_subcommand_from list show use probe help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and not __fish_seen_subcommand_from list show use probe help" -f -a "list" -d 'List all registered computer backends (default)'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and not __fish_seen_subcommand_from list show use probe help" -f -a "show" -d 'Show details of a specific computer backend'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and not __fish_seen_subcommand_from list show use probe help" -f -a "use" -d 'Set the active computer backend'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and not __fish_seen_subcommand_from list show use probe help" -f -a "probe" -d 'Probe connectivity to a computer backend'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and not __fish_seen_subcommand_from list show use probe help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from list" -l json -d 'Output results as JSON'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from show" -l json -d 'Output results as JSON'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from show" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from use" -l json -d 'Output results as JSON'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from use" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from probe" -l json -d 'Output results as JSON'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from probe" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from help" -f -a "list" -d 'List all registered computer backends (default)'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from help" -f -a "show" -d 'Show details of a specific computer backend'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from help" -f -a "use" -d 'Set the active computer backend'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from help" -f -a "probe" -d 'Probe connectivity to a computer backend'
complete -c xencode -n "__fish_xencode_using_subcommand computers; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and not __fish_seen_subcommand_from run list show pick help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and not __fish_seen_subcommand_from run list show pick help" -f -a "run" -d 'Build each candidate arm in its own worktree and branch, run the verification checklist on every one of them, and print the table'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and not __fish_seen_subcommand_from run list show pick help" -f -a "list" -d 'List recorded competing runs, newest first'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and not __fish_seen_subcommand_from run list show pick help" -f -a "show" -d 'Re-print the verification table of a recorded run, from its saved report'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and not __fish_seen_subcommand_from run list show pick help" -f -a "pick" -d 'Switch the repository onto one arm\'s branch, leaving every other candidate branch and all evidence files on disk untouched'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and not __fish_seen_subcommand_from run list show pick help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from run" -l arm -d 'A candidate arm as `id` or `id=label`. Give it twice for two arms, or three times for three. Omit it to compete `arm-a` against `arm-b`' -r
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from run" -l edit -d 'Write content into a file inside one arm\'s worktree: the arm id, the path relative to that worktree, then the file\'s full text. Repeat it once per file per arm' -r
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from run" -l command -d 'Run a shell command inside one arm\'s worktree after its edits: the arm id, then the command' -r
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from run" -l skip -d 'Skip a check by name (repeatable): test, lint, or fmt. A skipped check is reported as skipped, never as passed' -r
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from run" -l timeout -d 'Wall-clock ceiling in seconds for each check' -r
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from run" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from run" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from list" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from show" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from show" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from pick" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from help" -f -a "run" -d 'Build each candidate arm in its own worktree and branch, run the verification checklist on every one of them, and print the table'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from help" -f -a "list" -d 'List recorded competing runs, newest first'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from help" -f -a "show" -d 'Re-print the verification table of a recorded run, from its saved report'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from help" -f -a "pick" -d 'Switch the repository onto one arm\'s branch, leaving every other candidate branch and all evidence files on disk untouched'
complete -c xencode -n "__fish_xencode_using_subcommand compete; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and not __fish_seen_subcommand_from precheck plan land help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and not __fish_seen_subcommand_from precheck plan land help" -f -a "precheck" -d 'Speculatively precheck a candidate branch against a base branch using git merge-tree'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and not __fish_seen_subcommand_from precheck plan land help" -f -a "plan" -d 'Build a multi-branch merge plan with conflict prechecks and worker checks'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and not __fish_seen_subcommand_from precheck plan land help" -f -a "land" -d 'Land branches into base branch guarded by a named human decision'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and not __fish_seen_subcommand_from precheck plan land help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from precheck" -l base -d 'Target base branch' -r
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from precheck" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from precheck" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from plan" -l branch -d 'Candidate branches to evaluate' -r
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from plan" -l base -d 'Target base branch' -r
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from plan" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from plan" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from land" -l branch -d 'Candidate branches to merge' -r
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from land" -l base -d 'Target base branch' -r
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from land" -l approved-by -d 'Full name of the human approving the merge (required gate)' -r
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from land" -l test-cmd -d 'Post-integration test commands to re-run on the integrated tree' -r
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from land" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from land" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from help" -f -a "precheck" -d 'Speculatively precheck a candidate branch against a base branch using git merge-tree'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from help" -f -a "plan" -d 'Build a multi-branch merge plan with conflict prechecks and worker checks'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from help" -f -a "land" -d 'Land branches into base branch guarded by a named human decision'
complete -c xencode -n "__fish_xencode_using_subcommand merge; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand advise" -l limit -d 'Maximum findings to show (0 shows all)' -r
complete -c xencode -n "__fish_xencode_using_subcommand advise" -l json -d 'Machine-readable output'
complete -c xencode -n "__fish_xencode_using_subcommand advise" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand server" -l port -d 'Port to listen on' -r
complete -c xencode -n "__fish_xencode_using_subcommand server" -l host -d 'Address to bind (default: loopback only)' -r
complete -c xencode -n "__fish_xencode_using_subcommand server" -l cert -d 'TLS certificate in PEM form; requires --key' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand server" -l key -d 'TLS private key in PEM form; requires --cert' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand server" -l audit-path -d 'Audit log path; "none" disables (default: <state dir>/audit.jsonl)' -r
complete -c xencode -n "__fish_xencode_using_subcommand server" -l allow-insecure-public -d 'Allow a non-loopback bind without TLS — tokens and activity then travel in clear text; read the warning before reaching for this'
complete -c xencode -n "__fish_xencode_using_subcommand server" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand analyze" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand analyze" -l runtime -d 'Report async and concurrency hazards: blocking calls on the reactor thread, unbounded channels, and dropped task handles. Needs the `ast-grep` binary; without it the report says the check did not run rather than reporting nothing found'
complete -c xencode -n "__fish_xencode_using_subcommand analyze" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand fetch" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand fetch" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand interop" -l agent -d 'Probe only these agents (repeatable); default is every installed agent the operator has not stood down' -r
complete -c xencode -n "__fish_xencode_using_subcommand interop" -l timeout -d 'Per-agent wall-clock limit in seconds' -r
complete -c xencode -n "__fish_xencode_using_subcommand interop" -l out -d 'Where to write the JSON report; omit to print it' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand interop" -l format -d 'Output format for the printed summary' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand interop" -l repeat -d 'Run each agent this many times and compare. One run is a reading; two is a check, and a fact that differs between runs is reported rather than smoothed over' -r
complete -c xencode -n "__fish_xencode_using_subcommand interop" -l capture-dir -d 'Keep each agent\'s whole run on disk, in `<dir>/<agent>/capture/`: `raw.jsonl` (every line the vendor printed, unredacted), `normalized.jsonl` (the common events, each naming its raw line) and `metadata.json` (what was run and what it said it cost). Written `0600` and only when you ask. Costs no extra run — it keeps what the probe already received' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand interop" -l trace -d 'Print the trace view of a capture written earlier and probe nothing. Takes one capture (`../captures/opencode`) or the whole root (`../captures`) to render every vendor in the same view' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand interop" -l fan-out -d 'Run every selected agent at the same time instead of one after another, and report what the overlap saved. Cannot be combined with `--repeat`, which needs sequential runs to compare'
complete -c xencode -n "__fish_xencode_using_subcommand interop" -l check-auth -d 'Report, read-only, which agents look configured on this machine, and what to run if one is not. Starts no login and reads no credential'
complete -c xencode -n "__fish_xencode_using_subcommand interop" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c xencode -n "__fish_xencode_using_subcommand anchor" -l timeout -d 'Per-command wall-clock limit in seconds. A command that runs out of time is recorded as unverified, never as a pass' -r
complete -c xencode -n "__fish_xencode_using_subcommand anchor" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand anchor" -l dry-run -d 'List what was found without running anything'
complete -c xencode -n "__fish_xencode_using_subcommand anchor" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand toolchain" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand toolchain" -l allow-dirty -d 'Allow `fix` on a tree with uncommitted edits'
complete -c xencode -n "__fish_xencode_using_subcommand toolchain" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand doctor" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand doctor" -l env -d 'Machine environment facts'
complete -c xencode -n "__fish_xencode_using_subcommand doctor" -l deps -d 'Dependency health: outdated list plus advisory state'
complete -c xencode -n "__fish_xencode_using_subcommand doctor" -l selfcheck -d 'Self-debug slice: index, git, providers, MCP servers, metrics, cache'
complete -c xencode -n "__fish_xencode_using_subcommand doctor" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand session; and not __fish_seen_subcommand_from name resolve export help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand session; and not __fish_seen_subcommand_from name resolve export help" -f -a "name" -d 'Name a run so it can be resumed without its id'
complete -c xencode -n "__fish_xencode_using_subcommand session; and not __fish_seen_subcommand_from name resolve export help" -f -a "resolve" -d 'Resolve a name, id prefix, or `latest` to a full run id'
complete -c xencode -n "__fish_xencode_using_subcommand session; and not __fish_seen_subcommand_from name resolve export help" -f -a "export" -d 'Print a session\'s transcript; `--redacted` scrubs secrets'
complete -c xencode -n "__fish_xencode_using_subcommand session; and not __fish_seen_subcommand_from name resolve export help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand session; and __fish_seen_subcommand_from name" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand session; and __fish_seen_subcommand_from resolve" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand session; and __fish_seen_subcommand_from export" -l redacted -d 'Scrub secrets with the trace module\'s patterns'
complete -c xencode -n "__fish_xencode_using_subcommand session; and __fish_seen_subcommand_from export" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand session; and __fish_seen_subcommand_from help" -f -a "name" -d 'Name a run so it can be resumed without its id'
complete -c xencode -n "__fish_xencode_using_subcommand session; and __fish_seen_subcommand_from help" -f -a "resolve" -d 'Resolve a name, id prefix, or `latest` to a full run id'
complete -c xencode -n "__fish_xencode_using_subcommand session; and __fish_seen_subcommand_from help" -f -a "export" -d 'Print a session\'s transcript; `--redacted` scrubs secrets'
complete -c xencode -n "__fish_xencode_using_subcommand session; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand paths" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand paths" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand migrate" -l dry-run -d 'Print what would move and change nothing'
complete -c xencode -n "__fish_xencode_using_subcommand migrate" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand verify" -l skip -d 'Skip these checks (repeatable); skipped is reported, never passed' -r
complete -c xencode -n "__fish_xencode_using_subcommand verify" -l timeout -d 'Wall-clock ceiling in seconds for the test slot' -r
complete -c xencode -n "__fish_xencode_using_subcommand verify" -l session -d 'Session ID to file checks under in the verification ledger (defaults to active session or \'cli\')' -r
complete -c xencode -n "__fish_xencode_using_subcommand verify" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand verify" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand envcheck" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand envcheck" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l agent -d 'Specific agent to inspect or resume with (e.g. claude, agy, cursor-agent)' -r
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l build-package -d 'Build a worker continuation package from workspace diff and test runs (AR-7)' -r
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l package -d 'Path to inspect or load a worker continuation package (AR-7)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l test-cmd -d 'Test commands to execute for package verification (defaults to \'cargo test\')' -r
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l route -d 'Route a task based strictly on probed capabilities, load, and cost ceiling (OR-6)' -r
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l require-cap -d 'Capabilities required for the routed task (e.g. stream, acp, mcp, resume, daemon, approval) (OR-6)' -r
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l max-cost -d 'Maximum cost ceiling allowed for the routed task (OR-6)' -r
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l contract -d 'Verify each roster claim against the agent\'s live --help'
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l health -d 'Report worker health (installed, version, authenticated, responsive, rate-limited) (AR-8)'
complete -c xencode -n "__fish_xencode_using_subcommand agents" -l resume -d 'Resume a task from a continuation package using the specified agent (AR-7)'
complete -c xencode -n "__fish_xencode_using_subcommand agents" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand hotspots" -l limit -d 'How many files to list' -r
complete -c xencode -n "__fish_xencode_using_subcommand hotspots" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand hotspots" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand impact" -l limit -d 'How many entries to list in each section' -r
complete -c xencode -n "__fish_xencode_using_subcommand impact" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand impact" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand removal" -l limit -d 'How many entries to list in each section' -r
complete -c xencode -n "__fish_xencode_using_subcommand removal" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand removal" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand generate" -l shell -d 'Shell for completions (ignored for man)' -r -f -a "bash\t''
fish\t''
zsh\t''
powershell\t''
elvish\t''"
complete -c xencode -n "__fish_xencode_using_subcommand generate" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c xencode -n "__fish_xencode_using_subcommand mutants" -l diff -d 'Only mutants in the diff against this ref' -r
complete -c xencode -n "__fish_xencode_using_subcommand mutants" -l timeout -d 'Wall-clock limit per mutant, in seconds' -r
complete -c xencode -n "__fish_xencode_using_subcommand mutants" -l check-repair -d 'Judge a proposed repair instead of running anything' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand mutants" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand mutants" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand cov" -l base -d 'Compare against this ref instead of the working tree' -r
complete -c xencode -n "__fish_xencode_using_subcommand cov" -l test -d 'Run this test command instead of the repository\'s own verified one' -r
complete -c xencode -n "__fish_xencode_using_subcommand cov" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand cov" -l show-missing-lines -d 'List only the file and line numbers'
complete -c xencode -n "__fish_xencode_using_subcommand cov" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and not __fish_seen_subcommand_from record check show help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and not __fish_seen_subcommand_from record check show help" -f -a "record" -d 'Measure every hot path and store those samples as the baseline'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and not __fish_seen_subcommand_from record check show help" -f -a "check" -d 'Measure the hot paths and compare them against the stored baseline'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and not __fish_seen_subcommand_from record check show help" -f -a "show" -d 'Show the recorded baseline without measuring anything'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and not __fish_seen_subcommand_from record check show help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from record" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from record" -l force -d 'Store it even if a path was measured too widely to support a verdict'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from record" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from check" -l filter -d 'Measure only the paths whose name contains this' -r
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from check" -l alert-pct -d 'The slowdown that raises a flag, as a percentage of the baseline' -r
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from check" -l alpha -d 'The significance level a verdict is judged at' -r
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from check" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from check" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from show" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from show" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from help" -f -a "record" -d 'Measure every hot path and store those samples as the baseline'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from help" -f -a "check" -d 'Measure the hot paths and compare them against the stored baseline'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from help" -f -a "show" -d 'Show the recorded baseline without measuring anything'
complete -c xencode -n "__fish_xencode_using_subcommand perf; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand prices; and not __fish_seen_subcommand_from show fetch help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand prices; and not __fish_seen_subcommand_from show fetch help" -f -a "show" -d 'Show every price a cost report would use, which document each came from, and which of the models this project has actually run are unpriced'
complete -c xencode -n "__fish_xencode_using_subcommand prices; and not __fish_seen_subcommand_from show fetch help" -f -a "fetch" -d 'Read the public catalogue again and replace the cached listing with it'
complete -c xencode -n "__fish_xencode_using_subcommand prices; and not __fish_seen_subcommand_from show fetch help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand prices; and __fish_seen_subcommand_from show" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand prices; and __fish_seen_subcommand_from show" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand prices; and __fish_seen_subcommand_from fetch" -l url -d 'Read from somewhere other than OpenRouter\'s listing — a gateway that publishes the same document, or an address for testing' -r
complete -c xencode -n "__fish_xencode_using_subcommand prices; and __fish_seen_subcommand_from fetch" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c xencode -n "__fish_xencode_using_subcommand prices; and __fish_seen_subcommand_from help" -f -a "show" -d 'Show every price a cost report would use, which document each came from, and which of the models this project has actually run are unpriced'
complete -c xencode -n "__fish_xencode_using_subcommand prices; and __fish_seen_subcommand_from help" -f -a "fetch" -d 'Read the public catalogue again and replace the cached listing with it'
complete -c xencode -n "__fish_xencode_using_subcommand prices; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand test" -l package -d 'Only these packages (repeatable)' -r
complete -c xencode -n "__fish_xencode_using_subcommand test" -l retries -d 'Retries allowed per failing test. Kept at 0 by default: a retry-pass is not a pass' -r
complete -c xencode -n "__fish_xencode_using_subcommand test" -l stress-count -d 'Run each test this many times, to surface flakes and order dependence' -r
complete -c xencode -n "__fish_xencode_using_subcommand test" -l timeout -d 'Wall-clock ceiling in seconds' -r
complete -c xencode -n "__fish_xencode_using_subcommand test" -l isolate -d 'Classify one failing test against the clean base tree instead of running the suite: PRE_EXISTING_FAILURE, INTRODUCED, or FLAKY' -r
complete -c xencode -n "__fish_xencode_using_subcommand test" -l base -d 'The ref the base tree is taken at for --isolate' -r
complete -c xencode -n "__fish_xencode_using_subcommand test" -l repeat -d 'Runs per side for --isolate; a pass on any run means flaky' -r
complete -c xencode -n "__fish_xencode_using_subcommand test" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand test" -l session -d 'Session ID to file checks under in the verification ledger (defaults to active session or \'cli\')' -r
complete -c xencode -n "__fish_xencode_using_subcommand test" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand release-notes" -l from -d 'Start the range here instead of at the newest tag. An empty value means no lower bound: every commit reachable from --to' -r
complete -c xencode -n "__fish_xencode_using_subcommand release-notes" -l to -d 'End the range here (default: HEAD)' -r
complete -c xencode -n "__fish_xencode_using_subcommand release-notes" -l release -d 'Label the draft heading with this version instead of `[Unreleased]`' -r
complete -c xencode -n "__fish_xencode_using_subcommand release-notes" -l out -d 'Write the draft here instead of printing it. A file that already exists is not replaced unless --force says so' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand release-notes" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand release-notes" -l force -d 'Replace the file named by --out even if it is already there'
complete -c xencode -n "__fish_xencode_using_subcommand release-notes" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand review" -l base -d 'Base branch, tag, commit — or HEAD for uncommitted changes. Defaults to origin/HEAD, init.defaultBranch, or \'main\'' -r
complete -c xencode -n "__fish_xencode_using_subcommand review" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand review" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand replay" -l tool-root -d 'The tree the replay\'s tool calls work against (default: this one)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand replay" -l out -d 'Where to write tool_calls.jsonl and the replay\'s own recording (default: <project>/.xencode/cache/replays/<run id>)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand replay" -l list -d 'List recorded runs, newest first'
complete -c xencode -n "__fish_xencode_using_subcommand replay" -l run-tools -d 'Let the replay\'s tool calls really run. Without this the permission gate stays in charge, so a call it would have asked a person about comes back denied'
complete -c xencode -n "__fish_xencode_using_subcommand replay" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and not __fish_seen_subcommand_from list show trailer help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and not __fish_seen_subcommand_from list show trailer help" -f -a "list" -d 'List recent runs, oldest first. The window is not a cap: `run_by_id` reaches past it, and so does `show` with a full id'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and not __fish_seen_subcommand_from list show trailer help" -f -a "show" -d 'Show one run: its model, every question a person answered while it went, and the verification rows its session left behind'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and not __fish_seen_subcommand_from list show trailer help" -f -a "trailer" -d 'Print the commit trailer block naming one run, for pasting into a commit message. Every line is a `Token: value` trailer, so `git interpret-trailers` reads it as trailers'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and not __fish_seen_subcommand_from list show trailer help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and __fish_seen_subcommand_from list" -l limit -d 'How many of the newest runs to print' -r
complete -c xencode -n "__fish_xencode_using_subcommand runs; and __fish_seen_subcommand_from list" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand runs; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and __fish_seen_subcommand_from show" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand runs; and __fish_seen_subcommand_from show" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and __fish_seen_subcommand_from trailer" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and __fish_seen_subcommand_from help" -f -a "list" -d 'List recent runs, oldest first. The window is not a cap: `run_by_id` reaches past it, and so does `show` with a full id'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and __fish_seen_subcommand_from help" -f -a "show" -d 'Show one run: its model, every question a person answered while it went, and the verification rows its session left behind'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and __fish_seen_subcommand_from help" -f -a "trailer" -d 'Print the commit trailer block naming one run, for pasting into a commit message. Every line is a `Token: value` trailer, so `git interpret-trailers` reads it as trailers'
complete -c xencode -n "__fish_xencode_using_subcommand runs; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand run" -l resume -d 'Continue a crashed run from its last completed round. Refuses a run that finished, was stopped, or is still going' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l show -d 'Show one run: its spec, its status, its rounds and its exit' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l log -d 'Print the tail of one run\'s log' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l stop -d 'Ask one running run to stop. Reads as stopped, not crashed' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l model -d 'Run with this model instead of the configured default' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l tool-root -d 'The tree the run works in (default: this project)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand run" -l max-rounds -d 'Stop after this many completed rounds' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l max-minutes -d 'Stop after this many minutes of wall-clock time. System time, so a suspended laptop counts — a cap that slept through suspend would be a way past it' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l max-cost -d 'Stop after spending this many dollars. Needs a model `pricing.json` (or the fetched listing) names and a route that reports token counts; without both the run is refused, because a cap that cannot count cannot stop' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l tail -d 'How many log lines `run --log` prints' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l ollama-url -d 'Where an Ollama server is, for a model id with no prefix' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l llamacpp-url -d 'Where a llama.cpp server is, for a `llamacpp:` model id' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l child -d 'The detached worker itself. Forked by `run --detach`, never typed' -r
complete -c xencode -n "__fish_xencode_using_subcommand run" -l xencode-dir -d 'Where the detached worker\'s run lives. Passed by the forking parent, because the worker\'s own working directory is the run\'s tree rather than the project' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand run" -l detach -d 'Start the run in the background and print its id. The terminal may go away; the run keeps going under its caps'
complete -c xencode -n "__fish_xencode_using_subcommand run" -l list -d 'List detached runs and what each is doing'
complete -c xencode -n "__fish_xencode_using_subcommand run" -l allow-shell -d 'Pre-approve shell commands. Without this a detached run has nobody to ask, so shell calls are refused where they stand'
complete -c xencode -n "__fish_xencode_using_subcommand run" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and not __fish_seen_subcommand_from list run help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and not __fish_seen_subcommand_from list run help" -f -a "list" -d 'List the shapes that can be run, and every run recorded so far'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and not __fish_seen_subcommand_from list run help" -f -a "run" -d 'Seed the defects, let the agent work on them, and grade what it changed'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and not __fish_seen_subcommand_from list run help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -s c -l case -d 'Which defect to seed (repeatable; default: all eight)' -r
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -s m -l model -d 'Model to score, in the form the router understands: a plain name for Ollama, `llamacpp:<name>`, or `remote:<name>` (default: the config\'s)' -r
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -l repeats -d 'How many times to run each defect. Eight shapes three times is twenty-four cases, which is the smallest sample worth reading' -r
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -l max-rounds -d 'Tool rounds a single case gets before the loop has to answer' -r
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -l out -d 'Where the seeded repositories and their diffs go (default: a stamped directory under the system temporary directory)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -l ollama-url -d 'Where an Ollama server is, for a model id with no prefix' -r
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -l llamacpp-url -d 'Where a llama.cpp server is, for a `llamacpp:` model id' -r
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -l timeout -d 'Seconds one model request may take before the case gives up on it' -r
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -l max-tokens -d 'How long one answer may be. Default 1024; a small model that will not stop talking otherwise holds a case for minutes. `0` leaves the limit to the server' -r
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -l judge-model -d 'Rank with a different model than the one under test, which is the only thing here that does anything about a judge favouring its own style. Defaults to the model being scored, and says so in the report' -r
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -l allow-shell -d 'Let the model really run shell commands. Without this a `run_command` is refused and the refusal is recorded like any other call'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -l judge -d 'After every verdict is in, ask a model to rank the attempts that came close. Two more requests per run, and no verdict changes'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from run" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from help" -f -a "list" -d 'List the shapes that can be run, and every run recorded so far'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from help" -f -a "run" -d 'Seed the defects, let the agent work on them, and grade what it changed'
complete -c xencode -n "__fish_xencode_using_subcommand eval; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and not __fish_seen_subcommand_from list install update remove help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and not __fish_seen_subcommand_from list install update remove help" -f -a "list" -d 'List installed plugins, whether each one loads, and what it contributes'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and not __fish_seen_subcommand_from list install update remove help" -f -a "install" -d 'Install a plugin from a git URL or a local path'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and not __fish_seen_subcommand_from list install update remove help" -f -a "update" -d 'Fetch a plugin\'s own repository again and show what changed before it is applied'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and not __fish_seen_subcommand_from list install update remove help" -f -a "remove" -d 'Remove a plugin by name'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and not __fish_seen_subcommand_from list install update remove help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from install" -l rev -d 'Install this branch, tag or commit instead of the repository\'s default branch. What is installed is pinned to the one commit the name resolved to, and `update` follows this same name' -r
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from install" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from update" -l rev -d 'Move to this branch, tag or commit rather than the one the plugin was installed at' -r
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from update" -l yes -d 'Apply the fetched version even though it changes what the plugin puts in front of the agent. Without this, such an update is only shown'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from update" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from remove" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from help" -f -a "list" -d 'List installed plugins, whether each one loads, and what it contributes'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from help" -f -a "install" -d 'Install a plugin from a git URL or a local path'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from help" -f -a "update" -d 'Fetch a plugin\'s own repository again and show what changed before it is applied'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from help" -f -a "remove" -d 'Remove a plugin by name'
complete -c xencode -n "__fish_xencode_using_subcommand plugin; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand mcp; and not __fish_seen_subcommand_from serve help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand mcp; and not __fish_seen_subcommand_from serve help" -f -a "serve" -d 'Serve xencode\'s tools as an MCP server on standard input and output'
complete -c xencode -n "__fish_xencode_using_subcommand mcp; and not __fish_seen_subcommand_from serve help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand mcp; and __fish_seen_subcommand_from serve" -l workspace -d 'The directory the tools work on; a `path` or `cwd` argument that leaves it is refused' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand mcp; and __fish_seen_subcommand_from serve" -l allow -d 'Permit this one tool to run despite the read-only default. Repeat it per tool; the name must be one of the six xencode publishes, so a typo is reported instead of doing nothing' -r
complete -c xencode -n "__fish_xencode_using_subcommand mcp; and __fish_seen_subcommand_from serve" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c xencode -n "__fish_xencode_using_subcommand mcp; and __fish_seen_subcommand_from help" -f -a "serve" -d 'Serve xencode\'s tools as an MCP server on standard input and output'
complete -c xencode -n "__fish_xencode_using_subcommand mcp; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and not __fish_seen_subcommand_from status start stop load unload list set-path help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and not __fish_seen_subcommand_from status start stop load unload list set-path help" -f -a "status" -d 'Show llama.cpp server status, loaded model, and token throughput'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and not __fish_seen_subcommand_from status start stop load unload list set-path help" -f -a "start" -d 'Start a llama-server process hosting the configured GGUF model'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and not __fish_seen_subcommand_from status start stop load unload list set-path help" -f -a "stop" -d 'Stop a llama-server process started by xencode'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and not __fish_seen_subcommand_from status start stop load unload list set-path help" -f -a "load" -d 'Load / switch a model on a running llama-server'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and not __fish_seen_subcommand_from status start stop load unload list set-path help" -f -a "unload" -d 'Unload the currently loaded model'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and not __fish_seen_subcommand_from status start stop load unload list set-path help" -f -a "list" -d 'List models available on a running llama-server'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and not __fish_seen_subcommand_from status start stop load unload list set-path help" -f -a "set-path" -d 'Set the configured GGUF model path used for auto-start/load'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and not __fish_seen_subcommand_from status start stop load unload list set-path help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from status" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from start" -l model -d 'GGUF model path (overrides config llama_cpp_model_path)' -r
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from start" -l port -d 'Port to bind (defaults to 8080)' -r
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from start" -l exec -d 'llama-server executable path (overrides config / PATH lookup)' -r
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from start" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from stop" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from load" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from unload" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from set-path" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from help" -f -a "status" -d 'Show llama.cpp server status, loaded model, and token throughput'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from help" -f -a "start" -d 'Start a llama-server process hosting the configured GGUF model'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from help" -f -a "stop" -d 'Stop a llama-server process started by xencode'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from help" -f -a "load" -d 'Load / switch a model on a running llama-server'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from help" -f -a "unload" -d 'Unload the currently loaded model'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from help" -f -a "list" -d 'List models available on a running llama-server'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from help" -f -a "set-path" -d 'Set the configured GGUF model path used for auto-start/load'
complete -c xencode -n "__fish_xencode_using_subcommand llamacpp; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand hw; and not __fish_seen_subcommand_from probe help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand hw; and not __fish_seen_subcommand_from probe help" -f -a "probe" -d 'Read RAM, cores and compute devices, and recommend launch flags'
complete -c xencode -n "__fish_xencode_using_subcommand hw; and not __fish_seen_subcommand_from probe help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand hw; and __fish_seen_subcommand_from probe" -l model -d 'GGUF file to size the answer against (defaults to the configured model, then to any GGUF in the usual cache directory)' -r
complete -c xencode -n "__fish_xencode_using_subcommand hw; and __fish_seen_subcommand_from probe" -l exec -d 'llama-server binary to ask what it can offload to' -r
complete -c xencode -n "__fish_xencode_using_subcommand hw; and __fish_seen_subcommand_from probe" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand hw; and __fish_seen_subcommand_from help" -f -a "probe" -d 'Read RAM, cores and compute devices, and recommend launch flags'
complete -c xencode -n "__fish_xencode_using_subcommand hw; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand history; and not __fish_seen_subcommand_from status digest setup help" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand history; and not __fish_seen_subcommand_from status digest setup help" -f -a "status" -d 'Show which history indexes exist here and time the queries that use them'
complete -c xencode -n "__fish_xencode_using_subcommand history; and not __fish_seen_subcommand_from status digest setup help" -f -a "digest" -d 'Print the ~250-token history digest for one file'
complete -c xencode -n "__fish_xencode_using_subcommand history; and not __fish_seen_subcommand_from status digest setup help" -f -a "setup" -d 'Write the commit-graph and the multi-pack-index, then time them'
complete -c xencode -n "__fish_xencode_using_subcommand history; and not __fish_seen_subcommand_from status digest setup help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from status" -l path -d 'Repository to look at (default: the current directory)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from status" -l file -d 'File to run a blame probe on (default: README.md, else the first tracked file)' -r
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from status" -l json -d 'Emit JSON instead of a table'
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from status" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from digest" -l path -d 'Repository to read (default: the current directory)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from digest" -l json -d 'Emit JSON instead of text'
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from digest" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from setup" -l path -d 'Repository to write into (default: the current directory)' -r -F
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from setup" -l file -d 'File to run a blame probe on (default: README.md, else the first tracked file)' -r
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from setup" -l json -d 'Emit JSON instead of a table'
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from setup" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from help" -f -a "status" -d 'Show which history indexes exist here and time the queries that use them'
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from help" -f -a "digest" -d 'Print the ~250-token history digest for one file'
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from help" -f -a "setup" -d 'Write the commit-graph and the multi-pack-index, then time them'
complete -c xencode -n "__fish_xencode_using_subcommand history; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand bootstrap" -l format -d 'Output format' -r -f -a "text\t''
json\t''"
complete -c xencode -n "__fish_xencode_using_subcommand bootstrap" -l check -d 'Report what would be written and create nothing'
complete -c xencode -n "__fish_xencode_using_subcommand bootstrap" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand tui" -s h -l help -d 'Print help'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "scan" -d 'Scan a workspace and list all entries'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "config" -d 'Configuration management'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "models" -d 'Local model management (Ollama & llama.cpp)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "cache" -d 'Response cache management'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "audit" -d 'The session server\'s audit log'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "advisories" -d 'Known security advisories for the crates this project depends on'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "deps" -d 'Supply-chain report: shell to the installed dependency checkers (cargo-shear, cargo-deny) and stream their findings. Report only — it never edits a manifest or auto-fixes a dependency'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "query" -d 'Send a query to a model'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "memory" -d 'Manage conversation memory'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "tasks" -d 'Manage background tasks (file-backed, survives this process)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "worktree" -d 'Manage git worktrees of the current repository'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "colab" -d 'Google Colab bridge: preflight, then up / status / down for a model server running on a Colab VM'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "remote" -d 'Manage remote inference hosts reached over SSH: add, list, use, forget'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "computers" -d 'Manage and inspect registered computer backends (AF-4)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "compete" -d 'Compete candidate implementations on isolated branches, verify each, and let a person pick one (AF-5)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "merge" -d 'Evaluate merge conflicts with git merge-tree and land branches under a human approval gate (OR-5)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "advise" -d 'Repository insights from the .xencode snapshot: broken imports, import cycles, hub files and orphans'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "server" -d 'Start the collaboration server'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "analyze" -d 'Analyze code for issues and vulnerabilities'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "fetch" -d 'Fetch a web page and extract research-ready text'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "interop" -d 'Measure what the coding-agent CLIs on this machine actually do'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "anchor" -d 'Find this repository\'s build and test commands, run them, and record only the ones that actually worked'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "toolchain" -d 'Run the project\'s own toolchain checks, and report structured evidence'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "doctor" -d 'Write one bug report: configuration, secrets, disk, providers, models, MCP servers and the Colab bridge. Flags narrow it to one part'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "session" -d 'Name sessions, resolve them, and export redacted transcripts'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "paths" -d 'Where xencode keeps its own files: settings, session records, cache and downloaded models, and whether they are still in `~/.xencode`'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "migrate" -d 'Move the files in `~/.xencode` to the four directories they belong in. Nothing is overwritten and the old directory is only removed once empty'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "verify" -d 'Run the machine-checkable checklist: test, lint, fmt — each verified, none graded'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "envcheck" -d 'Report environment keys read in code against the templates that document them'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "agents" -d 'List installed agents with versions and install provenance, inspect worker health (AR-8), or manage continuation packages (AR-7)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "hotspots" -d 'Rank files by churn times size with bus factor and owners'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "impact" -d 'What a change to one file affects: the crates that depend on its crate, the files that link it, and the files its history is coupled to'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "removal" -d 'What deleting one file would cost: the links it holds up, and the modules that become dead code the moment it is taken out'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "generate" -d 'Print shell completions or the man page; both are generated from the clap definition, never written by hand'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "mutants" -d 'Find code whose tests cannot tell right from wrong'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "cov" -d 'Report which lines this diff added were never executed'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "perf" -d 'Measure the hot paths against a stored baseline, and refuse the verdict when the run is too noisy to support one'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "prices" -d 'Where the prices a cost report uses come from, and reading them again'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "test" -d 'Run the tests, and never call a test that only passed on retry a pass'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "release-notes" -d 'Draft the release notes from the commits since the last release and the changelog block this project keeps, and report where the two disagree'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "review" -d 'Review the diff between a base branch and HEAD, file by file'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "replay" -d 'Run a recorded session again from the bytes it was made of'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "runs" -d 'Which runs happened, what each asked a person, and the commit trailer naming it. Reads `.xencode/cache/runs.jsonl` only, so it works with every model server down'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "run" -d 'Run an agent turn from the command line, in the foreground or detached so it survives the terminal. A detached run persists every completed round under `.xencode/cache/detached/<run-id>/`, so a kill is resumed with `--resume` instead of restarted, and stops on round, wall-clock and cost caps as well as the model finishing'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "eval" -d 'Score the agent on defects that were seeded on purpose'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "plugin" -d 'Manage plugins'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "mcp" -d 'Let another program drive xencode\'s tools over Model Context Protocol'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "llamacpp" -d 'llama.cpp server management (status/start/stop/load/unload)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "hw" -d 'What this machine can serve, read from the machine'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "history" -d 'How fast this repository\'s history is to ask about, and how to speed it up'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "bootstrap" -d 'Write the files a project xencode has never seen is missing'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "tui" -d 'Launch the Terminal User Interface'
complete -c xencode -n "__fish_xencode_using_subcommand help; and not __fish_seen_subcommand_from scan config models cache audit advisories deps query memory tasks worktree colab remote computers compete merge advise server analyze fetch interop anchor toolchain doctor session paths migrate verify envcheck agents hotspots impact removal generate mutants cov perf prices test release-notes review replay runs run eval plugin mcp llamacpp hw history bootstrap tui help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from config" -f -a "show" -d 'Display current configuration'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from config" -f -a "dump" -d 'Dump the engine composition and resolved configuration as JSON (AF-3)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from config" -f -a "set" -d 'Set a configuration value'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from config" -f -a "reset" -d 'Reset configuration to defaults'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from models" -f -a "list" -d 'List all installed Ollama models'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from models" -f -a "health" -d 'Check health of a specific model'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from models" -f -a "default" -d 'Show the smart-selected default model'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from models" -f -a "advice" -d 'Say which GGUF this machine can serve, from the dated advice table, with the pinned address and checksum to fetch it by'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from cache" -f -a "stats" -d 'Show cache statistics'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from cache" -f -a "clear" -d 'Clear all cached responses'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from cache" -f -a "gc" -d 'Drop the oldest cached responses until the cache directory fits under a size. The downloaded advisory corpora are not counted and cannot be removed by this command'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from audit" -f -a "verify" -d 'Check an audit log for records that were changed after they were written'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from advisories" -f -a "sync" -d 'Download both corpora: 6.3 MB of RustSec text plus a 3.5 MB OSV archive, about 20 MB unpacked'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from advisories" -f -a "show" -d 'What the corpus says about one crate, judged against a version when given'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from advisories" -f -a "check" -d 'Judge every package in this project\'s Cargo.lock against the corpus'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from advisories" -f -a "status" -d 'Whether a corpus exists here, how big it is, and when it was taken'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "list" -d 'List all conversation sessions'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "show" -d 'Show transcript of a session'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "fork" -d 'Fork a conversation session into a child holding an exact event prefix'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "prune" -d 'Delete conversation sessions that have no messages'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "gc" -d 'List the durable facts this repository contradicts, with how long each has been contradicted for. `--apply` retires the ones past a year'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "evidence" -d 'How many revisions each durable fact has been re-checked against, and what that evidence supports saying. Prints an interval, never a confidence'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from tasks" -f -a "list" -d 'List known tasks with their derived status'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from tasks" -f -a "start" -d 'Start a background task (survives this CLI process)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from tasks" -f -a "poll" -d 'Show a task\'s status and captured output'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from tasks" -f -a "stop" -d 'Ask a running task to stop'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from tasks" -f -a "rm" -d 'Forget a finished task and delete its files'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from worktree" -f -a "list" -d 'List git worktrees of the current repository'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from worktree" -f -a "add" -d 'Create a worktree at <path>, checking out <branch> (or a new branch named after the directory when omitted)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from worktree" -f -a "remove" -d 'Remove a worktree (git refuses dirty worktrees; main never removable)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from colab" -f -a "preflight" -d 'Verify the google-colab-cli bridge is usable before bringing a VM up'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from colab" -f -a "up" -d 'Bring a Colab VM up: create the session, install the runtime, and hold an SSH forward so the VM\'s OpenAI endpoint appears on the laptop'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from colab" -f -a "status" -d 'Report the Colab bridge state: forward pid, `colab sessions`, and a /v1/models probe on the forward'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from colab" -f -a "down" -d 'Tear the Colab bridge down: kill the forward, `colab stop`, clear state'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from remote" -f -a "add" -d 'Record a remote host profile'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from remote" -f -a "list" -d 'List recorded remote host profiles'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from remote" -f -a "use" -d 'Select the active remote host profile used by default'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from remote" -f -a "forget" -d 'Remove a recorded remote host profile'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from remote" -f -a "show" -d 'Show details of a remote host profile (or the active profile if omitted)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from computers" -f -a "list" -d 'List all registered computer backends (default)'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from computers" -f -a "show" -d 'Show details of a specific computer backend'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from computers" -f -a "use" -d 'Set the active computer backend'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from computers" -f -a "probe" -d 'Probe connectivity to a computer backend'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from compete" -f -a "run" -d 'Build each candidate arm in its own worktree and branch, run the verification checklist on every one of them, and print the table'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from compete" -f -a "list" -d 'List recorded competing runs, newest first'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from compete" -f -a "show" -d 'Re-print the verification table of a recorded run, from its saved report'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from compete" -f -a "pick" -d 'Switch the repository onto one arm\'s branch, leaving every other candidate branch and all evidence files on disk untouched'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from merge" -f -a "precheck" -d 'Speculatively precheck a candidate branch against a base branch using git merge-tree'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from merge" -f -a "plan" -d 'Build a multi-branch merge plan with conflict prechecks and worker checks'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from merge" -f -a "land" -d 'Land branches into base branch guarded by a named human decision'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from session" -f -a "name" -d 'Name a run so it can be resumed without its id'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from session" -f -a "resolve" -d 'Resolve a name, id prefix, or `latest` to a full run id'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from session" -f -a "export" -d 'Print a session\'s transcript; `--redacted` scrubs secrets'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from perf" -f -a "record" -d 'Measure every hot path and store those samples as the baseline'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from perf" -f -a "check" -d 'Measure the hot paths and compare them against the stored baseline'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from perf" -f -a "show" -d 'Show the recorded baseline without measuring anything'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from prices" -f -a "show" -d 'Show every price a cost report would use, which document each came from, and which of the models this project has actually run are unpriced'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from prices" -f -a "fetch" -d 'Read the public catalogue again and replace the cached listing with it'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from runs" -f -a "list" -d 'List recent runs, oldest first. The window is not a cap: `run_by_id` reaches past it, and so does `show` with a full id'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from runs" -f -a "show" -d 'Show one run: its model, every question a person answered while it went, and the verification rows its session left behind'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from runs" -f -a "trailer" -d 'Print the commit trailer block naming one run, for pasting into a commit message. Every line is a `Token: value` trailer, so `git interpret-trailers` reads it as trailers'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from eval" -f -a "list" -d 'List the shapes that can be run, and every run recorded so far'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from eval" -f -a "run" -d 'Seed the defects, let the agent work on them, and grade what it changed'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from plugin" -f -a "list" -d 'List installed plugins, whether each one loads, and what it contributes'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from plugin" -f -a "install" -d 'Install a plugin from a git URL or a local path'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from plugin" -f -a "update" -d 'Fetch a plugin\'s own repository again and show what changed before it is applied'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from plugin" -f -a "remove" -d 'Remove a plugin by name'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from mcp" -f -a "serve" -d 'Serve xencode\'s tools as an MCP server on standard input and output'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from llamacpp" -f -a "status" -d 'Show llama.cpp server status, loaded model, and token throughput'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from llamacpp" -f -a "start" -d 'Start a llama-server process hosting the configured GGUF model'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from llamacpp" -f -a "stop" -d 'Stop a llama-server process started by xencode'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from llamacpp" -f -a "load" -d 'Load / switch a model on a running llama-server'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from llamacpp" -f -a "unload" -d 'Unload the currently loaded model'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from llamacpp" -f -a "list" -d 'List models available on a running llama-server'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from llamacpp" -f -a "set-path" -d 'Set the configured GGUF model path used for auto-start/load'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from hw" -f -a "probe" -d 'Read RAM, cores and compute devices, and recommend launch flags'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from history" -f -a "status" -d 'Show which history indexes exist here and time the queries that use them'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from history" -f -a "digest" -d 'Print the ~250-token history digest for one file'
complete -c xencode -n "__fish_xencode_using_subcommand help; and __fish_seen_subcommand_from history" -f -a "setup" -d 'Write the commit-graph and the multi-pack-index, then time them'
