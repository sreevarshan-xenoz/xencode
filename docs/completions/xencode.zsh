#compdef xencode

autoload -U is-at-least

_xencode() {
    typeset -A opt_args
    typeset -a _arguments_options
    local ret=1

    if is-at-least 5.2; then
        _arguments_options=(-s -S -C)
    else
        _arguments_options=(-s -C)
    fi

    local context curcontext="$curcontext" state line
    _arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
'-V[Print version]' \
'--version[Print version]' \
":: :_xencode_commands" \
"*::: :->xencode" \
&& ret=0
    case $state in
    (xencode)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-command-$line[1]:"
        case $line[1] in
            (scan)
_arguments "${_arguments_options[@]}" : \
'--max-depth=[Maximum directory depth to traverse]:MAX_DEPTH:_default' \
'--format=[Output format]:FORMAT:(text json)' \
'--hidden[Include hidden files and directories]' \
'-h[Print help]' \
'--help[Print help]' \
'::path -- Path to scan (defaults to current directory):_files' \
&& ret=0
;;
(config)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__config_commands" \
"*::: :->config" \
&& ret=0

    case $state in
    (config)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-config-command-$line[1]:"
        case $line[1] in
            (show)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(set)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':key -- Configuration key (e.g., default_model, ollama_url):_default' \
':value -- Value to set. A leading hyphen is allowed because the value people set most often is `llama_cpp_args`, which is a server command line, and the line `xencode hw probe` hands them to paste starts with a flag:_default' \
&& ret=0
;;
(reset)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__config__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-config-help-command-$line[1]:"
        case $line[1] in
            (show)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(set)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(reset)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(models)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__models_commands" \
"*::: :->models" \
&& ret=0

    case $state in
    (models)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-models-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(health)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':model -- Model name to check:_default' \
&& ret=0
;;
(default)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(advice)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__models__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-models-help-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(health)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(default)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(advice)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(cache)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__cache_commands" \
"*::: :->cache" \
&& ret=0

    case $state in
    (cache)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-cache-command-$line[1]:"
        case $line[1] in
            (stats)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(clear)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__cache__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-cache-help-command-$line[1]:"
        case $line[1] in
            (stats)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(clear)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(audit)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__audit_commands" \
"*::: :->audit" \
&& ret=0

    case $state in
    (audit)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-audit-command-$line[1]:"
        case $line[1] in
            (verify)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
'::path -- Log to check (default\: ~/.xencode/audit.jsonl):_files' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__audit__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-audit-help-command-$line[1]:"
        case $line[1] in
            (verify)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(advisories)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__advisories_commands" \
"*::: :->advisories" \
&& ret=0

    case $state in
    (advisories)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-advisories-command-$line[1]:"
        case $line[1] in
            (sync)
_arguments "${_arguments_options[@]}" : \
'--dir=[Where to keep them (default\: <config dir>/advisories)]:DIR:_files' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(show)
_arguments "${_arguments_options[@]}" : \
'--version=[Version to judge; without it the records are listed but not assessed]:VERSION:_default' \
'--dir=[Corpus location (default\: <config dir>/advisories)]:DIR:_files' \
'-h[Print help]' \
'--help[Print help]' \
':crate_name -- Crate name, as it appears in Cargo.lock:_default' \
&& ret=0
;;
(check)
_arguments "${_arguments_options[@]}" : \
'--path=[Project to read Cargo.lock from (default\: the current directory)]:PATH:_files' \
'--dir=[Corpus location (default\: <config dir>/advisories)]:DIR:_files' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(status)
_arguments "${_arguments_options[@]}" : \
'--dir=[Corpus location (default\: <config dir>/advisories)]:DIR:_files' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__advisories__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-advisories-help-command-$line[1]:"
        case $line[1] in
            (sync)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(show)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(check)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(status)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(query)
_arguments "${_arguments_options[@]}" : \
'-m+[Model to use (overrides config default)]:MODEL:_default' \
'--model=[Model to use (overrides config default)]:MODEL:_default' \
'--session=[Session ID to attach to]:SESSION:_default' \
'--temperature=[llama.cpp sampling\: temperature (e.g. 0.7)]:TEMPERATURE:_default' \
'--top-k=[llama.cpp sampling\: top-k]:top-k:_default' \
'--min-p=[llama.cpp sampling\: min-p (e.g. 0.05)]:min-p:_default' \
'--mirostat=[llama.cpp sampling\: mirostat mode (0, 1, or 2)]:MIROSTAT:_default' \
'--seed=[llama.cpp sampling\: seed, for output that can be produced again]:SEED:_default' \
'--max-tokens=[llama.cpp sampling\: max generated tokens]:MAX_TOKENS:_default' \
'--grammar=[llama.cpp sampling\: GBNF grammar file/string]:GRAMMAR:_default' \
'--json-schema=[llama.cpp sampling\: JSON schema for structured output]:JSON_SCHEMA:_default' \
'--format=[How to write the answer\: plain words, or one JSON event per line]:FORMAT:(text ndjson)' \
'--no-cache[Do not use cached responses]' \
'-h[Print help]' \
'--help[Print help]' \
':prompt -- The prompt to send:_default' \
&& ret=0
;;
(memory)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__memory_commands" \
"*::: :->memory" \
&& ret=0

    case $state in
    (memory)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-memory-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(show)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':session -- Session ID:_default' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__memory__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-memory-help-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(show)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(tasks)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__tasks_commands" \
"*::: :->tasks" \
&& ret=0

    case $state in
    (tasks)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-tasks-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
'--json[Emit JSON instead of a table]' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(start)
_arguments "${_arguments_options[@]}" : \
'-n+[Human-friendly label (defaults to the command)]:NAME:_default' \
'--name=[Human-friendly label (defaults to the command)]:NAME:_default' \
'-h[Print help]' \
'--help[Print help]' \
':command -- Shell command to run:_default' \
&& ret=0
;;
(poll)
_arguments "${_arguments_options[@]}" : \
'--lines=[Trailing output lines to print]:LINES:_default' \
'-h[Print help]' \
'--help[Print help]' \
':id -- Task ID:_default' \
&& ret=0
;;
(stop)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':id -- Task ID:_default' \
&& ret=0
;;
(rm)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':id -- Task ID:_default' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__tasks__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-tasks-help-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(start)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(poll)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(stop)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(rm)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(worktree)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__worktree_commands" \
"*::: :->worktree" \
&& ret=0

    case $state in
    (worktree)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-worktree-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(add)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':path -- Directory for the new worktree:_default' \
'::branch -- Existing branch or commit to check out:_default' \
&& ret=0
;;
(remove)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':path -- Path of the worktree to remove:_default' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__worktree__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-worktree-help-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(add)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(remove)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(colab)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__colab_commands" \
"*::: :->colab" \
&& ret=0

    case $state in
    (colab)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-colab-command-$line[1]:"
        case $line[1] in
            (preflight)
_arguments "${_arguments_options[@]}" : \
'--generate-key[Generate the ed25519 key pair into the xencode config dir if absent]' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(up)
_arguments "${_arguments_options[@]}" : \
'--session=[Session name (defaults to config colab.session)]:SESSION:_default' \
'--gpu=[GPU accelerator (T4, L4, G4, H100, A100) for colab new]:GPU:_default' \
'--runtime=[Inference runtime on the VM\: llama.cpp (pinned llama-server + GGUF, heavier install, one-shot) or ollama (tags flow into the model picker)]:RUNTIME:_default' \
'--model=[Model repo/tag installed on the VM (defaults to config colab.model)]:MODEL:_default' \
'--weights=[Weights source\: hf (llama.cpp only; drive/gcs are refused with a fix)]:WEIGHTS:_default' \
'--quant=[GGUF quantization fragment to serve (llama.cpp only, e.g. Q4_K_M; defaults to config colab.quant)]:QUANT:_default' \
'--local-port=[Local port the SSH forward exposes (defaults to config / 18000)]:LOCAL_PORT:_default' \
'--remote-port=[VM-side port the runtime binds (0 = runtime-native\: llama.cpp 18080, ollama 11434; defaults to config)]:REMOTE_PORT:_default' \
'--reconnect[Rebuild a broken bridge from the recorded colab.json (re-spawn the forward, or re-create the VM if it was reaped) instead of a full up]' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(status)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(down)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__colab__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-colab-help-command-$line[1]:"
        case $line[1] in
            (preflight)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(up)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(status)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(down)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(advise)
_arguments "${_arguments_options[@]}" : \
'--limit=[Maximum findings to show (0 shows all)]:LIMIT:_default' \
'--json[Machine-readable output]' \
'-h[Print help]' \
'--help[Print help]' \
'::filter -- Only report findings whose file path contains this substring:_default' \
&& ret=0
;;
(server)
_arguments "${_arguments_options[@]}" : \
'--port=[Port to listen on]:PORT:_default' \
'--host=[Address to bind (default\: loopback only)]:HOST:_default' \
'--cert=[TLS certificate in PEM form; requires --key]:CERT:_files' \
'--key=[TLS private key in PEM form; requires --cert]:KEY:_files' \
'--audit-path=[Audit log path; "none" disables (default\: ~/.xencode/audit.jsonl)]:AUDIT_PATH:_default' \
'--allow-insecure-public[Allow a non-loopback bind without TLS — tokens and activity then travel in clear text; read the warning before reaching for this]' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(analyze)
_arguments "${_arguments_options[@]}" : \
'--format=[Output format]:FORMAT:(text json)' \
'--runtime[Report async and concurrency hazards\: blocking calls on the reactor thread, unbounded channels, and dropped task handles. Needs the \`ast-grep\` binary; without it the report says the check did not run rather than reporting nothing found]' \
'-h[Print help]' \
'--help[Print help]' \
':path -- Path to analyze (file or directory):_files' \
&& ret=0
;;
(fetch)
_arguments "${_arguments_options[@]}" : \
'--format=[Output format]:FORMAT:(text json)' \
'-h[Print help]' \
'--help[Print help]' \
':url -- URL to fetch (http/https only):_default' \
&& ret=0
;;
(interop)
_arguments "${_arguments_options[@]}" : \
'*--agent=[Probe only these agents (repeatable); default is the whole roster]:AGENTS:_default' \
'--timeout=[Per-agent wall-clock limit in seconds]:TIMEOUT:_default' \
'--out=[Where to write the JSON report; omit to print it]:OUT:_files' \
'--format=[Output format for the printed summary]:FORMAT:(text json)' \
'--repeat=[Run each agent this many times and compare. One run is a reading; two is a check, and a fact that differs between runs is reported rather than smoothed over]:REPEAT:_default' \
'--check-auth[Report, read-only, which agents look configured on this machine, and what to run if one is not. Starts no login and reads no credential]' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
&& ret=0
;;
(anchor)
_arguments "${_arguments_options[@]}" : \
'--timeout=[Per-command wall-clock limit in seconds. A command that runs out of time is recorded as unverified, never as a pass]:TIMEOUT:_default' \
'--format=[Output format]:FORMAT:(text json)' \
'--dry-run[List what was found without running anything]' \
'-h[Print help]' \
'--help[Print help]' \
'::path -- Path to the repository (defaults to current directory):_files' \
&& ret=0
;;
(toolchain)
_arguments "${_arguments_options[@]}" : \
'--format=[Output format]:FORMAT:(text json)' \
'--allow-dirty[Allow \`fix\` on a tree with uncommitted edits]' \
'-h[Print help]' \
'--help[Print help]' \
':action -- What to run\: lint, fix, fmt, or shear:_default' \
&& ret=0
;;
(doctor)
_arguments "${_arguments_options[@]}" : \
'--format=[Output format]:FORMAT:(text json)' \
'--env[Machine environment facts]' \
'--deps[Dependency health\: outdated list plus advisory state]' \
'--selfcheck[Self-debug slice\: index, git, providers, MCP servers, metrics, cache]' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(session)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__session_commands" \
"*::: :->session" \
&& ret=0

    case $state in
    (session)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-session-command-$line[1]:"
        case $line[1] in
            (name)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':run -- The run id (or a prefix of one):_default' \
':name -- The name to give it:_default' \
&& ret=0
;;
(resolve)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':target -- Name, id prefix, or `latest`:_default' \
&& ret=0
;;
(export)
_arguments "${_arguments_options[@]}" : \
'--redacted[Scrub secrets with the trace module'\''s patterns]' \
'-h[Print help]' \
'--help[Print help]' \
':target -- Name, id prefix, or `latest`:_default' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__session__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-session-help-command-$line[1]:"
        case $line[1] in
            (name)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(resolve)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(export)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(verify)
_arguments "${_arguments_options[@]}" : \
'*--skip=[Skip these checks (repeatable); skipped is reported, never passed]:SKIP:_default' \
'--timeout=[Wall-clock ceiling in seconds for the test slot]:TIMEOUT:_default' \
'--format=[Output format]:FORMAT:(text json)' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(envcheck)
_arguments "${_arguments_options[@]}" : \
'--format=[Output format]:FORMAT:(text json)' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(agents)
_arguments "${_arguments_options[@]}" : \
'--format=[Output format]:FORMAT:(text json)' \
'--contract[Verify each roster claim against the agent'\''s live --help]' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(hotspots)
_arguments "${_arguments_options[@]}" : \
'--limit=[How many files to list]:LIMIT:_default' \
'--format=[Output format]:FORMAT:(text json)' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(generate)
_arguments "${_arguments_options[@]}" : \
'--shell=[Shell for completions (ignored for man)]:SHELL:(bash fish zsh powershell elvish)' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
':artifact -- What to emit\: completions or man:((completions\:"Shell completion script for \`--shell\`"
man\:"Roff man page for \`xencode(1)\`"))' \
&& ret=0
;;
(mutants)
_arguments "${_arguments_options[@]}" : \
'--diff=[Only mutants in the diff against this ref]:DIFF:_default' \
'--timeout=[Wall-clock limit per mutant, in seconds]:TIMEOUT:_default' \
'--check-repair=[Judge a proposed repair instead of running anything]:CHECK_REPAIR:_files' \
'--format=[Output format]:FORMAT:(text json)' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(cov)
_arguments "${_arguments_options[@]}" : \
'--base=[Compare against this ref instead of the working tree]:BASE:_default' \
'--test=[Run this test command instead of the repository'\''s own verified one]:TEST:_default' \
'--format=[Output format]:FORMAT:(text json)' \
'--show-missing-lines[List only the file and line numbers]' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(test)
_arguments "${_arguments_options[@]}" : \
'*--package=[Only these packages (repeatable)]:PACKAGES:_default' \
'--retries=[Retries allowed per failing test. Kept at 0 by default\: a retry-pass is not a pass]:RETRIES:_default' \
'--stress-count=[Run each test this many times, to surface flakes and order dependence]:STRESS_COUNT:_default' \
'--timeout=[Wall-clock ceiling in seconds]:TIMEOUT:_default' \
'--isolate=[Classify one failing test against the clean base tree instead of running the suite\: PRE_EXISTING_FAILURE, INTRODUCED, or FLAKY]:ISOLATE:_default' \
'--base=[The ref the base tree is taken at for --isolate]:BASE:_default' \
'--repeat=[Runs per side for --isolate; a pass on any run means flaky]:REPEAT:_default' \
'--format=[Output format]:FORMAT:(text json)' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(review)
_arguments "${_arguments_options[@]}" : \
'--base=[Base branch, tag, commit — or HEAD for uncommitted changes]:BASE:_default' \
'--format=[Output format]:FORMAT:(text json)' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(replay)
_arguments "${_arguments_options[@]}" : \
'--tool-root=[The tree the replay'\''s tool calls work against (default\: this one)]:TOOL_ROOT:_files' \
'--out=[Where to write tool_calls.jsonl and the replay'\''s own recording (default\: <project>/.xencode/cache/replays/<run id>)]:OUT:_files' \
'--list[List recorded runs, newest first]' \
'--run-tools[Let the replay'\''s tool calls really run. Without this the permission gate stays in charge, so a call it would have asked a person about comes back denied]' \
'-h[Print help]' \
'--help[Print help]' \
'::run_id -- Which run\: its full id, or enough of the start to be unique. Omit it with --list to see what has been recorded:_default' \
&& ret=0
;;
(eval)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__eval_commands" \
"*::: :->eval" \
&& ret=0

    case $state in
    (eval)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-eval-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(run)
_arguments "${_arguments_options[@]}" : \
'*-c+[Which defect to seed (repeatable; default\: all eight)]:CASES:_default' \
'*--case=[Which defect to seed (repeatable; default\: all eight)]:CASES:_default' \
'-m+[Model to score, in the form the router understands\: a plain name for Ollama, \`llamacpp\:<name>\`, or \`remote\:<name>\` (default\: the config'\''s)]:MODEL:_default' \
'--model=[Model to score, in the form the router understands\: a plain name for Ollama, \`llamacpp\:<name>\`, or \`remote\:<name>\` (default\: the config'\''s)]:MODEL:_default' \
'--repeats=[How many times to run each defect. Eight shapes three times is twenty-four cases, which is the smallest sample worth reading]:REPEATS:_default' \
'--max-rounds=[Tool rounds a single case gets before the loop has to answer]:MAX_ROUNDS:_default' \
'--out=[Where the seeded repositories and their diffs go (default\: a stamped directory under the system temporary directory)]:OUT:_files' \
'--ollama-url=[Where an Ollama server is, for a model id with no prefix]:OLLAMA_URL:_default' \
'--llamacpp-url=[Where a llama.cpp server is, for a \`llamacpp\:\` model id]:LLAMACPP_URL:_default' \
'--timeout=[Seconds one model request may take before the case gives up on it]:TIMEOUT:_default' \
'--max-tokens=[How long one answer may be. Default 1024; a small model that will not stop talking otherwise holds a case for minutes. \`0\` leaves the limit to the server]:MAX_TOKENS:_default' \
'--judge-model=[Rank with a different model than the one under test, which is the only thing here that does anything about a judge favouring its own style. Defaults to the model being scored, and says so in the report]:JUDGE_MODEL:_default' \
'--allow-shell[Let the model really run shell commands. Without this a \`run_command\` is refused and the refusal is recorded like any other call]' \
'--judge[After every verdict is in, ask a model to rank the attempts that came close. Two more requests per run, and no verdict changes]' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__eval__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-eval-help-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(run)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(plugin)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__plugin_commands" \
"*::: :->plugin" \
&& ret=0

    case $state in
    (plugin)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-plugin-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(install)
_arguments "${_arguments_options[@]}" : \
'--rev=[Install this branch, tag or commit instead of the repository'\''s default branch. What is installed is pinned to the one commit the name resolved to, and \`update\` follows this same name]:REV:_default' \
'-h[Print help]' \
'--help[Print help]' \
':source -- A git URL to clone (`https\://…`, `git@…\:…`, `file\:///…`) or a path to a plugin directory or manifest file:_default' \
&& ret=0
;;
(update)
_arguments "${_arguments_options[@]}" : \
'--rev=[Move to this branch, tag or commit rather than the one the plugin was installed at]:REV:_default' \
'--yes[Apply the fetched version even though it changes what the plugin puts in front of the agent. Without this, such an update is only shown]' \
'-h[Print help]' \
'--help[Print help]' \
':name -- Name of an installed plugin:_default' \
&& ret=0
;;
(remove)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':name -- Name of the plugin:_default' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__plugin__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-plugin-help-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(install)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(update)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(remove)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(mcp)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__mcp_commands" \
"*::: :->mcp" \
&& ret=0

    case $state in
    (mcp)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-mcp-command-$line[1]:"
        case $line[1] in
            (serve)
_arguments "${_arguments_options[@]}" : \
'--workspace=[The directory the tools work on; a \`path\` or \`cwd\` argument that leaves it is refused]:WORKSPACE:_files' \
'*--allow=[Permit this one tool to run despite the read-only default. Repeat it per tool; the name must be one of the six xencode publishes, so a typo is reported instead of doing nothing]:ALLOW:_default' \
'-h[Print help (see more with '\''--help'\'')]' \
'--help[Print help (see more with '\''--help'\'')]' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__mcp__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-mcp-help-command-$line[1]:"
        case $line[1] in
            (serve)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(llamacpp)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__llamacpp_commands" \
"*::: :->llamacpp" \
&& ret=0

    case $state in
    (llamacpp)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-llamacpp-command-$line[1]:"
        case $line[1] in
            (status)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(start)
_arguments "${_arguments_options[@]}" : \
'--model=[GGUF model path (overrides config llama_cpp_model_path)]:MODEL:_default' \
'--port=[Port to bind (defaults to 8080)]:PORT:_default' \
'--exec=[llama-server executable path (overrides config / PATH lookup)]:EXEC:_default' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(stop)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(load)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':model -- GGUF model path to load:_default' \
&& ret=0
;;
(unload)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(list)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(set-path)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
':path -- GGUF model path:_default' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__llamacpp__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-llamacpp-help-command-$line[1]:"
        case $line[1] in
            (status)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(start)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(stop)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(load)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(unload)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(set-path)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(hw)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__hw_commands" \
"*::: :->hw" \
&& ret=0

    case $state in
    (hw)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-hw-command-$line[1]:"
        case $line[1] in
            (probe)
_arguments "${_arguments_options[@]}" : \
'--model=[GGUF file to size the answer against (defaults to the configured model, then to any GGUF in the usual cache directory)]:MODEL:_default' \
'--exec=[llama-server binary to ask what it can offload to]:EXEC:_default' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__hw__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-hw-help-command-$line[1]:"
        case $line[1] in
            (probe)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(history)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
":: :_xencode__subcmd__history_commands" \
"*::: :->history" \
&& ret=0

    case $state in
    (history)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-history-command-$line[1]:"
        case $line[1] in
            (status)
_arguments "${_arguments_options[@]}" : \
'--path=[Repository to look at (default\: the current directory)]:PATH:_files' \
'--file=[File to run a blame probe on (default\: README.md, else the first tracked file)]:FILE:_default' \
'--json[Emit JSON instead of a table]' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(digest)
_arguments "${_arguments_options[@]}" : \
'--path=[Repository to read (default\: the current directory)]:PATH:_files' \
'--json[Emit JSON instead of text]' \
'-h[Print help]' \
'--help[Print help]' \
':file -- File to digest, repository-relative:_default' \
&& ret=0
;;
(setup)
_arguments "${_arguments_options[@]}" : \
'--path=[Repository to write into (default\: the current directory)]:PATH:_files' \
'--file=[File to run a blame probe on (default\: README.md, else the first tracked file)]:FILE:_default' \
'--json[Emit JSON instead of a table]' \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__history__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-history-help-command-$line[1]:"
        case $line[1] in
            (status)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(digest)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(setup)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
;;
(tui)
_arguments "${_arguments_options[@]}" : \
'-h[Print help]' \
'--help[Print help]' \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help_commands" \
"*::: :->help" \
&& ret=0

    case $state in
    (help)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-command-$line[1]:"
        case $line[1] in
            (scan)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(config)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__config_commands" \
"*::: :->config" \
&& ret=0

    case $state in
    (config)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-config-command-$line[1]:"
        case $line[1] in
            (show)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(set)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(reset)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(models)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__models_commands" \
"*::: :->models" \
&& ret=0

    case $state in
    (models)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-models-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(health)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(default)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(advice)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(cache)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__cache_commands" \
"*::: :->cache" \
&& ret=0

    case $state in
    (cache)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-cache-command-$line[1]:"
        case $line[1] in
            (stats)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(clear)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(audit)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__audit_commands" \
"*::: :->audit" \
&& ret=0

    case $state in
    (audit)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-audit-command-$line[1]:"
        case $line[1] in
            (verify)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(advisories)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__advisories_commands" \
"*::: :->advisories" \
&& ret=0

    case $state in
    (advisories)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-advisories-command-$line[1]:"
        case $line[1] in
            (sync)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(show)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(check)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(status)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(query)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(memory)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__memory_commands" \
"*::: :->memory" \
&& ret=0

    case $state in
    (memory)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-memory-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(show)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(tasks)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__tasks_commands" \
"*::: :->tasks" \
&& ret=0

    case $state in
    (tasks)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-tasks-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(start)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(poll)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(stop)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(rm)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(worktree)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__worktree_commands" \
"*::: :->worktree" \
&& ret=0

    case $state in
    (worktree)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-worktree-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(add)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(remove)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(colab)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__colab_commands" \
"*::: :->colab" \
&& ret=0

    case $state in
    (colab)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-colab-command-$line[1]:"
        case $line[1] in
            (preflight)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(up)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(status)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(down)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(advise)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(server)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(analyze)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(fetch)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(interop)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(anchor)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(toolchain)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(doctor)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(session)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__session_commands" \
"*::: :->session" \
&& ret=0

    case $state in
    (session)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-session-command-$line[1]:"
        case $line[1] in
            (name)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(resolve)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(export)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(verify)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(envcheck)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(agents)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(hotspots)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(generate)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(mutants)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(cov)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(test)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(review)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(replay)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(eval)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__eval_commands" \
"*::: :->eval" \
&& ret=0

    case $state in
    (eval)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-eval-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(run)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(plugin)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__plugin_commands" \
"*::: :->plugin" \
&& ret=0

    case $state in
    (plugin)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-plugin-command-$line[1]:"
        case $line[1] in
            (list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(install)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(update)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(remove)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(mcp)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__mcp_commands" \
"*::: :->mcp" \
&& ret=0

    case $state in
    (mcp)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-mcp-command-$line[1]:"
        case $line[1] in
            (serve)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(llamacpp)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__llamacpp_commands" \
"*::: :->llamacpp" \
&& ret=0

    case $state in
    (llamacpp)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-llamacpp-command-$line[1]:"
        case $line[1] in
            (status)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(start)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(stop)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(load)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(unload)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(list)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(set-path)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(hw)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__hw_commands" \
"*::: :->hw" \
&& ret=0

    case $state in
    (hw)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-hw-command-$line[1]:"
        case $line[1] in
            (probe)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(history)
_arguments "${_arguments_options[@]}" : \
":: :_xencode__subcmd__help__subcmd__history_commands" \
"*::: :->history" \
&& ret=0

    case $state in
    (history)
        words=($line[1] "${words[@]}")
        (( CURRENT += 1 ))
        curcontext="${curcontext%:*:*}:xencode-help-history-command-$line[1]:"
        case $line[1] in
            (status)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(digest)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(setup)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
(tui)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
(help)
_arguments "${_arguments_options[@]}" : \
&& ret=0
;;
        esac
    ;;
esac
;;
        esac
    ;;
esac
}

(( $+functions[_xencode_commands] )) ||
_xencode_commands() {
    local commands; commands=(
'scan:Scan a workspace and list all entries' \
'config:Configuration management' \
'models:Local model management (Ollama & llama.cpp)' \
'cache:Response cache management' \
'audit:The session server'\''s audit log' \
'advisories:Known security advisories for the crates this project depends on' \
'query:Send a query to a model' \
'memory:Manage conversation memory' \
'tasks:Manage background tasks (file-backed, survives this process)' \
'worktree:Manage git worktrees of the current repository' \
'colab:Google Colab bridge\: preflight, then up / status / down for a model server running on a Colab VM' \
'advise:Repository insights from the .xencode snapshot\: broken imports, import cycles, hub files and orphans' \
'server:Start the collaboration server' \
'analyze:Analyze code for issues and vulnerabilities' \
'fetch:Fetch a web page and extract research-ready text' \
'interop:Measure what the coding-agent CLIs on this machine actually do' \
'anchor:Find this repository'\''s build and test commands, run them, and record only the ones that actually worked' \
'toolchain:Run the project'\''s own toolchain checks, and report structured evidence' \
'doctor:Probe and display this machine\: cores, memory, GPUs, logs, colab route' \
'session:Name sessions, resolve them, and export redacted transcripts' \
'verify:Run the machine-checkable checklist\: test, lint, fmt — each verified, none graded' \
'envcheck:Report environment keys read in code against the templates that document them' \
'agents:List installed agents with versions and install provenance' \
'hotspots:Rank files by churn times size with bus factor and owners' \
'generate:Print shell completions or the man page; both are generated from the clap definition, never written by hand' \
'mutants:Find code whose tests cannot tell right from wrong' \
'cov:Report which lines this diff added were never executed' \
'test:Run the tests, and never call a test that only passed on retry a pass' \
'review:Review the diff between a base branch and HEAD, file by file' \
'replay:Run a recorded session again from the bytes it was made of' \
'eval:Score the agent on defects that were seeded on purpose' \
'plugin:Manage plugins' \
'mcp:Let another program drive xencode'\''s tools over Model Context Protocol' \
'llamacpp:llama.cpp server management (status/start/stop/load/unload)' \
'hw:What this machine can serve, read from the machine' \
'history:How fast this repository'\''s history is to ask about, and how to speed it up' \
'tui:Launch the Terminal User Interface' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advise_commands] )) ||
_xencode__subcmd__advise_commands() {
    local commands; commands=()
    _describe -t commands 'xencode advise commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories_commands] )) ||
_xencode__subcmd__advisories_commands() {
    local commands; commands=(
'sync:Download both corpora\: 6.3 MB of RustSec text plus a 3.5 MB OSV archive, about 20 MB unpacked' \
'show:What the corpus says about one crate, judged against a version when given' \
'check:Judge every package in this project'\''s Cargo.lock against the corpus' \
'status:Whether a corpus exists here, how big it is, and when it was taken' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode advisories commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories__subcmd__check_commands] )) ||
_xencode__subcmd__advisories__subcmd__check_commands() {
    local commands; commands=()
    _describe -t commands 'xencode advisories check commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories__subcmd__help_commands] )) ||
_xencode__subcmd__advisories__subcmd__help_commands() {
    local commands; commands=(
'sync:Download both corpora\: 6.3 MB of RustSec text plus a 3.5 MB OSV archive, about 20 MB unpacked' \
'show:What the corpus says about one crate, judged against a version when given' \
'check:Judge every package in this project'\''s Cargo.lock against the corpus' \
'status:Whether a corpus exists here, how big it is, and when it was taken' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode advisories help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories__subcmd__help__subcmd__check_commands] )) ||
_xencode__subcmd__advisories__subcmd__help__subcmd__check_commands() {
    local commands; commands=()
    _describe -t commands 'xencode advisories help check commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__advisories__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode advisories help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories__subcmd__help__subcmd__show_commands] )) ||
_xencode__subcmd__advisories__subcmd__help__subcmd__show_commands() {
    local commands; commands=()
    _describe -t commands 'xencode advisories help show commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories__subcmd__help__subcmd__status_commands] )) ||
_xencode__subcmd__advisories__subcmd__help__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode advisories help status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories__subcmd__help__subcmd__sync_commands] )) ||
_xencode__subcmd__advisories__subcmd__help__subcmd__sync_commands() {
    local commands; commands=()
    _describe -t commands 'xencode advisories help sync commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories__subcmd__show_commands] )) ||
_xencode__subcmd__advisories__subcmd__show_commands() {
    local commands; commands=()
    _describe -t commands 'xencode advisories show commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories__subcmd__status_commands] )) ||
_xencode__subcmd__advisories__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode advisories status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__advisories__subcmd__sync_commands] )) ||
_xencode__subcmd__advisories__subcmd__sync_commands() {
    local commands; commands=()
    _describe -t commands 'xencode advisories sync commands' commands "$@"
}
(( $+functions[_xencode__subcmd__agents_commands] )) ||
_xencode__subcmd__agents_commands() {
    local commands; commands=()
    _describe -t commands 'xencode agents commands' commands "$@"
}
(( $+functions[_xencode__subcmd__analyze_commands] )) ||
_xencode__subcmd__analyze_commands() {
    local commands; commands=()
    _describe -t commands 'xencode analyze commands' commands "$@"
}
(( $+functions[_xencode__subcmd__anchor_commands] )) ||
_xencode__subcmd__anchor_commands() {
    local commands; commands=()
    _describe -t commands 'xencode anchor commands' commands "$@"
}
(( $+functions[_xencode__subcmd__audit_commands] )) ||
_xencode__subcmd__audit_commands() {
    local commands; commands=(
'verify:Check an audit log for records that were changed after they were written' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode audit commands' commands "$@"
}
(( $+functions[_xencode__subcmd__audit__subcmd__help_commands] )) ||
_xencode__subcmd__audit__subcmd__help_commands() {
    local commands; commands=(
'verify:Check an audit log for records that were changed after they were written' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode audit help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__audit__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__audit__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode audit help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__audit__subcmd__help__subcmd__verify_commands] )) ||
_xencode__subcmd__audit__subcmd__help__subcmd__verify_commands() {
    local commands; commands=()
    _describe -t commands 'xencode audit help verify commands' commands "$@"
}
(( $+functions[_xencode__subcmd__audit__subcmd__verify_commands] )) ||
_xencode__subcmd__audit__subcmd__verify_commands() {
    local commands; commands=()
    _describe -t commands 'xencode audit verify commands' commands "$@"
}
(( $+functions[_xencode__subcmd__cache_commands] )) ||
_xencode__subcmd__cache_commands() {
    local commands; commands=(
'stats:Show cache statistics' \
'clear:Clear all cached responses' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode cache commands' commands "$@"
}
(( $+functions[_xencode__subcmd__cache__subcmd__clear_commands] )) ||
_xencode__subcmd__cache__subcmd__clear_commands() {
    local commands; commands=()
    _describe -t commands 'xencode cache clear commands' commands "$@"
}
(( $+functions[_xencode__subcmd__cache__subcmd__help_commands] )) ||
_xencode__subcmd__cache__subcmd__help_commands() {
    local commands; commands=(
'stats:Show cache statistics' \
'clear:Clear all cached responses' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode cache help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__cache__subcmd__help__subcmd__clear_commands] )) ||
_xencode__subcmd__cache__subcmd__help__subcmd__clear_commands() {
    local commands; commands=()
    _describe -t commands 'xencode cache help clear commands' commands "$@"
}
(( $+functions[_xencode__subcmd__cache__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__cache__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode cache help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__cache__subcmd__help__subcmd__stats_commands] )) ||
_xencode__subcmd__cache__subcmd__help__subcmd__stats_commands() {
    local commands; commands=()
    _describe -t commands 'xencode cache help stats commands' commands "$@"
}
(( $+functions[_xencode__subcmd__cache__subcmd__stats_commands] )) ||
_xencode__subcmd__cache__subcmd__stats_commands() {
    local commands; commands=()
    _describe -t commands 'xencode cache stats commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab_commands] )) ||
_xencode__subcmd__colab_commands() {
    local commands; commands=(
'preflight:Verify the google-colab-cli bridge is usable before bringing a VM up' \
'up:Bring a Colab VM up\: create the session, install the runtime, and hold an SSH forward so the VM'\''s OpenAI endpoint appears on the laptop' \
'status:Report the Colab bridge state\: forward pid, \`colab sessions\`, and a /v1/models probe on the forward' \
'down:Tear the Colab bridge down\: kill the forward, \`colab stop\`, clear state' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode colab commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab__subcmd__down_commands] )) ||
_xencode__subcmd__colab__subcmd__down_commands() {
    local commands; commands=()
    _describe -t commands 'xencode colab down commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab__subcmd__help_commands] )) ||
_xencode__subcmd__colab__subcmd__help_commands() {
    local commands; commands=(
'preflight:Verify the google-colab-cli bridge is usable before bringing a VM up' \
'up:Bring a Colab VM up\: create the session, install the runtime, and hold an SSH forward so the VM'\''s OpenAI endpoint appears on the laptop' \
'status:Report the Colab bridge state\: forward pid, \`colab sessions\`, and a /v1/models probe on the forward' \
'down:Tear the Colab bridge down\: kill the forward, \`colab stop\`, clear state' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode colab help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab__subcmd__help__subcmd__down_commands] )) ||
_xencode__subcmd__colab__subcmd__help__subcmd__down_commands() {
    local commands; commands=()
    _describe -t commands 'xencode colab help down commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__colab__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode colab help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab__subcmd__help__subcmd__preflight_commands] )) ||
_xencode__subcmd__colab__subcmd__help__subcmd__preflight_commands() {
    local commands; commands=()
    _describe -t commands 'xencode colab help preflight commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab__subcmd__help__subcmd__status_commands] )) ||
_xencode__subcmd__colab__subcmd__help__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode colab help status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab__subcmd__help__subcmd__up_commands] )) ||
_xencode__subcmd__colab__subcmd__help__subcmd__up_commands() {
    local commands; commands=()
    _describe -t commands 'xencode colab help up commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab__subcmd__preflight_commands] )) ||
_xencode__subcmd__colab__subcmd__preflight_commands() {
    local commands; commands=()
    _describe -t commands 'xencode colab preflight commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab__subcmd__status_commands] )) ||
_xencode__subcmd__colab__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode colab status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__colab__subcmd__up_commands] )) ||
_xencode__subcmd__colab__subcmd__up_commands() {
    local commands; commands=()
    _describe -t commands 'xencode colab up commands' commands "$@"
}
(( $+functions[_xencode__subcmd__config_commands] )) ||
_xencode__subcmd__config_commands() {
    local commands; commands=(
'show:Display current configuration' \
'set:Set a configuration value' \
'reset:Reset configuration to defaults' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode config commands' commands "$@"
}
(( $+functions[_xencode__subcmd__config__subcmd__help_commands] )) ||
_xencode__subcmd__config__subcmd__help_commands() {
    local commands; commands=(
'show:Display current configuration' \
'set:Set a configuration value' \
'reset:Reset configuration to defaults' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode config help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__config__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__config__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode config help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__config__subcmd__help__subcmd__reset_commands] )) ||
_xencode__subcmd__config__subcmd__help__subcmd__reset_commands() {
    local commands; commands=()
    _describe -t commands 'xencode config help reset commands' commands "$@"
}
(( $+functions[_xencode__subcmd__config__subcmd__help__subcmd__set_commands] )) ||
_xencode__subcmd__config__subcmd__help__subcmd__set_commands() {
    local commands; commands=()
    _describe -t commands 'xencode config help set commands' commands "$@"
}
(( $+functions[_xencode__subcmd__config__subcmd__help__subcmd__show_commands] )) ||
_xencode__subcmd__config__subcmd__help__subcmd__show_commands() {
    local commands; commands=()
    _describe -t commands 'xencode config help show commands' commands "$@"
}
(( $+functions[_xencode__subcmd__config__subcmd__reset_commands] )) ||
_xencode__subcmd__config__subcmd__reset_commands() {
    local commands; commands=()
    _describe -t commands 'xencode config reset commands' commands "$@"
}
(( $+functions[_xencode__subcmd__config__subcmd__set_commands] )) ||
_xencode__subcmd__config__subcmd__set_commands() {
    local commands; commands=()
    _describe -t commands 'xencode config set commands' commands "$@"
}
(( $+functions[_xencode__subcmd__config__subcmd__show_commands] )) ||
_xencode__subcmd__config__subcmd__show_commands() {
    local commands; commands=()
    _describe -t commands 'xencode config show commands' commands "$@"
}
(( $+functions[_xencode__subcmd__cov_commands] )) ||
_xencode__subcmd__cov_commands() {
    local commands; commands=()
    _describe -t commands 'xencode cov commands' commands "$@"
}
(( $+functions[_xencode__subcmd__doctor_commands] )) ||
_xencode__subcmd__doctor_commands() {
    local commands; commands=()
    _describe -t commands 'xencode doctor commands' commands "$@"
}
(( $+functions[_xencode__subcmd__envcheck_commands] )) ||
_xencode__subcmd__envcheck_commands() {
    local commands; commands=()
    _describe -t commands 'xencode envcheck commands' commands "$@"
}
(( $+functions[_xencode__subcmd__eval_commands] )) ||
_xencode__subcmd__eval_commands() {
    local commands; commands=(
'list:List the shapes that can be run, and every run recorded so far' \
'run:Seed the defects, let the agent work on them, and grade what it changed' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode eval commands' commands "$@"
}
(( $+functions[_xencode__subcmd__eval__subcmd__help_commands] )) ||
_xencode__subcmd__eval__subcmd__help_commands() {
    local commands; commands=(
'list:List the shapes that can be run, and every run recorded so far' \
'run:Seed the defects, let the agent work on them, and grade what it changed' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode eval help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__eval__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__eval__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode eval help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__eval__subcmd__help__subcmd__list_commands] )) ||
_xencode__subcmd__eval__subcmd__help__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode eval help list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__eval__subcmd__help__subcmd__run_commands] )) ||
_xencode__subcmd__eval__subcmd__help__subcmd__run_commands() {
    local commands; commands=()
    _describe -t commands 'xencode eval help run commands' commands "$@"
}
(( $+functions[_xencode__subcmd__eval__subcmd__list_commands] )) ||
_xencode__subcmd__eval__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode eval list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__eval__subcmd__run_commands] )) ||
_xencode__subcmd__eval__subcmd__run_commands() {
    local commands; commands=()
    _describe -t commands 'xencode eval run commands' commands "$@"
}
(( $+functions[_xencode__subcmd__fetch_commands] )) ||
_xencode__subcmd__fetch_commands() {
    local commands; commands=()
    _describe -t commands 'xencode fetch commands' commands "$@"
}
(( $+functions[_xencode__subcmd__generate_commands] )) ||
_xencode__subcmd__generate_commands() {
    local commands; commands=()
    _describe -t commands 'xencode generate commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help_commands] )) ||
_xencode__subcmd__help_commands() {
    local commands; commands=(
'scan:Scan a workspace and list all entries' \
'config:Configuration management' \
'models:Local model management (Ollama & llama.cpp)' \
'cache:Response cache management' \
'audit:The session server'\''s audit log' \
'advisories:Known security advisories for the crates this project depends on' \
'query:Send a query to a model' \
'memory:Manage conversation memory' \
'tasks:Manage background tasks (file-backed, survives this process)' \
'worktree:Manage git worktrees of the current repository' \
'colab:Google Colab bridge\: preflight, then up / status / down for a model server running on a Colab VM' \
'advise:Repository insights from the .xencode snapshot\: broken imports, import cycles, hub files and orphans' \
'server:Start the collaboration server' \
'analyze:Analyze code for issues and vulnerabilities' \
'fetch:Fetch a web page and extract research-ready text' \
'interop:Measure what the coding-agent CLIs on this machine actually do' \
'anchor:Find this repository'\''s build and test commands, run them, and record only the ones that actually worked' \
'toolchain:Run the project'\''s own toolchain checks, and report structured evidence' \
'doctor:Probe and display this machine\: cores, memory, GPUs, logs, colab route' \
'session:Name sessions, resolve them, and export redacted transcripts' \
'verify:Run the machine-checkable checklist\: test, lint, fmt — each verified, none graded' \
'envcheck:Report environment keys read in code against the templates that document them' \
'agents:List installed agents with versions and install provenance' \
'hotspots:Rank files by churn times size with bus factor and owners' \
'generate:Print shell completions or the man page; both are generated from the clap definition, never written by hand' \
'mutants:Find code whose tests cannot tell right from wrong' \
'cov:Report which lines this diff added were never executed' \
'test:Run the tests, and never call a test that only passed on retry a pass' \
'review:Review the diff between a base branch and HEAD, file by file' \
'replay:Run a recorded session again from the bytes it was made of' \
'eval:Score the agent on defects that were seeded on purpose' \
'plugin:Manage plugins' \
'mcp:Let another program drive xencode'\''s tools over Model Context Protocol' \
'llamacpp:llama.cpp server management (status/start/stop/load/unload)' \
'hw:What this machine can serve, read from the machine' \
'history:How fast this repository'\''s history is to ask about, and how to speed it up' \
'tui:Launch the Terminal User Interface' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__advise_commands] )) ||
_xencode__subcmd__help__subcmd__advise_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help advise commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__advisories_commands] )) ||
_xencode__subcmd__help__subcmd__advisories_commands() {
    local commands; commands=(
'sync:Download both corpora\: 6.3 MB of RustSec text plus a 3.5 MB OSV archive, about 20 MB unpacked' \
'show:What the corpus says about one crate, judged against a version when given' \
'check:Judge every package in this project'\''s Cargo.lock against the corpus' \
'status:Whether a corpus exists here, how big it is, and when it was taken' \
    )
    _describe -t commands 'xencode help advisories commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__advisories__subcmd__check_commands] )) ||
_xencode__subcmd__help__subcmd__advisories__subcmd__check_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help advisories check commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__advisories__subcmd__show_commands] )) ||
_xencode__subcmd__help__subcmd__advisories__subcmd__show_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help advisories show commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__advisories__subcmd__status_commands] )) ||
_xencode__subcmd__help__subcmd__advisories__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help advisories status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__advisories__subcmd__sync_commands] )) ||
_xencode__subcmd__help__subcmd__advisories__subcmd__sync_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help advisories sync commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__agents_commands] )) ||
_xencode__subcmd__help__subcmd__agents_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help agents commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__analyze_commands] )) ||
_xencode__subcmd__help__subcmd__analyze_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help analyze commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__anchor_commands] )) ||
_xencode__subcmd__help__subcmd__anchor_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help anchor commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__audit_commands] )) ||
_xencode__subcmd__help__subcmd__audit_commands() {
    local commands; commands=(
'verify:Check an audit log for records that were changed after they were written' \
    )
    _describe -t commands 'xencode help audit commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__audit__subcmd__verify_commands] )) ||
_xencode__subcmd__help__subcmd__audit__subcmd__verify_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help audit verify commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__cache_commands] )) ||
_xencode__subcmd__help__subcmd__cache_commands() {
    local commands; commands=(
'stats:Show cache statistics' \
'clear:Clear all cached responses' \
    )
    _describe -t commands 'xencode help cache commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__cache__subcmd__clear_commands] )) ||
_xencode__subcmd__help__subcmd__cache__subcmd__clear_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help cache clear commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__cache__subcmd__stats_commands] )) ||
_xencode__subcmd__help__subcmd__cache__subcmd__stats_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help cache stats commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__colab_commands] )) ||
_xencode__subcmd__help__subcmd__colab_commands() {
    local commands; commands=(
'preflight:Verify the google-colab-cli bridge is usable before bringing a VM up' \
'up:Bring a Colab VM up\: create the session, install the runtime, and hold an SSH forward so the VM'\''s OpenAI endpoint appears on the laptop' \
'status:Report the Colab bridge state\: forward pid, \`colab sessions\`, and a /v1/models probe on the forward' \
'down:Tear the Colab bridge down\: kill the forward, \`colab stop\`, clear state' \
    )
    _describe -t commands 'xencode help colab commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__colab__subcmd__down_commands] )) ||
_xencode__subcmd__help__subcmd__colab__subcmd__down_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help colab down commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__colab__subcmd__preflight_commands] )) ||
_xencode__subcmd__help__subcmd__colab__subcmd__preflight_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help colab preflight commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__colab__subcmd__status_commands] )) ||
_xencode__subcmd__help__subcmd__colab__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help colab status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__colab__subcmd__up_commands] )) ||
_xencode__subcmd__help__subcmd__colab__subcmd__up_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help colab up commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__config_commands] )) ||
_xencode__subcmd__help__subcmd__config_commands() {
    local commands; commands=(
'show:Display current configuration' \
'set:Set a configuration value' \
'reset:Reset configuration to defaults' \
    )
    _describe -t commands 'xencode help config commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__config__subcmd__reset_commands] )) ||
_xencode__subcmd__help__subcmd__config__subcmd__reset_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help config reset commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__config__subcmd__set_commands] )) ||
_xencode__subcmd__help__subcmd__config__subcmd__set_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help config set commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__config__subcmd__show_commands] )) ||
_xencode__subcmd__help__subcmd__config__subcmd__show_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help config show commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__cov_commands] )) ||
_xencode__subcmd__help__subcmd__cov_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help cov commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__doctor_commands] )) ||
_xencode__subcmd__help__subcmd__doctor_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help doctor commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__envcheck_commands] )) ||
_xencode__subcmd__help__subcmd__envcheck_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help envcheck commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__eval_commands] )) ||
_xencode__subcmd__help__subcmd__eval_commands() {
    local commands; commands=(
'list:List the shapes that can be run, and every run recorded so far' \
'run:Seed the defects, let the agent work on them, and grade what it changed' \
    )
    _describe -t commands 'xencode help eval commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__eval__subcmd__list_commands] )) ||
_xencode__subcmd__help__subcmd__eval__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help eval list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__eval__subcmd__run_commands] )) ||
_xencode__subcmd__help__subcmd__eval__subcmd__run_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help eval run commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__fetch_commands] )) ||
_xencode__subcmd__help__subcmd__fetch_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help fetch commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__generate_commands] )) ||
_xencode__subcmd__help__subcmd__generate_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help generate commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__history_commands] )) ||
_xencode__subcmd__help__subcmd__history_commands() {
    local commands; commands=(
'status:Show which history indexes exist here and time the queries that use them' \
'digest:Print the ~250-token history digest for one file' \
'setup:Write the commit-graph and the multi-pack-index, then time them' \
    )
    _describe -t commands 'xencode help history commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__history__subcmd__digest_commands] )) ||
_xencode__subcmd__help__subcmd__history__subcmd__digest_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help history digest commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__history__subcmd__setup_commands] )) ||
_xencode__subcmd__help__subcmd__history__subcmd__setup_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help history setup commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__history__subcmd__status_commands] )) ||
_xencode__subcmd__help__subcmd__history__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help history status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__hotspots_commands] )) ||
_xencode__subcmd__help__subcmd__hotspots_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help hotspots commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__hw_commands] )) ||
_xencode__subcmd__help__subcmd__hw_commands() {
    local commands; commands=(
'probe:Read RAM, cores and compute devices, and recommend launch flags' \
    )
    _describe -t commands 'xencode help hw commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__hw__subcmd__probe_commands] )) ||
_xencode__subcmd__help__subcmd__hw__subcmd__probe_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help hw probe commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__interop_commands] )) ||
_xencode__subcmd__help__subcmd__interop_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help interop commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__llamacpp_commands] )) ||
_xencode__subcmd__help__subcmd__llamacpp_commands() {
    local commands; commands=(
'status:Show llama.cpp server status, loaded model, and token throughput' \
'start:Start a llama-server process hosting the configured GGUF model' \
'stop:Stop a llama-server process started by xencode' \
'load:Load / switch a model on a running llama-server' \
'unload:Unload the currently loaded model' \
'list:List models available on a running llama-server' \
'set-path:Set the configured GGUF model path used for auto-start/load' \
    )
    _describe -t commands 'xencode help llamacpp commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__llamacpp__subcmd__list_commands] )) ||
_xencode__subcmd__help__subcmd__llamacpp__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help llamacpp list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__llamacpp__subcmd__load_commands] )) ||
_xencode__subcmd__help__subcmd__llamacpp__subcmd__load_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help llamacpp load commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__llamacpp__subcmd__set-path_commands] )) ||
_xencode__subcmd__help__subcmd__llamacpp__subcmd__set-path_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help llamacpp set-path commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__llamacpp__subcmd__start_commands] )) ||
_xencode__subcmd__help__subcmd__llamacpp__subcmd__start_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help llamacpp start commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__llamacpp__subcmd__status_commands] )) ||
_xencode__subcmd__help__subcmd__llamacpp__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help llamacpp status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__llamacpp__subcmd__stop_commands] )) ||
_xencode__subcmd__help__subcmd__llamacpp__subcmd__stop_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help llamacpp stop commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__llamacpp__subcmd__unload_commands] )) ||
_xencode__subcmd__help__subcmd__llamacpp__subcmd__unload_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help llamacpp unload commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__mcp_commands] )) ||
_xencode__subcmd__help__subcmd__mcp_commands() {
    local commands; commands=(
'serve:Serve xencode'\''s tools as an MCP server on standard input and output' \
    )
    _describe -t commands 'xencode help mcp commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__mcp__subcmd__serve_commands] )) ||
_xencode__subcmd__help__subcmd__mcp__subcmd__serve_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help mcp serve commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__memory_commands] )) ||
_xencode__subcmd__help__subcmd__memory_commands() {
    local commands; commands=(
'list:List all conversation sessions' \
'show:Show transcript of a session' \
    )
    _describe -t commands 'xencode help memory commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__memory__subcmd__list_commands] )) ||
_xencode__subcmd__help__subcmd__memory__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help memory list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__memory__subcmd__show_commands] )) ||
_xencode__subcmd__help__subcmd__memory__subcmd__show_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help memory show commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__models_commands] )) ||
_xencode__subcmd__help__subcmd__models_commands() {
    local commands; commands=(
'list:List all installed Ollama models' \
'health:Check health of a specific model' \
'default:Show the smart-selected default model' \
'advice:Say which GGUF this machine can serve, from the dated advice table, with the pinned address and checksum to fetch it by' \
    )
    _describe -t commands 'xencode help models commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__models__subcmd__advice_commands] )) ||
_xencode__subcmd__help__subcmd__models__subcmd__advice_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help models advice commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__models__subcmd__default_commands] )) ||
_xencode__subcmd__help__subcmd__models__subcmd__default_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help models default commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__models__subcmd__health_commands] )) ||
_xencode__subcmd__help__subcmd__models__subcmd__health_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help models health commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__models__subcmd__list_commands] )) ||
_xencode__subcmd__help__subcmd__models__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help models list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__mutants_commands] )) ||
_xencode__subcmd__help__subcmd__mutants_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help mutants commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__plugin_commands] )) ||
_xencode__subcmd__help__subcmd__plugin_commands() {
    local commands; commands=(
'list:List installed plugins, whether each one loads, and what it contributes' \
'install:Install a plugin from a git URL or a local path' \
'update:Fetch a plugin'\''s own repository again and show what changed before it is applied' \
'remove:Remove a plugin by name' \
    )
    _describe -t commands 'xencode help plugin commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__plugin__subcmd__install_commands] )) ||
_xencode__subcmd__help__subcmd__plugin__subcmd__install_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help plugin install commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__plugin__subcmd__list_commands] )) ||
_xencode__subcmd__help__subcmd__plugin__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help plugin list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__plugin__subcmd__remove_commands] )) ||
_xencode__subcmd__help__subcmd__plugin__subcmd__remove_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help plugin remove commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__plugin__subcmd__update_commands] )) ||
_xencode__subcmd__help__subcmd__plugin__subcmd__update_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help plugin update commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__query_commands] )) ||
_xencode__subcmd__help__subcmd__query_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help query commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__replay_commands] )) ||
_xencode__subcmd__help__subcmd__replay_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help replay commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__review_commands] )) ||
_xencode__subcmd__help__subcmd__review_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help review commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__scan_commands] )) ||
_xencode__subcmd__help__subcmd__scan_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help scan commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__server_commands] )) ||
_xencode__subcmd__help__subcmd__server_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help server commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__session_commands] )) ||
_xencode__subcmd__help__subcmd__session_commands() {
    local commands; commands=(
'name:Name a run so it can be resumed without its id' \
'resolve:Resolve a name, id prefix, or \`latest\` to a full run id' \
'export:Print a session'\''s transcript; \`--redacted\` scrubs secrets' \
    )
    _describe -t commands 'xencode help session commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__session__subcmd__export_commands] )) ||
_xencode__subcmd__help__subcmd__session__subcmd__export_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help session export commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__session__subcmd__name_commands] )) ||
_xencode__subcmd__help__subcmd__session__subcmd__name_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help session name commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__session__subcmd__resolve_commands] )) ||
_xencode__subcmd__help__subcmd__session__subcmd__resolve_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help session resolve commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__tasks_commands] )) ||
_xencode__subcmd__help__subcmd__tasks_commands() {
    local commands; commands=(
'list:List known tasks with their derived status' \
'start:Start a background task (survives this CLI process)' \
'poll:Show a task'\''s status and captured output' \
'stop:Ask a running task to stop' \
'rm:Forget a finished task and delete its files' \
    )
    _describe -t commands 'xencode help tasks commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__tasks__subcmd__list_commands] )) ||
_xencode__subcmd__help__subcmd__tasks__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help tasks list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__tasks__subcmd__poll_commands] )) ||
_xencode__subcmd__help__subcmd__tasks__subcmd__poll_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help tasks poll commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__tasks__subcmd__rm_commands] )) ||
_xencode__subcmd__help__subcmd__tasks__subcmd__rm_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help tasks rm commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__tasks__subcmd__start_commands] )) ||
_xencode__subcmd__help__subcmd__tasks__subcmd__start_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help tasks start commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__tasks__subcmd__stop_commands] )) ||
_xencode__subcmd__help__subcmd__tasks__subcmd__stop_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help tasks stop commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__test_commands] )) ||
_xencode__subcmd__help__subcmd__test_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help test commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__toolchain_commands] )) ||
_xencode__subcmd__help__subcmd__toolchain_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help toolchain commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__tui_commands] )) ||
_xencode__subcmd__help__subcmd__tui_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help tui commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__verify_commands] )) ||
_xencode__subcmd__help__subcmd__verify_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help verify commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__worktree_commands] )) ||
_xencode__subcmd__help__subcmd__worktree_commands() {
    local commands; commands=(
'list:List git worktrees of the current repository' \
'add:Create a worktree at <path>, checking out <branch> (or a new branch named after the directory when omitted)' \
'remove:Remove a worktree (git refuses dirty worktrees; main never removable)' \
    )
    _describe -t commands 'xencode help worktree commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__worktree__subcmd__add_commands] )) ||
_xencode__subcmd__help__subcmd__worktree__subcmd__add_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help worktree add commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__worktree__subcmd__list_commands] )) ||
_xencode__subcmd__help__subcmd__worktree__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help worktree list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__help__subcmd__worktree__subcmd__remove_commands] )) ||
_xencode__subcmd__help__subcmd__worktree__subcmd__remove_commands() {
    local commands; commands=()
    _describe -t commands 'xencode help worktree remove commands' commands "$@"
}
(( $+functions[_xencode__subcmd__history_commands] )) ||
_xencode__subcmd__history_commands() {
    local commands; commands=(
'status:Show which history indexes exist here and time the queries that use them' \
'digest:Print the ~250-token history digest for one file' \
'setup:Write the commit-graph and the multi-pack-index, then time them' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode history commands' commands "$@"
}
(( $+functions[_xencode__subcmd__history__subcmd__digest_commands] )) ||
_xencode__subcmd__history__subcmd__digest_commands() {
    local commands; commands=()
    _describe -t commands 'xencode history digest commands' commands "$@"
}
(( $+functions[_xencode__subcmd__history__subcmd__help_commands] )) ||
_xencode__subcmd__history__subcmd__help_commands() {
    local commands; commands=(
'status:Show which history indexes exist here and time the queries that use them' \
'digest:Print the ~250-token history digest for one file' \
'setup:Write the commit-graph and the multi-pack-index, then time them' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode history help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__history__subcmd__help__subcmd__digest_commands] )) ||
_xencode__subcmd__history__subcmd__help__subcmd__digest_commands() {
    local commands; commands=()
    _describe -t commands 'xencode history help digest commands' commands "$@"
}
(( $+functions[_xencode__subcmd__history__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__history__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode history help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__history__subcmd__help__subcmd__setup_commands] )) ||
_xencode__subcmd__history__subcmd__help__subcmd__setup_commands() {
    local commands; commands=()
    _describe -t commands 'xencode history help setup commands' commands "$@"
}
(( $+functions[_xencode__subcmd__history__subcmd__help__subcmd__status_commands] )) ||
_xencode__subcmd__history__subcmd__help__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode history help status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__history__subcmd__setup_commands] )) ||
_xencode__subcmd__history__subcmd__setup_commands() {
    local commands; commands=()
    _describe -t commands 'xencode history setup commands' commands "$@"
}
(( $+functions[_xencode__subcmd__history__subcmd__status_commands] )) ||
_xencode__subcmd__history__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode history status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__hotspots_commands] )) ||
_xencode__subcmd__hotspots_commands() {
    local commands; commands=()
    _describe -t commands 'xencode hotspots commands' commands "$@"
}
(( $+functions[_xencode__subcmd__hw_commands] )) ||
_xencode__subcmd__hw_commands() {
    local commands; commands=(
'probe:Read RAM, cores and compute devices, and recommend launch flags' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode hw commands' commands "$@"
}
(( $+functions[_xencode__subcmd__hw__subcmd__help_commands] )) ||
_xencode__subcmd__hw__subcmd__help_commands() {
    local commands; commands=(
'probe:Read RAM, cores and compute devices, and recommend launch flags' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode hw help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__hw__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__hw__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode hw help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__hw__subcmd__help__subcmd__probe_commands] )) ||
_xencode__subcmd__hw__subcmd__help__subcmd__probe_commands() {
    local commands; commands=()
    _describe -t commands 'xencode hw help probe commands' commands "$@"
}
(( $+functions[_xencode__subcmd__hw__subcmd__probe_commands] )) ||
_xencode__subcmd__hw__subcmd__probe_commands() {
    local commands; commands=()
    _describe -t commands 'xencode hw probe commands' commands "$@"
}
(( $+functions[_xencode__subcmd__interop_commands] )) ||
_xencode__subcmd__interop_commands() {
    local commands; commands=()
    _describe -t commands 'xencode interop commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp_commands] )) ||
_xencode__subcmd__llamacpp_commands() {
    local commands; commands=(
'status:Show llama.cpp server status, loaded model, and token throughput' \
'start:Start a llama-server process hosting the configured GGUF model' \
'stop:Stop a llama-server process started by xencode' \
'load:Load / switch a model on a running llama-server' \
'unload:Unload the currently loaded model' \
'list:List models available on a running llama-server' \
'set-path:Set the configured GGUF model path used for auto-start/load' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode llamacpp commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__help_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__help_commands() {
    local commands; commands=(
'status:Show llama.cpp server status, loaded model, and token throughput' \
'start:Start a llama-server process hosting the configured GGUF model' \
'stop:Stop a llama-server process started by xencode' \
'load:Load / switch a model on a running llama-server' \
'unload:Unload the currently loaded model' \
'list:List models available on a running llama-server' \
'set-path:Set the configured GGUF model path used for auto-start/load' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode llamacpp help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__help__subcmd__list_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__help__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp help list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__help__subcmd__load_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__help__subcmd__load_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp help load commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__help__subcmd__set-path_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__help__subcmd__set-path_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp help set-path commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__help__subcmd__start_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__help__subcmd__start_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp help start commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__help__subcmd__status_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__help__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp help status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__help__subcmd__stop_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__help__subcmd__stop_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp help stop commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__help__subcmd__unload_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__help__subcmd__unload_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp help unload commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__list_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__load_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__load_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp load commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__set-path_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__set-path_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp set-path commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__start_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__start_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp start commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__status_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__status_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp status commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__stop_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__stop_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp stop commands' commands "$@"
}
(( $+functions[_xencode__subcmd__llamacpp__subcmd__unload_commands] )) ||
_xencode__subcmd__llamacpp__subcmd__unload_commands() {
    local commands; commands=()
    _describe -t commands 'xencode llamacpp unload commands' commands "$@"
}
(( $+functions[_xencode__subcmd__mcp_commands] )) ||
_xencode__subcmd__mcp_commands() {
    local commands; commands=(
'serve:Serve xencode'\''s tools as an MCP server on standard input and output' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode mcp commands' commands "$@"
}
(( $+functions[_xencode__subcmd__mcp__subcmd__help_commands] )) ||
_xencode__subcmd__mcp__subcmd__help_commands() {
    local commands; commands=(
'serve:Serve xencode'\''s tools as an MCP server on standard input and output' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode mcp help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__mcp__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__mcp__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode mcp help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__mcp__subcmd__help__subcmd__serve_commands] )) ||
_xencode__subcmd__mcp__subcmd__help__subcmd__serve_commands() {
    local commands; commands=()
    _describe -t commands 'xencode mcp help serve commands' commands "$@"
}
(( $+functions[_xencode__subcmd__mcp__subcmd__serve_commands] )) ||
_xencode__subcmd__mcp__subcmd__serve_commands() {
    local commands; commands=()
    _describe -t commands 'xencode mcp serve commands' commands "$@"
}
(( $+functions[_xencode__subcmd__memory_commands] )) ||
_xencode__subcmd__memory_commands() {
    local commands; commands=(
'list:List all conversation sessions' \
'show:Show transcript of a session' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode memory commands' commands "$@"
}
(( $+functions[_xencode__subcmd__memory__subcmd__help_commands] )) ||
_xencode__subcmd__memory__subcmd__help_commands() {
    local commands; commands=(
'list:List all conversation sessions' \
'show:Show transcript of a session' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode memory help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__memory__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__memory__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode memory help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__memory__subcmd__help__subcmd__list_commands] )) ||
_xencode__subcmd__memory__subcmd__help__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode memory help list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__memory__subcmd__help__subcmd__show_commands] )) ||
_xencode__subcmd__memory__subcmd__help__subcmd__show_commands() {
    local commands; commands=()
    _describe -t commands 'xencode memory help show commands' commands "$@"
}
(( $+functions[_xencode__subcmd__memory__subcmd__list_commands] )) ||
_xencode__subcmd__memory__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode memory list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__memory__subcmd__show_commands] )) ||
_xencode__subcmd__memory__subcmd__show_commands() {
    local commands; commands=()
    _describe -t commands 'xencode memory show commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models_commands] )) ||
_xencode__subcmd__models_commands() {
    local commands; commands=(
'list:List all installed Ollama models' \
'health:Check health of a specific model' \
'default:Show the smart-selected default model' \
'advice:Say which GGUF this machine can serve, from the dated advice table, with the pinned address and checksum to fetch it by' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode models commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models__subcmd__advice_commands] )) ||
_xencode__subcmd__models__subcmd__advice_commands() {
    local commands; commands=()
    _describe -t commands 'xencode models advice commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models__subcmd__default_commands] )) ||
_xencode__subcmd__models__subcmd__default_commands() {
    local commands; commands=()
    _describe -t commands 'xencode models default commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models__subcmd__health_commands] )) ||
_xencode__subcmd__models__subcmd__health_commands() {
    local commands; commands=()
    _describe -t commands 'xencode models health commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models__subcmd__help_commands] )) ||
_xencode__subcmd__models__subcmd__help_commands() {
    local commands; commands=(
'list:List all installed Ollama models' \
'health:Check health of a specific model' \
'default:Show the smart-selected default model' \
'advice:Say which GGUF this machine can serve, from the dated advice table, with the pinned address and checksum to fetch it by' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode models help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models__subcmd__help__subcmd__advice_commands] )) ||
_xencode__subcmd__models__subcmd__help__subcmd__advice_commands() {
    local commands; commands=()
    _describe -t commands 'xencode models help advice commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models__subcmd__help__subcmd__default_commands] )) ||
_xencode__subcmd__models__subcmd__help__subcmd__default_commands() {
    local commands; commands=()
    _describe -t commands 'xencode models help default commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models__subcmd__help__subcmd__health_commands] )) ||
_xencode__subcmd__models__subcmd__help__subcmd__health_commands() {
    local commands; commands=()
    _describe -t commands 'xencode models help health commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__models__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode models help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models__subcmd__help__subcmd__list_commands] )) ||
_xencode__subcmd__models__subcmd__help__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode models help list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__models__subcmd__list_commands] )) ||
_xencode__subcmd__models__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode models list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__mutants_commands] )) ||
_xencode__subcmd__mutants_commands() {
    local commands; commands=()
    _describe -t commands 'xencode mutants commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin_commands] )) ||
_xencode__subcmd__plugin_commands() {
    local commands; commands=(
'list:List installed plugins, whether each one loads, and what it contributes' \
'install:Install a plugin from a git URL or a local path' \
'update:Fetch a plugin'\''s own repository again and show what changed before it is applied' \
'remove:Remove a plugin by name' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode plugin commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin__subcmd__help_commands] )) ||
_xencode__subcmd__plugin__subcmd__help_commands() {
    local commands; commands=(
'list:List installed plugins, whether each one loads, and what it contributes' \
'install:Install a plugin from a git URL or a local path' \
'update:Fetch a plugin'\''s own repository again and show what changed before it is applied' \
'remove:Remove a plugin by name' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode plugin help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__plugin__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode plugin help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin__subcmd__help__subcmd__install_commands] )) ||
_xencode__subcmd__plugin__subcmd__help__subcmd__install_commands() {
    local commands; commands=()
    _describe -t commands 'xencode plugin help install commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin__subcmd__help__subcmd__list_commands] )) ||
_xencode__subcmd__plugin__subcmd__help__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode plugin help list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin__subcmd__help__subcmd__remove_commands] )) ||
_xencode__subcmd__plugin__subcmd__help__subcmd__remove_commands() {
    local commands; commands=()
    _describe -t commands 'xencode plugin help remove commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin__subcmd__help__subcmd__update_commands] )) ||
_xencode__subcmd__plugin__subcmd__help__subcmd__update_commands() {
    local commands; commands=()
    _describe -t commands 'xencode plugin help update commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin__subcmd__install_commands] )) ||
_xencode__subcmd__plugin__subcmd__install_commands() {
    local commands; commands=()
    _describe -t commands 'xencode plugin install commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin__subcmd__list_commands] )) ||
_xencode__subcmd__plugin__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode plugin list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin__subcmd__remove_commands] )) ||
_xencode__subcmd__plugin__subcmd__remove_commands() {
    local commands; commands=()
    _describe -t commands 'xencode plugin remove commands' commands "$@"
}
(( $+functions[_xencode__subcmd__plugin__subcmd__update_commands] )) ||
_xencode__subcmd__plugin__subcmd__update_commands() {
    local commands; commands=()
    _describe -t commands 'xencode plugin update commands' commands "$@"
}
(( $+functions[_xencode__subcmd__query_commands] )) ||
_xencode__subcmd__query_commands() {
    local commands; commands=()
    _describe -t commands 'xencode query commands' commands "$@"
}
(( $+functions[_xencode__subcmd__replay_commands] )) ||
_xencode__subcmd__replay_commands() {
    local commands; commands=()
    _describe -t commands 'xencode replay commands' commands "$@"
}
(( $+functions[_xencode__subcmd__review_commands] )) ||
_xencode__subcmd__review_commands() {
    local commands; commands=()
    _describe -t commands 'xencode review commands' commands "$@"
}
(( $+functions[_xencode__subcmd__scan_commands] )) ||
_xencode__subcmd__scan_commands() {
    local commands; commands=()
    _describe -t commands 'xencode scan commands' commands "$@"
}
(( $+functions[_xencode__subcmd__server_commands] )) ||
_xencode__subcmd__server_commands() {
    local commands; commands=()
    _describe -t commands 'xencode server commands' commands "$@"
}
(( $+functions[_xencode__subcmd__session_commands] )) ||
_xencode__subcmd__session_commands() {
    local commands; commands=(
'name:Name a run so it can be resumed without its id' \
'resolve:Resolve a name, id prefix, or \`latest\` to a full run id' \
'export:Print a session'\''s transcript; \`--redacted\` scrubs secrets' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode session commands' commands "$@"
}
(( $+functions[_xencode__subcmd__session__subcmd__export_commands] )) ||
_xencode__subcmd__session__subcmd__export_commands() {
    local commands; commands=()
    _describe -t commands 'xencode session export commands' commands "$@"
}
(( $+functions[_xencode__subcmd__session__subcmd__help_commands] )) ||
_xencode__subcmd__session__subcmd__help_commands() {
    local commands; commands=(
'name:Name a run so it can be resumed without its id' \
'resolve:Resolve a name, id prefix, or \`latest\` to a full run id' \
'export:Print a session'\''s transcript; \`--redacted\` scrubs secrets' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode session help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__session__subcmd__help__subcmd__export_commands] )) ||
_xencode__subcmd__session__subcmd__help__subcmd__export_commands() {
    local commands; commands=()
    _describe -t commands 'xencode session help export commands' commands "$@"
}
(( $+functions[_xencode__subcmd__session__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__session__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode session help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__session__subcmd__help__subcmd__name_commands] )) ||
_xencode__subcmd__session__subcmd__help__subcmd__name_commands() {
    local commands; commands=()
    _describe -t commands 'xencode session help name commands' commands "$@"
}
(( $+functions[_xencode__subcmd__session__subcmd__help__subcmd__resolve_commands] )) ||
_xencode__subcmd__session__subcmd__help__subcmd__resolve_commands() {
    local commands; commands=()
    _describe -t commands 'xencode session help resolve commands' commands "$@"
}
(( $+functions[_xencode__subcmd__session__subcmd__name_commands] )) ||
_xencode__subcmd__session__subcmd__name_commands() {
    local commands; commands=()
    _describe -t commands 'xencode session name commands' commands "$@"
}
(( $+functions[_xencode__subcmd__session__subcmd__resolve_commands] )) ||
_xencode__subcmd__session__subcmd__resolve_commands() {
    local commands; commands=()
    _describe -t commands 'xencode session resolve commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks_commands] )) ||
_xencode__subcmd__tasks_commands() {
    local commands; commands=(
'list:List known tasks with their derived status' \
'start:Start a background task (survives this CLI process)' \
'poll:Show a task'\''s status and captured output' \
'stop:Ask a running task to stop' \
'rm:Forget a finished task and delete its files' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode tasks commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__help_commands] )) ||
_xencode__subcmd__tasks__subcmd__help_commands() {
    local commands; commands=(
'list:List known tasks with their derived status' \
'start:Start a background task (survives this CLI process)' \
'poll:Show a task'\''s status and captured output' \
'stop:Ask a running task to stop' \
'rm:Forget a finished task and delete its files' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode tasks help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__tasks__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__help__subcmd__list_commands] )) ||
_xencode__subcmd__tasks__subcmd__help__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks help list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__help__subcmd__poll_commands] )) ||
_xencode__subcmd__tasks__subcmd__help__subcmd__poll_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks help poll commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__help__subcmd__rm_commands] )) ||
_xencode__subcmd__tasks__subcmd__help__subcmd__rm_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks help rm commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__help__subcmd__start_commands] )) ||
_xencode__subcmd__tasks__subcmd__help__subcmd__start_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks help start commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__help__subcmd__stop_commands] )) ||
_xencode__subcmd__tasks__subcmd__help__subcmd__stop_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks help stop commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__list_commands] )) ||
_xencode__subcmd__tasks__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__poll_commands] )) ||
_xencode__subcmd__tasks__subcmd__poll_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks poll commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__rm_commands] )) ||
_xencode__subcmd__tasks__subcmd__rm_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks rm commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__start_commands] )) ||
_xencode__subcmd__tasks__subcmd__start_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks start commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tasks__subcmd__stop_commands] )) ||
_xencode__subcmd__tasks__subcmd__stop_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tasks stop commands' commands "$@"
}
(( $+functions[_xencode__subcmd__test_commands] )) ||
_xencode__subcmd__test_commands() {
    local commands; commands=()
    _describe -t commands 'xencode test commands' commands "$@"
}
(( $+functions[_xencode__subcmd__toolchain_commands] )) ||
_xencode__subcmd__toolchain_commands() {
    local commands; commands=()
    _describe -t commands 'xencode toolchain commands' commands "$@"
}
(( $+functions[_xencode__subcmd__tui_commands] )) ||
_xencode__subcmd__tui_commands() {
    local commands; commands=()
    _describe -t commands 'xencode tui commands' commands "$@"
}
(( $+functions[_xencode__subcmd__verify_commands] )) ||
_xencode__subcmd__verify_commands() {
    local commands; commands=()
    _describe -t commands 'xencode verify commands' commands "$@"
}
(( $+functions[_xencode__subcmd__worktree_commands] )) ||
_xencode__subcmd__worktree_commands() {
    local commands; commands=(
'list:List git worktrees of the current repository' \
'add:Create a worktree at <path>, checking out <branch> (or a new branch named after the directory when omitted)' \
'remove:Remove a worktree (git refuses dirty worktrees; main never removable)' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode worktree commands' commands "$@"
}
(( $+functions[_xencode__subcmd__worktree__subcmd__add_commands] )) ||
_xencode__subcmd__worktree__subcmd__add_commands() {
    local commands; commands=()
    _describe -t commands 'xencode worktree add commands' commands "$@"
}
(( $+functions[_xencode__subcmd__worktree__subcmd__help_commands] )) ||
_xencode__subcmd__worktree__subcmd__help_commands() {
    local commands; commands=(
'list:List git worktrees of the current repository' \
'add:Create a worktree at <path>, checking out <branch> (or a new branch named after the directory when omitted)' \
'remove:Remove a worktree (git refuses dirty worktrees; main never removable)' \
'help:Print this message or the help of the given subcommand(s)' \
    )
    _describe -t commands 'xencode worktree help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__worktree__subcmd__help__subcmd__add_commands] )) ||
_xencode__subcmd__worktree__subcmd__help__subcmd__add_commands() {
    local commands; commands=()
    _describe -t commands 'xencode worktree help add commands' commands "$@"
}
(( $+functions[_xencode__subcmd__worktree__subcmd__help__subcmd__help_commands] )) ||
_xencode__subcmd__worktree__subcmd__help__subcmd__help_commands() {
    local commands; commands=()
    _describe -t commands 'xencode worktree help help commands' commands "$@"
}
(( $+functions[_xencode__subcmd__worktree__subcmd__help__subcmd__list_commands] )) ||
_xencode__subcmd__worktree__subcmd__help__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode worktree help list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__worktree__subcmd__help__subcmd__remove_commands] )) ||
_xencode__subcmd__worktree__subcmd__help__subcmd__remove_commands() {
    local commands; commands=()
    _describe -t commands 'xencode worktree help remove commands' commands "$@"
}
(( $+functions[_xencode__subcmd__worktree__subcmd__list_commands] )) ||
_xencode__subcmd__worktree__subcmd__list_commands() {
    local commands; commands=()
    _describe -t commands 'xencode worktree list commands' commands "$@"
}
(( $+functions[_xencode__subcmd__worktree__subcmd__remove_commands] )) ||
_xencode__subcmd__worktree__subcmd__remove_commands() {
    local commands; commands=()
    _describe -t commands 'xencode worktree remove commands' commands "$@"
}

if [ "$funcstack[1]" = "_xencode" ]; then
    _xencode "$@"
else
    compdef _xencode xencode
fi
