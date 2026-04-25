use std::env;
use std::path::PathBuf;

use xencode_core_rs::{scan_workspace, ScanOptions};

fn main() {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_help();
        return;
    };

    let result = match command.as_str() {
        "scan" => run_scan(args.collect()),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        _ => Err(format!("unknown command: {command}")),
    };

    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run_scan(args: Vec<String>) -> Result<(), String> {
    let mut root = PathBuf::from(".");
    let mut options = ScanOptions::default();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--hidden" => {
                options.include_hidden = true;
                index += 1;
            }
            "--max-depth" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--max-depth requires a value".to_string())?;
                options.max_depth = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --max-depth value: {value}"))?,
                );
                index += 2;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown scan option: {value}"));
            }
            value => {
                root = PathBuf::from(value);
                index += 1;
            }
        }
    }

    let entries = scan_workspace(root, &options).map_err(|error| error.to_string())?;
    for entry in entries {
        let bytes = entry
            .bytes
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        println!("{}\t{}\t{}", entry.kind, bytes, entry.path.display());
    }

    Ok(())
}

fn print_help() {
    println!(
        "xencode rust prototype\n\n\
Commands:\n\
  scan [path] [--hidden] [--max-depth N]   List workspace entries\n\
  help                                    Show this help"
    );
}

