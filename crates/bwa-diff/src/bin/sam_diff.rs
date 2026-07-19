//! `sam-diff <oracle.sam> <ours.sam> [--json <path>]`
//!
//! Prints a JSON concordance report (and optionally writes it to a file).
//!
//! Argument order is significant: the first file is the ORACLE (bwa-mem2's output) and the second
//! is ours. See `bwa_diff::compare` for why the two roles are not interchangeable, and the crate
//! docs for why a clean report here is not the same as byte-identity.
//!
//! Exit codes: 0 always on a successful comparison, 2 on a usage or I/O error. Note that a report
//! full of divergences still exits 0, so a CI gate must inspect the JSON, not the exit status.
//! Argument parsing is hand-rolled (no clap) to keep this diagnostic dependency-light; `--json` is
//! the only flag and anything else is taken as a positional path.

use std::path::Path;
use std::process::ExitCode;

/// Exit status for a usage or I/O failure. A comparison that RAN still exits 0 however divergent
/// its report, so callers must inspect the JSON rather than the status.
const EXIT_USAGE_OR_IO: u8 = 2;

/// How many divergence examples the report carries: enough to see a pattern in the failing reads,
/// few enough to read in a terminal. Chosen here, not inherited from any upstream tool.
const MAX_EXAMPLES: usize = 20;

fn main() -> ExitCode {
    // ---- Hand-rolled argv walk: `--json <path>`, everything else positional ----
    // Full argv including args[0], the program name, which the walk below skips by starting at 1.
    let args: Vec<String> = std::env::args().collect();
    // Non-flag arguments in the order given. Exactly two are required: oracle path then our path.
    let mut positional = Vec::new();
    // Destination for the JSON report if `--json <path>` was given; `None` means stdout only.
    // Either way the report is always printed to stdout as well.
    let mut json_out: Option<String> = None;
    // Cursor into `args`. Advanced by 2 for `--json` (flag plus its value), 1 otherwise.
    // Invariant at the top of each iteration: every argument before `arg_idx` has been classified
    // into `positional` or consumed by a flag.
    let mut arg_idx = 1;
    while arg_idx < args.len() {
        match args[arg_idx].as_str() {
            "--json" => {
                arg_idx += 1;
                if arg_idx >= args.len() {
                    eprintln!("--json requires a path");
                    return ExitCode::from(EXIT_USAGE_OR_IO);
                }
                json_out = Some(args[arg_idx].clone());
            }
            other => positional.push(other.to_string()),
        }
        arg_idx += 1;
    }
    if positional.len() != 2 {
        eprintln!("usage: sam-diff <oracle.sam> <ours.sam> [--json <path>]");
        return ExitCode::from(EXIT_USAGE_OR_IO);
    }
    // Positional order is the tool's contract: oracle first, candidate second.
    // `oracle_path` is bwa-mem2's SAM (the trusted side), `our_path` is the candidate under test.
    // Both are borrowed from `positional`, which is known to hold exactly two entries here.
    let (oracle_path, our_path) = (&positional[0], &positional[1]);

    // ---- Parse both files into "primary alignments keyed by read" ----
    // Primary records keyed `"<qname>/<mate>"`. Any I/O or UTF-8 failure is terminal: without both
    // sides there is nothing to compare, so it reports the offending path and exits 2.
    let oracle = match bwa_diff::parse_primary(Path::new(oracle_path)) {
        Ok(records) => records,
        Err(e) => {
            eprintln!("error reading {oracle_path}: {e}");
            return ExitCode::from(EXIT_USAGE_OR_IO);
        }
    };
    let ours = match bwa_diff::parse_primary(Path::new(our_path)) {
        Ok(records) => records,
        Err(e) => {
            eprintln!("error reading {our_path}: {e}");
            return ExitCode::from(EXIT_USAGE_OR_IO);
        }
    };

    // ---- Compare and report ----
    // The concordance summary: counts plus at most MAX_EXAMPLES divergences.
    let report = bwa_diff::compare(&oracle, &ours, MAX_EXAMPLES);
    // Pretty-printed JSON, the tool's sole output format. The `expect` is sound because `Report`
    // derives `Serialize` over plain owned counts, strings and a `Vec`, none of which can fail to
    // serialise.
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    if let Some(path) = &json_out {
        if let Err(e) = std::fs::write(path, &json) {
            eprintln!("error writing {path}: {e}");
            return ExitCode::from(EXIT_USAGE_OR_IO);
        }
    }
    println!("{json}");
    ExitCode::SUCCESS
}
