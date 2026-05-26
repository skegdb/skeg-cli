#![deny(unsafe_code)]

//! `skeg-cli` CLI dispatch. Routes to `build`, `inspect`, and `stats`
//! subcommands. The library lives in `lib.rs`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use skeg_cli::{BuildStats, build_index, inspect, stats};
use skeg_vector::VamanaConfig;

const USAGE: &str = concat!(
    "skeg-cli ",
    env!("CARGO_PKG_VERSION"),
    " - operator tools for skeg

USAGE:
    skeg-cli <COMMAND> [OPTIONS]

COMMANDS:
    build      Build a Vamana index offline from a vector dataset.
    inspect    Walk a data directory and report VINDEXes and sizes.
    stats      Fetch SKEG.STATS / SKEG.SHARDS / SKEG.VINDEX.LIST from a
               running skeg-resp3 server.

GLOBAL OPTIONS:
    -h, --help     Print this help.
    -V, --version  Print the version.

Run 'skeg-cli <COMMAND> --help' for command-specific options.
"
);

const BUILD_USAGE: &str = "\
skeg-cli build - offline Vamana index builder

USAGE:
    skeg-cli build --input <FILE> --output <DIR> [OPTIONS]

Reads a vector dataset, constructs a Vamana graph once, and writes a
ready-to-serve skeg data directory. Serve it with:

    skeg --mode serve --data-dir <DIR>
    skeg-resp3 --mode serve --data-dir <DIR>

OPTIONS:
    --input  <FILE>   Dataset: .npy (NumPy f32) or .fbin ([u32 n][u32 dim][f32])
    --output <DIR>    Output data directory (created if missing)
    --name   <NAME>   VINDEX name [default: default]
    --r      <R>      Max graph out-degree [default: 64]
    --l      <L>      Query-time search-list size [default: 100]
";

const INSPECT_USAGE: &str = "\
skeg-cli inspect - offline introspection of a data directory

USAGE:
    skeg-cli inspect <DATA-DIR>

Walks <DATA-DIR>, enumerates every shard-<N>/ subdirectory, lists the
VINDEXes registered in each, and prints their dim, vector count, and
on-disk sizes. The server does not need to be running.
";

const STATS_USAGE: &str = "\
skeg-cli stats - fetch live stats from a running server

USAGE:
    skeg-cli stats <HOST:PORT>

Connects over RESP3 to <HOST:PORT> (default skeg-resp3 port is 6379),
runs HELLO 3, SKEG.STATS, SKEG.SHARDS, and SKEG.VINDEX.LIST, and prints
the aggregated result. Read-only; the connection is closed before exit.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("skeg-cli: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("build") => run_build(&args[1..]),
        Some("inspect") => run_inspect(&args[1..]),
        Some("stats") => run_stats(&args[1..]),
        None | Some("-h" | "--help") => {
            print!("{USAGE}");
            Ok(())
        }
        Some("-V" | "--version") => {
            println!("skeg-cli {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(other) => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    }
}

#[allow(clippy::cast_precision_loss)] // byte counts well under 2^53
fn run_build(args: &[String]) -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut name = "default".to_owned();
    let mut config = VamanaConfig::default();

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "-h" || flag == "--help" {
            print!("{BUILD_USAGE}");
            return Ok(());
        }
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for '{flag}'"))?;
        match flag {
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--name" => name.clone_from(value),
            "--r" => config.r = parse_usize(value, flag)?,
            "--l" => config.l_search = parse_usize(value, flag)?,
            other => return Err(format!("unknown flag '{other}'\n\n{BUILD_USAGE}")),
        }
        i += 2;
    }

    let input = input.ok_or("'build' requires --input")?;
    let output = output.ok_or("'build' requires --output")?;
    if name.is_empty() {
        return Err("--name must not be empty".to_owned());
    }

    eprintln!("building '{name}' from {} ...", input.display());
    let t0 = std::time::Instant::now();
    let stats: BuildStats =
        build_index(&input, &output, &name, &config).map_err(|e| format!("build failed: {e}"))?;
    let secs = t0.elapsed().as_secs_f64();
    let mib = (stats.graph_bytes + stats.vectors_bytes) as f64 / (1024.0 * 1024.0);
    eprintln!(
        "done in {secs:.1}s - {} vectors, dim {}, index {mib:.1} MiB",
        stats.n, stats.dim
    );
    eprintln!(
        "serve with: skeg --mode serve --data-dir {}",
        output.display()
    );
    Ok(())
}

fn run_inspect(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{INSPECT_USAGE}");
        return Ok(());
    }
    let dir = args
        .first()
        .ok_or_else(|| format!("'inspect' requires <DATA-DIR>\n\n{INSPECT_USAGE}"))?;
    let report = inspect::inspect(Path::new(dir)).map_err(|e| format!("inspect failed: {e}"))?;
    print!("{report}");
    Ok(())
}

fn run_stats(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{STATS_USAGE}");
        return Ok(());
    }
    let addr = args
        .first()
        .ok_or_else(|| format!("'stats' requires <HOST:PORT>\n\n{STATS_USAGE}"))?;
    let s = stats::fetch(addr).map_err(|e| format!("stats failed: {e}"))?;
    print!("{s}");
    Ok(())
}

fn parse_usize(s: &str, flag: &str) -> Result<usize, String> {
    s.parse()
        .map_err(|_| format!("'{flag}' expects an integer, got '{s}'"))
}
