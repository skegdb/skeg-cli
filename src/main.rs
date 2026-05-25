#![deny(unsafe_code)]

//! `skeg-cli` CLI - offline index builder.

use std::path::PathBuf;
use std::process::ExitCode;

use skeg_cli::{BuildStats, build_index};
use skeg_vector::VamanaConfig;

const USAGE: &str = "\
skeg-cli - offline index builder for skeg

USAGE:
    skeg-cli build --input <FILE> --output <DIR> [OPTIONS]

The 'build' command reads a vector dataset, constructs a Vamana graph once,
and writes a ready-to-serve skeg data directory. Serve it with:

    skeg-server --mode serve --data-dir <DIR>

OPTIONS for 'build':
    --input  <FILE>   Dataset: .npy (NumPy f32) or .fbin ([u32 n][u32 dim][f32])
    --output <DIR>    Output data directory (created if missing)
    --name   <NAME>   VINDEX name [default: default]
    --r      <R>      Max graph out-degree [default: 64]
    --l      <L>      Query-time search-list size [default: 100]
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
        None | Some("-h" | "--help") => {
            print!("{USAGE}");
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
            print!("{USAGE}");
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
            other => return Err(format!("unknown flag '{other}'\n\n{USAGE}")),
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
        "serve with: skeg-server --mode serve --data-dir {}",
        output.display()
    );
    Ok(())
}

fn parse_usize(s: &str, flag: &str) -> Result<usize, String> {
    s.parse()
        .map_err(|_| format!("'{flag}' expects an integer, got '{s}'"))
}
