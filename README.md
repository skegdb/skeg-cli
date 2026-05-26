# skeg-cli

Operator tools for [skeg](https://github.com/skegdb/skeg). Three
subcommands: `build` (offline Vamana index builder), `inspect` (data
directory introspection), and `stats` (RESP3 client for live server
counters).

```sh
cargo install skeg-cli
brew tap skegdb/tap && brew install skeg-cli
```

Pre-built aarch64 tarballs (Apple Silicon, Linux ARM) are attached to
every GitHub Release.

## Commands

### `build`: offline Vamana index

Reads a vector dataset, constructs a Vamana graph, and writes a
ready-to-serve skeg data directory. For a static corpus (RAG dump,
embedding snapshot, scientific dataset) building offline gives the
server a clean index instead of paying the streaming-insert path's
consolidation cost.

```sh
skeg-cli build --input vectors.npy --output ./data --name docs --r 64 --l 100
```

| Flag             | Meaning                                                                                              | Default   |
| ---------------- | ---------------------------------------------------------------------------------------------------- | --------- |
| `--input <FILE>` | Dataset: `.npy` (NumPy v1.0 little-endian f32, C order) or `.fbin` (`[u32 n][u32 dim][f32 data]`)    | required  |
| `--output <DIR>` | Output data directory (created if missing)                                                           | required  |
| `--name <NAME>`  | VINDEX name the server will register                                                                 | `default` |
| `--r <R>`        | Max graph out-degree                                                                                 | `64`      |
| `--l <L>`        | Query-time search-list size                                                                          | `100`     |

Then serve it:

```sh
skeg --mode serve --data-dir ./data
# or
skeg-resp3 --mode serve --data-dir ./data
```

The input file is memory-mapped: vectors are read from the mapping
rather than copied into the heap, so a dataset close to or larger than
RAM can still be indexed.

### `inspect`: what's in a data directory

Walks a data directory, enumerates every `shard-<N>/` subdirectory,
lists the VINDEXes registered in each, and prints their dim, vector
count, and on-disk sizes. The server does not need to be running.

```sh
$ skeg-cli inspect ./data
data_dir: ./data
shards:   1
kv_bytes: 0 B
vindexes: 1

[shard-0]
  kv_bytes: 0 B
  vindex=docs dim=1024 n=1000000 graph=512.00 MiB vectors=3.81 GiB
```

`n` is `?` when the VINDEX directory exists but `DiskVamanaIndex::open`
fails (interrupted build, corrupted graph, partial write). `kv_bytes`
sums every regular file directly under `shard-<N>/` that is not a
`vindex-*` subdirectory: the vLog segments and the index snapshot.

### `stats`: live server counters

Connects to a running `skeg-resp3` server, runs `HELLO 3`,
`SKEG.STATS`, `SKEG.SHARDS`, and `SKEG.VINDEX.LIST`, and prints the
combined result. Read-only; the TCP connection is closed before exit.

```sh
$ skeg-cli stats 127.0.0.1:6379
server     version=0.1.1 mode=standalone
aggregate  cache_bytes=68 evictions=0 n_keys=1 budget=268435456
shards     8
  shard-0: cache_bytes=0 evictions=0 n_keys=0
  shard-1: cache_bytes=0 evictions=0 n_keys=0
  ...
  shard-5: cache_bytes=68 evictions=0 n_keys=1
vindexes   1
  docs (dim=4)
```

A live dashboard over the same data points (`skeg-top`) will ship in
a separate package alongside future releases.

## Global flags

| Flag              | Meaning                                              |
| ----------------- | ---------------------------------------------------- |
| `-h`, `--help`    | Print help (top-level or for a specific subcommand)  |
| `-V`, `--version` | Print the version                                    |

## Library use

`skeg-cli` also ships as a library. The `skeg_cli` crate exposes:

- `build_index`, `build_index_from`, `read_vectors`, `read_header`, `BuildStats`
  for building indexes from Rust code.
- `inspect::inspect` returning an `InspectReport` for embedding the
  data-directory walker in another tool.
- `stats::fetch` returning a `ServerStats` for embedding the RESP3
  client.

```rust
use std::path::Path;
use skeg_cli::{build_index, inspect, stats};
use skeg_vector::VamanaConfig;

build_index(Path::new("vectors.npy"), Path::new("./data"),
            "docs", &VamanaConfig::default())?;

let report = inspect::inspect(Path::new("./data"))?;
println!("{report}");

let s = stats::fetch("127.0.0.1:6379")?;
println!("{s}");
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
