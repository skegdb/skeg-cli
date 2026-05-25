#![deny(unsafe_code)]

//! `skeg-cli` - offline index builder for skeg.
//!
//! The `build` command reads a vector dataset (`.npy` or `.fbin`), constructs
//! a Vamana graph once, and writes a ready-to-serve skeg data directory. A
//! server started with `--mode serve` against that directory serves VSEARCH
//! at the clean resident footprint - the streaming-insert path's
//! consolidation churn is never paid (see PLAN-POST-Q10 Step 1.5).
//!
//! The output layout mirrors a single-shard data directory:
//!
//! ```text
//! <output>/shard-0/vindexes.registry
//! <output>/shard-0/vindex-<name>/graph.vmn
//! <output>/shard-0/vindex-<name>/vectors.bin
//! ```

use std::io;
use std::path::Path;

use skeg_vector::{MmapVectorSource, VamanaConfig, VamanaIndex};

/// Single-shard subdirectory an offline build writes into. A server started
/// with `--mode serve` opens the output directory with one shard, so the
/// layout matches `shard-0/` of an ordinary data directory.
const SERVE_SHARD_DIR: &str = "shard-0";

/// VINDEX registry file name. The byte layout mirrors `skeg-server`'s
/// `shard::VINDEX_REGISTRY`: `[u32 count]` then, per entry,
/// `[u16 name_len][name][u32 dim]`.
const VINDEX_REGISTRY: &str = "vindexes.registry";

/// On-disk file names produced by `VamanaIndex::save`. Hardcoded here because
/// they are not exported by `skeg-vector`; they are a stable part of the
/// format.
const GRAPH_FILE: &str = "graph.vmn";
const VECTORS_FILE: &str = "vectors.bin";

/// Summary of a completed offline build.
pub struct BuildStats {
    /// Number of vectors indexed.
    pub n: usize,
    /// Vector dimension.
    pub dim: usize,
    /// Size of `graph.vmn` in bytes.
    pub graph_bytes: u64,
    /// Size of `vectors.bin` in bytes.
    pub vectors_bytes: u64,
}

fn bad_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Read a row-major f32 dataset, returning `(data, n, dim)`.
///
/// The format is selected by extension: `.npy` (`NumPy` v1.0, little-endian
/// f32, C order) or `.fbin`/`.bin` (`[u32 n][u32 dim][f32 data]`, the
/// `big-ann-benchmarks` layout).
///
/// # Errors
///
/// Returns an error if the file is missing, the extension is unknown, or the
/// contents do not parse as the expected format.
pub fn read_vectors(path: &Path) -> io::Result<(Vec<f32>, usize, usize)> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("npy") => read_npy(path),
        Some("fbin" | "bin") => read_fbin(path),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unsupported input extension {:?} (expected .npy or .fbin)",
                other.unwrap_or("")
            ),
        )),
    }
}

/// Parse the `(n, dim)` shape from a `NumPy` header dict string, rejecting
/// anything that is not a 2-D little-endian float32 array.
fn parse_npy_shape(header: &str) -> io::Result<(usize, usize)> {
    if !header.contains("<f4") {
        return Err(bad_data(
            "only little-endian float32 (<f4) .npy is supported",
        ));
    }
    let sh = header
        .find("'shape':")
        .ok_or_else(|| bad_data("no shape in .npy header"))?;
    let lp = header[sh..]
        .find('(')
        .ok_or_else(|| bad_data("malformed .npy shape"))?
        + sh
        + 1;
    let rp = header[lp..]
        .find(')')
        .ok_or_else(|| bad_data("malformed .npy shape"))?
        + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if dims.len() != 2 {
        return Err(bad_data("expected a 2-D .npy array"));
    }
    Ok((dims[0], dims[1]))
}

fn read_npy(path: &Path) -> io::Result<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(bad_data("not a .npy file (bad magic)"));
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    if 10 + header_len > bytes.len() {
        return Err(bad_data("truncated .npy header"));
    }
    let header = std::str::from_utf8(&bytes[10..10 + header_len])
        .map_err(|_| bad_data("non-utf8 .npy header"))?;
    let (n, dim) = parse_npy_shape(header)?;
    let payload = &bytes[10 + header_len..];
    let need = n.checked_mul(dim).and_then(|v| v.checked_mul(4));
    match need {
        Some(need) if payload.len() >= need => {
            let data = payload[..need]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok((data, n, dim))
        }
        _ => Err(bad_data("truncated .npy payload")),
    }
}

fn read_fbin(path: &Path) -> io::Result<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 {
        return Err(bad_data("truncated .fbin header"));
    }
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let need = n.checked_mul(dim).and_then(|v| v.checked_mul(4));
    match need {
        Some(need) if bytes.len() >= 8 + need => {
            let data = bytes[8..8 + need]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok((data, n, dim))
        }
        _ => Err(bad_data("truncated .fbin payload")),
    }
}

/// Parse only the format header of `path`, returning `(byte_offset, n, dim)`:
/// where the f32 payload begins and its shape. The payload itself is not
/// read, so a large dataset is not pulled into the heap.
///
/// # Errors
///
/// Returns an error if the extension is unknown or the header does not parse.
pub fn read_header(path: &Path) -> io::Result<(usize, usize, usize)> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("npy") => read_npy_header(path),
        Some("fbin" | "bin") => read_fbin_header(path),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unsupported input extension {:?} (expected .npy or .fbin)",
                other.unwrap_or("")
            ),
        )),
    }
}

fn read_npy_header(path: &Path) -> io::Result<(usize, usize, usize)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut pre = [0u8; 10];
    f.read_exact(&mut pre)
        .map_err(|_| bad_data("truncated .npy header"))?;
    if &pre[0..6] != b"\x93NUMPY" {
        return Err(bad_data("not a .npy file (bad magic)"));
    }
    let header_len = u16::from_le_bytes([pre[8], pre[9]]) as usize;
    let mut header_bytes = vec![0u8; header_len];
    f.read_exact(&mut header_bytes)
        .map_err(|_| bad_data("truncated .npy header"))?;
    let header =
        std::str::from_utf8(&header_bytes).map_err(|_| bad_data("non-utf8 .npy header"))?;
    let (n, dim) = parse_npy_shape(header)?;
    Ok((10 + header_len, n, dim))
}

fn read_fbin_header(path: &Path) -> io::Result<(usize, usize, usize)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr)
        .map_err(|_| bad_data("truncated .fbin header"))?;
    let n = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    Ok((8, n, dim))
}

/// Save a built index under `output` as a ready-to-serve data directory and
/// report its on-disk sizes.
fn finish_build(index: &VamanaIndex, output: &Path, name: &str) -> io::Result<BuildStats> {
    let shard_dir = output.join(SERVE_SHARD_DIR);
    let vindex_dir = shard_dir.join(format!("vindex-{name}"));
    index.save(&vindex_dir)?;
    write_registry(&shard_dir, name, index.dim())?;
    let graph_bytes = std::fs::metadata(vindex_dir.join(GRAPH_FILE))?.len();
    let vectors_bytes = std::fs::metadata(vindex_dir.join(VECTORS_FILE))?.len();
    Ok(BuildStats {
        n: index.len(),
        dim: index.dim(),
        graph_bytes,
        vectors_bytes,
    })
}

/// Build a Vamana index over an already-loaded dataset and write it under
/// `output` as a ready-to-serve data directory.
///
/// # Errors
///
/// Returns an error if the dataset is empty, `vectors.len()` disagrees with
/// `n * dim`, or any file cannot be written.
pub fn build_index_from(
    vectors: Vec<f32>,
    n: usize,
    dim: usize,
    output: &Path,
    name: &str,
    config: &VamanaConfig,
) -> io::Result<BuildStats> {
    if n == 0 || dim == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty dataset"));
    }
    if vectors.len() != n * dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected {} f32 values, got {}", n * dim, vectors.len()),
        ));
    }
    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(vectors, ids, dim, config);
    finish_build(&index, output, name)
}

/// Read a dataset from `input` and build a ready-to-serve index under
/// `output`.
///
/// The input file is memory-mapped: the build draws vectors from the mapping
/// rather than copying the whole dataset into the heap, so a dataset close to
/// or larger than RAM can still be indexed (post-Q10 Step 2, Build
/// Foundation).
///
/// # Errors
///
/// Returns an error if the input cannot be read or the index cannot be built.
pub fn build_index(
    input: &Path,
    output: &Path,
    name: &str,
    config: &VamanaConfig,
) -> io::Result<BuildStats> {
    let (byte_offset, n, dim) = read_header(input)?;
    if n == 0 || dim == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty dataset"));
    }
    let source = MmapVectorSource::open(input, byte_offset, n, dim)?;
    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build_from_source(Box::new(source), ids, config);
    finish_build(&index, output, name)
}

/// Write the single-entry VINDEX registry the server reads on startup.
fn write_registry(shard_dir: &Path, name: &str, dim: usize) -> io::Result<()> {
    std::fs::create_dir_all(shard_dir)?;
    let name_len = u16::try_from(name.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "index name too long"))?;
    let dim = u32::try_from(dim)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "dim too large"))?;
    let mut buf = Vec::new();
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&name_len.to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(&dim.to_le_bytes());
    std::fs::write(shard_dir.join(VINDEX_REGISTRY), &buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use skeg_vector::DiskVamanaIndex;
    use tempfile::TempDir;

    /// Encode a minimal valid `NumPy` v1.0 array of `n * dim` f32 values.
    fn make_npy(n: usize, dim: usize, data: &[f32]) -> Vec<u8> {
        let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({n}, {dim}), }}");
        // 10 fixed bytes + dict + trailing '\n', padded so the total is a
        // multiple of 64 (NumPy v1.0 alignment requirement).
        let unpadded = 10 + dict.len() + 1;
        let pad = (64 - unpadded % 64) % 64;
        let header_len = dict.len() + 1 + pad;
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&u16::try_from(header_len).unwrap().to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        out.extend(std::iter::repeat_n(b' ', pad));
        out.push(b'\n');
        for &x in data {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn make_fbin(n: usize, dim: usize, data: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&u32::try_from(n).unwrap().to_le_bytes());
        out.extend_from_slice(&u32::try_from(dim).unwrap().to_le_bytes());
        for &x in data {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    /// Deterministic 8-dim test vector, distinct per seed.
    #[allow(clippy::cast_precision_loss)]
    fn tvec(seed: u64) -> Vec<f32> {
        let mut s = (seed << 1) | 1;
        (0..8)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFFFF) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn npy_roundtrip() {
        let dir = TempDir::new().unwrap();
        let data: Vec<f32> = (0..12u8).map(f32::from).collect();
        let path = dir.path().join("d.npy");
        std::fs::write(&path, make_npy(3, 4, &data)).unwrap();
        let (got, n, dim) = read_vectors(&path).unwrap();
        assert_eq!((n, dim), (3, 4));
        assert_eq!(got, data);
    }

    #[test]
    fn fbin_roundtrip() {
        let dir = TempDir::new().unwrap();
        let data: Vec<f32> = (0..10u8).map(|i| f32::from(i) * 0.5).collect();
        let path = dir.path().join("d.fbin");
        std::fs::write(&path, make_fbin(2, 5, &data)).unwrap();
        let (got, n, dim) = read_vectors(&path).unwrap();
        assert_eq!((n, dim), (2, 5));
        assert_eq!(got, data);
    }

    #[test]
    fn truncated_fbin_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut bytes = make_fbin(2, 5, &[0.0; 10]);
        bytes.truncate(bytes.len() - 8);
        let path = dir.path().join("bad.fbin");
        std::fs::write(&path, bytes).unwrap();
        assert!(read_vectors(&path).is_err());
    }

    #[test]
    fn unknown_extension_is_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("d.csv");
        std::fs::write(&path, b"1,2,3").unwrap();
        assert!(read_vectors(&path).is_err());
    }

    #[test]
    fn build_produces_a_servable_index() {
        let n = 64;
        let dim = 8;
        let flat: Vec<f32> = (0..n).flat_map(|i| tvec(i as u64 + 1)).collect();
        let out = TempDir::new().unwrap();
        let stats =
            build_index_from(flat, n, dim, out.path(), "docs", &VamanaConfig::default()).unwrap();
        assert_eq!((stats.n, stats.dim), (n, dim));
        assert!(stats.graph_bytes > 0 && stats.vectors_bytes > 0);

        // The output must open as a DiskVamanaIndex and recover every vector.
        let vindex_dir = out.path().join("shard-0").join("vindex-docs");
        let index = DiskVamanaIndex::open(&vindex_dir).unwrap();
        assert_eq!(index.len(), n);
        assert_eq!(index.dim(), dim);
        let hits = index.search(&tvec(43), 5).unwrap();
        assert_eq!(hits[0].0, 42, "querying a stored vector returns its id");

        // The registry must name the index so the server recovers it.
        let registry = out.path().join("shard-0").join("vindexes.registry");
        assert!(registry.exists());
    }

    #[test]
    fn build_rejects_a_length_mismatch() {
        let out = TempDir::new().unwrap();
        let err = build_index_from(
            vec![0.0; 10],
            3,
            4,
            out.path(),
            "x",
            &VamanaConfig::default(),
        );
        assert!(err.is_err());
    }

    #[test]
    fn build_index_mmaps_input_and_serves() {
        // The file-reading `build_index` memory-maps the input rather than
        // slurping it; the result must still open and answer queries.
        let n = 64;
        let dim = 8;
        let flat: Vec<f32> = (0..n).flat_map(|i| tvec(i as u64 + 1)).collect();
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("data.fbin");
        std::fs::write(&input, make_fbin(n, dim, &flat)).unwrap();

        let out = TempDir::new().unwrap();
        let stats = build_index(&input, out.path(), "docs", &VamanaConfig::default()).unwrap();
        assert_eq!((stats.n, stats.dim), (n, dim));

        let vindex_dir = out.path().join("shard-0").join("vindex-docs");
        let index = DiskVamanaIndex::open(&vindex_dir).unwrap();
        assert_eq!(index.len(), n);
        let hits = index.search(&tvec(43), 5).unwrap();
        assert_eq!(hits[0].0, 42, "querying a stored vector returns its id");
    }
}
