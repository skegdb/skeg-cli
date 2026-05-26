//! Offline inspection of a skeg data directory.
//!
//! Walks the on-disk layout produced by `skeg-server` (or `skeg-cli build`)
//! and reports what is in it: the VINDEX names per shard, vector counts,
//! graph/vectors file sizes. The server does not need to be running.
//!
//! The layout this expects mirrors what the server writes:
//!
//! ```text
//! <data-dir>/shard-<N>/
//!   vindexes.registry        <- [u32 count] + per entry [u16 name_len][name][u32 dim]
//!   vindex-<name>/
//!     graph.vmn
//!     vectors.bin
//! ```

use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use skeg_vector::DiskVamanaIndex;

/// One VINDEX as recorded in a shard's registry, with on-disk sizes.
#[derive(Debug, Clone)]
pub struct VindexEntry {
    pub name: String,
    pub dim: usize,
    /// Number of live vectors. `None` if the directory exists but the
    /// `DiskVamanaIndex` could not be opened (corrupted / partial write).
    pub n_vectors: Option<usize>,
    /// Size of `graph.vmn` in bytes. 0 if missing.
    pub graph_bytes: u64,
    /// Size of `vectors.bin` in bytes. 0 if missing.
    pub vectors_bytes: u64,
}

/// One shard's contribution to the inspect report.
#[derive(Debug, Clone)]
pub struct ShardReport {
    pub shard_id: usize,
    pub vindexes: Vec<VindexEntry>,
    /// Sum of every regular file directly under `shard-<N>/` that is not a
    /// `vindex-*` directory. Mostly: the vLog segments and the index
    /// snapshot.
    pub kv_bytes: u64,
}

/// Full report on a data directory.
#[derive(Debug, Clone)]
pub struct InspectReport {
    pub data_dir: PathBuf,
    pub shards: Vec<ShardReport>,
}

/// Inspect a data directory. Returns an empty report if the directory
/// contains no `shard-*` subdirectories.
///
/// # Errors
///
/// Returns an error only on filesystem errors that are not "missing file"
/// (e.g. permission denied). A truncated or missing `vindexes.registry`
/// is reported by leaving the shard's `vindexes` list empty.
pub fn inspect(data_dir: &Path) -> io::Result<InspectReport> {
    let mut shards = Vec::new();
    if !data_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("data directory not found: {}", data_dir.display()),
        ));
    }
    for entry in fs::read_dir(data_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(rest) = name.strip_prefix("shard-") else {
            continue;
        };
        let Ok(shard_id) = rest.parse::<usize>() else {
            continue;
        };
        shards.push(inspect_shard(shard_id, &entry.path())?);
    }
    shards.sort_by_key(|s| s.shard_id);
    Ok(InspectReport {
        data_dir: data_dir.to_path_buf(),
        shards,
    })
}

fn inspect_shard(shard_id: usize, shard_dir: &Path) -> io::Result<ShardReport> {
    let entries = read_registry(&shard_dir.join("vindexes.registry"))?;
    let mut vindexes = Vec::with_capacity(entries.len());
    for (name, dim) in entries {
        let vindex_dir = shard_dir.join(format!("vindex-{name}"));
        let graph_bytes = fs::metadata(vindex_dir.join("graph.vmn"))
            .map(|m| m.len())
            .unwrap_or(0);
        let vectors_bytes = fs::metadata(vindex_dir.join("vectors.bin"))
            .map(|m| m.len())
            .unwrap_or(0);
        let n_vectors = DiskVamanaIndex::open(&vindex_dir).ok().map(|i| i.len());
        vindexes.push(VindexEntry {
            name,
            dim,
            n_vectors,
            graph_bytes,
            vectors_bytes,
        });
    }
    let kv_bytes = kv_size_in_shard(shard_dir)?;
    Ok(ShardReport {
        shard_id,
        vindexes,
        kv_bytes,
    })
}

/// Sum every regular file directly inside `shard-<N>/` that is not the
/// registry and not a `vindex-*` subdirectory.
fn kv_size_in_shard(shard_dir: &Path) -> io::Result<u64> {
    let mut total = 0;
    for entry in fs::read_dir(shard_dir)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_file() {
            let name = entry.file_name();
            if name == "vindexes.registry" {
                continue;
            }
            total += entry.metadata()?.len();
        }
    }
    Ok(total)
}

/// Parse a `vindexes.registry` file into `(name, dim)` pairs. Missing file
/// yields an empty list.
fn read_registry(path: &Path) -> io::Result<Vec<(String, usize)>> {
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    if buf.len() < 4 {
        return Ok(Vec::new());
    }
    let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    let mut i = 4;
    for _ in 0..count {
        if i + 2 > buf.len() {
            break;
        }
        let name_len = u16::from_le_bytes([buf[i], buf[i + 1]]) as usize;
        i += 2;
        if i + name_len + 4 > buf.len() {
            break;
        }
        let name = String::from_utf8_lossy(&buf[i..i + name_len]).into_owned();
        i += name_len;
        let dim = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        out.push((name, dim));
    }
    Ok(out)
}

impl fmt::Display for InspectReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "data_dir: {}", self.data_dir.display())?;
        writeln!(f, "shards:   {}", self.shards.len())?;
        if self.shards.is_empty() {
            writeln!(f, "(no shard-* subdirectories found)")?;
            return Ok(());
        }
        let total_kv: u64 = self.shards.iter().map(|s| s.kv_bytes).sum();
        let total_vindexes: usize = self.shards.iter().map(|s| s.vindexes.len()).sum();
        writeln!(f, "kv_bytes: {}", human_bytes(total_kv))?;
        writeln!(f, "vindexes: {total_vindexes}")?;
        writeln!(f)?;
        for shard in &self.shards {
            writeln!(f, "[shard-{}]", shard.shard_id)?;
            writeln!(f, "  kv_bytes: {}", human_bytes(shard.kv_bytes))?;
            if shard.vindexes.is_empty() {
                writeln!(f, "  (no vindexes)")?;
                continue;
            }
            for v in &shard.vindexes {
                let n = v
                    .n_vectors
                    .map_or_else(|| "?".to_string(), |n| n.to_string());
                writeln!(
                    f,
                    "  vindex={} dim={} n={} graph={} vectors={}",
                    v.name,
                    v.dim,
                    n,
                    human_bytes(v.graph_bytes),
                    human_bytes(v.vectors_bytes),
                )?;
            }
        }
        Ok(())
    }
}

#[allow(clippy::cast_precision_loss)]
fn human_bytes(n: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let x = n as f64;
    if x >= GB {
        format!("{:.2} GiB", x / GB)
    } else if x >= MB {
        format!("{:.2} MiB", x / MB)
    } else if x >= KB {
        format!("{:.2} KiB", x / KB)
    } else {
        format!("{n} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_index_from;
    use skeg_vector::VamanaConfig;
    use tempfile::TempDir;

    #[allow(clippy::cast_precision_loss)]
    fn tvec(seed: u64, dim: usize) -> Vec<f32> {
        let mut s = (seed << 1) | 1;
        (0..dim)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFFFF) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn inspect_of_a_built_index_reports_vindex_and_dim() {
        let n = 32;
        let dim = 8;
        let flat: Vec<f32> = (0..n).flat_map(|i| tvec(i as u64 + 1, dim)).collect();
        let out = TempDir::new().unwrap();
        build_index_from(flat, n, dim, out.path(), "docs", &VamanaConfig::default()).unwrap();

        let report = inspect(out.path()).unwrap();
        assert_eq!(report.shards.len(), 1);
        let shard = &report.shards[0];
        assert_eq!(shard.shard_id, 0);
        assert_eq!(shard.vindexes.len(), 1);
        let v = &shard.vindexes[0];
        assert_eq!(v.name, "docs");
        assert_eq!(v.dim, dim);
        assert_eq!(v.n_vectors, Some(n));
        assert!(v.graph_bytes > 0);
        assert!(v.vectors_bytes > 0);
    }

    #[test]
    fn inspect_of_an_empty_dir_returns_empty_report() {
        let dir = TempDir::new().unwrap();
        let report = inspect(dir.path()).unwrap();
        assert!(report.shards.is_empty());
    }

    #[test]
    fn inspect_of_a_missing_dir_is_an_error() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("not-here");
        assert!(inspect(&missing).is_err());
    }

    #[test]
    fn registry_with_one_entry_roundtrips() {
        let dir = TempDir::new().unwrap();
        let shard = dir.path().join("shard-0");
        fs::create_dir_all(&shard).unwrap();
        // count=1, name="docs", dim=8 -- mirrors what build_index writes.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u16.to_le_bytes());
        buf.extend_from_slice(b"docs");
        buf.extend_from_slice(&8u32.to_le_bytes());
        fs::write(shard.join("vindexes.registry"), &buf).unwrap();

        let entries = read_registry(&shard.join("vindexes.registry")).unwrap();
        assert_eq!(entries, vec![("docs".to_string(), 8)]);
    }
}
