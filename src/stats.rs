//! Minimal RESP3 client for the `skeg-cli stats` subcommand.
//!
//! Connects to a running `skeg-resp3` server, runs `HELLO 3`,
//! `SKEG.STATS`, `SKEG.SHARDS`, and `SKEG.VINDEX.LIST`, and parses the
//! responses into the structs below. No persistent connection, no
//! pipelining: open, query, close. This is the read-only "what's the
//! server doing right now" path; for live dashboards use `skeg-top`.

use std::fmt;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use bytes::BytesMut;
use skeg_resp3::{Frame, ProtoVersion, encode_frame, parse_frame};

const READ_CAP: usize = 64 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const RW_TIMEOUT: Duration = Duration::from_secs(10);

/// Server-side info gathered from `HELLO 3`.
#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub version: String,
    pub mode: String,
}

/// Aggregate cache + KV counters from `SKEG.STATS`.
///
/// The server returns a flat `cache_bytes=X evictions=Y n_keys=Z budget=B`
/// status string; we parse the four fields. Unknown fields are ignored
/// so that a newer server adding more counters does not break the CLI.
#[derive(Debug, Clone, Default)]
pub struct AggregateStats {
    pub cache_bytes: u64,
    pub evictions: u64,
    pub n_keys: u64,
    pub budget: u64,
}

/// One `SKEG.SHARDS` line, parsed into the four counters the server
/// emits today. Unknown fields are ignored (same logic as
/// [`AggregateStats`]).
#[derive(Debug, Clone, Default)]
pub struct ShardStats {
    pub shard_id: u32,
    pub cache_bytes: u64,
    pub evictions: u64,
    pub n_keys: u64,
}

/// Full report assembled from a single `stats` call.
#[derive(Debug, Clone)]
pub struct ServerStats {
    pub info: ServerInfo,
    pub aggregate: AggregateStats,
    pub shards: Vec<ShardStats>,
    /// `(name, dim)` pairs from `SKEG.VINDEX.LIST`. Empty if the server
    /// has no indexes or the command returned a non-array response.
    pub vindexes: Vec<(String, u32)>,
}

/// Connect to `addr` (e.g. `127.0.0.1:6379`), gather server stats, and
/// return them. The TCP connection is closed before this returns.
///
/// # Errors
///
/// Returns an error if the connection fails, a read times out, the
/// server returns a `-ERR` frame for `HELLO 3`, or a response cannot
/// be parsed.
pub fn fetch(addr: &str) -> io::Result<ServerStats> {
    let mut sock = open(addr)?;
    let info = do_hello(&mut sock)?;
    let aggregate = parse_aggregate(&do_call(&mut sock, &["SKEG.STATS"])?);
    let shards = parse_shards(&do_call(&mut sock, &["SKEG.SHARDS"])?);
    let vindexes = parse_vindexes(&do_call(&mut sock, &["SKEG.VINDEX.LIST"])?);
    Ok(ServerStats {
        info,
        aggregate,
        shards,
        vindexes,
    })
}

fn open(addr: &str) -> io::Result<TcpStream> {
    let socket_addr = addr.parse().map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("bad address {addr:?}: {e}"),
        )
    })?;
    let sock = TcpStream::connect_timeout(&socket_addr, CONNECT_TIMEOUT)?;
    sock.set_read_timeout(Some(RW_TIMEOUT))?;
    sock.set_write_timeout(Some(RW_TIMEOUT))?;
    Ok(sock)
}

fn do_hello(sock: &mut TcpStream) -> io::Result<ServerInfo> {
    let frame = do_call(sock, &["HELLO", "3"])?;
    let map = match &frame {
        Frame::Map(m) => m,
        Frame::Error(e) => {
            return Err(io::Error::other(format!("HELLO failed: {e}")));
        }
        _ => return Err(io::Error::other("HELLO returned a non-map frame")),
    };
    let mut version = String::new();
    let mut mode = String::new();
    for (k, v) in map {
        let key = frame_as_string(k);
        let val = frame_as_string(v);
        match key.as_str() {
            "version" => version = val,
            "mode" => mode = val,
            _ => {}
        }
    }
    Ok(ServerInfo { version, mode })
}

/// Send one command (an array of bulk strings) and read one response
/// frame. The server is expected to answer exactly one frame; extra
/// bytes left in the socket are discarded by the next `do_call`.
fn do_call(sock: &mut TcpStream, args: &[&str]) -> io::Result<Frame> {
    let req = Frame::Array(
        args.iter()
            .map(|a| Frame::Bulk(a.as_bytes().to_vec().into()))
            .collect(),
    );
    let mut buf = BytesMut::new();
    encode_frame(&req, ProtoVersion::Resp3, &mut buf);
    sock.write_all(&buf)?;
    read_one_frame(sock)
}

fn read_one_frame(sock: &mut TcpStream) -> io::Result<Frame> {
    let mut buf = BytesMut::with_capacity(READ_CAP);
    let mut tmp = [0u8; 4096];
    loop {
        match parse_frame(&buf) {
            Ok(Some((frame, _consumed))) => return Ok(frame),
            Ok(None) => {
                let n = sock.read(&mut tmp)?;
                if n == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "server closed the connection mid-frame",
                    ));
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            Err(e) => return Err(io::Error::other(format!("RESP3 parse error: {e}"))),
        }
    }
}

/// Best-effort conversion of a frame into a string, mirroring how
/// `redis-cli` prints scalars. Anything not naturally a string yields
/// an empty string -- the caller decides whether that is an error.
fn frame_as_string(f: &Frame) -> String {
    match f {
        Frame::Simple(s) => s.clone(),
        Frame::Bulk(b) => String::from_utf8_lossy(b).into_owned(),
        Frame::Verbatim { data, .. } => String::from_utf8_lossy(data).into_owned(),
        Frame::Integer(i) => i.to_string(),
        Frame::Double(d) => d.to_string(),
        Frame::Boolean(b) => b.to_string(),
        _ => String::new(),
    }
}

fn parse_aggregate(frame: &Frame) -> AggregateStats {
    let s = frame_as_string(frame);
    let mut out = AggregateStats::default();
    apply_kv_pairs(&s, |key, value| match key {
        "cache_bytes" => out.cache_bytes = value,
        "evictions" => out.evictions = value,
        "n_keys" => out.n_keys = value,
        "budget" => out.budget = value,
        _ => {}
    });
    out
}

/// Parse the multi-line `SKEG.SHARDS` output, one line per shard.
fn parse_shards(frame: &Frame) -> Vec<ShardStats> {
    let s = frame_as_string(frame);
    let mut out = Vec::new();
    for line in s.lines() {
        if line.is_empty() {
            continue;
        }
        let mut shard = ShardStats::default();
        apply_kv_pairs(line, |key, value| match key {
            "shard" => {
                if let Ok(v) = u32::try_from(value) {
                    shard.shard_id = v;
                }
            }
            "cache_bytes" => shard.cache_bytes = value,
            "evictions" => shard.evictions = value,
            "n_keys" => shard.n_keys = value,
            _ => {}
        });
        out.push(shard);
    }
    out
}

/// Parse `SKEG.VINDEX.LIST` output. The server returns one
/// `name=<n> dim=<d> kind=<k> backend=<b> n_vectors=<v>` line per
/// index; we keep only `(name, dim)` for the summary.
fn parse_vindexes(frame: &Frame) -> Vec<(String, u32)> {
    let s = frame_as_string(frame);
    let mut out = Vec::new();
    for line in s.lines() {
        if line.is_empty() {
            continue;
        }
        let mut name = String::new();
        let mut dim = 0u32;
        for tok in line.split_whitespace() {
            let Some((k, v)) = tok.split_once('=') else {
                continue;
            };
            match k {
                "name" => name = v.to_string(),
                "dim" => {
                    if let Ok(d) = v.parse() {
                        dim = d;
                    }
                }
                _ => {}
            }
        }
        if !name.is_empty() {
            out.push((name, dim));
        }
    }
    out
}

/// Drive `f(key, value)` for every `key=value` token in `s`. Tokens
/// whose value is not a non-negative integer are skipped silently.
fn apply_kv_pairs<F: FnMut(&str, u64)>(s: &str, mut f: F) {
    for tok in s.split_whitespace() {
        let Some((k, v)) = tok.split_once('=') else {
            continue;
        };
        if let Ok(v) = v.parse::<u64>() {
            f(k, v);
        }
    }
}

impl fmt::Display for ServerStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "server     version={} mode={}",
            self.info.version, self.info.mode
        )?;
        writeln!(
            f,
            "aggregate  cache_bytes={} evictions={} n_keys={} budget={}",
            self.aggregate.cache_bytes,
            self.aggregate.evictions,
            self.aggregate.n_keys,
            self.aggregate.budget,
        )?;
        if !self.shards.is_empty() {
            writeln!(f, "shards     {}", self.shards.len())?;
            for s in &self.shards {
                writeln!(
                    f,
                    "  shard-{}: cache_bytes={} evictions={} n_keys={}",
                    s.shard_id, s.cache_bytes, s.evictions, s.n_keys
                )?;
            }
        }
        if self.vindexes.is_empty() {
            writeln!(f, "vindexes   (none)")?;
        } else {
            writeln!(f, "vindexes   {}", self.vindexes.len())?;
            for (name, dim) in &self.vindexes {
                writeln!(f, "  {name} (dim={dim})")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_parses_known_keys_ignores_unknown() {
        let f = Frame::Simple("cache_bytes=128 evictions=2 n_keys=4 budget=1024 future=99".into());
        let agg = parse_aggregate(&f);
        assert_eq!(agg.cache_bytes, 128);
        assert_eq!(agg.evictions, 2);
        assert_eq!(agg.n_keys, 4);
        assert_eq!(agg.budget, 1024);
    }

    #[test]
    fn shards_parses_one_line_per_shard() {
        let f = Frame::Simple(
            "shard=0 cache_bytes=10 evictions=0 n_keys=1\nshard=1 cache_bytes=20 evictions=0 n_keys=2".into(),
        );
        let v = parse_shards(&f);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].shard_id, 0);
        assert_eq!(v[0].cache_bytes, 10);
        assert_eq!(v[1].shard_id, 1);
        assert_eq!(v[1].n_keys, 2);
    }

    #[test]
    fn vindexes_parses_name_and_dim() {
        let f = Frame::Simple(
            "name=docs dim=4 kind=int8 backend=flat n_vectors=0\nname=other dim=128 kind=f32 backend=disk n_vectors=10".into(),
        );
        let v = parse_vindexes(&f);
        assert_eq!(v, vec![("docs".into(), 4), ("other".into(), 128)]);
    }

    #[test]
    fn vindexes_empty_on_no_output() {
        let f = Frame::Simple(String::new());
        assert!(parse_vindexes(&f).is_empty());
    }
}
