//! The `bwa-mem4` command implementations, callable without a subprocess.
//!
//! The crate has always built a binary. Everything the binary does lives in
//! these modules; exposing them means an embedder can run an alignment in
//! process, with the same code path and therefore the same byte-identical
//! output, instead of spawning `bwa-mem4` and parsing its stdout.
//!
//! ```no_run
//! use bwa_mem4::cmd_mem::{run, MemArgs};
//!
//! let args = MemArgs {
//!     index: "hg38".into(),
//!     reads: vec!["R1.fq.gz".into(), "R2.fq.gz".into()],
//!     threads: 8,
//!     ..Default::default()
//! };
//! run(args, &["bwa-mem4".to_string(), "mem".to_string()])?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! `main.rs` uses these same modules, so there is one copy of each and no risk
//! of the library and the binary drifting apart.

pub mod cmd_index;
pub mod cmd_longread;
pub mod cmd_mem;
pub mod stage_time;
