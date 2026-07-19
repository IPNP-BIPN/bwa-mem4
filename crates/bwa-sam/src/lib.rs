//! **Empty placeholder. Nothing lives here and nothing depends on this crate.**
//!
//! It was reserved early on for "phases 5 to 7" (primary/secondary marking, MAPQ, CIGAR, SAM
//! tags), but that work ended up next to the code it needs, and this crate was never filled in.
//! Its earlier doc comment still advertised those phases, which would send a reader looking here
//! for code that has never existed. The real locations are:
//!
//! | What you are looking for | Where it actually is |
//! |---|---|
//! | Primary/secondary marking, MAPQ | `crates/bwa-mem/src/primary.rs` |
//! | CIGAR generation and clipping | `crates/bwa-mem/src/cigar.rs` |
//! | SAM header and record formatting | `crates/bwa-io/src/sam.rs` |
//! | XA/SA tags, ALT contig handling | `crates/bwa-mem/src/alt.rs` |
//!
//! Kept only so the workspace layout stays stable. Deleting it is safe: no `Cargo.toml` in the
//! workspace lists `bwa-sam` as a dependency.
