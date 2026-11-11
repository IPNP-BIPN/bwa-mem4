//! `bwa-mem4` command-line entry point.
//!
//! Two subcommands, mirroring the two things you do with a read aligner: `index` prepares a
//! reference genome once, `mem` aligns reads against it. Both are drop-in compatible with their
//! `bwa-mem2` equivalents, which is the whole point: the same command line must produce the same
//! bytes. See [`cmd_mem::MemArgs`] for the option surface and the three bwa options not carried
//! over.

//! # Rust mechanics used in this file
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `main.rs` | by convention the EXECUTABLE's entry point, as opposed to `lib.rs`, which is a library's root. This is the function the operating system starts. |
//! | `#[derive(Parser)]` | generates a whole command-line parser from a struct definition, using the `clap` crate. The fields become options and the doc comments become the help text, so the declaration and the documentation cannot drift apart. |
//! | `#[derive(Subcommand)]` | the same for an enum, where each variant is one subcommand (`index`, `mem`). |
//! | `#[global_allocator]` | designates which allocator the whole program uses for every heap allocation. Declaring one replaces the system default process-wide, including inside dependencies. |
//! | `static GLOBAL: ... = ...;` | a single value existing for the program's whole run. Here it is the allocator itself. |
//! | `use a::{B, C};` | imports two names from one module, so they can be written unqualified below. |

use clap::{Parser, Subcommand};

// mimalloc as the global allocator: the pipeline makes many small short-lived allocations (per-job
// query/target buffers, per-chunk DP scratch, per-read region vectors); a fast allocator with good
// locality cuts wall time noticeably. Does not affect output bytes (byte-identity preserved).
//
// Behind a feature because a global allocator is a whole-program decision and
// this crate is also a library now; see Cargo.toml. On by default, so this
// binary is unchanged.
#[cfg(all(feature = "mimalloc", not(feature = "stage-alloc")))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// mimalloc, called the way C's `malloc` calls it. The `mimalloc` crate sends every request through
/// `mi_malloc_aligned`, whose fast path only covers blocks of at most 1 KiB with a free slot ready:
/// any other size that is not a power of two falls to its over-allocation path (`size + align - 1`,
/// often one size class up) and leaves the page flagged for interior pointers. Every mimalloc block
/// is at least 8-byte aligned, so an alignment of 8 or less needs none of that.
#[cfg(feature = "mimalloc")]
struct MiMalloc;

#[cfg(feature = "mimalloc")]
// SAFETY: every path returns a block of at least `size` bytes aligned to `align` (mimalloc's
// word alignment covers `align <= 8`; the aligned calls cover the rest), or null on failure.
unsafe impl std::alloc::GlobalAlloc for MiMalloc {
    #[inline]
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        use libmimalloc_sys::{mi_malloc, mi_malloc_aligned};
        if l.align() <= 8 {
            mi_malloc(l.size()).cast()
        } else {
            mi_malloc_aligned(l.size(), l.align()).cast()
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        use libmimalloc_sys::{mi_zalloc, mi_zalloc_aligned};
        if l.align() <= 8 {
            mi_zalloc(l.size()).cast()
        } else {
            mi_zalloc_aligned(l.size(), l.align()).cast()
        }
    }

    #[inline]
    unsafe fn dealloc(&self, p: *mut u8, _: std::alloc::Layout) {
        libmimalloc_sys::mi_free(p.cast());
    }

    #[inline]
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, size: usize) -> *mut u8 {
        use libmimalloc_sys::{mi_realloc, mi_realloc_aligned};
        if l.align() <= 8 {
            mi_realloc(p.cast(), size).cast()
        } else {
            mi_realloc_aligned(p.cast(), size, l.align()).cast()
        }
    }
}

/// mimalloc returns freed pages to the OS a second after they empty (`purge_delay`), and the
/// next chunk faults them back in: on Linux that is `MADV_DONTNEED` plus a zeroed page per touch,
/// measured at 1.8% of the cycles on a Neoverse N2 (0.5% on an M4), for +8% peak RSS on chr21 and
/// none on a human index. Never purging keeps them; `MIMALLOC_PURGE_DELAY` still overrides.
#[cfg(feature = "mimalloc")]
fn keep_freed_pages() {
    extern "C" {
        fn mi_option_set(option: std::ffi::c_int, value: std::ffi::c_long);
    }
    // `mi_option_purge_delay` in mimalloc v3's `mi_option_t` (libmimalloc-sys's own constant is
    // v2's numbering).
    const MI_OPTION_PURGE_DELAY: std::ffi::c_int = 15;
    if std::env::var_os("MIMALLOC_PURGE_DELAY").is_none() {
        // SAFETY: plain option setter; mimalloc reads it when it next considers a purge.
        unsafe { mi_option_set(MI_OPTION_PURGE_DELAY, -1) };
    }
}

// With `--features stage-alloc`, the same allocator wrapped in `stage_alloc::Counting`, which
// forwards every request and counts the bytes against the pipeline stage that asked for them. Off
// by default because that wrapper costs a measured +0.5% even disarmed, one branch per allocation
// times 167M allocations per 500k pairs; see the feature's entry in Cargo.toml and `stage_alloc`
// for why issue #25 needs a counter inside the allocator rather than another peak-RSS snapshot.
#[cfg(all(feature = "mimalloc", feature = "stage-alloc"))]
#[global_allocator]
static GLOBAL: bwa_mem4::stage_alloc::Counting<MiMalloc> =
    bwa_mem4::stage_alloc::Counting::new(MiMalloc);

// Without the `mimalloc` feature the binary declares no allocator at all and gets the system one.
// The probe still wraps it, so a build made to compare allocators can measure both with the same
// instrument.
#[cfg(all(not(feature = "mimalloc"), feature = "stage-alloc"))]
#[global_allocator]
static GLOBAL: bwa_mem4::stage_alloc::Counting<std::alloc::System> =
    bwa_mem4::stage_alloc::Counting::new(std::alloc::System);

// The command implementations live in the library target so an embedder can
// call them without spawning this binary. Declared there, used here, so there
// is exactly one copy of each.
use bwa_mem4::{cmd_index, cmd_mem};

// Top-level parsed command line. Holds nothing but the chosen subcommand: every real option lives
// on `cmd_index::IndexArgs` or `cmd_mem::MemArgs`.
//
// Deliberately a `//` comment, not `///`: clap derives `long_about` from a struct's doc comment, so
// a `///` here would appear in `bwa-mem4 --help`. The `about` string in the attribute below is the
// only help text this struct should contribute.
#[derive(Parser)]
#[command(
    name = "bwa-mem4",
    version,
    about = "Native Rust reimplementation of bwa-mem2"
)]
struct Cli {
    // Which subcommand was invoked, with its own already-parsed arguments. Always set: clap
    // refuses the command line outright when no subcommand is named.
    //
    // `//` rather than `///` so clap cannot pick this up as help text.
    #[command(subcommand)]
    cmd: Cmd,
}

/// The two subcommands. Note that clap's `about` strings here are ours, not bwa's: bwa has no
/// structured help for its subcommands.
#[derive(Subcommand)]
enum Cmd {
    // Variant payload: the one FASTA path to index. Run once per reference, minutes and tens of GB.
    /// Build the FMD index from a FASTA reference. Output is byte-identical to `bwa-mem2 index`.
    Index(cmd_index::IndexArgs),
    // Variant payload: the whole `mem` option surface (~35 flags plus 2-3 positionals). This is the
    // variant every production run takes.
    /// Align reads to an indexed reference.
    Mem(cmd_mem::MemArgs),
    // No payload. Exists purely because `bwa-mem2 version` is a SUBCOMMAND, not a flag, and a
    // drop-in replacement that only answers `--version` breaks any script that calls the former.
    // It prints the bare version and nothing else, as bwa-mem2 does ("2.3"), so a caller can use it
    // unquoted; `--version` keeps clap's "bwa-mem4 4.3.3" form for humans.
    /// Print the version number, as `bwa-mem2 version` does.
    Version,
}

/// A baseline x86 build hands the process over to the `x86-64-v3` build shipped beside it
/// (`bwa-mem4-x86-64-v3`, same release archive) when this CPU runs it: identical output, 11%
/// faster on real reads (5.45M GIAB pairs on an EPYC 9V74: 41.3 s against 46.4 s). `argv[0]` is
/// passed through, so the @PG line is the same. Returns only when there is nothing to hand over to,
/// or the `exec` failed, and the baseline build runs on.
#[cfg(all(unix, target_arch = "x86_64", not(target_feature = "avx2")))]
fn exec_v3() {
    use std::os::unix::process::CommandExt;
    let v3 = is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("bmi2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
        && is_x86_feature_detected!("lzcnt")
        && is_x86_feature_detected!("movbe");
    let Some(exe) = v3.then(std::env::current_exe).and_then(Result::ok) else {
        return;
    };
    let exe = exe.with_file_name("bwa-mem4-x86-64-v3");
    if exe.is_file() {
        let mut args = std::env::args_os();
        let arg0 = args.next().unwrap_or_default();
        let _ = std::process::Command::new(exe).arg0(arg0).args(args).exec();
    }
}

/// Process entry point: capture argv, parse it, dispatch to the chosen subcommand.
///
/// # Returns
///
/// `Ok(())` after the subcommand has run to completion (for `mem`, after the SAM sink has been
/// flushed and finalized). Any error propagates out of `main`, so the process exits non-zero and
/// prints the anyhow chain; there is no partial-success exit code.
fn main() -> anyhow::Result<()> {
    #[cfg(all(unix, target_arch = "x86_64", not(target_feature = "avx2")))]
    exec_v3();
    // Capture the raw command line for the @PG CL tag before clap consumes it. It must be the raw
    // argv, not a reconstruction from the parsed args: the CL field is meant to record what the
    // user actually typed, defaults and all, so the SAM file documents its own provenance.
    let argv: Vec<String> = std::env::args().collect();
    // Arms the allocation probe if `BWA4_STAGE_ALLOC` is set. First thing after argv, because it
    // reads the environment (which allocates) and the allocator must never do that itself; every
    // allocation before this line goes uncounted, which is clap's parsing and nothing else.
    bwa_mem4::stage_alloc::init();
    #[cfg(feature = "mimalloc")]
    keep_freed_pages();
    match Cli::parse().cmd {
        Cmd::Index(args) => cmd_index::run(args),
        Cmd::Mem(args) => cmd_mem::run(args, &argv),
        Cmd::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
    }
}
