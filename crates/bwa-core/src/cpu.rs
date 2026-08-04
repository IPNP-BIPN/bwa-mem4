//! CPU topology probing, for the one case where it changes how work is scheduled.
//!
//! Apple Silicon is asymmetric: a handful of Efficiency cores sit alongside the Performance cores
//! and appear in `hw.ncpu` as if they were equals. They are not. Measured on an M4 Max (12 P + 4 E)
//! aligning 500k pairs against GRCh38, `-t16` against `-t12`:
//!
//! | `-t` | wall | CPU |
//! |---|---|---|
//! | 12 | 6.17 s | 49.5 s |
//! | 16 | 6.10 s | 54.5 s |
//!
//! The four E cores bought nothing measurable (1%, inside the run-to-run spread of three
//! repetitions) and cost 10% more CPU. Driven onto E cores deliberately with `taskpolicy -b`, the
//! same work took 5.75x longer, so an E core is worth a fraction of a P core here and the even
//! split rayon applies leaves its E-core chunks straggling while everything else waits.
//!
//! There is no way to *forbid* a core on this platform: `thread_policy_set` with
//! `THREAD_AFFINITY_POLICY` is a no-op on arm64, and a QoS class is a preference the scheduler may
//! ignore once there are more runnable threads than P cores. Capping the worker count is the only
//! lever that actually keeps work off them.
//!
//! **QoS on top of the cap was tried and measured at zero**, so it is deliberately not implemented.
//! Setting `QOS_CLASS_USER_INITIATED` or `QOS_CLASS_USER_INTERACTIVE` on each rayon worker via
//! `pthread_set_qos_class_self_np`, at `-t12` on the same 500k-pair workload, five interleaved
//! pairs: 6.18 s default against 6.12 s interactive, a 1% difference inside the 0.36 s spread of
//! each condition, with the last two pairs identical to the hundredth. Interleaving mattered, since
//! the runs drift upward with heat (5.97 s to 6.33 s across the series) and a sequential A-then-B
//! comparison would have manufactured a result. The reading is that once the pool is capped, the
//! twelve workers already land on the twelve P cores and there is nothing left for a scheduling
//! hint to fix. This is the "QoS hints for performance cores" item from `fg-labs/bwa-mem3`'s tuning
//! list; it is spent.
//!
//! # Rust mechanics used in this file
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `Option<T>` | a value that is either `Some(value)` or `None`. It is how Rust says "there may be no answer" without a null pointer: the compiler forces every reader to handle the `None` case, so a missing value cannot be dereferenced by mistake. |
//! | `usize` | an unsigned integer the width of a machine pointer, the type used for counts and indices. |
//! | `#[cfg(...)]` | conditional COMPILATION. The block that follows exists only when the condition holds; on other platforms it is not compiled at all, rather than compiled and skipped. The two blocks below are mutually exclusive, so exactly one is present in any given build. |
//! | `{ ... }` as a value | a braced block is an expression: its last line, written without a semicolon, is the block's value. That is how each `#[cfg]` block below supplies the function's return value with no `return` keyword. |
//! | `.ok()` | converts a `Result` (succeeded-or-failed) into an `Option` (have-a-value-or-not), discarding the error detail. Used here because the caller only needs to know that the probe failed, not why. |
//! | `?` on an `Option` | the propagation operator again: "if this is `None`, stop and return `None` from the whole function". Chaining `.ok()?` four times is how the four separate ways this probe can fail all collapse into one `None`. |
//! | `let n: usize = ...` | an explicit type annotation. It is needed here because `.parse()` can produce many numeric types and nothing else in the line says which one is wanted. |
//! | `.then_some(v)` | on a `bool`: yields `Some(v)` when true and `None` when false. A compact way to turn a validity test into an optional answer. |
//! | `if let Some(n) = ...` | run this block only if the value is `Some`, binding its contents to `n`. The `None` case is silently skipped, which in the test below is deliberate. |
//! | `{n}` inside a message | inline interpolation: prints the variable named `n`. Same idea as a format placeholder, with the name written where the value goes. |

/// Number of Performance cores, or `None` when the question does not apply or cannot be answered.
///
/// `None` on every platform except macOS, and on macOS whenever the probe fails: an Intel Mac has
/// no `hw.perflevel0` at all (its cores are symmetric, so there is nothing to cap), and a future
/// layout that stops reporting it should degrade to "use every core" rather than to an error.
///
/// # Returns
///
/// `Some(n)` with `n >= 1` on an asymmetric Apple Silicon host, where `n` is the logical CPU count
/// of performance level 0. `None` otherwise; the caller then uses the thread count it was given.
pub fn performance_core_count() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        // `hw.perflevel0` is the fastest level and `hw.perflevel1` the slowest, so level 0 is the P
        // cores. Subprocess rather than a libc `sysctlbyname` call to keep this crate
        // dependency-free, matching `sysram::total_ram_bytes`; it runs once per process.
        //
        // Rust: this builds a child process step by step, then runs it. `.output()` waits for it to
        // finish and hands back its captured stdout, stderr and exit status, wrapped in a `Result`
        // because spawning can fail (no such binary, no permission). `.ok()?` says: on failure,
        // abandon the probe and return `None` from `performance_core_count` immediately.
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.logicalcpu"])
            .output()
            .ok()?;
        // Turn the captured bytes into a number, giving up at the first step that does not work.
        // Read left to right: interpret stdout as text (it may not be valid UTF-8), drop the
        // trailing newline `sysctl` prints, then parse what is left as an integer (it may not be
        // one). Each `.ok()?` is an independent bail-out, so a host that answers with garbage
        // yields `None` rather than a wrong core count or a crash.
        //
        // Rust: `&out.stdout` borrows the captured bytes instead of copying them. The `: usize`
        // annotation is what tells `.parse()` which kind of number to produce.
        let n: usize = std::str::from_utf8(&out.stdout).ok()?.trim().parse().ok()?;
        // A host reporting zero P cores is nonsense; treat it as "cannot answer" rather than
        // capping the pool to nothing.
        //
        // Rust: `(n >= 1)` is a plain boolean, and `.then_some(n)` converts it into the answer:
        // `Some(n)` when the test holds, `None` when it does not. No semicolon, so this is the
        // value of the whole `#[cfg]` block and therefore the function's return value.
        (n >= 1).then_some(n)
    }
    // The non-macOS build. This block and the one above are the same function body written twice,
    // and only one of the two is ever compiled: on Linux the `sysctl` code above does not exist in
    // the binary at all, which is why this file needs no runtime platform check.
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe must never return a value that would starve the pool, on any host. It is allowed
    /// to return `None` (non-macOS, Intel Mac, or an unparseable answer); what it may not do is
    /// return `Some(0)`.
    #[test]
    fn performance_core_count_is_never_zero() {
        // Rust: `if let Some(n) = ...` runs the body only when the probe answered, binding the
        // count to `n`. There is no `else`, so a `None` answer passes the test silently, which is
        // the point: this must pass on a Linux CI runner and on an Intel Mac, where `None` is the
        // correct result. The test asserts the shape of a positive answer, not that one exists.
        if let Some(n) = performance_core_count() {
            assert!(n >= 1, "probe returned {n} performance cores");
        }
    }
}
