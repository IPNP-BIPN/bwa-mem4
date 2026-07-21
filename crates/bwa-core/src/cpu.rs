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
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.logicalcpu"])
            .output()
            .ok()?;
        let n: usize = std::str::from_utf8(&out.stdout).ok()?.trim().parse().ok()?;
        // A host reporting zero P cores is nonsense; treat it as "cannot answer" rather than
        // capping the pool to nothing.
        (n >= 1).then_some(n)
    }
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
        if let Some(n) = performance_core_count() {
            assert!(n >= 1, "probe returned {n} performance cores");
        }
    }
}
