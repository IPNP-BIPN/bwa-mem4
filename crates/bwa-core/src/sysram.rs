//! System-RAM detection, for the learned-index auto-select: LISA seeding needs a large in-memory
//! index (~78 GB at hg38 scale), worth it only on big-memory hosts, so the aligner picks LISA when
//! enough RAM is present and otherwise falls back to the classic FM-index path, transparently, so a
//! 16-64 GB machine is never penalized.
//!
//! Nothing here affects SAM bytes: both index paths produce the same alignments, so this file is
//! purely a performance-policy decision. Reading order: [`total_ram_bytes`] then
//! [`learned_index_fits`], its only caller-facing use.

/// Headroom multiplier on the index footprint: the host must have 1.25x the index size before the
/// learned index is chosen, leaving room for the OS, the page cache and per-thread scratch. The
/// figure is a policy choice made here, not a number taken from any upstream source.
const RAM_HEADROOM_FACTOR: f64 = 1.25;

/// Bytes per gigabyte as used by the comparison below: decimal (1e9), matching how RAM sizes are
/// quoted on spec sheets, NOT the binary 2^30.
const BYTES_PER_GB: f64 = 1e9;

/// Total physical RAM in bytes, or `None` if it can't be determined. macOS reads `hw.memsize`
/// (`sysctl`); Linux reads `/proc/meminfo`'s `MemTotal`.
///
/// # Returns
///
/// Total PHYSICAL RAM in bytes, not free or available RAM: the caller wants the machine's capacity,
/// since the index is mapped once and shared. `None` on any other platform, and on macOS/Linux
/// whenever the probe fails (no `sysctl` binary, unreadable `/proc`, unparseable output). Every
/// failure collapses to `None` via `ok()?` rather than surfacing a distinct error, because the only
/// caller treats all of them the same way: fall back to the FM index.
pub fn total_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // `sysctl -n hw.memsize` prints the byte count and nothing else, so the whole trimmed
        // stdout parses directly as the answer. Subprocess rather than a libc `sysctlbyname` call
        // to keep this crate dependency-free; it runs once per process, so the fork is free.
        let sysctl_output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        std::str::from_utf8(&sysctl_output.stdout)
            .ok()?
            .trim()
            .parse()
            .ok()
    }
    #[cfg(target_os = "linux")]
    {
        // The whole pseudo-file, a few kilobytes of `Label: value kB` lines. Read in one go rather
        // than streamed: `MemTotal` is the first line in practice, but the format guarantees only
        // that it is present.
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        // Scans for the one line that matters; falls out to `None` if the file lacks it entirely.
        for line in meminfo.lines() {
            // `MemTotal:` is reported in kibibytes, hence the * 1024.
            if let Some(after_label) = line.strip_prefix("MemTotal:") {
                // The numeric field of `MemTotal:    16384000 kB`: leading spaces dropped by
                // `split_whitespace`, the trailing `kB` unit token discarded by taking only the
                // first token.
                let kibibytes: u64 = after_label.split_whitespace().next()?.parse().ok()?;
                return Some(kibibytes * 1024);
            }
        }
        None
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// Whether to use the learned index: total RAM must comfortably exceed the index footprint (a 1.25x
/// headroom for the OS, page cache, and per-thread scratch). Unknown RAM => `false` (use FM), so the
/// safe classic path is always the default when detection fails.
///
/// # Parameters
///
/// - `index_gb`: the learned index's in-memory footprint in DECIMAL gigabytes (1e9 bytes, see
///   [`BYTES_PER_GB`]), which must be the same unit the caller's estimate is in. Positive; roughly
///   78.0 at hg38 scale. Supplied by the index-selection code, which derives it from the reference
///   length rather than measuring it.
///
/// # Returns
///
/// True only when detected RAM is at least `index_gb * `[`RAM_HEADROOM_FACTOR`]. False whenever RAM
/// cannot be detected, so the conservative FM path is the default on unknown platforms. Purely a
/// performance decision: neither branch changes the SAM output.
pub fn learned_index_fits(index_gb: f64) -> bool {
    match total_ram_bytes() {
        Some(total_bytes) => (total_bytes as f64) / BYTES_PER_GB >= index_gb * RAM_HEADROOM_FACTOR,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_detection_plausible() {
        // On the CI/dev host this must return a sane figure (>= 1 GB, < 100 TB).
        if let Some(total_bytes) = total_ram_bytes() {
            assert!(total_bytes >= 1 << 30, "implausibly small: {total_bytes}");
            assert!(
                total_bytes < (100u64 << 40),
                "implausibly large: {total_bytes}"
            );
        }
        // A 1 TB index never fits a normal host; a 1 GB index fits any host with detectable RAM.
        assert!(!learned_index_fits(1000.0));
    }
}
