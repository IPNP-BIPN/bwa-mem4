//! System-RAM detection, for the learned-index auto-select: LISA seeding needs a large in-memory
//! index (~78 GB at hg38 scale), worth it only on big-memory hosts, so the aligner picks LISA when
//! enough RAM is present and otherwise falls back to the classic FM-index path — transparently, so a
//! 16-64 GB machine is never penalized.

/// Total physical RAM in bytes, or `None` if it can't be determined. macOS reads `hw.memsize`
/// (`sysctl`); Linux reads `/proc/meminfo`'s `MemTotal`.
pub fn total_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        std::str::from_utf8(&out.stdout).ok()?.trim().parse().ok()
    }
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
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
pub fn learned_index_fits(index_gb: f64) -> bool {
    match total_ram_bytes() {
        Some(b) => (b as f64) / 1e9 >= index_gb * 1.25,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_detection_plausible() {
        // On the CI/dev host this must return a sane figure (>= 1 GB, < 100 TB).
        if let Some(b) = total_ram_bytes() {
            assert!(b >= 1 << 30, "implausibly small: {b}");
            assert!(b < (100u64 << 40), "implausibly large: {b}");
        }
        // A 1 TB index never fits a normal host; a 1 GB index fits any host with detectable RAM.
        assert!(!learned_index_fits(1000.0));
    }
}
