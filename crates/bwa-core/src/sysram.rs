//! System-RAM detection, for the learned-index auto-select: LISA seeding needs a large in-memory
//! index (~78 GB at hg38 scale), worth it only on big-memory hosts, so the aligner picks LISA when
//! enough RAM is present and otherwise falls back to the classic FM-index path, transparently, so a
//! 16-64 GB machine is never penalized.
//!
//! Nothing here affects SAM bytes: both index paths produce the same alignments, so this file is
//! purely a performance-policy decision. Reading order: [`total_ram_bytes`] then
//! [`learned_index_fits`], its only caller-facing use.
//!
//! # Rust mechanics used in this file
//!
//! The `Option`, `?`, `.ok()` and `#[cfg]` machinery is the same as in `cpu.rs`, which glosses it
//! in full. What is new here is the string handling and the number types.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `u64` | an unsigned 64-bit integer. Byte counts need it: a 32-bit count tops out at 4 GB, which modern hosts exceed. |
//! | `f64` | a double-precision floating-point number. Used for the ratio below because the comparison is fractional (1.25x), not a whole number of bytes. |
//! | `as f64` | an explicit conversion from integer to float. Rust never converts number types on its own, so the cast has to be written even where C would do it silently. |
//! | `1e9` | scientific notation for a float, here one decimal gigabyte. |
//! | `1 << 30` | bit shift: one, shifted left thirty places, which is 2^30, a binary gigabyte. A compact way to write large round powers of two. |
//! | `.lines()` | walks a block of text one line at a time, without allocating a new string per line. |
//! | `.strip_prefix("x")` | returns `Some(rest)` when the text starts with `x`, and `None` when it does not. It tests and trims in one step, so the label match and the value extraction cannot drift apart. |
//! | `.split_whitespace()` | walks the space-separated tokens of a line, ignoring how many spaces separate them. `.next()` then takes the first token. |
//! | `return` | an EARLY exit. Rust only needs the keyword when leaving from the middle of a function, as in the loop below; the value at the end of a body needs no keyword and no semicolon. |
//! | `match` on an `Option` | handles both cases explicitly, `Some(v)` and `None`, and the compiler refuses the code if either is forgotten. That exhaustiveness is what guarantees the "RAM unknown" path below is never left undecided. |
//! | `!expr` | boolean negation, "not". |

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
        // Read top to bottom as four steps, each of which may give up: interpret the captured bytes
        // as text, drop the trailing newline, parse the digits, hand back the answer. The final
        // `.ok()` has no `?` because this is the block's last expression: its `Option` IS the
        // function's return value, so a parse failure becomes the `None` the caller expects.
        //
        // Rust: nothing states `u64` here. The compiler works it out backwards from the function's
        // declared return type, `Option<u64>`, and tells `.parse()` what to produce.
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
                //
                // Rust: `.next()?` takes the first token and bails out with `None` if the line was
                // empty after the label. `.parse().ok()?` bails out again if that token is not a
                // number. The `: u64` annotation is what tells `.parse()` which type to build.
                let kibibytes: u64 = after_label.split_whitespace().next()?.parse().ok()?;
                // Rust: an early `return`, needed because we are inside a loop rather than at the
                // end of the body. It exits `total_ram_bytes` entirely, not just the loop.
                return Some(kibibytes * 1024);
            }
        }
        // Reached only when the loop ran off the end without ever seeing `MemTotal:`. No semicolon,
        // so this is the block's value: the file existed but did not say what we needed.
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
    // The whole policy in one comparison: detected RAM converted to decimal gigabytes, against the
    // index footprint inflated by the headroom factor.
    //
    // Rust: `match` forces both cases to be written out, and the compiler rejects the function if
    // one is missing. That is what makes "RAM could not be detected" an explicit decision (fall
    // back to the classic FM path) instead of an accident. `as f64` converts the byte count to a
    // float so the division is fractional rather than truncated integer division.
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
        //
        // Rust: this assertion holds on EVERY host, which is what makes it a usable test. Where RAM
        // is detectable the comparison fails on size; where it is not, the `None` arm returns false
        // anyway. `!` negates, so the test reads "assert that a 1 TB index does not fit".
        assert!(!learned_index_fits(1000.0));
    }
}
