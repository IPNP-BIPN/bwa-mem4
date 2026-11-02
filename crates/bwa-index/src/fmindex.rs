//! FM-index loading and traversal, mirroring bwa-mem2's `FMI_search`.
//!
//! Loads `.bwt.2bit.64` (checkpointed occurrences + compressed suffix array) and `.0123` (the
//! forward++reverse-complement binary reference), and provides the primitives seeding needs:
//! [`FmIndex::get_occ`], [`FmIndex::backward_ext`] and [`FmIndex::get_sa`]. See
//! `reference/bwa-mem2/src/FMI_search.cpp` (`load_index`, `GET_OCC`, `backwardExt`,
//! `get_sa_entry_compressed`).
//!
//! # What an FM index is, in the four lines needed to read this file
//!
//! Sort every suffix of the text; `sa[i]` is where the `i`-th smallest suffix starts. The BWT is
//! the character just before each of those suffixes. Any pattern occupies one CONTIGUOUS range of
//! rows `[k, k+s)`, so "how many times does P occur" is just `s`. [`FmIndex::backward_ext`] grows
//! the pattern one character to the LEFT and updates that range in O(1); [`FmIndex::get_sa`] then
//! turns a row number back into a reference position.
//!
//! The one primitive that makes the O(1) step work is `occ(c, p)`: how many `c`s appear in the
//! first `p` characters of the BWT. Storing that for every `(c, p)` would cost more than the
//! genome, so bwa-mem2 stores it every 64 rows and recovers the remainder with a popcount over a
//! bitmap. That is the `CP_OCC` record, and it is why almost everything here is expressed in
//! `>> 6` and `& 63`.
//!
//! This index is bidirectional (an "FMD index"): it tracks the interval of the pattern AND of its
//! reverse complement at once (`k` and `l`), which is what lets bwa find maximal exact matches by
//! extending in both directions without a second index.
//!
//! See [`crate::build`] for the on-disk byte layout this module reads.
//!
//! # Glossary: names kept identical to the C
//!
//! | name | C origin | plain-language meaning |
//! |---|---|---|
//! | `k` | `SMEM.k` | First BWT ROW of an interval. A "row" is a rank in the sorted list of all suffixes of the doubled reference, `0 ..= ref_seq_len`. |
//! | `l` | `SMEM.l` | Same interval for the reverse-complemented pattern. Only [`FmIndex::backward_ext`] maintains it; nothing resolves positions from it. |
//! | `s` | `SMEM.s` | Interval SIZE, i.e. how many times the pattern occurs. Both `[k, k+s)` and `[l, l+s)` have this length. |
//! | `occ` | `GET_OCC` | "How many `c`s appear in `bwt[0..p)`". The one primitive that makes each search step O(1). Exclusive of row `p` itself. |
//! | `count` / `counts` | `count[]` | The classic FM-index `C[]`: `count[c]` = number of rows whose suffix starts with a base smaller than `c`. `count[c] + occ(c, row)` is the LF-mapping. |
//! | `cp_occ` | `cp_occ` | The checkpoint array: one record per 64 BWT rows, holding the cumulative counts up to that block plus the block's own 64 rows. 64 bytes in the file, 32 in memory: see [`CpOcc`]. |
//! | `sa` | `suffix_array` | The suffix array: `sa[row]` is the reference position the suffix at `row` starts at. Only every 8th entry is stored; the rest are recovered by walking. |
//! | `pac` / `l_pac` | `bns->l_pac` | Length of the FORWARD reference in bases. The searched text is `2 * l_pac` long (forward genome then its reverse complement), so a position `>= l_pac` is a reverse-strand hit. |
//! | `sp` / `ep` | `sp` / `ep` | Start and (exclusive) end row of an interval, i.e. `k` and `k + s`. |
//! | `pp` | `pp` | The row an occurrence query is made at. Valid range is `[0, ref_seq_len]` INCLUSIVE, because interval ENDS are passed here. |
//!
//! # Reading order for this file
//!
//! 1. [`Smem`] and [`CpOcc`]: what an interval is and what one checkpoint block holds in memory.
//! 2. [`FmIndex::load`]: the on-disk layout, read field by field.
//! 3. [`FmIndex::get_occ`]: the single primitive everything else is built from.
//! 4. [`FmIndex::backward_ext`]: one search step, including the fiddly `l` recurrence.
//! 5. [`FmIndex::get_sa`]: turning a row back into a reference position.
//! 6. [`FmIndex::get_sa_batch`] and [`FmIndex::prefetch_occ`]: latency hiding only, no new logic.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use bwa_core::Result;
use memmap2::Mmap;
use rayon::prelude::*;

/// Rows per occurrence-checkpoint block, the C's `CP_BLOCK_SIZE` (`FMI_search.h:49`). Also the
/// record's size in BYTES in the file. It is baked into the index FILENAME (`.bwt.2bit.64`), so an index
/// built with a different block size simply fails to open rather than being misread.
const CP_BLOCK_SIZE: i64 = 64;
/// `log2(CP_BLOCK_SIZE)`: the C's `CP_SHIFT` (`FMI_search.h:51`). `row >> CP_SHIFT` is the block.
const CP_SHIFT: i64 = 6;
/// `CP_BLOCK_SIZE - 1`: the C's `CP_MASK`. `row & CP_MASK` is the row's offset within its block.
const CP_MASK: i64 = 63;

/// `log2` of the suffix-array sampling stride: the C's `SA_COMPX = 03` octal = 3 (`macro.h:65`).
/// Only every 8th suffix-array entry is stored, an 8x space saving paid for by a short LF-walk.
const SA_SHIFT: i64 = 3;
/// Stride minus one, so `row & SA_MASK == 0` tests "this row was sampled outright".
const SA_MASK: i64 = 7;

/// A bidirectional FM-index interval, mirroring bwa-mem2's `SMEM` (`FMI_search.h:75`).
///
/// "SMEM" is Super-Maximal Exact Match: a substring of the read that matches the reference exactly
/// and cannot be extended in either direction without losing matches. The struct is really two
/// things at once: the read-coordinate span (`m`, `n`) and the reference-side FM interval
/// (`k`, `l`, `s`).
///
/// INVARIANT tying the three interval fields together: `[k, k+s)` and `[l, l+s)` are ranges of BWT
/// rows of the SAME length `s`, the first for the pattern and the second for its reverse
/// complement. `s == 0` means the pattern does not occur; `s` never grows under extension.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Smem {
    /// Read index within the batch this SMEM came from. Not a reference contig id (that is `rid`
    /// in the bntseq sense); the name is inherited from the C's `SMEM.rid`.
    pub rid: u32,
    /// Start offset of the match in the read, inclusive.
    pub m: u32,
    /// End offset of the match in the read, inclusive in bwa's usage. `n - m + 1` is the match
    /// length, which is why seeding code compares against `min_seed_len` after a `+ 1`.
    pub n: u32,
    /// Forward interval start: the first BWT row matching the pattern.
    pub k: i64,
    /// Reverse-complement interval start: the first BWT row matching `revcomp(pattern)`. Maintained
    /// purely so the search can also be extended rightward; never used to look up positions.
    pub l: i64,
    /// Interval size (occurrence count), shared by both intervals.
    pub s: i64,
}

/// One 64-base checkpoint block, as held in memory: 32 bytes, half of what the file spends.
///
/// bwa-mem2's on-disk `CP_OCC` (`FMI_search.h:55`) is 64 bytes: four `i64` counts and four one-hot
/// bitmaps. Neither half needs that much. A row holds one of four symbols, so two bit-planes say what
/// four one-hot words say; and a count is at most the row number, so 32 bits reach any row of a
/// genome below 4 Gbp of a single base, with `FmIndex::count_base` carrying the rest for the larger
/// ones. On GRCh38 that is 3.1 GB of resident memory that held nothing, and it is read back only
/// through [`FmIndex::get_occ`], [`FmIndex::backward_ext`] and the LF-walks, so no output can move
/// unless they do, and the tests below compare them against the file's own one-hot form.
///
/// The planes are also faster to use, not merely smaller. `bwt[row]` is two shifts and an OR rather
/// than up to four unpredictable branches, and [`FmIndex::backward_ext`] gets its four counts from
/// three popcounts because the fourth is what remains of the rows.
///
/// `hi` and `lo` are the two bits of the base code (A=00 C=01 G=10 T=11), bit `63 - j` for the block's
/// row `j`, MSB-first as in the file. The sentinel row has no base and is stored as A; the two places
/// that could tell are `FmIndex::sentinel_before` and the row test in the LF-walks. Padding past the
/// last row is stored as A too, and no query reaches it, because a count stops at its row.
#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct CpOcc {
    /// Occurrences of A, C, G, T in `bwt[0 .. block_start)`, EXCLUSIVE of this block's own rows, less
    /// the `FmIndex::count_base` entry for the block's span. So always below 2^32.
    cp_count: [u32; 4],
    /// High bit of each row's base code. Set for G and T.
    hi: u64,
    /// Low bit of each row's base code. Set for C and T.
    lo: u64,
}

/// `log2` of the rows one `FmIndex::count_base` entry spans. A 32-bit [`CpOcc::cp_count`] measured
/// from that span's start cannot overflow, because a span has fewer than 2^32 rows.
const COUNT_SPAN_SHIFT: i64 = 32;

/// Turn each record's low 32 count bits into the count less its span's base, for a `wide` genome.
///
/// Wrapping subtraction of the low words is exact: the true difference is below 2^32, because a span
/// has fewer rows than that.
///
/// # Parameters
/// * `cp_occ`: the records, holding the low 32 bits of their full counts.
/// * `count_base`: full counts at each span's first block.
/// * `span_shift`: `log2` of a span's rows. [`COUNT_SPAN_SHIFT`] outside the tests, which cannot
///   afford 2^32 rows.
fn rebase_counts(cp_occ: &mut [CpOcc], count_base: &[[i64; 4]], span_shift: i64) {
    cp_occ.par_iter_mut().enumerate().for_each(|(block, rec)| {
        let base = &count_base[block >> (span_shift - CP_SHIFT)];
        for c in 0..4 {
            rec.cp_count[c] = rec.cp_count[c].wrapping_sub(base[c] as u32);
        }
    });
}

/// `bits` bits per sampled suffix-array entry, as a shift. Chosen by the row count in
/// [`FmIndex::load`], and the packing it describes is why a word is never straddled.
const fn sa_hi_log2_bits(ref_seq_len: i64) -> u32 {
    // Positions run up to `2L`, so this is how many bits the part above the low 32 ever needs.
    let bits = 64 - ((ref_seq_len - 1) as u64 >> 32).leading_zeros();
    if bits <= 1 {
        0
    } else if bits <= 2 {
        1
    } else if bits <= 4 {
        2
    } else {
        3
    }
}

/// The rows strictly before offset `y` in a block: the top `y` bits. `y == 0` gives 0, which is why
/// the shift is the complement of an all-ones right shift rather than a left shift by `64 - y`.
#[inline(always)]
const fn rows_before(y: i64) -> u64 {
    !(u64::MAX >> y)
}

/// Loaded FM-index plus the binary reference for O(1) base access.
pub struct FmIndex {
    /// `reference_seq_len` = 2L + 1, i.e. the number of BWT rows: `2 * l_pac` reference positions
    /// plus the one sentinel row. NOT the length of `reference` below, which is `2L`.
    pub ref_seq_len: i64,
    /// Cumulative base counts, already incremented by 1 as bwa-mem2's `load_index` does.
    /// `count[c]` is "number of BWT rows whose suffix starts with a base < c", so `count[c] + occ`
    /// is the classic LF-mapping. The `+1` on every entry (`FMI_search.cpp:424`) accounts for the
    /// sentinel row, which sorts before every base and therefore occupies row 0.
    count: [i64; 5],
    /// One checkpoint per 64 BWT rows, `(ref_seq_len >> 6) + 1` of them. Indexed by `pos >> 6`.
    cp_occ: Vec<CpOcc>,
    /// What [`CpOcc::cp_count`] leaves out when `wide`: the counts at the start of each 2^32-row
    /// span, indexed by `row >> COUNT_SPAN_SHIFT`. One unused entry otherwise.
    count_base: Vec<[i64; 4]>,
    /// The bits above the low 32 of every 8th suffix-array entry, packed. The file gives them a byte
    /// each, and a human position has one such bit: 0.78 GB in the file, 97 MB here, and one fewer
    /// random line per lookup.
    sa_hi: Vec<u64>,
    /// `log2` of the bits per entry in `sa_hi`. See [`sa_hi_log2_bits`].
    sa_hi_log2: u32,
    /// One entry's worth of low bits, `(1 << (1 << sa_hi_log2)) - 1`.
    sa_hi_mask: u64,
    /// Low 32 bits of every 8th suffix-array entry. Paired index-for-index with `sa_hi`.
    sa_ls_word: Vec<u32>,
    /// The unique BWT row with `sa[row] == 0`. Needed by `backward_ext` to keep `l` correct.
    sentinel_index: i64,
    /// `sentinel_index >> CP_SHIFT`, the one block whose A counts need `FmIndex::sentinel_before`.
    sentinel_block: i64,
    /// Whether any base occurs 2^32 times or more, so that a `u32` count alone is not the count and
    /// `count_base` has to be added. False on every genome up to a few times the human one, where the
    /// hot paths pay one predicted test for a table they then never read.
    wide: bool,
    /// `log2` of the rows one `count_base` entry spans. [`COUNT_SPAN_SHIFT`] outside the tests.
    count_span_shift: i64,
    /// The `.pac` reference (forward strand only, 2 bits/base, ~L/4 bytes), memory-mapped. This is
    /// the 4x-smaller archival packing; the aligner used to load the 4x-larger one-byte-per-base
    /// `.0123` (2L bytes, ~6.2 GB for hg38) on the hot path and mmap kept those pages resident. We
    /// unpack from `.pac` in [`unpack_pac_base`] instead: a shift-and-mask per base, and the
    /// reverse-complement half is reconstructed rather than stored, so nothing keeps 6.2 GB
    /// resident. `.0123` is still written by the indexer and still on disk (the index stays
    /// byte-identical to bwa-mem2), it is simply not loaded.
    pac: Mmap,
}

/// Unpack the reference base at 2L-space position `pos` from the 2-bit-packed, forward-only `.pac`.
///
/// The old `.0123` stored `forward ++ reverse_complement(forward)` as one byte per base (2L bytes);
/// `.pac` stores only the forward strand at 2 bits per base. So the forward half `[0, L)` is a
/// direct unpack, and the reverse-complement half `[L, 2L)` is reconstructed exactly the way the
/// indexer built `.0123`: a position `p >= L` maps to forward coordinate `2L - 1 - p` (`bns_depos`)
/// with the base complemented, which for the 2-bit codes (A=0,C=1,G=2,T=3) is `3 - b`.
///
/// `pac` is the whole mapped file including its one or two trailer bytes; those are never indexed,
/// because a valid `pos` gives `fwd_idx < L` and therefore `fwd_idx >> 2` stays inside the payload.
#[inline]
pub(crate) fn unpack_pac_base(pac: &[u8], l_pac: i64, pos: i64) -> u8 {
    let (fwd_idx, rc) = if pos < l_pac {
        (pos, false)
    } else {
        (2 * l_pac - 1 - pos, true)
    };
    // bwa-mem2's `_get_pac`: base i lives in byte `i>>2`, most-significant pair first, 2 bits wide.
    let b = (pac[(fwd_idx >> 2) as usize] >> ((3 - (fwd_idx & 3)) << 1)) & 3;
    if rc {
        3 - b
    } else {
        b
    }
}

/// Unpack the forward-strand bases at `[start, start + len)` from `.pac` into `out`.
///
/// The scalar form is one branch, one shift, one mask and a conditional complement per base, and
/// `bns_fetch_seq` calls it ~1400 times per mate-rescue job, ~5 G bases over a real paired-end run.
/// Four packed bytes hold sixteen bases, so the whole expansion is one table lookup (spread each
/// byte over four lanes), one variable shift (`[6,4,2,0]` per group, as a negative left shift) and
/// one mask: six instructions per sixteen bases instead of roughly ninety.
///
/// # Parameters
/// - `pac`: the whole mapped `.pac`, including its trailer bytes, which are never indexed here.
/// - `start`: first FORWARD-strand position to unpack; `start + len` must not exceed `l_pac`.
/// - `out`: destination, filled with exactly `len` bases, one per byte, codes 0..=3.
#[inline]
fn unpack_pac_fwd(pac: &[u8], start: i64, len: usize, out: &mut Vec<u8>) {
    // Bases before the first whole packed byte, done one at a time so the block loop below can
    // assume `pos % 4 == 0`.
    let mut pos = start;
    let end = start + len as i64;
    while pos < end && (pos & 3) != 0 {
        out.push((pac[(pos >> 2) as usize] >> ((3 - (pos & 3)) << 1)) & 3);
        pos += 1;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            use std::arch::aarch64::*;
            // SAFETY: neon detected. Every load reads 4 bytes at `pos >> 2` with
            // `pos + 16 <= end <= l_pac`, so the byte index is inside the payload; every store
            // writes 16 bytes into capacity reserved by the caller's `with_capacity`.
            unsafe {
                // Spread byte `k` of the four loaded bytes across lanes `4k..4k+4`.
                let spread = vld1q_u8([0u8, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].as_ptr());
                // Right shifts, as the negative left shifts `vshlq_u8` wants: within a byte the
                // most significant pair is base 0, so the four bases need `>>6, >>4, >>2, >>0`.
                let shifts = vld1q_s8(
                    [-6i8, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0].as_ptr(),
                );
                let three = vdupq_n_u8(3);
                while pos + 16 <= end {
                    // The four packed bytes holding these sixteen bases, broadcast so the table
                    // lookup can pick any of them into any lane.
                    let word =
                        (pac.as_ptr().add((pos >> 2) as usize) as *const u32).read_unaligned();
                    let v = vreinterpretq_u8_u32(vdupq_n_u32(word));
                    let v = vqtbl1q_u8(v, spread);
                    let v = vandq_u8(vshlq_u8(v, shifts), three);
                    let n = out.len();
                    vst1q_u8(out.as_mut_ptr().add(n), v);
                    out.set_len(n + 16);
                    pos += 16;
                }
            }
        }
    }

    // The same sixteen-bases-per-iteration unpack on x86. It was missing, and the cost of missing
    // it is measurable: a side-by-side `perf` profile against fg-labs/bwa-mem3 on an EPYC 7763 put
    // this function at 3.50% of our run against 1.39% for their `bns_get_seq_into`, because on x86
    // every base went through the scalar tail below while aarch64 took the vector path above.
    //
    // SSSE3 rather than AVX2, and chosen at runtime: `pshufb` is the only instruction here that is
    // not in SSE2, every x86-64 part since 2006 has it, and the release artefact is built at the
    // x86-64 baseline, so a compile-time gate would leave the shipped binary on the scalar loop.
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("ssse3") {
            use std::arch::x86_64::*;
            // SAFETY: ssse3 detected. Each iteration reads the 4 packed bytes at `pos >> 2` with
            // `pos + 16 <= end <= 2 * l_pac`, so the byte index is inside the payload, and writes
            // 16 bytes into capacity the caller reserved (`bases_into` reserves `len + 16`).
            unsafe {
                // Byte k of the loaded word into lanes 4k..4k+3, the same spread the NEON arm uses.
                let spread = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
                let three = _mm_set1_epi8(3);
                // x86 has no per-byte variable shift before AVX-512, so the four shifts are done
                // whole-vector and then selected by lane. A 16-bit shift is safe here even though
                // the data is bytes: the bits that leak in from the neighbouring byte land in bit
                // positions 8-s and above, and the `& 3` that follows discards every one of them.
                // Within a packed byte the most significant pair is base 0, hence 6, 4, 2, 0.
                let m0 = _mm_setr_epi8(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0);
                let m1 = _mm_setr_epi8(0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0);
                let m2 = _mm_setr_epi8(0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0);
                let m3 = _mm_setr_epi8(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);
                while pos + 16 <= end {
                    let word =
                        (pac.as_ptr().add((pos >> 2) as usize) as *const u32).read_unaligned();
                    let v = _mm_shuffle_epi8(_mm_set1_epi32(word as i32), spread);
                    let s6 = _mm_and_si128(_mm_srli_epi16(v, 6), m0);
                    let s4 = _mm_and_si128(_mm_srli_epi16(v, 4), m1);
                    let s2 = _mm_and_si128(_mm_srli_epi16(v, 2), m2);
                    let s0 = _mm_and_si128(v, m3);
                    let out_v = _mm_and_si128(
                        _mm_or_si128(_mm_or_si128(s6, s4), _mm_or_si128(s2, s0)),
                        three,
                    );
                    let n = out.len();
                    _mm_storeu_si128(out.as_mut_ptr().add(n) as *mut __m128i, out_v);
                    out.set_len(n + 16);
                    pos += 16;
                }
            }
        }
    }

    // Tail, and the whole range on a build with neither vector path.
    while pos < end {
        out.push((pac[(pos >> 2) as usize] >> ((3 - (pos & 3)) << 1)) & 3);
        pos += 1;
    }
}

/// Unpack `[rb, re)` of the DOUBLED 2L coordinate space into a fresh byte vector, one base per byte.
///
/// Equivalent to `(rb..re).map(|p| unpack_pac_base(pac, l_pac, p)).collect()`, and tested against
/// exactly that. The two halves are handled separately because they are different traversals of the
/// same packed forward strand: `[0, L)` is a direct ascending unpack, while `[L, 2L)` is the
/// forward range `[2L - re, 2L - rb)` read backwards with every base complemented (`3 - b`, which
/// for 2-bit codes is `b ^ 3`). A range straddling `L` cannot arise from a contig-clamped window and
/// falls back to the per-base loop.
///
/// # Parameters
/// - `pac`: the whole mapped `.pac`.
/// - `l_pac`: forward reference length `L`.
/// - `rb`, `re`: half-open range in 2L space, `0 <= rb <= re <= 2L`.
///
/// # Returns
/// `re - rb` bytes, base codes 0..=3.
pub(crate) fn unpack_pac_range(pac: &[u8], l_pac: i64, rb: i64, re: i64) -> Vec<u8> {
    let mut out = Vec::new();
    unpack_pac_range_into(pac, l_pac, rb, re, &mut out);
    out
}

/// [`unpack_pac_range`] into a caller-owned buffer. See [`FmIndex::bases_into`] for why the
/// buffer-reusing form exists.
///
/// # Parameters
/// * `pac`, `l_pac`: the packed reference and the forward length, as for [`unpack_pac_range`].
/// * `rb`, `re`: half-open 2L-space range.
/// * `out`: cleared, then filled with `re - rb` bytes. The `+ 16` of reserve is what lets the NEON
///   loop store a whole 16-lane vector past the last full block without a bounds check.
pub(crate) fn unpack_pac_range_into(pac: &[u8], l_pac: i64, rb: i64, re: i64, out: &mut Vec<u8>) {
    let len = (re - rb).max(0) as usize;
    out.clear();
    if len == 0 {
        return;
    }
    out.reserve(len + 16);
    if re <= l_pac {
        unpack_pac_fwd(pac, rb, len, out);
    } else if rb >= l_pac {
        // Reverse-complement half: same packed bytes, read forward then flipped end for end.
        //
        // Two separate passes, `reverse()` then a complement loop, and that is deliberate: fusing
        // them into one two-cursor swap loop was tried and measured 31.4 s of CPU against 11.1 s
        // for the pair, on a 500k-pair `-t16` run. `reverse()` is a vectorised memory reversal and
        // the `^= 3` loop auto-vectorises; a hand-fused loop is neither, and two vector passes beat
        // one scalar pass by a wide margin here.
        unpack_pac_fwd(pac, 2 * l_pac - re, len, out);
        out.reverse();
        for b in out.iter_mut() {
            // Complement in bwa's A=0 C=1 G=2 T=3 order, where complementary codes sum to 3.
            *b ^= 3;
        }
    } else {
        for p in rb..re {
            out.push(unpack_pac_base(pac, l_pac, p));
        }
    }
    debug_assert_eq!(out.len(), len);
}

/// Build the path `<prefix>.<ext>`, the naming convention every index file follows.
///
/// # Parameters
/// * `prefix`: the FASTA path given at index-build time, e.g. `/data/hg38.fa`. Used verbatim as a
///   string prefix, NOT as a directory: the extension is appended to the whole path, so
///   `hg38.fa` + `bwt.2bit.64` gives `hg38.fa.bwt.2bit.64`. Supplied by the caller of
///   [`FmIndex::load`], ultimately the `-x`/positional index argument on the command line.
/// * `ext`: the extension to append, WITHOUT a leading dot (the dot is added here). Only
///   `"bwt.2bit.64"` and `"0123"` are used in this module.
///
/// # Returns
/// The concatenated path. Existence is not checked; the caller's `File::open` reports a missing
/// file. Works on non-UTF-8 paths because the join happens at the `OsString` level.
fn sibling(prefix: &Path, ext: &str) -> PathBuf {
    // Byte-level path concatenation, so a prefix that is not valid UTF-8 survives the round trip.
    let mut s: OsString = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

/// Read one little-endian `i64` at `*p` and advance the cursor. Every scalar in `.bwt.2bit.64` is
/// LE `i64`, so this is the whole header decoder.
///
/// # Parameters
/// * `bytes`: the whole memory-mapped `.bwt.2bit.64` file. Indexed, never copied.
/// * `cursor`: BYTE offset into `bytes` (not an element index) at which to read. Read-modify-write:
///   on return it has been advanced by exactly 8. The caller owns the running position, so a
///   sequence of `rd_i64` calls decodes consecutive fields in file order.
///
/// # Returns
/// The decoded value. Panics on a truncated file, because the slice `[*cursor .. *cursor + 8]`
/// goes out of bounds; there is no format validation anywhere in this module.
fn rd_i64(bytes: &[u8], cursor: &mut usize) -> i64 {
    // The 8 bytes at the cursor, reinterpreted LE. `try_into` cannot fail: the slice is 8 long.
    let v = i64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    v
}

/// Fill `dst` from `file` starting at byte `offset`, using every core.
///
/// The three payload arrays of a human index are 6.2 GB, 0.78 GB and 3.1 GB. Reading them with one
/// sequential `read_exact` is one thread doing a 10 GB memcpy out of the page cache, and on a warm
/// cache that is what the whole load costs: about 2.4 s against the C fork's 1.8 s, measured on a
/// GRCh38 index in tmpfs. Splitting the copy across the pool takes it to well under a second, and
/// the only thing that changes is which thread copies which byte.
///
/// `read_exact_at` (POSIX `pread`) is what makes that legal without a lock: it takes the offset as an
/// argument instead of using the file's shared cursor, so N threads can read N disjoint ranges of
/// the same descriptor with no seeking and no interference. There is no such thing on Windows, hence
/// the `cfg`; the sequential fallback there is correct, just slower.
///
/// # Parameters
/// * `file`: the open index file. Not advanced: every read is positional, so the caller's cursor is
///   untouched and the caller must therefore pass explicit offsets.
/// * `offset`: absolute byte offset in the file of `dst[0]`.
/// * `dst`: destination bytes, filled completely or an error is returned.
///
/// # Errors
/// The first chunk read that fails or hits end-of-file early. A short file is an error here, not a
/// partially filled buffer: the caller is about to `set_len` on these bytes.
#[cfg(unix)]
fn read_at_parallel(file: &std::fs::File, offset: u64, dst: &mut [u8]) -> Result<()> {
    use std::os::unix::fs::FileExt;

    // One chunk per worker, with a floor so a small array is not split into slivers whose per-chunk
    // syscall overhead exceeds the copy they perform.
    const MIN_CHUNK: usize = 8 << 20;
    let workers = rayon::current_num_threads().max(1);
    let chunk = (dst.len() / workers).max(MIN_CHUNK);
    dst.par_chunks_mut(chunk)
        .enumerate()
        .try_for_each(|(i, part)| -> Result<()> {
            // `chunk` is the stride every earlier part used, so part `i` starts exactly there.
            file.read_exact_at(part, offset + (i * chunk) as u64)?;
            Ok(())
        })
}

/// Sequential fallback: same contract as the parallel version, for targets without `pread`.
#[cfg(not(unix))]
fn read_at_parallel(file: &std::fs::File, offset: u64, dst: &mut [u8]) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = file.try_clone()?;
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(dst)?;
    Ok(())
}

/// Records converted per load chunk: 4 MB of file, small enough that the buffers stay out of the peak
/// and large enough that a chunk is not mostly syscall.
const LOAD_CHUNK_RECORDS: usize = 1 << 16;

/// One positional read, for a caller already running on the pool. See [`read_at_parallel`] for why it
/// has to be positional.
#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, offset: u64, dst: &mut [u8]) -> Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(dst, offset)?;
    Ok(())
}

/// Sequential fallback, for targets without `pread`. Each call clones the handle for its own cursor.
#[cfg(not(unix))]
fn read_exact_at(file: &std::fs::File, offset: u64, dst: &mut [u8]) -> Result<()> {
    read_at_parallel(file, offset, dst)
}

/// Cached `BWA4_SA_WINDOW`, read once: a `getenv` per batched lookup would be its own cost.
static SA_WINDOW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

impl FmIndex {
    /// Load `<prefix>.bwt.2bit.64` and `<prefix>.0123`.
    ///
    /// `prefix` is the FASTA path given at index time; both files are siblings named
    /// `<prefix>.<ext>`. Mirrors `FMI_search::load_index` (`FMI_search.cpp:384`) field for field,
    /// in the same order, including the `+1` fixup on `count`.
    ///
    /// The cursor arithmetic below IS the format spec: header (`i64` + 5 x `i64`), then
    /// `cp_size = (N >> 6) + 1` 64-byte checkpoints, then `sa_size = (N >> 3) + 1` `i8`s, then
    /// `sa_size` `u32`s, then the trailing sentinel `i64`. Nothing is self-describing, so a
    /// truncated or foreign-endian file is not detected here; it simply produces wrong answers or
    /// panics on the slice bounds.
    ///
    /// # Parameters
    /// * `prefix`: the index prefix, i.e. the FASTA path used at build time. Both `.bwt.2bit.64`
    ///   and `.0123` must already exist next to it (see [`crate::build`]); this function only
    ///   reads. Comes from the command line, so it is untrusted as a PATH but trusted as CONTENT:
    ///   a corrupt file is not detected.
    ///
    /// # Returns
    /// A fully usable `FmIndex` holding two live memory maps, so the returned value keeps the
    /// index files open for its whole lifetime. `Err` only for I/O failures (missing file,
    /// permission, `mmap` refusal); a truncated file panics instead.
    pub fn load(prefix: &Path) -> Result<Self> {
        // Load the BWT with plain reads straight into aligned buffers, NOT mmap + copy. mmap would
        // fault the whole ~10 GB file into OUR resident set while we copy it out, so peak RSS
        // transiently holds the mapped file AND the copy at once. That transient, not the `.0123`
        // pages, is what set our 19.2 GB peak (measured: dropping the 6.2 GB `.0123` mmap moved the
        // peak by 0). `read` leaves the source in the kernel page cache, which does not count toward
        // our RSS, so the peak is just the destination arrays. It is still one bulk memcpy (page
        // cache -> aligned buffer) per array, so load stays fast; `cp_occ`'s 64-byte alignment,
        // which the file's offset-48 layout denies an in-place mmap, is satisfied because the
        // destination `Vec<CpOcc>` is freshly allocated and therefore aligned.
        let bwt_file = std::fs::File::open(sibling(prefix, "bwt.2bit.64"))?;
        let bwt_len = bwt_file.metadata()?.len();
        // Named here because the alternative is a bare "failed to fill whole buffer".
        let damaged = |what: String| {
            bwa_core::Error::IndexFormat(format!(
                "index {}: {what}. The file is damaged or incomplete; re-run `bwa-mem4 index`.",
                prefix.display()
            ))
        };
        /// Bytes before the first payload array: `ref_seq_len` plus `count[5]`, six little-endian
        /// i64. Every payload offset below is measured from here.
        const HEADER_BYTES: u64 = 48;

        // ---- Header: ref_seq_len, then count[5] (six i64, little-endian, 48 bytes) -----------
        let mut header = [0u8; 48];
        if bwt_len < HEADER_BYTES {
            return Err(damaged(format!(".bwt.2bit.64 is only {bwt_len} bytes")));
        }
        read_at_parallel(&bwt_file, 0, &mut header)?;
        let mut cursor = 0usize;
        // First header field: the BWT ROW count, 2L + 1 (2L reference bases plus the sentinel row).
        // Every later size in the file is derived from it, so it is read before anything else.
        let ref_seq_len = rd_i64(&header, &mut cursor);
        if ref_seq_len < 1 {
            return Err(damaged(format!(".bwt.2bit.64 declares {ref_seq_len} rows")));
        }
        // `count[c]`: rows whose suffix starts with a base < c. Index 4 is the total (all rows).
        // Read as written by the builder, i.e. WITHOUT the sentinel adjustment applied below.
        let mut count = [0i64; 5];
        for c in &mut count {
            *c = rd_i64(&header, &mut cursor);
        }
        for c in &mut count {
            // `for(ii = 0; ii < 5; ii++) count[ii] = count[ii] + 1;` (`FMI_search.cpp:423-426`).
            // The builder wrote counts over the 2L reference, which has no sentinel; the BWT does,
            // and it sorts before everything, so every base's first row shifts down by one. Skip
            // this and every LF-mapping is off by one row, which corrupts every reported position.
            *c += 1; // as load_index does
        }
        // On-disk layout after the header: `cp_occ[cp_size]`, `sa_ms_byte[sa_size]` (i8),
        // `sa_ls_word[sa_size]` (u32 LE), `sentinel_index` (i64). All little-endian, matching the
        // in-memory representation on this (LE) target, so each block is one bulk copy.
        // ELEMENT counts (not byte sizes) of the two payload arrays, both derived from the row
        // count. `cp_size` = one 64-byte checkpoint per 64 rows, `+1` so that a query at row
        // `ref_seq_len` itself (interval ENDS are passed to `get_occ`) still indexes a real block.
        // `sa_size` = one sampled suffix-array entry per 8 rows, `+1` for the same edge row.
        let cp_size = ((ref_seq_len >> CP_SHIFT) + 1) as usize;
        let sa_size = ((ref_seq_len >> SA_SHIFT) + 1) as usize;

        // ---- Payload: checkpoints, then the two sampled-SA arrays, then the sentinel row ------
        // BYTE lengths in the file, which is what the offsets are made of. A file record is 64 bytes
        // whatever `CpOcc` is in memory.
        const FILE_CP_BYTES: usize = 64;
        let cp_bytes = cp_size * FILE_CP_BYTES;
        let ls_bytes = sa_size * 4;
        let bwt_need = HEADER_BYTES + cp_bytes as u64 + sa_size as u64 + ls_bytes as u64 + 8;
        if bwt_len < bwt_need {
            return Err(damaged(format!(
                ".bwt.2bit.64 is {bwt_len} bytes and its header needs {bwt_need}"
            )));
        }

        // Trailing header field, read first because the checkpoints are built with it in hand:
        // the one BWT ROW (not a reference position) whose suffix is the whole text, i.e. the row
        // with `sa[row] == 0`.
        let mut sentinel_buf = [0u8; 8];
        read_at_parallel(
            &bwt_file,
            HEADER_BYTES + cp_bytes as u64 + sa_size as u64 + ls_bytes as u64,
            &mut sentinel_buf,
        )?;
        let sentinel_index = i64::from_le_bytes(sentinel_buf);

        // Hinted BEFORE the fill, so the pages are faulted in at the promoted size instead of being
        // collapsed afterwards by khugepaged. This is the single most valuable of the hints: every
        // `get_occ` lands here at a computed, unpredictable address. See `crate::hugepage` for why
        // this cannot change output.
        let mut cp_occ = Vec::<CpOcc>::with_capacity(cp_size);
        unsafe {
            crate::hugepage::advise_hugepage(
                cp_occ.as_ptr() as *const u8,
                cp_size * std::mem::size_of::<CpOcc>(),
            );
        }
        // Converted a slice at a time through a small buffer per chunk, so the 64-byte file records
        // and the 32-byte ones are never both resident: the peak of the load is the destination
        // alone, which is what reading straight into the array was for when the two were the same.
        //
        // Rust: `spare_capacity_mut` is the reserved, uninitialised tail of the vector as
        // `MaybeUninit`s, which safe code may write and may not read. `set_len` below is the promise
        // that every one of them was written, and `try_for_each` covers the whole slice or errors.
        //
        // A chunk cannot know its span's start count, so each record first holds the low 32 bits of
        // its full count. `count_base` and the rebase below finish the job on a genome with more than
        // one span, and on any smaller one the low bits already are the count.
        cp_occ.spare_capacity_mut()[..cp_size]
            .par_chunks_mut(LOAD_CHUNK_RECORDS)
            .enumerate()
            .try_for_each_init(
                || vec![0u8; LOAD_CHUNK_RECORDS * FILE_CP_BYTES],
                |buf, (i, part)| -> Result<()> {
                    let bytes = &mut buf[..part.len() * FILE_CP_BYTES];
                    read_exact_at(
                        &bwt_file,
                        HEADER_BYTES + (i * LOAD_CHUNK_RECORDS * FILE_CP_BYTES) as u64,
                        bytes,
                    )?;
                    for (rec, dst) in bytes
                        .as_chunks::<FILE_CP_BYTES>()
                        .0
                        .iter()
                        .zip(part.iter_mut())
                    {
                        let word = |n: usize| {
                            u64::from_le_bytes(rec[8 * n..8 * n + 8].try_into().unwrap())
                        };
                        dst.write(CpOcc {
                            cp_count: [
                                word(0) as u32,
                                word(1) as u32,
                                word(2) as u32,
                                word(3) as u32,
                            ],
                            hi: word(6) | word(7),
                            lo: word(5) | word(7),
                        });
                    }
                    Ok(())
                },
            )?;
        // SAFETY: the `try_for_each_init` above returned Ok, so every record was written.
        unsafe { cp_occ.set_len(cp_size) };

        // Differences of adjacent `count` entries are the per-base totals, and a total below 2^32
        // means every count below it fits too.
        let wide = count.windows(2).any(|w| w[1] - w[0] >= 1 << 32);
        // One span entry per 2^32 rows, read straight from the file's full-width counts at each
        // span's first block. The `+ 1` is the span holding row `ref_seq_len` itself.
        let n_spans = if wide {
            (ref_seq_len >> COUNT_SPAN_SHIFT) as usize + 1
        } else {
            1
        };
        let mut count_base = vec![[0i64; 4]; n_spans];
        for (span, base) in count_base.iter_mut().enumerate().skip(1) {
            let block = span as u64 * (1 << (COUNT_SPAN_SHIFT - CP_SHIFT));
            let mut rec = [0u8; 32];
            read_at_parallel(
                &bwt_file,
                HEADER_BYTES + block * FILE_CP_BYTES as u64,
                &mut rec,
            )?;
            for (c, b) in base.iter_mut().enumerate() {
                *b = i64::from_le_bytes(rec[8 * c..8 * c + 8].try_into().unwrap());
            }
        }
        if wide {
            rebase_counts(&mut cp_occ, &count_base, COUNT_SPAN_SHIFT);
        }

        // The high part of each sampled SA entry: one byte each in the file, `1 << sa_hi_log2` bits
        // each here. A word holds a whole number of entries, so chunks of words are independent.
        let sa_hi_log2 = sa_hi_log2_bits(ref_seq_len);
        let entries_per_word = 64usize >> sa_hi_log2;
        let mut sa_hi = vec![0u64; sa_size.div_ceil(entries_per_word)];
        // 97 MB on GRCh38, walked randomly by `get_sa` alongside `sa_ls_word`.
        unsafe {
            crate::hugepage::advise_hugepage(sa_hi.as_ptr() as *const u8, sa_hi.len() * 8);
        }
        let words_per_chunk = (LOAD_CHUNK_RECORDS * 64) / entries_per_word;
        sa_hi
            .par_chunks_mut(words_per_chunk)
            .enumerate()
            .try_for_each_init(
                || vec![0u8; words_per_chunk * entries_per_word],
                |buf, (i, words)| -> Result<()> {
                    let first = i * words_per_chunk * entries_per_word;
                    let n = (words.len() * entries_per_word).min(sa_size - first);
                    let bytes = &mut buf[..n];
                    read_exact_at(
                        &bwt_file,
                        HEADER_BYTES + cp_bytes as u64 + first as u64,
                        bytes,
                    )?;
                    for (word, entries) in words.iter_mut().zip(bytes.chunks(entries_per_word)) {
                        for (j, &b) in entries.iter().enumerate() {
                            *word |= u64::from(b) << (j << sa_hi_log2);
                        }
                    }
                    Ok(())
                },
            )?;

        // Low 32 bits of each sampled SA entry; `ls_bytes` is its BYTE length (4 per element).
        //
        // Rust, for the read that follows. In plain terms: reserve the exact memory, aim the file
        // read straight at it, then declare it filled. `with_capacity(n)` reserves room without
        // creating any element, `from_raw_parts_mut` views that room as bytes, and `set_len(n)`
        // afterwards is the promise that all `n` now hold real data. Reading into a `Vec<u8>` and
        // converting would be safe, and would hold both copies at once.
        let mut sa_ls_word = Vec::<u32>::with_capacity(sa_size);
        // SAFETY: the slice covers exactly the capacity just reserved, `u32` has no invalid bit
        // patterns, and `read_at_parallel` fills all of it or returns an error before `set_len`.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(sa_ls_word.as_mut_ptr() as *mut u8, ls_bytes);
            // 3.1 GB on GRCh38: the other half of every `get_sa` lookup.
            crate::hugepage::advise_hugepage(sa_ls_word.as_ptr() as *const u8, ls_bytes);
            read_at_parallel(
                &bwt_file,
                HEADER_BYTES + cp_bytes as u64 + sa_size as u64,
                dst,
            )?;
            sa_ls_word.set_len(sa_size);
        }

        // ---- The reference, memory-mapped from the 2-bit `.pac` -------------------------------
        // `.pac` (forward only, 2 bits/base, ~L/4 bytes) rather than `.0123` (forward++RC, one
        // byte/base, 2L bytes): the same reference bases, unpacked on demand in `base`, at ~1/8th
        // the resident footprint. `.0123` is still on disk (index byte-identity is unchanged), just
        // not mapped.
        let pac_file = std::fs::File::open(sibling(prefix, "pac"))?;
        // SAFETY: the index files are not mutated while a run holds them open; a concurrent external
        // truncation is out of scope (same assumption as bwa-mem2's mmap'd index).
        let pac = unsafe { Mmap::map(&pac_file)? };
        let pac_need = ((ref_seq_len as u64 - 1) / 2).div_ceil(4) as usize;
        if pac.len() < pac_need {
            return Err(damaged(format!(
                ".pac is {} bytes and the reference needs {pac_need}",
                pac.len()
            )));
        }
        // ~800 MB on GRCh38, read at extension time at positions derived from seed hits, i.e. also
        // effectively random. A file-backed mapping only takes huge pages on a kernel and
        // filesystem that support it (Linux >= 5.7 plus a THP-capable fs), so treat this one as a
        // bonus rather than a load-bearing hint; it costs one ignored syscall where unsupported.
        unsafe { crate::hugepage::advise_hugepage(pac.as_ptr(), pac.len()) };

        Ok(FmIndex {
            ref_seq_len,
            count,
            cp_occ,
            count_base,
            sa_hi,
            sa_hi_log2,
            sa_hi_mask: (1u64 << (1u32 << sa_hi_log2)) - 1,
            sa_ls_word,
            sentinel_index,
            sentinel_block: sentinel_index >> CP_SHIFT,
            wide,
            count_span_shift: COUNT_SPAN_SHIFT,
            pac,
        })
    }

    /// Measure how many independent cache-line fetches this core can keep in flight, by chasing
    /// dependent random addresses **inside this index's own checkpoint array**.
    ///
    /// # Why the index and not a scratch buffer
    ///
    /// A probe over a freshly allocated buffer measures whatever that buffer fits in. A 64 MiB one
    /// sits inside this machine's system-level cache and reports 2.7 ns per access, four times
    /// faster than DRAM, which then picks the widest candidate for the wrong reason. `cp_occ` is
    /// gigabytes, already resident, already the array the seeding walk will hammer, and it carries
    /// the same TLB behaviour. Probing it costs no allocation and measures the real thing.
    ///
    /// # Method
    ///
    /// Each chain reads a word from a block, mixes it, and derives its next block index from the
    /// value read, so the address genuinely depends on the load and no prefetcher can run ahead.
    /// `k` chains advance in lockstep, which is the aligner's own access shape. The reported time
    /// per access falls while the core still has miss slots to spare and flattens when it does not.
    ///
    /// # Returns
    /// Nanoseconds per access for each width in `widths`, in order. **EMPTY when the index is too
    /// small to measure anything**, which the caller must read as "use the default width".
    ///
    /// The empty return replaces an `assert!`, and the assert was a crash on ordinary input: an
    /// index whose `cp_occ` holds 1024 blocks or fewer is about 64 kbp of reference, so aligning
    /// against a plasmid, a single gene, an amplicon panel or any small contig set aborted the
    /// process with `index too small to probe` where bwa-mem2 aligns normally. Nothing about the
    /// probe is load-bearing -- it picks a lockstep WIDTH, which is scheduling and cannot move a
    /// byte of output -- so having no measurement is a reason to fall back, never to stop.
    pub fn probe_concurrency(&self, widths: &[usize], accesses_per_width: usize) -> Vec<f64> {
        let n_blocks = self.cp_occ.len();
        // Below this the chains cannot be spread far enough apart to be independent, and the whole
        // array sits in cache anyway, so any number measured here would describe the cache rather
        // than the memory system the width is meant to match.
        if n_blocks <= 1024 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(widths.len());
        for &k in widths {
            // Chains start spread across the array so they do not share lines.
            let mut cur: Vec<usize> = (0..k).map(|i| i * (n_blocks / k.max(1))).collect();
            let steps = accesses_per_width / k.max(1);
            // One untimed pass, so every width starts from the same cache and TLB state.
            for _ in 0..64 {
                for c in cur.iter_mut() {
                    let v = self.cp_occ[*c % n_blocks].hi ^ (*c as u64);
                    *c = (v.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 24) as usize % n_blocks;
                }
            }
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                for c in cur.iter_mut() {
                    // The load's VALUE decides the next address: a true dependency, not a stride.
                    let v = self.cp_occ[*c].hi ^ (*c as u64);
                    *c = (v.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 24) as usize % n_blocks;
                }
            }
            let ns = t0.elapsed().as_secs_f64() * 1e9 / (steps * k) as f64;
            std::hint::black_box(&cur);
            out.push(ns);
        }
        out
    }

    /// Prefetch the two checkpoint blocks a future [`backward_ext`] on an interval `[sp, ep)` will
    /// touch (`cp_occ[sp>>6]` and `cp_occ[ep>>6]`). Issued one SMEM step ahead in the seeding walk to
    /// hide the DRAM latency of the data-dependent block loads, exactly as bwa-mem2's / nh13's
    /// `ENABLE_PREFETCH` does. A pure hint: results are unchanged. No-op off AArch64/x86_64.
    ///
    /// # Parameters
    /// * `sp`: BWT ROW (not a reference position) of the interval start, i.e. some future `smem.k`.
    ///   Must be in `[0, ref_seq_len]` so that `sp >> 6` indexes an existing checkpoint.
    /// * `ep`: BWT ROW of the interval end, exclusive, i.e. `k + s`. Same range. May equal `sp`
    ///   (empty interval), in which case the same line is prefetched twice, which is harmless.
    ///
    /// Supplied by the seeding walk one step before it needs them. Passing stale or wrong rows
    /// costs a wasted fetch and nothing else.
    #[inline]
    pub fn prefetch_occ(&self, sp: i64, ep: i64) {
        // SAFETY (both arches): forming the block pointers is in-bounds (`sp`/`ep` are valid BWT
        // positions, so `>>6` indexes an allocated `cp_occ` slot); the prefetch is a hint that never
        // faults or writes.
        #[cfg(target_arch = "aarch64")]
        {
            // Element-typed base pointer, so `.add(n)` steps whole checkpoint records.
            let base = self.cp_occ.as_ptr();
            // `pldl1keep` = prefetch-for-load into L1, keep (AArch64 equivalent of `_MM_HINT_T0`).
            unsafe {
                // Addresses of the two checkpoint BLOCKS covering rows `sp` and `ep`.
                let p_sp = base.add((sp >> CP_SHIFT) as usize);
                let p_ep = base.add((ep >> CP_SHIFT) as usize);
                std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p_sp, options(nostack, readonly, preserves_flags));
                std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p_ep, options(nostack, readonly, preserves_flags));
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            // Same two checkpoint blocks as the AArch64 arm; `_MM_HINT_T0` = fetch into L1.
            let base = self.cp_occ.as_ptr();
            unsafe {
                _mm_prefetch(
                    base.add((sp >> CP_SHIFT) as usize) as *const i8,
                    _MM_HINT_T0,
                );
                _mm_prefetch(
                    base.add((ep >> CP_SHIFT) as usize) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            // No portable prefetch intrinsic: discard the rows to silence the unused warning.
            let _ = (sp, ep);
        }
    }

    /// Occurrences of base `c` in `bwt[0..pp)`, i.e. bwa-mem2's `GET_OCC` (`FMI_search.h:66-73`).
    ///
    /// `pp` is a BWT row in `[0, ref_seq_len]` INCLUSIVE (interval ends are passed here, which is
    /// why the checkpoint array carries one spare block). `c` is a base code in `0..4`; codes 4
    /// (sentinel) and 6 (padding) have no bitmap and would index out of bounds.
    ///
    /// The whole query is one cache line and about five instructions:
    ///   block `pp >> 6` gives the count of `c` in every row before the block,
    ///   `pp & 63` gives the offset within the block,
    ///   masking the bitmap to the rows before that offset and popcounting adds the remainder.
    ///
    /// Worked micro-example, `pp = 130`, so block 2 and offset 2. `cp_count[c]` already covers rows
    /// 0..128. `rows_before(2) = 0xC000...` keeps bits 63 and 62, that is rows 128 and 129. Sum is
    /// the count over rows 0..130, exclusive of 130. Correct.
    ///
    /// # Parameters
    /// * `pp`: a BWT ROW, never a reference position. Range `[0, ref_seq_len]` inclusive at the
    ///   top end. Supplied by `backward_ext` (as `k` or `k + s`) and by the `get_sa` LF-walk.
    /// * `c`: base code, `0..4` for A, C, G, T. NOT a nucleotide character and not the sentinel:
    ///   code 4 or 6 indexes past the 4-element arrays and panics in debug / reads garbage in
    ///   release. Callers must have already screened out the sentinel row.
    ///
    /// # Returns
    /// The count, in `[0, pp]`. Exclusive of row `pp` itself, which is what makes
    /// `count[c] + get_occ(pp, c)` the correct LF-mapping.
    #[inline]
    pub fn get_occ(&self, pp: i64, c: usize) -> i64 {
        // The 64-row checkpoint covering row `pp`, and `pp`'s OFFSET within that block (0..64),
        // which doubles as the index into the mask table.
        let block = &self.cp_occ[(pp >> CP_SHIFT) as usize];
        self.occ_in(block, pp, c)
    }

    /// [`get_occ`](Self::get_occ) against a block the caller already loaded, which is what lets an
    /// LF-walk step read the row's base and its count from one load.
    #[inline(always)]
    fn occ_in(&self, block: &CpOcc, pp: i64, c: usize) -> i64 {
        // The rows holding `c`: each plane as is where `c` has that bit set, inverted where it does
        // not. `wrapping_sub(1)` of the bit is all ones for 0 and zero for 1.
        let hi = block.hi ^ ((c as u64 >> 1) & 1).wrapping_sub(1);
        let lo = block.lo ^ (c as u64 & 1).wrapping_sub(1);
        let rows = (hi & lo & rows_before(pp & CP_MASK)).count_ones() as i64;
        let mut occ = i64::from(block.cp_count[c]) + rows;
        if self.wide {
            occ += self.count_base[(pp >> self.count_span_shift) as usize][c];
        }
        // Compared before anything is known about `c`, because a walk's `c` is the genome's own base
        // and a branch on it is wrong a third of the time.
        if pp >> CP_SHIFT == self.sentinel_block {
            occ -= self.sentinel_before(pp, c);
        }
        occ
    }

    /// How many rows [`occ_in`](Self::occ_in) over-counted in the sentinel's own block: 1 for an A
    /// count at a row past the sentinel, else 0.
    ///
    /// The sentinel has no base and the planes cannot say so, so it reads as an A. Counting rows in
    /// the block is the only place that sees it: a checkpoint's count was taken from the file's
    /// one-hot form, where the sentinel set no bit. One block of 97M on GRCh38, hence out of line.
    #[cold]
    #[inline(never)]
    fn sentinel_before(&self, pp: i64, c: usize) -> i64 {
        i64::from(c == 0 && pp > self.sentinel_index)
    }

    /// The four counts of `bwt[0..row)`, from one block and three popcounts.
    ///
    /// `T` is where both planes are set, `G` and `C` are the rest of each plane, and `A` is whatever
    /// rows remain, so the fourth popcount the file's one-hot form would need is a subtraction.
    #[inline(always)]
    fn occ4(&self, block: &CpOcc, row: i64) -> [i64; 4] {
        let y = row & CP_MASK;
        let m = rows_before(y);
        let t = (block.hi & block.lo & m).count_ones() as i64;
        let g = (block.hi & m).count_ones() as i64 - t;
        let c = (block.lo & m).count_ones() as i64 - t;
        let mut a = y - t - g - c;
        if row >> CP_SHIFT == self.sentinel_block {
            a -= self.sentinel_before(row, 0);
        }
        let mut occ = [
            i64::from(block.cp_count[0]) + a,
            i64::from(block.cp_count[1]) + c,
            i64::from(block.cp_count[2]) + g,
            i64::from(block.cp_count[3]) + t,
        ];
        if self.wide {
            let base = &self.count_base[(row >> self.count_span_shift) as usize];
            for b in 0..4 {
                occ[b] += base[b];
            }
        }
        occ
    }

    /// `backward_ext`'s starts and sizes on a `wide` genome, out of line so that its scalar counts do
    /// not sit in the seeding walk's own code on every other genome.
    #[cold]
    #[inline(never)]
    fn ext_sizes_wide(
        &self,
        blk_sp: &CpOcc,
        blk_ep: &CpOcc,
        sp: i64,
        ep: i64,
    ) -> ([i64; 4], [i64; 4]) {
        let occ_sp = self.occ4(blk_sp, sp);
        let occ_ep = self.occ4(blk_ep, ep);
        (
            std::array::from_fn(|b| self.count[b] + occ_sp[b]),
            std::array::from_fn(|b| occ_ep[b] - occ_sp[b]),
        )
    }

    /// [`occ4`](Self::occ4) for a genome that is not `wide`, where 32 bits hold any count.
    ///
    /// All four in one vector on AArch64, where a scalar popcount has to cross to the vector unit and
    /// back: this is `backward_ext`'s whole cost, and the sums below are about 15 instructions for
    /// what four scalar counts take 40 for. Two registers hold the four masked one-hot words, `cnt`
    /// counts their bytes, and three pairwise adds fold each word's eight bytes into one lane, the
    /// last of them onto the checkpoint's counts.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn occ4_narrow(&self, block: &CpOcc, row: i64) -> [u32; 4] {
        use std::arch::aarch64::*;
        // SAFETY: NEON is mandatory on AArch64, and the one load reads the four `u32`s of `block`.
        let mut occ: [u32; 4] = unsafe {
            let m = vdupq_n_u64(rows_before(row & CP_MASK));
            let hi = vdupq_n_u64(block.hi);
            // `[!lo, lo]`, so the one register gives both the low-bit-clear and low-bit-set bases.
            let lo = vcombine_u64(vcreate_u64(!block.lo), vcreate_u64(block.lo));
            let ac = vandq_u64(vbicq_u64(m, hi), lo);
            let gt = vandq_u64(vandq_u64(m, hi), lo);
            let bytes = vpaddq_u8(
                vcntq_u8(vreinterpretq_u8_u64(ac)),
                vcntq_u8(vreinterpretq_u8_u64(gt)),
            );
            let halves = vpaddlq_u8(bytes);
            let counts = vld1q_u32(block.cp_count.as_ptr());
            std::mem::transmute(vpadalq_u16(counts, halves))
        };
        if row >> CP_SHIFT == self.sentinel_block {
            occ[0] -= self.sentinel_before(row, 0) as u32;
        }
        occ
    }

    /// [`occ4`](Self::occ4) for a genome that is not `wide`, where 32 bits hold any count.
    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn occ4_narrow(&self, block: &CpOcc, row: i64) -> [u32; 4] {
        self.occ4(block, row).map(|c| c as u32)
    }

    /// One sampled suffix-array entry, by its index in the sampled arrays (row / 8): a 2L-space
    /// reference position reassembled from its split storage.
    #[inline(always)]
    fn sa_sample(&self, idx: usize) -> i64 {
        let bit = idx << self.sa_hi_log2;
        let hi = (self.sa_hi[bit >> 6] >> (bit & 63)) & self.sa_hi_mask;
        ((hi as i64) << 32) + i64::from(self.sa_ls_word[idx])
    }

    /// The base at row `sp`, as a code: its two bits, one from each plane. Only meaningful off the
    /// sentinel row, which the caller tests for, since the planes store it as an A.
    #[inline(always)]
    fn base_at(block: &CpOcc, sp: i64) -> usize {
        // MSB-first: row `sp` sits at bit `63 - (sp & 63)`.
        let y = CP_BLOCK_SIZE - (sp & CP_MASK) - 1;
        (((block.hi >> y) & 1) << 1 | ((block.lo >> y) & 1)) as usize
    }

    /// The full interval `[0, ref_seq_len)` (the empty match), for starting a backward search.
    ///
    /// The empty pattern matches at every one of the `N` rows, so `k = l = 0` and `s = N`. Every
    /// backward search starts here and narrows.
    ///
    /// # Returns
    /// An `Smem` whose read-coordinate fields (`rid`, `m`, `n`) are all 0 placeholders: the caller
    /// fills them in once it knows which read and which span the seed came from. Only the interval
    /// fields are meaningful here.
    #[inline]
    pub fn full_interval(&self) -> Smem {
        Smem {
            rid: 0,
            m: 0,
            n: 0,
            k: 0,
            l: 0,
            s: self.ref_seq_len,
        }
    }

    /// Extend `smem` by one base `a` on the left, i.e. bwa-mem2's `backwardExt`
    /// (`FMI_search.cpp:1025`). `a` is a base code in `0..4`.
    ///
    /// Standard FM backward step for `k` and `s`: rows matching `a` + pattern start at
    /// `count[a] + occ(a, k)` and there are `occ(a, k+s) - occ(a, k)` of them.
    ///
    /// The interesting half is `l`, the reverse-complement interval, which is what makes this an
    /// FMD (bidirectional) index. All four candidate extensions partition the parent interval, and
    /// on the reverse-complement side they appear in the OPPOSITE base order, so `l` is recovered
    /// by accumulating the sibling sizes from base 3 downwards: `l[3]` first, then each lower base
    /// sits above the ones already accounted for. This is why the assignments run 3, 2, 1, 0 and
    /// cannot be reordered.
    ///
    /// `sentinel_offset` handles the row that has no base at all. If the sentinel row lies inside
    /// `[k, k+s)` it consumes one slot of the parent interval that none of the four `s[b]` counted,
    /// so every reverse-side start shifts up by one. Omitting it makes `l` drift by one exactly on
    /// the intervals that contain the sentinel, which is rare enough to pass small tests and then
    /// corrupt seeds on a real genome.
    ///
    /// All four bases are computed even though only `a` is returned, because the `l` recurrence
    /// needs the sibling sizes. That is also why loading the two checkpoint blocks once (rather
    /// than calling `get_occ` eight times) is a pure win: all 8 queries hit just 2 cache lines.
    ///
    /// # Parameters
    /// * `smem`: the interval for the pattern BEFORE this base is prepended. Taken by value (it is
    ///   6 words) and its read-coordinate fields are carried through untouched, so the caller
    ///   still owns the `m`/`n` bookkeeping. Must satisfy the struct invariant; typically the
    ///   previous step's result or [`full_interval`].
    /// * `a`: the base code to prepend, `0..4`. A code of 4 (sentinel) or 6 (ambiguous base N)
    ///   would index past the 4-element arrays, so callers must stop the search at an N instead of
    ///   passing it here.
    ///
    /// # Returns
    /// The interval for `a ++ pattern`, with `s == 0` if that extended pattern does not occur.
    /// Note `s == 0` is a normal, expected result, not an error, and the returned `k`/`l` are then
    /// meaningless.
    // Copied from the fork's PR #88 (`backwardExt` marked `__attribute__((always_inline))`): called
    // ~1e9x per run, and returning the ~40-byte `Smem` by value forces a hidden-pointer store on
    // both the SysV (x86) and AAPCS (arm64) ABIs unless the call is inlined into the lockstep walk.
    // On x86 the fork measured that store at ~42% of the function's samples. `always` (not a plain
    // `#[inline]` hint) because LTO's cost model otherwise declines this larger function, exactly the
    // gcc-14 regression that PR fixed. Byte-identical: inlining changes no arithmetic.
    #[inline(always)]
    pub fn backward_ext(&self, smem: Smem, a: usize) -> Smem {
        // Load the sp/ep checkpoint blocks once (all 4 bases share them), rather than re-deriving
        // the block index and re-indexing per base as `get_occ` would. Values are identical.
        // The parent interval as a half-open row range `[sp, ep)`: `sp` is the first matching BWT
        // ROW, `ep` the first row past it. Both are rows, not reference positions.
        let sp = smem.k;
        let ep = smem.k + smem.s;
        #[cfg(feature = "traffic")]
        if traffic::enabled() {
            // Four 32B blocks share one 128B line, and for a small interval sp>>6 == ep>>6, so calls
            // are a bad proxy for traffic. Count distinct LINES.
            // 128-byte LINE indices for the two blocks: block index >> 2, since four 32-byte
            // checkpoints share one line.
            let (l1, l2) = ((sp >> CP_SHIFT) >> 2, (ep >> CP_SHIFT) >> 2);
            traffic::EXT_LINES.fetch_add(
                if l1 == l2 { 1 } else { 2 },
                std::sync::atomic::Ordering::Relaxed,
            );
            traffic::EXT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // The two checkpoint blocks (often the same one for a narrow interval). Hoisted out so all 8
        // occurrence queries below reuse these 2 loads. Inlining `get_occ` here is not an optimisation
        // of the arithmetic: it is identical, only the loads are shared.
        let blk_sp = &self.cp_occ[(sp >> CP_SHIFT) as usize];
        let blk_ep = &self.cp_occ[(ep >> CP_SHIFT) as usize];
        // Per-base results, indexed by base code 0..4. `k[b]` is the first BWT ROW of the child
        // interval for `b ++ pattern`; `s[b]` is its size. All four are computed because the `l`
        // recurrence below needs every sibling's size, not just the requested base's.
        let mut k = [0i64; 4];
        let mut s = [0i64; 4];
        if self.wide {
            (k, s) = self.ext_sizes_wide(blk_sp, blk_ep, sp, ep);
        } else {
            // Occurrences of `b` in `bwt[0..sp)` and `bwt[0..ep)`. Their difference is how many of
            // the parent's rows carry `b`.
            let occ_sp = self.occ4_narrow(blk_sp, sp);
            let occ_ep = self.occ4_narrow(blk_ep, ep);
            for b in 0..4 {
                k[b] = self.count[b] + i64::from(occ_sp[b]);
                s[b] = i64::from(occ_ep[b].wrapping_sub(occ_sp[b]));
            }
        }
        // 1 exactly when the sentinel ROW falls inside the parent interval `[k, k+s)`, else 0. That
        // row carries no base, so none of the four `s[b]` counted it, yet it does occupy a slot on
        // the reverse-complement side; adding 1 shifts every `l` up to compensate.
        let sentinel_offset =
            i64::from((smem.k <= self.sentinel_index) & (smem.k + smem.s > self.sentinel_index));
        // Reverse-complement interval starts, filled 3 down to 0. Invariant after assigning `l[b]`:
        // `l[b]` is the first row of the RC interval for base `b`, sitting immediately above the
        // RC intervals of all bases GREATER than `b` (hence adding each higher sibling's `s`).
        // The descending order is forced by that dependency and must not be rewritten.
        let mut l = [0i64; 4];
        l[3] = smem.l + sentinel_offset;
        l[2] = l[3] + s[3];
        l[1] = l[2] + s[2];
        l[0] = l[1] + s[1];
        Smem {
            k: k[a],
            l: l[a],
            s: s[a],
            ..smem
        }
    }

    /// Suffix-array value at BWT row `pos`, decompressing via LF-walk to the nearest sample
    /// (bwa-mem2's `get_sa_entry_compressed`, `FMI_search.cpp:1103`).
    ///
    /// This is the step that turns "the pattern occupies rows k..k+s" into actual 2L-space
    /// reference positions, so it runs once per seed occurrence and is the crate's hottest random
    /// memory access.
    ///
    /// Only every 8th row is sampled (`SA_COMPX = 3`). For an unsampled row, walk BACKWARDS one
    /// reference position at a time via LF-mapping, counting steps, until landing on a row whose
    /// index is a multiple of 8; then `sa[pos] = sa[landing] + steps`. Each step costs one random
    /// `cp_occ` line, and the average walk is under 4 steps, so the 8x space saving buys a bounded
    /// time cost.
    ///
    /// The early return is the sentinel row: it has no preceding character, meaning the walk reached
    /// reference position 0, so the answer is exactly the number of steps taken.
    ///
    /// INVARIANT: `pos` must be a valid BWT row in `[0, ref_seq_len)`. The loop terminates because
    /// LF-mapping is a permutation with no cycles shorter than the text, so a multiple of 8 (or the
    /// sentinel) is always reached.
    ///
    /// # Parameters
    /// * `pos`: a BWT ROW in `[0, ref_seq_len)`, typically one row of a seed's `[k, k+s)`. This is
    ///   a rank in the sorted suffix list, NOT a coordinate in the genome; converting it is the
    ///   whole job of this function. Out-of-range rows panic on the `cp_occ` index.
    ///
    /// # Returns
    /// A position in 2L-space, `[0, 2 * l_pac)`: the forward genome occupies `[0, l_pac)` and its
    /// reverse complement `[l_pac, 2*l_pac)`, so a result at or above `l_pac` is a reverse-strand
    /// hit and must go through `BntSeq::depos` before it means anything on the forward strand.
    pub fn get_sa(&self, pos: i64) -> i64 {
        // Fast path: `pos & 7 == 0` means the row was sampled outright. Reassemble the 40-bit value
        // from its split storage, high byte shifted up 32 (see `build::write_fm_index`).
        if pos & SA_MASK == 0 {
            // Index into the SAMPLED arrays (row / 8), not a row and not a position.
            return self.sa_sample((pos >> SA_SHIFT) as usize);
        }
        // Walk state. `offset` counts LF steps taken so far, which equals how many reference
        // positions EARLIER than `sa[pos]` the current row sits; `sp` is the current BWT ROW.
        // Loop invariant at the top of each iteration: `sa[pos] == sa[sp] + offset`, and `sp` is
        // not a multiple of 8 (the exit test at the bottom guarantees it).
        let mut offset = 0i64;
        let mut sp = pos;
        loop {
            #[cfg(feature = "traffic")]
            if traffic::enabled() {
                // One line per step (the base and its count come from the same block).
                traffic::SA_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // The row with no base means the walk has reached reference position 0.
            if sp == self.sentinel_index {
                return offset;
            }
            // The checkpoint block holding row `sp`: the random DRAM access that dominates the
            // crate's runtime, which one load now answers both questions from.
            let block = &self.cp_occ[(sp >> CP_SHIFT) as usize];
            // LF-mapping: the row whose suffix is one character LONGER, i.e. one reference position
            // EARLIER. `count[b] + occ(b, sp)` is the standard formula, and it is the same
            // arithmetic `backward_ext` uses for `k`.
            let b = Self::base_at(block, sp);
            sp = self.count[b] + self.occ_in(block, sp, b);
            offset += 1;
            if sp & SA_MASK == 0 {
                break;
            }
        }
        // Here `sp` is a multiple of 8, so its SA value was stored outright. `sa_entry` is a
        // 2L-space reference POSITION; adding back the `offset` steps walked undoes the walk.
        self.sa_sample((sp >> SA_SHIFT) as usize) + offset
    }

    /// Prefetch the single `cp_occ` checkpoint block that the LF-walk step at `pos` will touch. A
    /// pure hint (never faults or writes); no-op off AArch64/x86_64.
    ///
    /// # Parameters
    /// * `pos`: the BWT ROW the walk is about to examine, i.e. a slot's current `sp`. Must be in
    ///   `[0, ref_seq_len]` so `pos >> 6` is an allocated checkpoint. Unlike [`prefetch_occ`] this
    ///   touches ONE block, because an LF-walk step queries a single row rather than an interval.
    #[inline]
    fn prefetch_cp(&self, pos: i64) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            // Address of the checkpoint block covering row `pos`.
            let p = self.cp_occ.as_ptr().add((pos >> CP_SHIFT) as usize);
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(
                self.cp_occ.as_ptr().add((pos >> CP_SHIFT) as usize) as *const i8,
                _MM_HINT_T0,
            );
        }
        // No portable prefetch intrinsic: discard the row to silence the unused warning.
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = pos;
    }

    /// Prefetch the two words [`sa_sample`](Self::sa_sample) will read for sampled index `idx`. A hint
    /// like [`prefetch_cp`](Self::prefetch_cp), with no effect on any result.
    #[inline]
    fn prefetch_sa_sample(&self, idx: usize) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let ls = self.sa_ls_word.as_ptr().add(idx);
            let hi = self.sa_hi.as_ptr().add((idx << self.sa_hi_log2) >> 6);
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ls, options(nostack, readonly, preserves_flags));
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) hi, options(nostack, readonly, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(self.sa_ls_word.as_ptr().add(idx) as *const i8, _MM_HINT_T0);
            _mm_prefetch(
                self.sa_hi.as_ptr().add((idx << self.sa_hi_log2) >> 6) as *const i8,
                _MM_HINT_T0,
            );
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = idx;
    }

    /// [`get_sa`] for many independent positions at once. Each `get_sa` is a data-dependent LF-walk
    /// (each step a random `cp_occ` block load), but distinct positions are independent, so running a
    /// **window** of them in lockstep (prefetch every active slot's next block, then advance all one
    /// step) keeps many DRAM misses in flight and hides the latency. Result-identical to calling
    /// [`get_sa`] per position; `out[i]` receives `get_sa(positions[i])`.
    ///
    /// `positions` and `out` must be the same length. Positions need not be distinct or sorted.
    ///
    /// Structure of the lockstep loop, since it is not obvious from the indices:
    /// * `W = 32` positions form a window. `sp[j]` is slot `j`'s current row, `off[j]` its step
    ///   count. `slot[0..nact]` is the compaction list of slots still walking, so finished slots
    ///   drop out without disturbing the rest.
    /// * Each round issues ALL prefetches first, then does the work. Splitting the loop is the
    ///   whole trick: the prefetches for round `r` overlap the loads of round `r`, so up to 32 DRAM
    ///   misses are in flight instead of one.
    /// * `wr` rewrites `slot` in place while iterating it, which is safe because `wr <= a` always
    ///   (a slot is only ever written to a position at or before the one being read).
    ///
    /// Results are bit-identical to calling [`get_sa`] in a loop; only the memory schedule differs.
    ///
    /// # Parameters
    /// * `positions`: BWT ROWS to resolve, each in `[0, ref_seq_len)`. Order is preserved but
    ///   irrelevant to correctness; duplicates are allowed and simply resolved twice. In practice
    ///   these are the rows of one or more seeds' `[k, k+s)` intervals.
    /// * `out`: destination, written in full; `out[i]` gets the 2L-space reference POSITION for
    ///   `positions[i]`. Must already be `positions.len()` long (checked only by `debug_assert`,
    ///   so a mismatch in release panics later on an index, or silently under-fills). Prior
    ///   contents are ignored, never read.
    pub fn get_sa_batch(&self, positions: &[i64], out: &mut [i64]) {
        debug_assert_eq!(positions.len(), out.len());
        // Window width: how many independent LF-walks are kept in flight, which is how many DRAM
        // misses are outstanding at once. This is the single hottest number in the aligner on a
        // whole-genome run: SA resolution is 42.5M random lookups per 500k pairs and it dominates
        // chain building, which in turn dominates the align stage.
        //
        // Swept on GRCh38 at -t16, 500k pairs, cost per lookup: W=8 369 ns, 16 272 ns, 32 211 ns,
        // 64 185 ns, 96 162 ns, 128 152 ns, 192 147 ns, 256 144 ns. It was 32. Past 128 the curve is
        // flat and the per-call stack arrays (`sp`/`off`/`slot`, 13 bytes a slot) start crowding L1,
        // so 128 is where the knee is rather than where the minimum is.
        //
        // Width changes how many walks overlap, never which rows are visited or in what order a
        // given walk proceeds, so it cannot move a byte of output. `BWA4_SA_WINDOW` narrows it at
        // run time for sweeping on another machine.
        // Compile-time CEILING on the window, and the size of the per-call stack arrays. Raised
        // from 128 to 256 so the width can be swept ABOVE the default as well as below it: the knee
        // was found on an M4 (W=32 costs 18%, 64 and 128 tie), and a core with more outstanding-miss
        // slots could want more than 128, which no environment variable could previously ask for.
        // Only the ceiling moves; the default below is unchanged, and the unused tail of the arrays
        // is never touched, so nothing about a default run changes.
        const W_MAX: usize = 256;
        /// Default width, i.e. what a run uses unless `BWA4_SA_WINDOW` says otherwise. The measured
        /// knee, and deliberately not `W_MAX`: past 128 the M4 curve is flat and the per-call arrays
        /// (`sp`/`off`/`slot`, 13 bytes a slot) start crowding L1.
        const W_DEFAULT: usize = 128;
        // Slots actually kept in flight this call. `BWA4_SA_WINDOW` sets it anywhere in
        // `1..=W_MAX`, so the width can be swept on a machine without rebuilding. Width changes how
        // many misses overlap, never which rows are visited, so it cannot move a byte of output.
        let width = *SA_WINDOW.get_or_init(|| {
            std::env::var("BWA4_SA_WINDOW")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|w| (1..=W_MAX).contains(w))
                .unwrap_or(W_DEFAULT)
        });
        // SA samples waiting to be read: `(position in out, sampled index, steps walked)`. A walk that
        // lands on a sample has one more random line to fetch, and read on the spot that miss holds up
        // every other slot in the window. So it is only prefetched here and read a round later, where
        // the misses overlap like the walk's own. Which entry is read, and what is added, is unchanged.
        let mut pending = [(0u32, 0usize, 0i32); W_MAX];
        let mut n_pending = 0usize;
        // Index into `positions`/`out` of the first element of the current window.
        let mut base = 0usize;
        while base < positions.len() {
            // Elements in this window: W, or fewer for the final partial window.
            let w = (positions.len() - base).min(width);
            // Per-slot walk state, indexed by the LOCAL index `j` in `0..w` (global index is
            // `base + j`). `sp[j]` is slot `j`'s current BWT ROW, `off[j]` its LF step count, so
            // the same invariant as `get_sa` holds per slot: answer == sa[sp[j]] + off[j].
            let mut sp = [0i64; W_MAX];
            let mut off = [0i32; W_MAX];
            // Compaction list: `slot[0..nact]` holds the local indices still walking, in no
            // particular order. Slots that finish are dropped from it, so the inner loops stay
            // dense as the window drains.
            let mut slot = [0u8; W_MAX]; // still-walking local indices [0, w)
            let mut nact = 0usize;
            for j in 0..w {
                // The BWT row this slot must resolve.
                let pos = positions[base + j];
                if pos & SA_MASK == 0 {
                    let idx = (pos >> SA_SHIFT) as usize;
                    self.prefetch_sa_sample(idx);
                    pending[n_pending] = ((base + j) as u32, idx, 0);
                    n_pending += 1;
                } else {
                    sp[j] = pos;
                    off[j] = 0;
                    slot[nact] = j as u8;
                    nact += 1;
                }
            }
            // Round loop. Invariant at the top: every slot listed in `slot[0..nact]` is mid-walk
            // with `sp[j]` not a multiple of 8, and every slot NOT listed has already had its
            // answer written to `out`. Each round advances all listed slots by exactly one LF step,
            // so the window's cost is the cost of its longest walk, not the sum of all of them.
            while nact > 0 {
                // Pass 1: issue every active slot's block prefetch before touching any data, so
                // the ~32 misses overlap instead of serialising. This split is the entire point.
                for a in 0..nact {
                    self.prefetch_cp(sp[slot[a] as usize]);
                }
                // Last round's samples, whose lines have had a round to arrive.
                for &(o, idx, steps) in &pending[..n_pending] {
                    out[o as usize] = self.sa_sample(idx) + i64::from(steps);
                }
                n_pending = 0;
                // Write cursor for the compacted `slot` list being rebuilt in place. `wr <= a`
                // always holds, so it never overwrites an entry this pass has yet to read.
                let mut wr = 0usize;
                // Pass 2: one LF step for each active slot, now that the lines are arriving.
                for a in 0..nact {
                    // `j` is the slot's local index, `spi` its current BWT row.
                    let j = slot[a] as usize;
                    let spi = sp[j];
                    #[cfg(feature = "traffic")]
                    if traffic::enabled() {
                        traffic::SA_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    if spi == self.sentinel_index {
                        // Walk hit reference position 0, so the answer is just the step count.
                        // `continue` drops the slot: it is not re-added to the compacted list.
                        out[base + j] = i64::from(off[j]);
                        continue;
                    }
                    // The row `spi` LF-maps to, i.e. one reference position EARLIER.
                    let block = &self.cp_occ[(spi >> CP_SHIFT) as usize];
                    let b = Self::base_at(block, spi);
                    let nsp = self.count[b] + self.occ_in(block, spi, b);
                    off[j] += 1;
                    if nsp & SA_MASK == 0 {
                        let idx = (nsp >> SA_SHIFT) as usize;
                        self.prefetch_sa_sample(idx);
                        pending[n_pending] = ((base + j) as u32, idx, off[j]);
                        n_pending += 1;
                    } else {
                        // Still unsampled: keep walking, and re-add the slot to the compacted list.
                        sp[j] = nsp;
                        slot[wr] = j as u8;
                        wr += 1;
                    }
                }
                // Survivors become the next round's active set.
                nact = wr;
            }
            for &(o, idx, steps) in &pending[..n_pending] {
                out[o as usize] = self.sa_sample(idx) + i64::from(steps);
            }
            n_pending = 0;
            base += w;
        }
    }

    /// The binary reference base (0-3) at position `pos` in `[0, 2L)`, from the memory-mapped
    /// `.0123`. Positions at or above `l_pac` are on the reverse-complement half; use
    /// `BntSeq::depos` to interpret them. One byte per base, so no unpacking is needed, which is
    /// the reason `.0123` exists alongside the 4x smaller `.pac`.
    ///
    /// # Parameters
    /// * `pos`: a 2L-space reference POSITION (as returned by [`get_sa`]), never a BWT row. Valid
    ///   range `[0, 2 * l_pac)`; out of range panics on the slice index.
    ///
    /// # Returns
    /// The base code, 0-3 for A, C, G, T. Ambiguous bases were randomised to a concrete code at
    /// build time, so 4 never appears in `.0123`.
    #[inline]
    pub fn base(&self, pos: i64) -> u8 {
        unpack_pac_base(&self.pac, self.l_pac(), pos)
    }

    /// [`base`](Self::base) over a whole range, vectorised. See [`unpack_pac_range`].
    ///
    /// # Parameters
    /// - `rb`, `re`: half-open range in the doubled 2L coordinate space.
    ///
    /// # Returns
    /// `re - rb` bytes, byte `i` equal to `self.base(rb + i as i64)`.
    pub fn bases(&self, rb: i64, re: i64) -> Vec<u8> {
        unpack_pac_range(&self.pac, self.l_pac(), rb, re)
    }

    /// [`bases`](Self::bases) into a caller-owned buffer, so a hot loop allocates once instead of
    /// once per call.
    ///
    /// The aligner materialises one window per chain, which on a whole-genome run is millions of
    /// short vectors; returning an owned `Vec` each time made the allocator, not the unpack, a
    /// visible cost. `out` is CLEARED first, so the result is exactly what [`bases`](Self::bases)
    /// would have returned and never a leftover tail from a longer previous window.
    ///
    /// # Parameters
    /// - `rb`, `re`: half-open range in the doubled 2L coordinate space.
    /// - `out`: destination, cleared then filled with `re - rb` bytes. Its capacity is kept.
    pub fn bases_into(&self, rb: i64, re: i64, out: &mut Vec<u8>) {
        unpack_pac_range_into(&self.pac, self.l_pac(), rb, re, out);
    }

    /// Prefetch the `.pac` lines of the window `[rb, re)` that [`bases_into`] will unpack. `rb` and
    /// `re` are 2L-space positions on one strand; out-of-range values waste a fetch at worst, because
    /// a prefetch never faults. Only the first and last line are hinted, which covers all of a
    /// short read's first-seed window (150 bases are 38 bytes, so one or two lines) and just the ends
    /// of a long one.
    #[inline]
    pub fn prefetch_pac(&self, rb: i64, re: i64) {
        let l_pac = self.l_pac();
        // A reverse-strand window reads the forward bytes it mirrors, as `unpack_pac_range_into`.
        let (lo, hi) = if rb >= l_pac {
            ((l_pac << 1) - re, (l_pac << 1) - rb)
        } else {
            (rb, re)
        };
        let base = self.pac.as_ptr();
        let line = |p: i64| base.wrapping_add((p.max(0) >> 2) as usize);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: a prefetch is a hint that never faults or writes.
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) line(lo), options(nostack, readonly, preserves_flags));
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) line(hi), options(nostack, readonly, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: a prefetch is a hint that never faults or writes.
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(line(lo) as *const i8, _MM_HINT_T0);
            _mm_prefetch(line(hi) as *const i8, _MM_HINT_T0);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            // No portable prefetch intrinsic: discard the values to silence the unused warning.
            let _ = (line, lo, hi);
        }
    }

    /// The loaded cumulative base counts (already `+1`, as bwa-mem2's `load_index`).
    ///
    /// # Returns
    /// A copy of `count[0..5]`: entry `c` is the number of BWT rows whose suffix starts with a
    /// base smaller than `c`, and entry 4 is the total row count. For callers that need to do
    /// their own LF-mapping outside this module.
    #[inline]
    pub fn counts(&self) -> [i64; 5] {
        self.count
    }

    /// The whole reference as one byte per base (forward ++ reverse-complement, 2L bytes),
    /// materialized from `.pac`.
    ///
    /// # Returns
    /// An owned `2 * l_pac`-byte vector, position `p` holding [`base`]`(p)`. This ALLOCATES the full
    /// 2L bytes (~6.2 GB on hg38), which is exactly the footprint the switch to `.pac` exists to
    /// avoid, so it is for tests and offline tooling only. The hot alignment path reads single bases
    /// through [`base`] and never calls this.
    pub fn reference(&self) -> Vec<u8> {
        (0..2 * self.l_pac()).map(|p| self.base(p)).collect()
    }

    /// Length of the forward reference `L` (`ref_seq_len` is `2L + 1`).
    ///
    /// Recomputed rather than read from `.ann`, so the FM index can be used without a `BntSeq`.
    /// The division is exact because `ref_seq_len - 1 == 2L` by construction.
    ///
    /// # Returns
    /// `L`, the forward-strand length in bases. This is the strand boundary: a 2L-space position
    /// below it is a forward hit, at or above it a reverse-complement hit.
    #[inline]
    pub fn l_pac(&self) -> i64 {
        (self.ref_seq_len - 1) / 2
    }
}

#[cfg(test)]
mod tests {

    /// The vectorised range unpack must equal the per-base one, byte for byte, in both halves of
    /// the doubled coordinate space and at every alignment of the range against the 4-base packing.
    #[test]
    fn unpack_range_matches_per_base() {
        // A synthetic `.pac`: 4 bases per byte, so 64 forward bases, plus a trailer byte the real
        // format carries and this code must never index.
        let l_pac: i64 = 64;
        let pac: Vec<u8> = (0..17u8)
            .map(|i| i.wrapping_mul(37).wrapping_add(11))
            .collect();
        for rb in 0..(2 * l_pac) {
            for len in [0i64, 1, 3, 4, 5, 15, 16, 17, 33, 64] {
                let re = (rb + len).min(2 * l_pac);
                let want: Vec<u8> = (rb..re)
                    .map(|p| super::unpack_pac_base(&pac, l_pac, p))
                    .collect();
                let got = super::unpack_pac_range(&pac, l_pac, rb, re);
                assert_eq!(got, want, "range [{rb}, {re}) of 2L = {}", 2 * l_pac);
            }
        }
    }
    use super::*;
    use crate::sais::suffix_array_with_sentinel;

    /// Load the checked-in miniature index from `testdata/tiny`. Small enough that the tests below
    /// can afford to compare against an O(n^2) naive computation over the whole reference.
    fn tiny() -> FmIndex {
        // Absolute path built at compile time, so the test does not depend on the working directory.
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        FmIndex::load(Path::new(prefix)).unwrap()
    }

    /// The 2-bit `.pac` (forward only) must reproduce the one-byte-per-base `.0123` (forward ++
    /// reverse-complement) for EVERY 2L-space position, or dropping the `.0123` mmap to save its
    /// 6.2 GB would corrupt every reference base the aligner reads. `.0123` is the ground truth
    /// here because it is the file the code trusted before this change; the test reads it raw so it
    /// does not depend on the very method it is validating.
    #[test]
    fn unpack_pac_reproduces_0123() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let pac = std::fs::read(format!("{prefix}.pac")).unwrap();
        let ref0123 = std::fs::read(format!("{prefix}.0123")).unwrap();
        let two_l = ref0123.len() as i64;
        let l_pac = two_l / 2;
        assert_eq!(two_l % 2, 0, "the .0123 reference must be 2L bytes");
        for pos in 0..two_l {
            assert_eq!(
                unpack_pac_base(&pac, l_pac, pos),
                ref0123[pos as usize],
                "pac unpack disagrees with .0123 at 2L-space position {pos} (l_pac={l_pac})"
            );
        }
    }

    /// Every row's `get_sa` matches an independently computed suffix array, and the map from rows
    /// to positions is a bijection. The permutation half catches errors the value comparison could
    /// mask, e.g. an off-by-one that maps two rows to one position.
    #[test]
    fn get_sa_matches_sais_and_is_permutation() {
        let fm = tiny();
        // The 2L reference, materialized once from `.pac` for this test.
        let reference = fm.reference();
        // `two_l` is the reference LENGTH in bases (2L); `n` the BWT ROW count (2L + 1).
        let two_l = reference.len();
        let n = fm.ref_seq_len;
        assert_eq!(n, two_l as i64 + 1);
        // Ground truth from the SA-IS builder: `sa[row]` is the reference position for that row.
        let sa = suffix_array_with_sentinel(&reference);
        // Which reference positions have been produced already, for the bijection check.
        let mut seen = vec![false; two_l + 1];
        for i in 0..n {
            // The 2L-space reference position this module reports for row `i`.
            let v = fm.get_sa(i);
            assert_eq!(v, sa[i as usize], "get_sa mismatch at row {i}");
            assert!(!seen[v as usize], "get_sa not a permutation at {i}");
            seen[v as usize] = true;
        }
    }

    /// The file's own record for one block: full counts, then the one-hot bitmaps.
    fn file_block(bwt: &[u8], block: usize) -> ([i64; 4], [u64; 4]) {
        let rec = &bwt[48 + block * 64..48 + (block + 1) * 64];
        let word = |n: usize| u64::from_le_bytes(rec[8 * n..8 * n + 8].try_into().unwrap());
        (
            [
                word(0) as i64,
                word(1) as i64,
                word(2) as i64,
                word(3) as i64,
            ],
            [word(4), word(5), word(6), word(7)],
        )
    }

    /// Every count the two-plane records can answer must equal the file's one-hot answer, at every
    /// row including the sentinel's block and the row past the end. The file form is the ground truth
    /// here because it is what bwa-mem2 reads; this is the test that says nothing was lost in memory.
    #[test]
    fn occ_matches_the_files_one_hot_form() {
        let fm = tiny();
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let bwt = std::fs::read(format!("{prefix}.bwt.2bit.64")).unwrap();
        for row in 0..=fm.ref_seq_len {
            let (counts, one_hot) = file_block(&bwt, (row >> CP_SHIFT) as usize);
            let m = rows_before(row & CP_MASK);
            let want: [i64; 4] =
                std::array::from_fn(|c| counts[c] + (one_hot[c] & m).count_ones() as i64);
            let block = &fm.cp_occ[(row >> CP_SHIFT) as usize];
            assert_eq!(fm.occ4(block, row), want, "occ4 at row {row}");
            assert_eq!(
                fm.occ4_narrow(block, row),
                want.map(|c| c as u32),
                "occ4_narrow at row {row}"
            );
            for (c, &w) in want.iter().enumerate() {
                assert_eq!(fm.get_occ(row, c), w, "get_occ of {c} at row {row}");
            }
        }
    }

    /// The packed high bits must give back the file's byte at every sampled entry.
    #[test]
    fn sa_samples_match_the_file() {
        let fm = tiny();
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let bwt = std::fs::read(format!("{prefix}.bwt.2bit.64")).unwrap();
        let ms = 48 + fm.cp_occ.len() * 64;
        let ls = ms + fm.sa_ls_word.len();
        for i in 0..fm.sa_ls_word.len() {
            let low = u32::from_le_bytes(bwt[ls + 4 * i..ls + 4 * i + 4].try_into().unwrap());
            let want = (i64::from(bwt[ms + i]) << 32) + i64::from(low);
            assert_eq!(fm.sa_sample(i), want, "sample {i}");
        }
    }

    /// The bits needed above the low 32 of a position, at the genome sizes that change the answer.
    #[test]
    fn sa_hi_width_follows_the_row_count() {
        // Any position fits the low 32: the width floors at one bit, not zero.
        assert_eq!(sa_hi_log2_bits(1000), 0);
        // GRCh38, doubled: 33 bits.
        assert_eq!(sa_hi_log2_bits(6_199_501_437), 0);
        assert_eq!(sa_hi_log2_bits((1 << 33) + 1), 1);
        assert_eq!(sa_hi_log2_bits((1 << 35) + 1), 2);
        assert_eq!(sa_hi_log2_bits((1 << 37) + 1), 3);
    }

    /// A genome holding 2^32 of one base cannot be built in a test, so this one shrinks the span
    /// instead and pushes counts past 2^32 by hand: the span tables must recover every full count
    /// through both counting paths.
    #[test]
    fn wide_counts_are_recovered_through_the_span_table() {
        // Eight 64-row blocks per span, and a base count that crosses 2^32 between spans.
        let span_shift = 9;
        let n_blocks = 40usize;
        let full = |block: usize| -> [i64; 4] {
            let b = block as i64;
            [(1 << 32) - 100 + 7 * b, 3 * b, (5 << 32) + 11 * b, b]
        };
        let mut cp_occ: Vec<CpOcc> = (0..n_blocks)
            .map(|b| CpOcc {
                cp_count: full(b).map(|c| c as u32),
                hi: 0x00FF_00FF_00FF_00FF,
                lo: 0x0F0F_0F0F_0F0F_0F0F,
            })
            .collect();
        let count_base: Vec<[i64; 4]> = (0..=n_blocks >> (span_shift - CP_SHIFT))
            .map(|span| full(span << (span_shift - CP_SHIFT)))
            .collect();
        rebase_counts(&mut cp_occ, &count_base, span_shift);
        let fm = FmIndex {
            ref_seq_len: (n_blocks as i64) << CP_SHIFT,
            count: [0; 5],
            cp_occ,
            count_base,
            sa_hi: vec![0],
            sa_hi_log2: 0,
            sa_hi_mask: 1,
            sa_ls_word: vec![0],
            // Past every queried row, so no block here reads as the sentinel's.
            sentinel_index: i64::MAX,
            sentinel_block: i64::MAX,
            wide: true,
            count_span_shift: span_shift,
            pac: memmap2::MmapMut::map_anon(1)
                .unwrap()
                .make_read_only()
                .unwrap(),
        };
        for block in 0..n_blocks {
            for y in [0i64, 1, 17, 63] {
                let row = ((block as i64) << CP_SHIFT) + y;
                let rec = &fm.cp_occ[block];
                let m = rows_before(y);
                let t = (rec.hi & rec.lo & m).count_ones() as i64;
                let g = (rec.hi & m).count_ones() as i64 - t;
                let c = (rec.lo & m).count_ones() as i64 - t;
                let a = y - t - g - c;
                let f = full(block);
                let want = [f[0] + a, f[1] + c, f[2] + g, f[3] + t];
                assert_eq!(fm.occ4(rec, row), want, "occ4 at row {row}");
                for (b, &w) in want.iter().enumerate() {
                    assert_eq!(fm.get_occ(row, b), w, "get_occ of {b} at row {row}");
                }
            }
        }
    }

    /// A full backward search reports the same occurrence count as brute-force window matching.
    /// Patterns are taken from the reference itself, so each is guaranteed to occur at least once
    /// and a spurious `s == 0` cannot pass.
    #[test]
    fn backward_search_counts_match_naive() {
        let fm = tiny();
        // The 2L reference as bytes; both the pattern source and the brute-force haystack.
        let bref = fm.reference();
        // Arbitrary (start position, length) probes spread across the reference and across
        // lengths, chosen only to exercise a few different intervals.
        for &(start, len) in &[(100usize, 20usize), (5000, 15), (123, 31), (77_777, 25)] {
            // The query, as base codes; `sm` is the interval, narrowed one base at a time.
            let pat = &bref[start..start + len];
            let mut sm = fm.full_interval();
            // Backward search: bases fed right to left, so after each step `sm` is the interval of
            // the pattern SUFFIX consumed so far.
            for &c in pat.iter().rev() {
                sm = fm.backward_ext(sm, c as usize);
            }
            // Ground-truth occurrence count by scanning every window of the reference.
            let naive = bref.windows(len).filter(|w| *w == pat).count() as i64;
            assert_eq!(sm.s, naive, "occurrence mismatch for pattern at {start}");
        }
    }
}

/// `BWA4_TRAFFIC=1`: count the 128-byte cache lines the FM index actually pulls, so the aligner's
/// DRAM bandwidth can be compared against the fabric ceiling (~293 GB/s random on M4 Max). Atomics
/// on the hot path: use at `-t1` and read the counts, not the wall clock.
pub mod traffic {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    /// Distinct 128-byte lines touched by `backward_ext`, process-wide since start. 1 or 2 per
    /// call: 2 only when the interval's start and end rows fall in different lines. Never reset.
    pub static EXT_LINES: AtomicU64 = AtomicU64::new(0);
    /// Number of `backward_ext` calls, i.e. total search steps. Only used as the denominator of
    /// the lines-per-call ratio printed by [`dump`].
    pub static EXT_CALLS: AtomicU64 = AtomicU64::new(0);
    /// Lines touched by `get_sa` / `get_sa_batch` LF-walks: exactly one per step, since a step's
    /// bitmap read and its `get_occ` hit the same block. Counts steps, not calls.
    pub static SA_LINES: AtomicU64 = AtomicU64::new(0);

    /// Whether `BWA4_TRAFFIC` was set in the environment. Read once and cached in a `OnceLock`, so
    /// the hot-path check is a relaxed atomic load rather than a `getenv`, and toggling the
    /// variable mid-run has no effect.
    ///
    /// # Returns
    /// `true` if the variable is present with ANY value, including empty: presence is the switch.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA4_TRAFFIC").is_some())
    }

    /// Print the accumulated line counts and the implied bandwidth to stderr. Call once at the end
    /// of a run; silently does nothing unless [`enabled`].
    ///
    /// # Parameters
    /// * `wall_s`: elapsed wall-clock time in SECONDS over which the counters accumulated, measured
    ///   by the caller. Used only as the divisor for the GB/s figure, and clamped away from zero.
    ///   The derived rate is per-process, so it is only meaningful for a single-threaded run.
    pub fn dump(wall_s: f64) {
        if !enabled() {
            return;
        }
        // Snapshot the three counters: ext lines, ext calls, sa-walk lines.
        let (el, ec, sl) = (
            EXT_LINES.load(Ordering::Relaxed),
            EXT_CALLS.load(Ordering::Relaxed),
            SA_LINES.load(Ordering::Relaxed),
        );
        // All FM-index lines pulled. Multiplied by 128 B below to get bytes moved.
        let total = el + sl;
        eprintln!(
            "[traffic] backward_ext: {ec} calls -> {el} lines ({:.2} lines/call)\n\
             [traffic] get_sa walk : {sl} lines\n\
             [traffic] TOTAL {} lines x 128 B = {:.1} GB in {:.2}s = {:.1} GB/s (1 thread)",
            el as f64 / ec.max(1) as f64,
            total,
            total as f64 * 128.0 / 1e9,
            wall_s,
            total as f64 * 128.0 / 1e9 / wall_s.max(1e-9),
        );
    }
}
