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
//! | `cp_occ` | `cp_occ` | The checkpoint array: one 64-byte record per 64 BWT rows, holding the cumulative counts up to that block plus a one-hot bitmap of the block's own 64 rows. |
//! | `sa` | `suffix_array` | The suffix array: `sa[row]` is the reference position the suffix at `row` starts at. Only every 8th entry is stored; the rest are recovered by walking. |
//! | `pac` / `l_pac` | `bns->l_pac` | Length of the FORWARD reference in bases. The searched text is `2 * l_pac` long (forward genome then its reverse complement), so a position `>= l_pac` is a reverse-strand hit. |
//! | `sp` / `ep` | `sp` / `ep` | Start and (exclusive) end row of an interval, i.e. `k` and `k + s`. |
//! | `pp` | `pp` | The row an occurrence query is made at. Valid range is `[0, ref_seq_len]` INCLUSIVE, because interval ENDS are passed here. |
//!
//! # Reading order for this file
//!
//! 1. [`Smem`] and [`CpOcc`]: what an interval is and what one checkpoint block holds.
//! 2. [`FmIndex::load`]: the on-disk layout, read field by field.
//! 3. [`FmIndex::get_occ`]: the single primitive everything else is built from.
//! 4. [`FmIndex::backward_ext`]: one search step, including the fiddly `l` recurrence.
//! 5. [`FmIndex::get_sa`]: turning a row back into a reference position.
//! 6. [`FmIndex::get_sa_batch`] and [`FmIndex::prefetch_occ`]: latency hiding only, no new logic.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use bwa_core::Result;
use memmap2::Mmap;

/// Rows per occurrence-checkpoint block, the C's `CP_BLOCK_SIZE` (`FMI_search.h:49`). Also the
/// record's size in BYTES, which is the point: one block is one cache line, so an occurrence query
/// is a single memory touch. It is baked into the index FILENAME (`.bwt.2bit.64`), so an index
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

/// The BWT symbol for the sentinel row (and, in the builder, for tail padding): a code outside
/// `0..4`, so it sets no bit in any of the four one-hot bitmaps. [`FmIndex::get_sa`] detects
/// exactly that "no bit set anywhere" condition to know the LF-walk has reached position 0.
const SENTINEL_CODE: usize = 4;

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

/// One 64-base checkpoint block, interleaved exactly as bwa-mem2's `CP_OCC`: the cumulative counts
/// and the one-hot BWT bitvectors for the 4 bases sit together in a single 64-byte, cache-line-aligned
/// record, so a single occ lookup touches one cache line instead of two.
///
/// `repr(C)` is mandatory, not stylistic: this struct is `copy_nonoverlapping`'d straight out of
/// the mapped file, so its field order and lack of padding must match the C's `CP_OCC`
/// (`FMI_search.h:55`) exactly. The 8 members are all 8 bytes, so `align(64)` adds no interior
/// padding and `size_of::<CpOcc>() == 64`, which the load path's offset arithmetic assumes.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct CpOcc {
    /// Occurrences of A, C, G, T in `bwt[0 .. block_start)`, EXCLUSIVE of this block's own rows.
    cp_count: [i64; 4],
    /// Bit `63 - j` of `one_hot[c]` is 1 iff `bwt[block_start + j] == c`. MSB-first, see
    /// [`crate::build::write_fm_index`]. Rows holding the sentinel or past-the-end padding set no
    /// bit in any of the four words, which is the "all zero" case `get_sa` uses as its stop signal.
    one_hot: [u64; 4],
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
    /// High byte (bits 32..40) of every 8th suffix-array entry. `i8`, matching the C's `int8_t`;
    /// the sign extension is harmless only because bit 39 is never set at genome scale.
    sa_ms_byte: Vec<i8>,
    /// Low 32 bits of every 8th suffix-array entry. Paired index-for-index with `sa_ms_byte`.
    sa_ls_word: Vec<u32>,
    /// The unique BWT row with `sa[row] == 0`. Needed by `backward_ext` to keep `l` correct.
    sentinel_index: i64,
    /// `one_hot_mask[y]` has the top `y` bits set (`y == 0` gives 0, `y == 63` gives all but the
    /// LSB). ANDed with a `one_hot` word it keeps exactly the rows BEFORE offset `y` in the block,
    /// which is what makes `occ` a half-open, exclusive count. Rebuilt at load rather than read
    /// from the file, exactly as `load_index` does (`FMI_search.cpp:386-393`).
    one_hot_mask: [u64; 64],
    /// The `.0123` reference (forward++reverse-complement, one byte/base, 2L bytes), memory-mapped:
    /// the 6.2 GB file is not copied into RAM at load, and its pages are shared via the OS page cache.
    reference: Mmap,
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
        // Memory-map the BWT and bulk-copy each array out in one memcpy: the `cp_occ` blocks need
        // 64-byte alignment the file doesn't provide, so they can't be borrowed in place, but the
        // copy is a single `memcpy` (from page-cached pages) instead of ~800M element-wise reads.
        // ---- Header: ref_seq_len, then count[5] ----------------------------------------------
        let bwt_file = std::fs::File::open(sibling(prefix, "bwt.2bit.64"))?;
        // SAFETY: index files are not mutated while a run holds them open (same as bwa-mem2's mmap).
        let bwt_map = unsafe { Mmap::map(&bwt_file)? };
        // Whole-file view, and the running BYTE offset that walks the format front to back.
        let bytes: &[u8] = &bwt_map;
        let mut cursor = 0usize;
        // First header field: the BWT ROW count, 2L + 1 (2L reference bases plus the sentinel row).
        // Every later size in the file is derived from it, so it is read before anything else.
        let ref_seq_len = rd_i64(bytes, &mut cursor);
        // `count[c]`: rows whose suffix starts with a base < c. Index 4 is the total (all rows).
        // Read as written by the builder, i.e. WITHOUT the sentinel adjustment applied below.
        let mut count = [0i64; 5];
        for c in &mut count {
            *c = rd_i64(bytes, &mut cursor);
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

        // SAFETY of the three copies below: each source range is in-bounds in the mapped file (its
        // length is exactly `header + cp_size*64 + sa_size*(1+4) + 8`), the destination `Vec` is
        // freshly allocated with matching capacity, `CpOcc`/`i8`/`u32` are all plain-old-data with no
        // padding-sensitive invariants, and `set_len` is reached only after the bytes are written.
        // ---- Payload: checkpoints, then the two sampled-SA arrays, then the sentinel row ------
        // Destination for the checkpoints, allocated 64-byte aligned by `CpOcc`'s `align(64)`.
        // `cp_bytes` is the BYTE length of that array in the file: cp_size * 64.
        let mut cp_occ = Vec::<CpOcc>::with_capacity(cp_size);
        let cp_bytes = cp_size * std::mem::size_of::<CpOcc>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes[cursor..].as_ptr(),
                cp_occ.as_mut_ptr() as *mut u8,
                cp_bytes,
            );
            cp_occ.set_len(cp_size);
        }
        cursor += cp_bytes;

        // High byte of each sampled SA entry, one byte per element, so element count == byte count.
        let mut sa_ms_byte = Vec::<i8>::with_capacity(sa_size);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes[cursor..].as_ptr(),
                sa_ms_byte.as_mut_ptr() as *mut u8,
                sa_size,
            );
            sa_ms_byte.set_len(sa_size);
        }
        cursor += sa_size;

        // Low 32 bits of each sampled SA entry; `ls_bytes` is its BYTE length (4 per element).
        let mut sa_ls_word = Vec::<u32>::with_capacity(sa_size);
        let ls_bytes = sa_size * 4;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes[cursor..].as_ptr(),
                sa_ls_word.as_mut_ptr() as *mut u8,
                ls_bytes,
            );
            sa_ls_word.set_len(sa_size);
        }
        cursor += ls_bytes;

        // Trailing header field: the one BWT ROW (not a reference position) whose suffix is the
        // whole text, i.e. the row with `sa[row] == 0`. Only `backward_ext` reads it.
        let sentinel_index = rd_i64(bytes, &mut cursor);

        // `one_hot_mask[y]` = the top `y` bits set. Built by the C's exact recurrence
        // (`FMI_search.cpp:386-393`): start from the MSB alone, then each step shifts the previous
        // mask right one and re-sets the MSB, growing the run of leading ones by one.
        //   [0] = 0x0000..., [1] = 0x8000..., [2] = 0xC000..., [3] = 0xE000..., [63] = 0xFFFE...
        // Because row `j` of a block lives at bit `63 - j`, ANDing with `one_hot_mask[y]` keeps
        // rows 0..y, i.e. everything STRICTLY BEFORE row `y`. That exclusivity is what makes
        // `get_occ(pp, c)` count `bwt[0..pp)` rather than `bwt[0..=pp]`.
        // ---- Derived: the "top y bits set" mask table -----------------------------------------
        // Indexed by an in-block row OFFSET `y` in 0..64; entry `y` has the top `y` bits set.
        // `[0]` stays zero (no rows before offset 0), which is why the loop starts at index 1.
        let mut one_hot_mask = [0u64; 64];
        // Bit 63 alone: row 0 of a block. Both the seed of the recurrence and the bit re-set on
        // every step, so each entry keeps a run of leading ones one longer than the previous.
        let msb_only = 0x8000_0000_0000_0000u64;
        one_hot_mask[1] = msb_only;
        for i in 2..64 {
            one_hot_mask[i] = (one_hot_mask[i - 1] >> 1) | msb_only;
        }

        // ---- The binary reference, memory-mapped ----------------------------------------------
        // Memory-map `.0123` instead of reading it: no 6.2 GB copy, pages shared via the page cache.
        let ref_file = std::fs::File::open(sibling(prefix, "0123"))?;
        // SAFETY: the index files are not mutated while a run holds them open; a concurrent external
        // truncation is out of scope (same assumption as bwa-mem2's mmap'd index).
        let reference = unsafe { Mmap::map(&ref_file)? };

        Ok(FmIndex {
            ref_seq_len,
            count,
            cp_occ,
            sa_ms_byte,
            sa_ls_word,
            sentinel_index,
            one_hot_mask,
            reference,
        })
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
            // Element-typed base pointer, so `.add(n)` steps whole 64-byte checkpoint records.
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
                _mm_prefetch(base.add((sp >> CP_SHIFT) as usize) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base.add((ep >> CP_SHIFT) as usize) as *const i8, _MM_HINT_T0);
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
    /// 0..128. `one_hot_mask[2] = 0xC000...` keeps bits 63 and 62, that is rows 128 and 129. Sum is
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
        let y = (pp & CP_MASK) as usize;
        block.cp_count[c] + (block.one_hot[c] & self.one_hot_mask[y]).count_ones() as i64
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
    pub fn backward_ext(&self, smem: Smem, a: usize) -> Smem {
        // Load the sp/ep checkpoint blocks once (all 4 bases share them), rather than re-deriving
        // the block index and re-indexing per base as `get_occ` would. Values are identical.
        // The parent interval as a half-open row range `[sp, ep)`: `sp` is the first matching BWT
        // ROW, `ep` the first row past it. Both are rows, not reference positions.
        let sp = smem.k;
        let ep = smem.k + smem.s;
        if traffic::enabled() {
            // Two 64B blocks share one 128B line, and for a small interval sp>>6 == ep>>6, so calls
            // are a bad proxy for traffic. Count distinct LINES.
            // 128-byte LINE indices for the two blocks: block index >> 1, since two 64-byte
            // checkpoints share one line.
            let (l1, l2) = ((sp >> CP_SHIFT) >> 1, (ep >> CP_SHIFT) >> 1);
            traffic::EXT_LINES.fetch_add(if l1 == l2 { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
            traffic::EXT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // The two checkpoint blocks (often the same one for a narrow interval) and the two
        // "keep rows before my offset" masks. Hoisted out of the per-base loop so all 8 occurrence
        // queries below reuse these 2 cache lines. Inlining `get_occ` here is not an optimisation
        // of the arithmetic: it is identical, only the loads are shared.
        let blk_sp = &self.cp_occ[(sp >> CP_SHIFT) as usize];
        let blk_ep = &self.cp_occ[(ep >> CP_SHIFT) as usize];
        let msk_sp = self.one_hot_mask[(sp & CP_MASK) as usize];
        let msk_ep = self.one_hot_mask[(ep & CP_MASK) as usize];
        // Per-base results, indexed by base code 0..4. `k[b]` is the first BWT ROW of the child
        // interval for `b ++ pattern`; `s[b]` is its size. All four are computed because the `l`
        // recurrence below needs every sibling's size, not just the requested base's.
        let mut k = [0i64; 4];
        let mut s = [0i64; 4];
        for b in 0..4 {
            // Occurrences of `b` in `bwt[0..sp)` and `bwt[0..ep)`: `get_occ` open-coded against the
            // preloaded blocks. Their difference is how many of the parent's rows carry `b`.
            let occ_sp = blk_sp.cp_count[b] + (blk_sp.one_hot[b] & msk_sp).count_ones() as i64;
            let occ_ep = blk_ep.cp_count[b] + (blk_ep.one_hot[b] & msk_ep).count_ones() as i64;
            k[b] = self.count[b] + occ_sp;
            s[b] = occ_ep - occ_sp;
        }
        // 1 exactly when the sentinel ROW falls inside the parent interval `[k, k+s)`, else 0. That
        // row carries no base, so none of the four `s[b]` counted it, yet it does occupy a slot on
        // the reverse-complement side; adding 1 shifts every `l` up to compensate.
        let sentinel_offset =
            i64::from(smem.k <= self.sentinel_index && smem.k + smem.s > self.sentinel_index);
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
    /// The `b == 4` early return is the sentinel row: it has no preceding character, meaning the
    /// walk reached reference position 0, so the answer is exactly the number of steps taken. That
    /// branch is the reason the builder must leave the sentinel row with no bit set in any of the
    /// four bitmaps.
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
            let idx = (pos >> SA_SHIFT) as usize;
            return (i64::from(self.sa_ms_byte[idx]) << 32) + i64::from(self.sa_ls_word[idx]);
        }
        // Walk state. `offset` counts LF steps taken so far, which equals how many reference
        // positions EARLIER than `sa[pos]` the current row sits; `sp` is the current BWT ROW.
        // Loop invariant at the top of each iteration: `sa[pos] == sa[sp] + offset`, and `sp` is
        // not a multiple of 8 (the exit test at the bottom guarantees it).
        let mut offset = 0i64;
        let mut sp = pos;
        loop {
            if traffic::enabled() {
                // One line per step (one_hot and get_occ hit the same 64B block).
                traffic::SA_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // Index of the checkpoint block holding row `sp`.
            let occ_id = (sp >> CP_SHIFT) as usize;
            // Row `sp` sits at BIT `63 - (sp & 63)` because the builder shifted the block in
            // MSB-first. The C writes the same thing as `CP_BLOCK_SIZE - (sp & CP_MASK) - 1`
            // (`FMI_search.cpp:1130`). Note this is a bit POSITION, whereas `get_occ` uses the raw
            // `sp & 63` as a mask INDEX; the two are different quantities over the same block.
            let y = CP_BLOCK_SIZE - (sp & CP_MASK) - 1;
            // Copy of the block's four one-hot BWT bitmaps (32 bytes, one cache line's worth): this
            // is the random DRAM access that dominates the crate's runtime.
            let oh = self.cp_occ[occ_id].one_hot;
            // Which base is `bwt[sp]`? Test the four bitmaps; exactly one can be set. No bit set at
            // all means the sentinel (or padding), encoded here as 4.
            let b = if (oh[0] >> y) & 1 == 1 {
                0
            } else if (oh[1] >> y) & 1 == 1 {
                1
            } else if (oh[2] >> y) & 1 == 1 {
                2
            } else if (oh[3] >> y) & 1 == 1 {
                3
            } else {
                SENTINEL_CODE
            };
            if b == SENTINEL_CODE {
                return offset;
            }
            // LF-mapping: the row whose suffix is one character LONGER, i.e. one reference position
            // EARLIER. `count[b] + occ(b, sp)` is the standard formula, and it is the same
            // arithmetic `backward_ext` uses for `k`.
            // Occurrences of `b` before row `sp`; `count[b] + occ_sp` is the row `sp` maps to.
            let occ_sp = self.get_occ(sp, b);
            sp = self.count[b] + occ_sp;
            offset += 1;
            if sp & SA_MASK == 0 {
                break;
            }
        }
        // Here `sp` is a multiple of 8, so its SA value was stored outright. `sa_entry` is a
        // 2L-space reference POSITION; adding back the `offset` steps walked undoes the walk.
        let idx = (sp >> SA_SHIFT) as usize;
        let sa_entry = (i64::from(self.sa_ms_byte[idx]) << 32) + i64::from(self.sa_ls_word[idx]);
        sa_entry + offset
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
            _mm_prefetch(self.cp_occ.as_ptr().add((pos >> CP_SHIFT) as usize) as *const i8, _MM_HINT_T0);
        }
        // No portable prefetch intrinsic: discard the row to silence the unused warning.
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = pos;
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
        // Window width: how many independent LF-walks are kept in flight. Chosen to be enough
        // outstanding misses to saturate the memory system while `sp`/`off`/`slot` stay small
        // enough to live in registers/L1. Changing it alters speed only, never results.
        const W: usize = 32;
        // Read one SAMPLED suffix-array entry: `p` is a BWT row that is a multiple of 8, and the
        // result is a 2L-space reference position reassembled from its split 8+32-bit storage.
        let sa = |p: i64| -> i64 {
            let idx = (p >> SA_SHIFT) as usize;
            (i64::from(self.sa_ms_byte[idx]) << 32) + i64::from(self.sa_ls_word[idx])
        };
        // Index into `positions`/`out` of the first element of the current window.
        let mut base = 0usize;
        while base < positions.len() {
            // Elements in this window: W, or fewer for the final partial window.
            let w = (positions.len() - base).min(W);
            // Per-slot walk state, indexed by the LOCAL index `j` in `0..w` (global index is
            // `base + j`). `sp[j]` is slot `j`'s current BWT ROW, `off[j]` its LF step count, so
            // the same invariant as `get_sa` holds per slot: answer == sa[sp[j]] + off[j].
            let mut sp = [0i64; W];
            let mut off = [0i32; W];
            // Compaction list: `slot[0..nact]` holds the local indices still walking, in no
            // particular order. Slots that finish are dropped from it, so the inner loops stay
            // dense as the window drains.
            let mut slot = [0u8; W]; // still-walking local indices [0, w)
            let mut nact = 0usize;
            for j in 0..w {
                // The BWT row this slot must resolve.
                let pos = positions[base + j];
                if pos & SA_MASK == 0 {
                    out[base + j] = sa(pos);
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
                // Write cursor for the compacted `slot` list being rebuilt in place. `wr <= a`
                // always holds, so it never overwrites an entry this pass has yet to read.
                let mut wr = 0usize;
                // Pass 2: one LF step for each active slot, now that the lines are arriving.
                for a in 0..nact {
                    // `j` is the slot's local index, `spi` its current BWT row, `occ_id` the
                    // checkpoint block holding that row, `y` the row's BIT position within the
                    // block's MSB-first bitmaps (same quantity as in `get_sa`).
                    let j = slot[a] as usize;
                    let spi = sp[j];
                    let occ_id = (spi >> CP_SHIFT) as usize;
                    let y = CP_BLOCK_SIZE - (spi & CP_MASK) - 1;
                    if traffic::enabled() {
                        traffic::SA_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    // The four one-hot bitmaps for this block, then `bwt[spi]` decoded from them
                    // exactly as in `get_sa`: no bit set anywhere means the sentinel row.
                    let oh = self.cp_occ[occ_id].one_hot;
                    let b = if (oh[0] >> y) & 1 == 1 {
                        0
                    } else if (oh[1] >> y) & 1 == 1 {
                        1
                    } else if (oh[2] >> y) & 1 == 1 {
                        2
                    } else if (oh[3] >> y) & 1 == 1 {
                        3
                    } else {
                        SENTINEL_CODE
                    };
                    if b == SENTINEL_CODE {
                        // Walk hit reference position 0, so the answer is just the step count.
                        // `continue` drops the slot: it is not re-added to the compacted list.
                        out[base + j] = i64::from(off[j]);
                        continue;
                    }
                    // The row `spi` LF-maps to, i.e. one reference position EARLIER.
                    let nsp = self.count[b] + self.get_occ(spi, b);
                    off[j] += 1;
                    if nsp & SA_MASK == 0 {
                        out[base + j] = sa(nsp) + i64::from(off[j]);
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
        self.reference[pos as usize]
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

    /// The `.0123` binary reference (forward ++ reverse-complement, 2L bytes).
    ///
    /// # Returns
    /// The whole mapped file as one byte-per-base slice, indexed by 2L-space POSITION. Borrowed
    /// from the mmap, so touching a page may fault it in; the length is `2 * l_pac`, one less than
    /// `ref_seq_len`. Used by extension code that needs a run of bases rather than single lookups.
    #[inline]
    pub fn reference(&self) -> &[u8] {
        &self.reference[..]
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
    use super::*;
    use crate::sais::suffix_array_with_sentinel;

    /// Load the checked-in miniature index from `testdata/tiny`. Small enough that the tests below
    /// can afford to compare against an O(n^2) naive computation over the whole reference.
    fn tiny() -> FmIndex {
        // Absolute path built at compile time, so the test does not depend on the working directory.
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        FmIndex::load(Path::new(prefix)).unwrap()
    }

    /// Every row's `get_sa` matches an independently computed suffix array, and the map from rows
    /// to positions is a bijection. The permutation half catches errors the value comparison could
    /// mask, e.g. an off-by-one that maps two rows to one position.
    #[test]
    fn get_sa_matches_sais_and_is_permutation() {
        let fm = tiny();
        // `two_l` is the reference LENGTH in bases (2L); `n` the BWT ROW count (2L + 1).
        let two_l = fm.reference.len();
        let n = fm.ref_seq_len;
        assert_eq!(n, two_l as i64 + 1);
        // Ground truth from the SA-IS builder: `sa[row]` is the reference position for that row.
        let sa = suffix_array_with_sentinel(&fm.reference);
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

    /// A full backward search reports the same occurrence count as brute-force window matching.
    /// Patterns are taken from the reference itself, so each is guaranteed to occur at least once
    /// and a spurious `s == 0` cannot pass.
    #[test]
    fn backward_search_counts_match_naive() {
        let fm = tiny();
        // The 2L reference as bytes; both the pattern source and the brute-force haystack.
        let bref = &fm.reference;
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


/// `BWA3_TRAFFIC=1`: count the 128-byte cache lines the FM index actually pulls, so the aligner's
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

    /// Whether `BWA3_TRAFFIC` was set in the environment. Read once and cached in a `OnceLock`, so
    /// the hot-path check is a relaxed atomic load rather than a `getenv`, and toggling the
    /// variable mid-run has no effect.
    ///
    /// # Returns
    /// `true` if the variable is present with ANY value, including empty: presence is the switch.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA3_TRAFFIC").is_some())
    }

    /// Print the accumulated line counts and the implied bandwidth to stderr. Call once at the end
    /// of a run; silently does nothing unless [`enabled`].
    ///
    /// # Parameters
    /// * `wall_s`: elapsed wall-clock time in SECONDS over which the counters accumulated, measured
    ///   by the caller. Used only as the divisor for the GB/s figure, and clamped away from zero.
    ///   The derived rate is per-process, so it is only meaningful for a single-threaded run.
    pub fn dump(wall_s: f64) {
        if !enabled() { return; }
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
            el as f64 / ec.max(1) as f64, total, total as f64 * 128.0 / 1e9, wall_s,
            total as f64 * 128.0 / 1e9 / wall_s.max(1e-9),
        );
    }
}
