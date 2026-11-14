//! Byte-level writers for the SAM record path: integers without an allocation, and SEQ/QUAL a
//! block at a time.
//!
//! Why this file exists is a measurement. On x86_64 the `sam_emit` stage is 13.1% of the wall and
//! **5.15x slower per read pair than the same stage on an M4**, against 2.57x for the vector
//! kernels; it is the single worst-scaling stage we have off Apple Silicon (ROADMAP, the x86
//! profile). Two causes, both visible in the code it replaced:
//!
//! 1. **Every integer field went through `.to_string()`**, i.e. a heap allocation per number. A SAM
//!    record has FLAG, POS, MAPQ, PNEXT, TLEN, one per CIGAR operation and one per numeric tag, so
//!    that is a dozen or more allocations per record. The allocation probe counts **40 allocations
//!    per read** in `sam_emit`, at a mean of 133 bytes.
//! 2. **SEQ and QUAL were written one byte at a time** with `Vec::push`, which is a capacity check
//!    and a branch per base, 150 times per read, twice.
//!
//! fg-labs/bwa-mem3 fixed the second one with a 16-byte `_mm_shuffle_epi8` / `vqtbl1q_u8` lookup
//! (`src/sam_encode.cpp`, compiled at five ISA tiers and dispatched at runtime) and the first one
//! by never building a string in the first place. This file does the same two things: integers
//! written straight into reserved capacity, and SEQ/QUAL 16 bytes per shuffle ([`encode16`]: NEON
//! on aarch64, SSSE3 in the x86-64-v3 build, a byte loop elsewhere).
//!
//! **Nothing here may change a byte of output.** Every function writes exactly the bytes the code
//! it replaced wrote: same digits, same alphabet, same order. That is checked by the parity gate,
//! not by this comment: 64 of 64 option combinations pass `scripts/opt_parity.sh` against the
//! oracle, and 1M chr21 pairs plus 500k real GIAB pairs are `cmp`-clean against the previous binary.
//!
//! **It is a wash on both platforms, and both of the first numbers here were wrong.** ARM: a single
//! run showed `sam_emit` at 2.787 s against 2.668 s and that was noise; five interleaved reps put
//! the minima at 2.641 and 2.637 s, -0.2%. x86, which is what it was written for: 28.736 s against
//! **28.530 s**, -0.7%, with the whole run inside its own noise.
//!
//! Why: LLVM never turns a 256-entry byte table into a shuffle, not even at x86-64-v3, where the
//! block loop compiled to 16 table loads spilled to the stack and 16 byte stores. That loop was
//! still 0.8% of an EPYC 9V74 and of a Neoverse N2 PE run (wave 12) before [`encode16`] spelled the
//! shuffle out.
//!
//! # Rust mechanics used in this file
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `Vec::reserve` | grows the buffer's capacity ONCE, up front, so the per-byte writes that follow cannot each trigger a reallocation. |
//! | `extend_from_slice` | appends a whole slice in one `memcpy`, rather than a loop of pushes with a capacity check each. |
//! | `[u8; N]` on the stack | a fixed-size array living in the function's frame: one encoded block of 16 bytes, appended in one fixed-size copy. |
//! | `spare_capacity_mut` + `set_len` | writes into the reserved bytes past the end, then declares them part of the vector: how a number's digits land in place with no copy. |
//! | `const fn` | a function the compiler can evaluate at compile time, here to build a 256-entry lookup table as a constant rather than at startup. |
//! | `as_chunks::<N>()` / `as_rchunks::<N>()` | a slice as `&[[u8; N]]` blocks from the front / from the back, with the remainder handed back separately. |
//! | `#[cfg(target_arch = ...)]` + `std::arch` | the per-ISA shuffle in [`encode16`], chosen at compile time; the byte loop is the fallback. |

/// Append `value`'s decimal representation to `out`, allocating nothing.
///
/// The replacement for `out.extend_from_slice(value.to_string().as_bytes())`, which allocated a
/// `String` per number. The digit count comes first, so the digits are written straight into the
/// spare capacity, least-significant last: a stack buffer copied out with `extend_from_slice` cost
/// a `memcpy` call per integer (18 per pair, 44% of every copy call on a PE run).
///
/// # Parameters
///
/// - `out`: the SAM record under construction.
/// - `value`: the number to write. Negative values are written with a leading `-`, as SAM's TLEN
///   requires.
#[inline]
pub fn push_int(out: &mut Vec<u8>, value: i64) {
    // `unsigned_abs`, not `abs`: `i64::MIN.abs()` overflows, and TLEN can legitimately be very
    // negative. The sign is remembered separately and written first.
    let negative = value < 0;
    let mut n = value.unsigned_abs();
    let digits = n.checked_ilog10().map_or(1, |d| d as usize + 1);
    let total = digits + usize::from(negative);
    out.reserve(total);
    let spare = &mut out.spare_capacity_mut()[..total];
    if negative {
        spare[0].write(b'-');
    }
    // Backward from the end: the loop produces the least-significant digit first. Zero writes one
    // '0'.
    for slot in spare[usize::from(negative)..].iter_mut().rev() {
        slot.write(b'0' + (n % 10) as u8);
        n /= 10;
    }
    // SAFETY: the `total` bytes past the old length were all written above.
    unsafe { out.set_len(out.len() + total) };
}

/// Same as [`push_int`] for a `String`, which the MD:Z and CIGAR builders accumulate into.
///
/// # Parameters
///
/// - `out`: the string under construction.
/// - `value`: the number to write.
#[inline]
pub fn push_int_str(out: &mut String, value: i64) {
    // SAFETY: `push_int` appends ASCII digits and an optional '-', so the string stays UTF-8. The
    // checked `from_utf8` cost 0.5G cycles per 2M pairs.
    push_int(unsafe { out.as_mut_vec() }, value);
}

/// Build a 256-entry table mapping an nt4 code to its SAM character, so the encoders below need no
/// bounds clamp in the inner loop.
///
/// bwa indexes `"ACGTN"` unguarded and relies on codes staying in `0..=4`; we clamp, because a code
/// above 4 is possible in principle and indexing out of bounds in Rust is a panic rather than a
/// quiet read. Folding the clamp into the table makes it free: every possible byte value has an
/// entry, so the lookup is a single load with no comparison.
///
/// # Parameters
///
/// - `alphabet`: the five characters for codes 0 to 4, i.e. `b"ACGTN"` or `b"TGCAN"`.
///
/// # Returns
///
/// A table where index `c` is `alphabet[min(c, 4)]`.
const fn base_table(alphabet: &[u8; 5]) -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        // `const fn` has no iterators and no `min` on this path, hence the explicit branch.
        table[i] = if i < 4 { alphabet[i] } else { alphabet[4] };
        i += 1;
    }
    table
}

/// Forward-strand alphabet, bwa's `F`.
static FWD_TABLE: [u8; 256] = base_table(b"ACGTN");
/// Reverse-strand alphabet, bwa's `R`. Indexing this one in reverse order performs the complement,
/// because `TGCAN[c]` is the complement of `ACGTN[c]`.
static REV_TABLE: [u8; 256] = base_table(b"TGCAN");

/// Block size for the chunked loops below.
///
/// 16 is what fg-labs/bwa-mem3's hand-written SSSE3 and NEON encoders use, and it is a good size
/// here for the same reason: it is one vector register on both architectures, so a compiler that
/// vectorises the body has a natural width to pick, and one that does not still amortises the
/// bounds and capacity checks over sixteen bases instead of paying them per base.
const BLOCK: usize = 16;

/// The first five entries of a [`base_table`] as a 16-byte shuffle table. Entries 5..16 are never
/// indexed: [`encode16`] clamps every code to 4 before the shuffle.
const fn shuffle_table(alphabet: &[u8; 5]) -> [u8; 16] {
    let mut t = [0u8; 16];
    let mut i = 0;
    while i < 5 {
        t[i] = alphabet[i];
        i += 1;
    }
    t
}

static FWD16: [u8; 16] = shuffle_table(b"ACGTN");
static REV16: [u8; 16] = shuffle_table(b"TGCAN");

/// `src`, reversed when `REV`, then mapped through `table` (identity when `LOOKUP` is false): 16
/// bytes in one or two shuffles where the target has a byte shuffle (NEON `tbl`, SSSE3 `pshufb`,
/// as fg-labs/bwa-mem3's `sam_encode.cpp`), byte by byte otherwise. LLVM turns neither the 256-entry
/// table nor the reversal into a shuffle on its own: at x86-64-v3 the old block loop was 16 table
/// loads spilled to the stack and 16 byte stores (0.8% of an EPYC 9V74 or Neoverse N2 PE run).
///
/// Every path computes `table16[min(c, 4)]`, which is [`base_table`]'s entry for `c` when
/// `table16` comes from [`shuffle_table`] on the same alphabet.
#[inline(always)]
fn encode16<const REV: bool, const LOOKUP: bool>(table16: &[u8; 16], src: &[u8; 16]) -> [u8; 16] {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline; every load and store is of 16 in-bounds bytes.
    unsafe {
        use std::arch::aarch64::*;
        let mut v = vld1q_u8(src.as_ptr());
        if REV {
            v = vrev64q_u8(v);
            v = vextq_u8::<8>(v, v);
        }
        if LOOKUP {
            v = vqtbl1q_u8(vld1q_u8(table16.as_ptr()), vminq_u8(v, vdupq_n_u8(4)));
        }
        let mut out = [0u8; 16];
        vst1q_u8(out.as_mut_ptr(), v);
        out
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
    // SAFETY: SSSE3 is enabled at compile time (the x86-64-v3 build); unaligned 16-byte accesses.
    unsafe {
        use std::arch::x86_64::*;
        let mut v = _mm_loadu_si128(src.as_ptr().cast());
        if REV {
            v = _mm_shuffle_epi8(
                v,
                _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            );
        }
        if LOOKUP {
            let t = _mm_loadu_si128(table16.as_ptr().cast());
            v = _mm_shuffle_epi8(t, _mm_min_epu8(v, _mm_set1_epi8(4)));
        }
        let mut out = [0u8; 16];
        _mm_storeu_si128(out.as_mut_ptr().cast(), v);
        out
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "ssse3")
    )))]
    {
        std::array::from_fn(|i| {
            let c = src[if REV { BLOCK - 1 - i } else { i }];
            if LOOKUP {
                table16[c.min(4) as usize]
            } else {
                c
            }
        })
    }
}

/// Append the forward-strand SEQ field for `codes` to `out`.
///
/// # Parameters
///
/// - `out`: the SAM record under construction.
/// - `codes`: nt4 codes for the bases to emit, in query order.
pub fn push_seq_fwd(out: &mut Vec<u8>, codes: &[u8]) {
    out.reserve(codes.len());
    let (chunks, remainder) = codes.as_chunks::<BLOCK>();
    for chunk in chunks {
        out.extend_from_slice(&encode16::<false, true>(&FWD16, chunk));
    }
    for &c in remainder {
        out.push(FWD_TABLE[c as usize]);
    }
}

/// Append the reverse-complement SEQ field for `codes` to `out`, i.e. the bases in reverse order,
/// each complemented.
///
/// # Parameters
///
/// - `out`: the SAM record under construction.
/// - `codes`: nt4 codes in query order; this function walks them backwards.
pub fn push_seq_rev(out: &mut Vec<u8>, codes: &[u8]) {
    out.reserve(codes.len());
    // Whole blocks from the end, each reversed inside `encode16`: together they walk `codes`
    // backwards. The head that is left over comes last.
    let (head, chunks) = codes.as_rchunks::<BLOCK>();
    for chunk in chunks.iter().rev() {
        out.extend_from_slice(&encode16::<true, true>(&REV16, chunk));
    }
    for &c in head.iter().rev() {
        out.push(REV_TABLE[c as usize]);
    }
}

/// Append `qual` reversed, which is what a reverse-strand record's QUAL field is.
///
/// # Parameters
///
/// - `out`: the SAM record under construction.
/// - `qual`: the quality string in query order.
pub fn push_qual_rev(out: &mut Vec<u8>, qual: &[u8]) {
    out.reserve(qual.len());
    let (head, chunks) = qual.as_rchunks::<BLOCK>();
    for chunk in chunks.iter().rev() {
        out.extend_from_slice(&encode16::<true, false>(&REV16, chunk));
    }
    for &c in head.iter().rev() {
        out.push(c);
    }
}

/// Append the `-V` tag for a contig's FASTA comment, or nothing when it has none (`bwamem.cpp:1721`).
///
/// # Parameters
///
/// - `out`: the SAM record under construction.
/// - `anno`: `Contig::comment` of the contig the record is placed on.
pub fn push_xr(out: &mut Vec<u8>, anno: &[u8]) {
    if anno.is_empty() {
        return;
    }
    out.extend_from_slice(b"\tXR:Z:");
    // A TAB would end the tag, so the C turns it into a space.
    out.extend(anno.iter().map(|&c| if c == b'\t' { b' ' } else { c }));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digits must match `to_string` for every shape of input, including the two that a
    /// hand-rolled writer gets wrong: zero, and `i64::MIN`, whose absolute value does not fit.
    #[test]
    fn integers_match_to_string() {
        let mut cases: Vec<i64> = vec![0, 1, -1, 9, 10, -10, 99, 100, i64::MAX, i64::MIN];
        // Every power of ten and its neighbours, where a digit-count bug lives.
        let mut p: i64 = 1;
        for _ in 0..18 {
            p *= 10;
            cases.extend_from_slice(&[p - 1, p, p + 1, -p]);
        }
        for v in cases {
            let mut out = Vec::new();
            push_int(&mut out, v);
            assert_eq!(out, v.to_string().as_bytes(), "push_int({v})");
            let mut s = String::new();
            push_int_str(&mut s, v);
            assert_eq!(s, v.to_string(), "push_int_str({v})");
        }
    }

    /// The encoders must agree with the per-byte loops they replaced, at every length around a
    /// block boundary: 0, a partial block, exactly one block, and a block plus a remainder are
    /// where a chunked rewrite goes wrong.
    #[test]
    fn seq_and_qual_match_the_byte_loops() {
        // Codes cycle through 0..=5, so the clamp (code 5 must print as N) is exercised too; the
        // second pattern reaches every byte, including the >= 128 a raw `pshufb` would zero.
        let patterns: [fn(usize) -> u8; 2] = [|i| (i % 6) as u8, |i| (i * 97 % 256) as u8];
        for (len, pattern) in (0..70usize).flat_map(|l| patterns.map(|p| (l, p))) {
            let codes: Vec<u8> = (0..len).map(pattern).collect();
            let qual: Vec<u8> = (0..len).map(|i| b'!' + (i % 40) as u8).collect();

            let mut got = Vec::new();
            push_seq_fwd(&mut got, &codes);
            let want: Vec<u8> = codes.iter().map(|&c| b"ACGTN"[c.min(4) as usize]).collect();
            assert_eq!(got, want, "forward SEQ at len {len}");

            let mut got = Vec::new();
            push_seq_rev(&mut got, &codes);
            let want: Vec<u8> = codes
                .iter()
                .rev()
                .map(|&c| b"TGCAN"[c.min(4) as usize])
                .collect();
            assert_eq!(got, want, "reverse SEQ at len {len}");

            let mut got = Vec::new();
            push_qual_rev(&mut got, &qual);
            let want: Vec<u8> = qual.iter().rev().copied().collect();
            assert_eq!(got, want, "reverse QUAL at len {len}");
        }
    }

    /// SAM column 10 is over `ACGTN` only, whatever the FASTQ held: lowercase comes back upper-cased
    /// and every byte that is not a base (IUPAC, `.`, `*`, a space) prints as `N`, as bwa-mem2 2.3
    /// does. A space leaking through would end the field and break the record.
    #[test]
    fn sam_seq_is_only_acgtn() {
        let codes: Vec<u8> = b"acgtNRY-.* ACGT"
            .iter()
            .map(|&b| bwa_core::dna::nt4(b))
            .collect();
        let mut fwd = Vec::new();
        push_seq_fwd(&mut fwd, &codes);
        assert_eq!(fwd, b"ACGTNNNNNNNACGT");
        // Complemented and reversed: this input happens to be its own reverse complement.
        let mut rev = Vec::new();
        push_seq_rev(&mut rev, &codes);
        assert_eq!(rev, b"ACGTNNNNNNNACGT");
    }
}
