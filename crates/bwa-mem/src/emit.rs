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
//! by never building a string in the first place. This file is the same two ideas in portable safe
//! Rust: no intrinsics, no `unsafe`, no per-tier build. Writing whole blocks into reserved capacity
//! is most of the win, and it costs nothing to maintain.
//!
//! **Nothing here may change a byte of output.** Every function writes exactly the bytes the code
//! it replaced wrote: same digits, same alphabet, same order. That is checked by the parity gate,
//! not by this comment: 64 of 64 option combinations pass `scripts/opt_parity.sh` against the
//! oracle, and 1M chr21 pairs plus 500k real GIAB pairs are `cmp`-clean against the previous binary.
//!
//! **On ARM this is a wash, and the first measurement here said otherwise.** A single run showed
//! `sam_emit` at 2.787 s against 2.668 s and that was noise; five interleaved repetitions put the
//! minima at 2.641 s and 2.637 s, i.e. -0.2%. That is the honest ARM number, and it is what should
//! be expected: the aarch64 baseline already lets the compiler vectorise, and this stage is 7.5% of
//! the wall there. The x86 case is the one this was written for, where the same stage is 13.1% of
//! the wall and 5.15x slower per pair, and it is measured in CI rather than here.
//!
//! # Rust mechanics used in this file
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `Vec::reserve` | grows the buffer's capacity ONCE, up front, so the per-byte writes that follow cannot each trigger a reallocation. |
//! | `extend_from_slice` | appends a whole slice in one `memcpy`, rather than a loop of pushes with a capacity check each. |
//! | `[u8; N]` on the stack | a fixed-size array living in the function's frame. Used as a scratch buffer so formatting a number allocates nothing. |
//! | `const fn` | a function the compiler can evaluate at compile time, here to build a 256-entry lookup table as a constant rather than at startup. |
//! | `chunks_exact(N)` | walks a slice in fixed-size pieces, with the remainder handed back separately. The fixed size is what lets the compiler unroll and vectorise the body. |

/// Largest number of decimal digits an `i64` can need, plus its sign.
///
/// `i64::MIN` is 19 digits and a minus sign, so 20 bytes is exact rather than generous.
const MAX_I64_DIGITS: usize = 20;

/// Append `value`'s decimal representation to `out`, allocating nothing.
///
/// The replacement for `out.extend_from_slice(value.to_string().as_bytes())`, which allocated a
/// `String` per number. Digits are generated least-significant first into a stack buffer and then
/// copied out in one `extend_from_slice`.
///
/// # Parameters
///
/// - `out`: the SAM record under construction.
/// - `value`: the number to write. Negative values are written with a leading `-`, as SAM's TLEN
///   requires.
#[inline]
pub fn push_int(out: &mut Vec<u8>, value: i64) {
    // 0 has no digits under the loop below (it terminates on `n == 0`), so it is written directly.
    if value == 0 {
        out.push(b'0');
        return;
    }
    let mut buf = [0u8; MAX_I64_DIGITS];
    // Write position, walking BACKWARD from the end of the buffer: the loop produces the
    // least-significant digit first, and this is what puts it last.
    let mut i = MAX_I64_DIGITS;
    // `unsigned_abs`, not `abs`: `i64::MIN.abs()` overflows, and TLEN can legitimately be very
    // negative. The sign is remembered separately and prepended below.
    let negative = value < 0;
    let mut n = value.unsigned_abs();
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    if negative {
        i -= 1;
        buf[i] = b'-';
    }
    out.extend_from_slice(&buf[i..]);
}

/// Same as [`push_int`] for a `String`, which the MD:Z and CIGAR builders accumulate into.
///
/// # Parameters
///
/// - `out`: the string under construction.
/// - `value`: the number to write.
#[inline]
pub fn push_int_str(out: &mut String, value: i64) {
    if value == 0 {
        out.push('0');
        return;
    }
    let mut buf = [0u8; MAX_I64_DIGITS];
    let mut i = MAX_I64_DIGITS;
    let negative = value < 0;
    let mut n = value.unsigned_abs();
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    if negative {
        i -= 1;
        buf[i] = b'-';
    }
    // SAFETY-free: the buffer holds ASCII digits and an optional '-', so it is valid UTF-8 by
    // construction and `from_utf8` cannot fail. `expect` rather than `unwrap_unchecked`: this is
    // once per number, not per byte, and a panic here would be a bug worth seeing.
    out.push_str(std::str::from_utf8(&buf[i..]).expect("decimal digits are ASCII"));
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

/// Append the forward-strand SEQ field for `codes` to `out`.
///
/// # Parameters
///
/// - `out`: the SAM record under construction.
/// - `codes`: nt4 codes for the bases to emit, in query order.
pub fn push_seq_fwd(out: &mut Vec<u8>, codes: &[u8]) {
    out.reserve(codes.len());
    let mut block = [0u8; BLOCK];
    let mut chunks = codes.chunks_exact(BLOCK);
    for chunk in &mut chunks {
        for (dst, &c) in block.iter_mut().zip(chunk) {
            *dst = FWD_TABLE[c as usize];
        }
        out.extend_from_slice(&block);
    }
    for &c in chunks.remainder() {
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
    let mut block = [0u8; BLOCK];
    // Walk from the end in whole blocks, so the reversal happens inside the block rather than
    // through a per-byte reverse iterator.
    let mut end = codes.len();
    while end >= BLOCK {
        let start = end - BLOCK;
        let src = &codes[start..end];
        for (i, dst) in block.iter_mut().enumerate() {
            // `BLOCK - 1 - i` reverses within the block; taking blocks from the end reverses across
            // them. Together they walk `codes` backwards.
            *dst = REV_TABLE[src[BLOCK - 1 - i] as usize];
        }
        out.extend_from_slice(&block);
        end = start;
    }
    for i in (0..end).rev() {
        out.push(REV_TABLE[codes[i] as usize]);
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
    let mut block = [0u8; BLOCK];
    let mut end = qual.len();
    while end >= BLOCK {
        let start = end - BLOCK;
        let src = &qual[start..end];
        for (i, dst) in block.iter_mut().enumerate() {
            *dst = src[BLOCK - 1 - i];
        }
        out.extend_from_slice(&block);
        end = start;
    }
    for i in (0..end).rev() {
        out.push(qual[i]);
    }
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
        for len in 0..70usize {
            // Codes cycle through 0..=5, so the clamp (code 5 must print as N) is exercised too.
            let codes: Vec<u8> = (0..len).map(|i| (i % 6) as u8).collect();
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
}
