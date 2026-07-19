//! Nucleotide encoding, matching bwa-mem2's `nst_nt4_table`.
//!
//! DNA is four bases (adenine, cytosine, guanine, thymine) and the aligner works on 2-bit codes
//! rather than ASCII throughout: it makes the scoring matrix a 5x5 array indexed directly by base
//! code, and it lets the reference be packed four bases to the byte. The fifth code, 4, absorbs
//! every other byte, which in practice means `N` (the sequencer could not call this base) and the
//! IUPAC ambiguity letters. Code 4 is never a match: see the N row of `opt::fill_scmat`.
//!
//! The A/C/G/T order is not arbitrary. It makes complementation the single operation `3 - c`
//! (A<->T, C<->G), which the reverse-complement paths rely on.
//!
//! # Glossary
//!
//! | Term | Plain language |
//! |------|----------------|
//! | nt4 code | a base as a small integer: A=0, C=1, G=2, T=3, anything else = 4 |
//! | N | "base unknown"; the sequencer could not decide which of the four it was |
//! | complement | the base pairing with it on the opposite DNA strand (A with T, C with G) |
//! | reverse complement | the same stretch of DNA read from the other strand: complement, backwards |
//!
//! Reading order: [`NT4_TABLE`] and [`nt4`] (ASCII in, code out), [`comp2`] (code in, code out),
//! then [`revcomp_ascii`] (ASCII in, ASCII out) which is the only one used at SAM-emission time.

/// The nt4 code standing for "unknown base" (N or any IUPAC ambiguity letter). It is one past the
/// four real bases, which is what makes the scoring matrix 5x5 rather than 4x4.
///
/// The value 4 is load-bearing, not a free choice: it must equal the number of real bases so that
/// N indexes the last row and column of `opt`'s 5x5 matrix, and it must exceed every real code so
/// that the `code < NT4_N` test in [`comp2`] separates the two cases.
pub const NT4_N: u8 = 4;

/// The largest real base code, used to complement by subtraction: `NT4_COMPLEMENT_PIVOT - code`
/// maps A(0)<->T(3) and C(1)<->G(2).
///
/// It is `NT4_N - 1`, the largest real code. The subtraction trick only works because the table
/// assigns the codes in A,C,G,T order so that complementary bases sit at mirrored positions;
/// reordering the table would silently produce wrong complements everywhere.
const NT4_COMPLEMENT_PIVOT: u8 = 3;

/// Maps an ASCII byte to a 2-bit base code (A=0, C=1, G=2, T=3), or 4 for anything else (N).
///
/// A full 256-entry table so lookup is one indexed load with no branch and no bounds concern: every
/// possible byte, including lowercase (soft-masked reference) and junk, has an entry.
pub const NT4_TABLE: [u8; 256] = build_nt4_table();

/// Build [`NT4_TABLE`] at compile time.
///
/// # Returns
///
/// All 256 entries: 4 ([`NT4_N`]) everywhere except the eight bytes `AaCcGgTt`, which map to
/// 0/0/1/1/2/2/3/3. `const fn` so the table is a static, not a runtime initialisation.
const fn build_nt4_table() -> [u8; 256] {
    // Every byte defaults to N; the eight ACGT/acgt entries are then overwritten.
    let mut table = [NT4_N; 256];
    table[b'A' as usize] = 0;
    table[b'a' as usize] = 0;
    table[b'C' as usize] = 1;
    table[b'c' as usize] = 1;
    table[b'G' as usize] = 2;
    table[b'g' as usize] = 2;
    table[b'T' as usize] = 3;
    table[b't' as usize] = 3;
    table
}

/// Encode a base to its 2-bit code (4 = N / other).
///
/// # Parameters
///
/// - `b`: one ASCII byte from a read or from the reference FASTA. Any of the 256 byte values is
///   accepted (the table is total), so there is no precondition and no error path: junk, IUPAC
///   ambiguity letters and lowercase soft-masked bases all resolve. Supplied by the FASTQ/FASTA
///   readers in `bwa-io` and by the index builder.
///
/// # Returns
///
/// The nt4 code in 0..=4: 0=A, 1=C, 2=G, 3=T, 4=N/other.
#[inline]
pub fn nt4(b: u8) -> u8 {
    NT4_TABLE[b as usize]
}

/// Complement of a 2-bit base code (0<->3, 1<->2); 4 (N) stays 4.
///
/// # Parameters
///
/// - `code`: an nt4 code, i.e. already the output of [`nt4`], NOT an ASCII byte. Valid range
///   0..=4; anything >= 4 is treated as N and returned as [`NT4_N`], so out-of-range input is
///   absorbed rather than rejected.
///
/// # Returns
///
/// The complementary nt4 code, in 0..=4.
#[inline]
pub fn comp2(code: u8) -> u8 {
    if code < NT4_N {
        NT4_COMPLEMENT_PIVOT - code
    } else {
        NT4_N
    }
}

/// Reverse-complement an ASCII nucleotide sequence (A<->T, C<->G, case-normalized to upper;
/// non-ACGT bytes pass through unchanged). Used for reverse-strand SAM SEQ output.
///
/// Both halves matter: DNA's two strands run antiparallel, so the reverse-strand rendering of a
/// read is its complement read backwards, not either operation alone. SAM stores SEQ relative to
/// the reference strand, so a read aligned with FLAG 0x10 must be written this way. Its QUAL is
/// reversed but NOT complemented (a quality score has no complement), which is why that reversal
/// lives at the call site rather than here.
///
/// Note the ASCII output is upper-cased, unlike [`nt4`]'s code path which is case-agnostic: a
/// soft-masked lowercase reference base would otherwise leak lowercase into SAM SEQ.
///
/// # Parameters
///
/// - `seq`: ASCII nucleotides, NOT nt4 codes (feeding it codes would leave them untouched, since
///   0..=4 are not ACGT bytes). Any length including empty; no other precondition. Supplied by the
///   SAM writer as the read's SEQ field when the alignment is on the reverse strand.
///
/// # Returns
///
/// A freshly allocated `Vec` of the same length, reversed and complemented, upper-cased for ACGT.
/// Bytes that are not ACGT/acgt (`N`, IUPAC letters, `*`) are reversed in place but passed through
/// uncomplemented and un-cased.
pub fn revcomp_ascii(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&base| match base {
            b'A' | b'a' => b'T',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            b'T' | b't' => b'A',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_bases() {
        assert_eq!(nt4(b'A'), 0);
        assert_eq!(nt4(b'c'), 1);
        assert_eq!(nt4(b'G'), 2);
        assert_eq!(nt4(b't'), 3);
        assert_eq!(nt4(b'N'), 4);
        assert_eq!(nt4(b'-'), 4);
    }

    #[test]
    fn complements() {
        assert_eq!(comp2(nt4(b'A')), nt4(b'T'));
        assert_eq!(comp2(nt4(b'C')), nt4(b'G'));
        assert_eq!(comp2(4), 4);
    }
}
