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
//!
//! # Rust mechanics used in this file
//!
//! This table is for readers who know the biology but not the language. It glosses every Rust
//! construct this file uses, once, so the comments in the code below can talk about the algorithm
//! instead of the syntax.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `u8` | one byte, an integer 0 to 255. Both an ASCII character and an nt4 code are stored in one. |
//! | `[u8; 256]` | an array of exactly 256 bytes. The length is part of the type, fixed at compile time, so it cannot grow. |
//! | `Vec<u8>` | a growable list of bytes that OWNS its memory. Returning one hands the caller a fresh copy it must free. |
//! | `&[u8]` | a BORROWED window onto bytes somebody else owns. Passing one copies no data and transfers no ownership; the caller keeps its buffer. |
//! | `const` | a value fixed at compile time. It is substituted into the code, not looked up in memory at run time. |
//! | `const fn` | a function the compiler is allowed to RUN during compilation. `build_nt4_table` below is one, so the finished 256-byte table is baked into the binary and costs nothing at startup. |
//! | `pub` | visible outside this file. Without it, an item is private to this module. |
//! | `b'A'` | a byte literal: the single ASCII code for the letter A (65), not a piece of text. |
//! | `as usize` | a cast, here from a byte to the integer type used for array indices. Required because Rust never converts number types implicitly. |
//! | `#[inline]` | a hint to paste the function body into each caller instead of making a call. Used on the two one-line hot functions. |
//! | `.iter()` | starts walking a slice, yielding a reference to each element in turn. It does not copy the slice. |
//! | `.rev()` | reverses the direction of that walk, so elements come out last-to-first. |
//! | `.map(...)` | applies a function to every element as it goes past, producing a new sequence. |
//! | `\|&base\| ...` | an anonymous inline function (a "closure"). `&base` in the argument position unwraps the reference `.iter()` handed over, so `base` is the byte value itself. |
//! | `match` | multi-way branch on a value. Arms are tried top to bottom; `b'A' \| b'a'` matches either; the final `other` arm catches everything left and binds it to a name. |
//! | `.collect()` | drains the walk and builds a container from it. Which container is decided by the function's declared return type, here `Vec<u8>`. |
//! | `#[cfg(test)]` | compile the item that follows ONLY when building tests. The test module below is absent from the shipped binary. |

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
    //
    // Rust: `[NT4_N; 256]` is "the value NT4_N, repeated 256 times", the array-fill form. `mut`
    // marks the variable as modifiable; without it Rust rejects every assignment below, because
    // bindings are read-only unless you say otherwise. All of this runs inside the compiler (the
    // function is `const fn`), so no instruction of it survives into the binary: only the finished
    // table does.
    let mut table = [NT4_N; 256];
    // Rust: `b'A'` is the byte 65, and `as usize` converts it to the index type. Two separate
    // entries per base because the index is the raw byte, so upper and lower case are different
    // slots that happen to hold the same code.
    table[b'A' as usize] = 0;
    table[b'a' as usize] = 0;
    table[b'C' as usize] = 1;
    table[b'c' as usize] = 1;
    table[b'G' as usize] = 2;
    table[b'g' as usize] = 2;
    table[b'T' as usize] = 3;
    table[b't' as usize] = 3;
    // Rust: a trailing expression with no semicolon IS the return value. There is no `return`
    // keyword here and none is needed; this line hands the finished array back.
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
    // The whole encoder is one array read. No `if`, no error case, because the table has an entry
    // for all 256 possible byte values, so the index can never be out of range.
    //
    // Rust: `as usize` is mandatory. Rust will not silently widen a `u8` into an index the way C
    // does; every numeric conversion is written out. No semicolon, so this is the return value.
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
    // Real base (0..=3): complement by subtracting from 3, which pairs A(0) with T(3) and C(1) with
    // G(2). N (4, or anything larger) has no complement and comes back unchanged.
    //
    // Rust: `if`/`else` is an EXPRESSION, not a statement. Each branch ends in a value with no
    // semicolon, and whichever branch runs supplies the function's return value. This is why there
    // is no `return` keyword and no temporary variable. Both branches must produce the same type,
    // `u8`, or the code does not compile.
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
    // Read as one sentence: walk the input backwards, swap each base for its partner, and gather
    // the result into a new buffer. The two halves of "reverse complement" are the `.rev()` and the
    // `match`; doing only one of them would be a silent, and biologically wrong, bug.
    //
    // Rust: this is a pipeline built from four steps, evaluated lazily. Nothing happens until
    // `.collect()` at the end pulls values through; there is no intermediate array per stage, so
    // despite reading like four passes it compiles to a single loop.
    seq.iter()
        // Walk the borrowed input. `seq` is `&[u8]`, a window onto the caller's buffer, so nothing
        // is copied and the caller's data is never modified.
        .rev()
        // Walk it last byte to first. This is the "reverse" half.
        .map(|&base| match base {
            // `|&base|` is an inline anonymous function. The `&` in front of the parameter name
            // undoes the reference `.iter()` yields, so `base` is the byte itself rather than a
            // pointer to it. This is called destructuring, and it is why the `match` arms below can
            // compare against plain byte literals.
            //
            // The `match` is the "complement" half. `b'A' | b'a'` means "either of these two
            // bytes", which folds the case-normalisation into the same table: both cases of A
            // produce upper-case T.
            b'A' | b'a' => b'T',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            b'T' | b't' => b'A',
            // The catch-all arm. `other` is a NAME being bound to whatever byte did not match
            // above (N, an IUPAC letter, `*`), and returning it unchanged is what makes those
            // bytes pass through reversed but neither complemented nor upper-cased. Rust requires
            // this arm: a `match` must cover every possible value or it does not compile, which is
            // what guarantees there is no unhandled byte.
            other => other,
        })
        // Run the pipeline and build the output. `.collect()` does not say which container to
        // build; the compiler reads the function's declared return type, `Vec<u8>`, and picks it.
        // The `Vec` is freshly allocated and OWNED by the caller, which is what lets this function
        // hand back a buffer that outlives the borrowed input.
        .collect()
}

// Rust: everything below is compiled only under `cargo test` and is absent from the shipped
// binary, so tests can sit in the same file as the code they check without costing anything at
// run time. `mod tests` opens a nested namespace; `use super::*` pulls in every item from the file
// above so the tests can call `nt4` and `comp2` by their bare names. Each `#[test]` function is
// discovered and run by the test harness, and it passes if it returns without panicking, which is
// what `assert_eq!` does when its two arguments differ.
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
