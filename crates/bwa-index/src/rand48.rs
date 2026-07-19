//! glibc `srand48`/`lrand48` reproduction (48-bit linear congruential generator).
//!
//! bwa-mem2 randomizes ambiguous (N) bases in `.pac` with `lrand48() & 3` after `srand48(11)`.
//! Reproducing glibc's generator exactly is required for byte-identical `.pac` on references that
//! contain N runs. (N-free references never call it.)

/// The 48-bit LCG state, matching glibc's `drand48` family.
pub struct Rand48 {
    /// The generator's whole state, `X(n)`. Only the low 48 bits are meaningful; bits 48..64 are
    /// always zero because every write is masked with [`MASK48`]. Initialised by [`Rand48::srand48`]
    /// and advanced by exactly one step per [`Rand48::lrand48`] call, so its value at any moment
    /// identifies how many draws have been made. Private: the ORDER of draws is what byte-identity
    /// depends on, so nothing may peek at or rewind it.
    state: u64, // only low 48 bits used
}

// The three constants are fixed by the SVID/POSIX definition of the `drand48` family, not chosen:
// `X(n+1) = (a * X(n) + c) mod 2^48` with `a = 0x5DEECE66D`, `c = 0xB`. Any other multiplier
// produces a valid LCG and an invalid `.pac`.
/// The LCG multiplier `a` = 0x5DEECE66D (25214903917 decimal), a 35-bit constant fixed by the
/// standard. Changing it changes every N replacement base and therefore every output file.
const MULT: u64 = 0x5DEECE66D;
/// The LCG increment `c` = 11. Coincidentally equal to `build::SEED`, which is also 11; the two
/// are unrelated and must not be conflated.
const ADD: u64 = 0xB;
/// `mod 2^48`, applied after every step: the generator's state is 48 bits, not 64.
const MASK48: u64 = 0xFFFF_FFFF_FFFF;

impl Rand48 {
    /// Equivalent to glibc `srand48(seed)`: state = `(seed << 16) | 0x330E`.
    ///
    /// The seed supplies only the HIGH 32 bits of the 48-bit state; the low 16 are the fixed
    /// constant `0x330E` mandated by the standard. That is why `srand48(11)` is not the same as
    /// starting from state 11, and why the first output is a large number rather than a small one.
    /// The `& 0xFFFF_FFFF` reproduces the C's truncation of `seed` to `long int` width.
    ///
    /// Called once per index build with `seed = 11` (see `build::SEED`).
    ///
    /// # Parameters
    ///
    /// * `seed`: the caller's seed. Only its low 32 bits are used (matching the C's `long int`
    ///   truncation on the platforms bwa targets); the sign is irrelevant for the same reason.
    ///   Supplied by `build_index` as the constant 11, never by the user.
    ///
    /// # Returns
    ///
    /// A generator positioned BEFORE the first draw, so the caller's first `lrand48()` yields the
    /// same value the C's first `lrand48()` after `srand48(seed)` does.
    pub fn srand48(seed: i64) -> Self {
        let state = (((seed as u64) & 0xFFFF_FFFF) << 16) | 0x330E;
        Rand48 {
            state: state & MASK48,
        }
    }

    /// Advance the state one LCG iteration: `X = (a*X + c) mod 2^48`.
    ///
    /// # Returns
    ///
    /// The NEW state (post-update), in `[0, 2^48)`. The pre-update state is not recoverable, so
    /// each call consumes exactly one position in the stream. `wrapping_mul`/`wrapping_add` cannot
    /// lose information here: the product's bits above 48 are discarded by the mask anyway, which
    /// is what makes the u64 arithmetic equivalent to the C's modular arithmetic.
    fn step(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(ADD) & MASK48;
        self.state
    }

    /// Equivalent to glibc `lrand48()`: the high 31 bits of the next state, in `[0, 2^31)`.
    ///
    /// `>> 17` discards the low 17 bits of the 48-bit state, which are the least random ones in any
    /// power-of-two-modulus LCG. Note the caller in `build` then takes `& 3`, i.e. the LOW two bits
    /// of this already-shifted value, so bits 17 and 18 of the raw state are what actually pick the
    /// replacement base. That is a quirk of bwa's usage, not of the generator, and it must be
    /// reproduced exactly rather than "improved".
    ///
    /// # Returns
    ///
    /// A value in `[0, 2^31)`, i.e. bit 31 is always clear, matching the C's non-negative `long`.
    /// Advances the state by exactly one step; there is no way to draw without advancing.
    pub fn lrand48(&mut self) -> u32 {
        (self.step() >> 17) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_libc_lrand48_seed11() {
        // Reference values from the system libc: srand48(11); lrand48() x8 (the oracle ran here).
        let mut r = Rand48::srand48(11);
        let got: Vec<u32> = (0..8).map(|_| r.lrand48()).collect();
        assert_eq!(
            got,
            vec![
                1609868485, 1074594562, 470884846, 2128573038, 960673312, 346697164, 303961605,
                444770020
            ]
        );
    }
}
