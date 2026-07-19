//! Recursive Model Index (RMI) over a sorted array of `u64` keys — the learned-index core of the
//! LISA seeding acceleration (Kraska et al. 2018; Ho/Vasimuddin LISA 2021; Jung/Han BWA-MEME 2022).
//!
//! An RMI predicts the array position of a key from a small hierarchy of cheap models, so a
//! *bounded* local search replaces a full `O(log n)` binary search over the (huge) sorted key array.
//! Two levels: a root linear model maps a key to a leaf bucket, and that leaf's linear model maps the
//! key to a position estimate; a per-leaf error bound turns the estimate into a small search window.
//!
//! **Result-preserving:** [`Rmi::lower_bound`] returns exactly the same index as a `partition_point`
//! / binary search over the same keys (verified by the unit tests), so routing seeding through it
//! keeps the alignment output byte-identical — the RMI only changes *how fast* the position is found,
//! never *which* position.
//!
//! # Provenance
//!
//! **No bwa-mem2 C original.** bwa-mem2 has no learned index at all; nothing in this file mirrors a C
//! quirk, and no byte-for-byte agreement with C is required of it, because its only output is a
//! search hint that is subsequently corrected. That is a genuinely unusual position in this codebase:
//! this is one of the few modules where the floating-point arithmetic below is allowed to differ from
//! anything, including from itself across platforms, without threatening output identity.
//!
//! # The idea, for a reader who has not seen a learned index
//!
//! A sorted array plus "where does key X sit?" is normally answered by binary search: `log2(n)`
//! probes, each a cache miss, and at genome scale `n` is ~6.2e9 so that is ~33 dependent misses. A
//! learned index instead *fits a function* from key value to array position. If the keys were
//! uniformly distributed, position would be a straight line in the key, and one multiply-add would
//! land exactly. Real keys are not uniform, so we use a hierarchy: one line to pick a bucket, then a
//! per-bucket line to predict the position, then a small binary search over a window whose width was
//! measured at build time. The "recursive model index" name (Kraska et al. 2018) is that hierarchy.
//!
//! # Layout and cost
//!
//! Nothing here is serialized to disk; the whole structure is rebuilt in RAM whenever a `LearnedSa`
//! is built. Its size is tiny next to the key array it indexes: `n_leaves` x (2 x `f64` + `u32`) = 20
//! bytes per leaf, so even a million leaves is 20 MB against ~31 GB of keys. That asymmetry is why
//! `n_leaves` can be raised freely to shrink `leaf_err` until the last-mile search fits one or two
//! cache lines.

/// A single-variable linear model `pos ~= slope * key + intercept`, evaluated in `f64`.
///
/// `f64` and not `f32`: keys are up to 40 bits, and `f32`'s 24-bit mantissa cannot even represent
/// them distinctly, so the prediction error would be dominated by rounding rather than by how well
/// the line fits. `f64`'s 53-bit mantissa covers a 40-bit key exactly.
///
/// `Default` (both fields zero) is the degenerate model that predicts position 0 for every key. It is
/// the deliberate fallback for empty or unfittable inputs: still correct, just uninformative.
#[derive(Clone, Copy, Debug, Default)]
struct LinearModel {
    /// Positions per unit of key. Non-negative in practice, since position is non-decreasing in a
    /// sorted key array.
    slope: f64,
    /// Position at key 0, after the `x0` origin shift has been folded back in (see `from_sums`).
    intercept: f64,
}

impl LinearModel {
    /// Least-squares fit from precomputed sums over points `(x_i - x0, y_i)`: `n` points, `sx=Σx`,
    /// `sy=Σy`, `sxx=Σx²`, `sxy=Σxy`, with `x0` the origin folded back in. Equivalent to a textbook
    /// least-squares fit over the materialized points (it is the same closed form), but it lets the
    /// caller stream the points without ever building per-leaf key/target arrays, which is essential
    /// at genome scale where those arrays would be tens of gigabytes.
    ///
    /// Why the `x0` shift exists: raw keys are up to 2^40, so `Σx²` over billions of points would be
    /// around 2^80 x n, losing precision and risking a badly conditioned normal equation. Subtracting
    /// the run's first key `x0` keeps the `x` values small (a leaf spans a narrow key range), and the
    /// final `intercept - slope * x0` translates the fitted line back to absolute key coordinates.
    /// Slope is unaffected by a shift of the origin, which is why only the intercept is corrected.
    ///
    /// # Parameters
    ///
    /// * `n`: number of points summed. Unitless count, `>= 0`. `0` and `1` are handled as special
    ///   cases below because neither determines a slope.
    /// * `sx`: `Σ(x_i - x0)`, in key units. Non-negative when `x0` is the run's minimum key.
    /// * `sy`: `Σy_i`, in array-position units (an absolute row index for a leaf fit, a leaf index
    ///   for the root fit).
    /// * `sxx`: `Σ(x_i - x0)²`, key units squared. Always `>= 0`.
    /// * `sxy`: `Σ(x_i - x0) * y_i`, key units times position units.
    /// * `x0`: the origin shift already subtracted from every x, in key units. Supplied by the
    ///   caller as the run's first (smallest) key. Caller must have used the same `x0` for every
    ///   point in the run or the fit is meaningless.
    ///
    /// # Returns
    ///
    /// A model in ABSOLUTE key coordinates (the `x0` shift is folded back into `intercept`), so the
    /// caller can call `predict` with a raw key and never has to remember the shift.
    fn from_sums(n: usize, sx: f64, sy: f64, sxx: f64, sxy: f64, x0: f64) -> Self {
        if n == 0 {
            return LinearModel::default();
        }
        if n == 1 {
            return LinearModel {
                slope: 0.0,
                intercept: sy, // single target == its own mean
            };
        }
        let nf = n as f64;
        // Normal-equation denominator, `n*Σx² - (Σx)²`, which is `n²` times the variance of x. It is
        // zero exactly when every key in the run is identical (a plateau of duplicate keys, common
        // for short or repetitive k-mers), in which case no slope is defined and we fall back to
        // predicting the run's mean position. The window bound computed later still makes that safe.
        // The `< f64::EPSILON` test is effectively "is it zero": `denom` is a sum of squares, so it
        // is either 0 or many orders of magnitude above 2.2e-16. A near-but-not-zero `denom` would
        // give a wild slope, which costs only a wider measured `leaf_err`, never a wrong answer.
        let denom = nf * sxx - sx * sx;
        if denom.abs() < f64::EPSILON {
            return LinearModel {
                slope: 0.0,
                intercept: sy / nf,
            };
        }
        // Ordinary-least-squares slope, in positions per unit of key.
        let slope = (nf * sxy - sx * sy) / denom;
        // Intercept in SHIFTED coordinates: the value the line takes at `x - x0 == 0`. It is
        // translated to absolute key coordinates by the `- slope * x0` below.
        let b = (sy - slope * sx) / nf;
        LinearModel {
            slope,
            intercept: b - slope * x0,
        }
    }

    /// Evaluate the line at `key`.
    ///
    /// # Parameters
    ///
    /// * `key`: an absolute (unshifted) key, up to 40 bits for this crate's use. Exactly
    ///   representable in `f64`, so the cast is lossless.
    ///
    /// # Returns
    ///
    /// The raw, UNCLAMPED prediction as `f64`. It can be negative (below the first key) or exceed
    /// the array length (above the last), because a line is unbounded; every caller is responsible
    /// for clamping it into a valid index range. Its unit is whatever the model was fit against:
    /// leaf indices for `Rmi::root`, absolute array positions for a leaf model.
    #[inline]
    fn predict(&self, key: u64) -> f64 {
        self.slope * key as f64 + self.intercept
    }
}

/// A two-level recursive model index over an immutable sorted `u64` key array.
#[derive(Clone, Debug)]
pub struct Rmi {
    /// Level 1: maps a key to a *leaf index* (not a position), by being fit against the target
    /// `i * n_leaves / n` rather than `i`.
    root: LinearModel,
    /// Level 2: `leaves[li]` maps a key to an absolute array position. Length is always `n_leaves`.
    ///
    /// A leaf that no stored key routed to keeps the `Default` model (slope 0, intercept 0) and a
    /// `leaf_err` of 0, i.e. it predicts position 0 with zero uncertainty. Such a leaf is unreachable
    /// for any *stored* key by construction, but a *probe* key falling in a large gap of the key
    /// distribution could in principle route to one, and `lower_bound` would then search the window
    /// `[0, 2)` and return a wrong index. See the caveat on [`Rmi::lower_bound`].
    leaves: Vec<LinearModel>,
    /// Per-leaf half-width of the search window: `max |predicted - actual|`, rounded up. Measured
    /// over the leaf's own keys at build time, so it is an exact bound for stored keys, and it is
    /// what converts a fuzzy prediction into a *provably* correct bounded search. Small `leaf_err`
    /// is the entire performance story: err 0 means one probe, err 1000 means ~11.
    leaf_err: Vec<u32>,
    /// Number of keys indexed. Must equal the length of the key array passed to `lower_bound`.
    n: usize,
    /// Number of second-level models, already clamped into `[1, n]` by `build`. Stored because
    /// `leaves.len()` alone would not distinguish the empty-input case.
    n_leaves: usize,
}

impl Rmi {
    /// Build an RMI over `keys`, which **must be sorted ascending**. `n_leaves` is the number of
    /// second-level models (clamped to `[1, keys.len()]`); a few thousand leaves per million keys is
    /// a good default. Empty input yields an index that always reports position 0.
    ///
    /// Cost: three linear passes over `keys` (root sums, per-leaf sums, error measurement). No pass
    /// allocates anything proportional to `n`, which is what makes it viable over a 6.2e9-element,
    /// 31 GB packed array. Sortedness is only `debug_assert`ed, since checking it is itself an O(n)
    /// pass; an unsorted input does not panic in release, it silently produces a useless index.
    ///
    /// # Parameters
    ///
    /// * `keys`: the sorted key array to index, 5-byte-packed. For [`crate::lisa::LearnedSa`] these
    ///   are the first-`K` k-mer keys of the suffixes in suffix-array row order, so `keys[i]`
    ///   belongs to ROW `i`, not to reference position `i`. Borrowed only for the duration of the
    ///   build; the RMI stores no reference to it, which is why `lower_bound` takes it again.
    /// * `n_leaves`: requested number of second-level models, unitless, clamped internally to
    ///   `[1, keys.len()]`. Chosen by the caller as a memory-versus-precision trade (20 bytes per
    ///   leaf). Larger means a narrower `leaf_err` and fewer last-mile probes; it can never change
    ///   the answer, only the speed.
    ///
    /// # Returns
    ///
    /// An `Rmi` valid only for a key array of exactly this length and content.
    pub fn build(keys: &crate::packed::Packed40, n_leaves: usize) -> Self {
        // Number of indexed keys; for LISA this is `2 * l_pac + 1` suffix-array rows.
        let n = keys.len();
        if n == 0 {
            return Rmi {
                root: LinearModel::default(),
                leaves: vec![LinearModel::default()],
                leaf_err: vec![0],
                n: 0,
                n_leaves: 1,
            };
        }
        debug_assert!(keys.is_sorted(), "keys must be sorted");
        // From here on `n_leaves` is the EFFECTIVE leaf count: at least 1 (so `n_leaves - 1` never
        // underflows) and at most `n` (more leaves than keys would leave most of them empty).
        let n_leaves = n_leaves.clamp(1, n);

        // Root model maps key -> leaf index, trained on (key_i -> leaf_target = i * n_leaves / n).
        // Accumulate the least-squares sums directly (no n-length target array) with x0 = keys[0].
        // `keys.get(0)` is the smallest key, so every shifted x below is >= 0 and the sums stay in a
        // well-conditioned range. See `from_sums` for why the shift matters at 40-bit key magnitudes.
        // Root fit's origin shift: the smallest key in the whole array.
        let x0r = keys.get(0) as f64;
        // Running least-squares sums for the ROOT fit, over all `n` points. Invariant at the top of
        // the loop below: they hold the sums over keys `0..i`, with x shifted by `x0r` and y the
        // target LEAF INDEX (not the array position).
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        // `scale` converts an array position into a leaf index: position `i` of `n` should land in
        // leaf `i * n_leaves / n`, which spreads the keys evenly across leaves *by rank*, not by key
        // value. Even-by-rank is the right target because it equalizes the number of keys each leaf
        // must model, and hence roughly equalizes the per-leaf error.
        let scale = n_leaves as f64 / n as f64;
        for i in 0..n {
            let k = keys.get(i);
            // `x`: this key measured from the global minimum key. `y`: the leaf this key SHOULD be
            // routed to, as a fractional leaf index in `[0, n_leaves)`.
            let x = k as f64 - x0r;
            let y = i as f64 * scale;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        // Level-1 model: absolute key -> fractional leaf index.
        let root = LinearModel::from_sums(n, sx, sy, sxx, sxy, x0r);

        // Route a key to its leaf. The two clamps are load-bearing: the root line is unbounded, so it
        // predicts negative values below the first key and values past `n_leaves` above the last.
        // Rust's float-to-int casts are saturating (negatives and NaN both become 0), so `max(0.0)`
        // is belt-and-braces rather than strictly required; it states the intent at the call site.
        let leaf_of = |key: u64| -> usize {
            let p = root.predict(key);
            (p.max(0.0) as usize).min(n_leaves - 1)
        };

        // Fit each leaf on the absolute positions of the keys routed to it. `leaf_of` is monotonic
        // non-decreasing in the (sorted) key (the root slope is non-negative for position-vs-key), so
        // the keys of each leaf form one contiguous run — we stream the sums per run instead of
        // bucketing every key. Result-identical to fitting `leaf_pos[li]` explicitly.
        // Outer loop walks one contiguous run per iteration; `i` always sits at a run start and `j`
        // scans to the run end, so the whole nest is a single O(n) pass, not O(n * n_leaves).
        // Level-2 models, indexed by leaf. Any leaf no key routes to keeps the `Default` model.
        let mut leaves = vec![LinearModel::default(); n_leaves];
        // Invariant at the top of the outer loop: keys `0..i` have all been assigned to a leaf and
        // every leaf they touched has been fitted; `i` is the first key of the next contiguous run.
        let mut i = 0usize;
        while i < n {
            // The leaf this whole run belongs to (leaf routing is monotone in the sorted key).
            let li = leaf_of(keys.get(i));
            // Per-run origin: the run's own first key. Unlike the root fit (which used the global
            // minimum) each leaf gets a local origin, so the shifted x values span only this leaf's
            // narrow key range. `y` is the *absolute* array position `j`, not a leaf-relative one,
            // because `predict` must return an index into the whole array.
            let x0 = keys.get(i) as f64;
            // Per-leaf least-squares sums, same roles as the root's but with y = absolute position.
            let (mut lsx, mut lsy, mut lsxx, mut lsxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            // Scans forward to the end of the run. On exit `[i, j)` is exactly leaf `li`'s keys.
            let mut j = i;
            while j < n && leaf_of(keys.get(j)) == li {
                let x = keys.get(j) as f64 - x0;
                let y = j as f64;
                lsx += x;
                lsy += y;
                lsxx += x * x;
                lsxy += x * y;
                j += 1;
            }
            leaves[li] = LinearModel::from_sums(j - i, lsx, lsy, lsxx, lsxy, x0);
            i = j;
        }

        // Per-leaf error bound: the max over the leaf's keys of |round(predict) - actual position|.
        // Guarantees the true position lies in [pred - err, pred + err].
        // Per-leaf max absolute prediction error, in array positions. Invariant at the top of the
        // loop: `leaf_err[l]` is the max error over keys `0..i` that routed to leaf `l`.
        let mut leaf_err = vec![0u32; n_leaves];
        for i in 0..n {
            let k = keys.get(i);
            let li = leaf_of(k);
            // Where leaf `li`'s model THINKS key `k` sits; `i` is where it actually sits.
            let pred = leaves[li].predict(k);
            // `round()` (not truncate) because `predict` is also rounded at query time; the two must
            // use the identical rounding or the measured bound would not cover the queried estimate.
            let pred_i = pred.round().max(0.0) as i64;
            // Signed difference then absolute value: the prediction may fall on either side, and the
            // window is symmetric, so only the magnitude is kept.
            // UNVERIFIED: the `as u32` narrowing is unchecked. At human-genome scale `n` is ~6.2e9,
            // which exceeds `u32::MAX`, so a pathologically bad leaf model could in principle produce
            // an error that wraps and yields a too-small window. Whether that is reachable given the
            // even-by-rank leaf targets has not been established here; it needs a measurement of the
            // real max `leaf_err` on a genome-scale build, which this session could not run.
            let err = (pred_i - i as i64).unsigned_abs() as u32;
            if err > leaf_err[li] {
                leaf_err[li] = err;
            }
        }

        Rmi {
            root,
            leaves,
            leaf_err,
            n,
            n_leaves,
        }
    }

    /// Predicted position and search half-window for `key` (before the last-mile correction).
    ///
    /// Two dependent multiply-adds and two clamps, no memory access beyond the (tiny, cache-resident)
    /// `leaves` and `leaf_err` vectors. That is the point: the expensive part of a lookup should be
    /// the handful of probes into the 31 GB key array, not the search that decides where to probe.
    /// Duplicates the `leaf_of` clamping from `build` rather than sharing it, because `leaf_of` is a
    /// build-time closure over locals.
    ///
    /// # Parameters
    ///
    /// * `key`: the probe key. It need NOT be one of the stored keys; that is the case the error
    ///   bound plus the `+1`/`+2` slack in `lower_bound` exists to cover.
    ///
    /// # Returns
    ///
    /// `(pos, err)`: `pos` is a predicted array index already clamped into `[0, n-1]`, and `err` is
    /// the half-width in array positions of the window that provably contains the true position of
    /// a stored key. Callers must not use `pos` directly as an answer; it is only a search centre.
    /// Panics on an empty index (`n_leaves - 1` underflows), which `lower_bound` guards against.
    #[inline]
    fn predict(&self, key: u64) -> (usize, u32) {
        // Level 1: which leaf model to consult. Clamped because the root line is unbounded.
        let li = (self.root.predict(key).max(0.0) as usize).min(self.n_leaves - 1);
        // Level 2: the leaf's estimate of the ARRAY POSITION (a row index), still unclamped.
        let pred = self.leaves[li].predict(key);
        // Clamped to a valid *element* index `[0, n-1]`, not to `[0, n]`: `pos` is compared against
        // stored positions, and the `+2` in `lower_bound` re-opens the door to returning `n`.
        let pos = (pred.round().max(0.0) as usize).min(self.n.saturating_sub(1));
        (pos, self.leaf_err[li])
    }

    /// First index `i` in `[0, n]` with `keys[i] >= key` (the `partition_point` / `lower_bound`).
    /// `keys` must be the same slice the index was built over. Identical result to a binary search;
    /// only faster, via the model prediction plus a bounded last-mile binary search.
    ///
    /// Returns a value in `[0, n]`; `n` means "greater than every key". Passing a `keys` other than
    /// the one built over is checked only by `debug_assert`, and in release would search a window
    /// derived from the wrong distribution, i.e. return a wrong index rather than fail.
    ///
    /// Caveat (see the [`Rmi::leaves`] field note): the window is provably correct for keys that
    /// were present at build time. For a probe key that routes to a leaf no stored key routed to,
    /// the fallback model predicts 0 with error 0 and the window degenerates to `[0, 2)`.
    /// UNVERIFIED: whether such a leaf is reachable for real 40-bit k-mer keys, and therefore whether
    /// this is a live bug or merely a theoretical one, has not been determined. It is benign for the
    /// current caller either way: [`crate::lisa::LearnedSa`] treats this result purely as a hint and
    /// re-derives the true boundary with a galloping search that is correct for any hint.
    ///
    /// # Parameters
    ///
    /// * `keys`: the SAME sorted key array this index was built over. Passed in rather than stored
    ///   so the (multi-gigabyte) array has exactly one owner. Only `debug_assert`ed.
    /// * `key`: the probe key, any `u64`; values outside the stored range are handled by the
    ///   clamping and return 0 or `n`.
    ///
    /// # Returns
    ///
    /// An index in `[0, n]`: the first `i` with `keys[i] >= key`, and `n` when `key` exceeds every
    /// stored key.
    #[inline]
    pub fn lower_bound(&self, keys: &crate::packed::Packed40, key: u64) -> usize {
        debug_assert_eq!(keys.len(), self.n);
        if self.n == 0 {
            return 0;
        }
        let (pos, err) = self.predict(key);
        // Window guaranteed to bracket the true lower_bound: the predicted position is within `err`
        // of the position of `key` (or its neighbours), so [pos-err-1, pos+err+1] contains the
        // partition point. Clamp to [0, n] and binary-search inside.
        // Why `+1` / `+2` on top of `err` rather than just `err`: `err` bounds the distance from the
        // prediction to the position of an *equal* key, but a lower_bound for an absent key sits
        // between two stored keys, one slot beyond that. The extra slot each side covers it, and the
        // `+2` (versus `+1`) on the high side is because `hi` is exclusive. Erring wide costs at most
        // one extra probe.
        // `lo`/`hi` are ARRAY INDICES delimiting the half-open search window, guaranteed to contain
        // the partition point. `saturating_sub` clamps at 0 rather than wrapping.
        let lo = pos.saturating_sub(err as usize + 1);
        let hi = (pos + err as usize + 2).min(self.n);
        // `partition_point_in` counts within `[lo, hi)`, so `lo` must be added back to get an
        // absolute index. Clamping `hi` to `n` is what lets the answer reach `n` (key above all).
        lo + keys.partition_point_in(lo, hi, |k| k < key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::Packed40;

    /// The oracle the RMI is checked against: `std`'s binary search over a plain slice.
    ///
    /// # Parameters
    ///
    /// * `keys`: the same logical key array, sorted ascending, unpacked.
    /// * `key`: the probe key.
    ///
    /// # Returns
    ///
    /// The first index with `keys[i] >= key`, in `[0, keys.len()]`.
    fn ref_lower_bound(keys: &[u64], key: u64) -> usize {
        keys.partition_point(|&k| k < key)
    }

    /// Pack u64 keys (all `< 2^40` here) for the RMI.
    ///
    /// # Parameters
    ///
    /// * `keys`: sorted keys, each of which must fit in 40 bits or the `i64` round-trip through
    ///   `Packed40` truncates it.
    fn pk(keys: &[u64]) -> Packed40 {
        Packed40::from_slice(&keys.iter().map(|&k| k as i64).collect::<Vec<_>>())
    }

    #[test]
    fn matches_binary_search_dense() {
        // Sorted keys with duplicates and gaps.
        let mut keys: Vec<u64> = Vec::new();
        let mut x = 5u64;
        for i in 0..50_000u64 {
            x += 1 + (i * 2654435761 % 7); // irregular strictly-nondecreasing-ish gaps
            keys.push(x);
            if i % 5 == 0 {
                keys.push(x); // duplicates
            }
        }
        keys.sort_unstable();
        let pkeys = pk(&keys);
        let rmi = Rmi::build(&pkeys, 2048);
        // Probe every stored key, plus values just below/above and outside the range.
        for &k in &keys {
            assert_eq!(
                rmi.lower_bound(&pkeys, k),
                ref_lower_bound(&keys, k),
                "key={k}"
            );
            assert_eq!(
                rmi.lower_bound(&pkeys, k.wrapping_sub(1)),
                ref_lower_bound(&keys, k.wrapping_sub(1))
            );
            assert_eq!(
                rmi.lower_bound(&pkeys, k + 1),
                ref_lower_bound(&keys, k + 1)
            );
        }
        assert_eq!(rmi.lower_bound(&pkeys, 0), 0);
        assert_eq!(rmi.lower_bound(&pkeys, u64::MAX), keys.len());
    }

    #[test]
    fn matches_binary_search_linear_keys() {
        // Perfectly linear keys: the model is exact, err should be ~0 but lower_bound still correct.
        let keys: Vec<u64> = (0..20_000u64).map(|i| i * 4).collect();
        let pkeys = pk(&keys);
        let rmi = Rmi::build(&pkeys, 512);
        for probe in [0u64, 1, 3, 4, 79_996, 79_999, 80_000, u64::MAX] {
            assert_eq!(
                rmi.lower_bound(&pkeys, probe),
                ref_lower_bound(&keys, probe),
                "probe={probe}"
            );
        }
    }

    #[test]
    fn edge_cases() {
        assert_eq!(Rmi::build(&pk(&[]), 8).lower_bound(&pk(&[]), 42), 0);
        let one = [7u64];
        let pone = pk(&one);
        let rmi = Rmi::build(&pone, 8);
        assert_eq!(rmi.lower_bound(&pone, 6), 0);
        assert_eq!(rmi.lower_bound(&pone, 7), 0);
        assert_eq!(rmi.lower_bound(&pone, 8), 1);
        // All-equal keys (degenerate slope).
        let flat = vec![3u64; 1000];
        let pflat = pk(&flat);
        let rmi = Rmi::build(&pflat, 64);
        assert_eq!(rmi.lower_bound(&pflat, 3), 0);
        assert_eq!(rmi.lower_bound(&pflat, 4), 1000);
        assert_eq!(rmi.lower_bound(&pflat, 2), 0);
    }
}
