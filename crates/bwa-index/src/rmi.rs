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

/// A single-variable linear model `pos ~= slope * key + intercept`, evaluated in `f64`.
#[derive(Clone, Copy, Debug, Default)]
struct LinearModel {
    slope: f64,
    intercept: f64,
}

impl LinearModel {

    /// Least-squares fit from precomputed sums over points `(x_i - x0, y_i)`: `n` points, `sx=Σx`,
    /// `sy=Σy`, `sxx=Σx²`, `sxy=Σxy`, with `x0` the origin folded back in. Bit-identical to [`fit`]
    /// (which computes these very sums), but lets the caller stream the points without materializing
    /// per-leaf key/target arrays — essential at genome scale.
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
        let denom = nf * sxx - sx * sx;
        if denom.abs() < f64::EPSILON {
            return LinearModel {
                slope: 0.0,
                intercept: sy / nf,
            };
        }
        let slope = (nf * sxy - sx * sy) / denom;
        let b = (sy - slope * sx) / nf;
        LinearModel {
            slope,
            intercept: b - slope * x0,
        }
    }

    #[inline]
    fn predict(&self, key: u64) -> f64 {
        self.slope * key as f64 + self.intercept
    }
}

/// A two-level recursive model index over an immutable sorted `u64` key array.
#[derive(Clone, Debug)]
pub struct Rmi {
    root: LinearModel,
    leaves: Vec<LinearModel>,
    /// Per-leaf half-width of the search window: `max |predicted - actual|`, rounded up.
    leaf_err: Vec<u32>,
    n: usize,
    n_leaves: usize,
}

impl Rmi {
    /// Build an RMI over `keys`, which **must be sorted ascending**. `n_leaves` is the number of
    /// second-level models (clamped to `[1, keys.len()]`); a few thousand leaves per million keys is
    /// a good default. Empty input yields an index that always reports position 0.
    pub fn build(keys: &crate::packed::Packed40, n_leaves: usize) -> Self {
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
        let n_leaves = n_leaves.clamp(1, n);

        // Root model maps key -> leaf index, trained on (key_i -> leaf_target = i * n_leaves / n).
        // Accumulate the least-squares sums directly (no n-length target array) with x0 = keys[0].
        let x0r = keys.get(0) as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let scale = n_leaves as f64 / n as f64;
        for i in 0..n {
            let k = keys.get(i);
            let x = k as f64 - x0r;
            let y = i as f64 * scale;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let root = LinearModel::from_sums(n, sx, sy, sxx, sxy, x0r);

        let leaf_of = |key: u64| -> usize {
            let p = root.predict(key);
            (p.max(0.0) as usize).min(n_leaves - 1)
        };

        // Fit each leaf on the absolute positions of the keys routed to it. `leaf_of` is monotonic
        // non-decreasing in the (sorted) key (the root slope is non-negative for position-vs-key), so
        // the keys of each leaf form one contiguous run — we stream the sums per run instead of
        // bucketing every key. Result-identical to fitting `leaf_pos[li]` explicitly.
        let mut leaves = vec![LinearModel::default(); n_leaves];
        let mut i = 0usize;
        while i < n {
            let li = leaf_of(keys.get(i));
            let x0 = keys.get(i) as f64;
            let (mut lsx, mut lsy, mut lsxx, mut lsxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
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
        let mut leaf_err = vec![0u32; n_leaves];
        for i in 0..n {
            let k = keys.get(i);
            let li = leaf_of(k);
            let pred = leaves[li].predict(k);
            let pred_i = pred.round().max(0.0) as i64;
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
    #[inline]
    fn predict(&self, key: u64) -> (usize, u32) {
        let li = (self.root.predict(key).max(0.0) as usize).min(self.n_leaves - 1);
        let pred = self.leaves[li].predict(key);
        let pos = (pred.round().max(0.0) as usize).min(self.n.saturating_sub(1));
        (pos, self.leaf_err[li])
    }

    /// First index `i` in `[0, n]` with `keys[i] >= key` (the `partition_point` / `lower_bound`).
    /// `keys` must be the same slice the index was built over. Identical result to a binary search;
    /// only faster, via the model prediction plus a bounded last-mile binary search.
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
        let lo = pos.saturating_sub(err as usize + 1);
        let hi = (pos + err as usize + 2).min(self.n);
        lo + keys.partition_point_in(lo, hi, |k| k < key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::Packed40;

    fn ref_lower_bound(keys: &[u64], key: u64) -> usize {
        keys.partition_point(|&k| k < key)
    }

    /// Pack u64 keys (all `< 2^40` here) for the RMI.
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
            assert_eq!(rmi.lower_bound(&pkeys, k), ref_lower_bound(&keys, k), "key={k}");
            assert_eq!(
                rmi.lower_bound(&pkeys, k.wrapping_sub(1)),
                ref_lower_bound(&keys, k.wrapping_sub(1))
            );
            assert_eq!(rmi.lower_bound(&pkeys, k + 1), ref_lower_bound(&keys, k + 1));
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
