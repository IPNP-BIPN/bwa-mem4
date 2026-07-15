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
    /// Least-squares fit of `(keys[i] as f64, targets[i] as f64)`. Falls back to a flat model (slope
    /// 0, intercept = mean target) when the keys are degenerate (all equal / single point), which
    /// keeps prediction finite and the error bound correct.
    fn fit(keys: &[u64], targets: &[f64]) -> Self {
        let n = keys.len();
        if n == 0 {
            return LinearModel::default();
        }
        let mean_target = targets.iter().sum::<f64>() / n as f64;
        if n == 1 {
            return LinearModel {
                slope: 0.0,
                intercept: mean_target,
            };
        }
        // Use the min key as the origin to keep the sums well-conditioned on genome-scale keys.
        let x0 = keys[0] as f64;
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        for (i, &k) in keys.iter().enumerate() {
            let x = k as f64 - x0;
            let y = targets[i];
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let nf = n as f64;
        let denom = nf * sxx - sx * sx;
        if denom.abs() < f64::EPSILON {
            return LinearModel {
                slope: 0.0,
                intercept: mean_target,
            };
        }
        let slope = (nf * sxy - sx * sy) / denom;
        // intercept is relative to x0; fold x0 back in: pos = slope*(key - x0) + b  =>
        // pos = slope*key + (b - slope*x0).
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
    pub fn build(keys: &[u64], n_leaves: usize) -> Self {
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
        debug_assert!(keys.windows(2).all(|w| w[0] <= w[1]), "keys must be sorted");
        let n_leaves = n_leaves.clamp(1, n);

        // Root model maps key -> leaf index. Train it on (key_i -> leaf target), where the target
        // leaf is the position scaled into [0, n_leaves): leaf_target = i * n_leaves / n.
        let root_targets: Vec<f64> = (0..n)
            .map(|i| (i as f64) * (n_leaves as f64) / (n as f64))
            .collect();
        let root = LinearModel::fit(keys, &root_targets);

        // Assign each key to a leaf via the (clamped) root prediction, then fit each leaf on the
        // absolute positions of the keys routed to it.
        let leaf_of = |key: u64| -> usize {
            let p = root.predict(key);
            (p.max(0.0) as usize).min(n_leaves - 1)
        };
        let mut leaf_keys: Vec<Vec<u64>> = vec![Vec::new(); n_leaves];
        let mut leaf_pos: Vec<Vec<f64>> = vec![Vec::new(); n_leaves];
        for (i, &k) in keys.iter().enumerate() {
            let li = leaf_of(k);
            leaf_keys[li].push(k);
            leaf_pos[li].push(i as f64);
        }
        let mut leaves = Vec::with_capacity(n_leaves);
        for li in 0..n_leaves {
            leaves.push(LinearModel::fit(&leaf_keys[li], &leaf_pos[li]));
        }

        // Per-leaf error bound: the max over the leaf's keys of |round(predict) - actual position|.
        // Guarantees the true position lies in [pred - err, pred + err].
        let mut leaf_err = vec![0u32; n_leaves];
        for (i, &k) in keys.iter().enumerate() {
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
    pub fn lower_bound(&self, keys: &[u64], key: u64) -> usize {
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
        lo + keys[lo..hi].partition_point(|&k| k < key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_lower_bound(keys: &[u64], key: u64) -> usize {
        keys.partition_point(|&k| k < key)
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
        let rmi = Rmi::build(&keys, 2048);
        // Probe every stored key, plus values just below/above and outside the range.
        for &k in &keys {
            assert_eq!(rmi.lower_bound(&keys, k), ref_lower_bound(&keys, k), "key={k}");
            assert_eq!(
                rmi.lower_bound(&keys, k.wrapping_sub(1)),
                ref_lower_bound(&keys, k.wrapping_sub(1))
            );
            assert_eq!(rmi.lower_bound(&keys, k + 1), ref_lower_bound(&keys, k + 1));
        }
        assert_eq!(rmi.lower_bound(&keys, 0), 0);
        assert_eq!(rmi.lower_bound(&keys, u64::MAX), keys.len());
    }

    #[test]
    fn matches_binary_search_linear_keys() {
        // Perfectly linear keys: the model is exact, err should be ~0 but lower_bound still correct.
        let keys: Vec<u64> = (0..20_000u64).map(|i| i * 4).collect();
        let rmi = Rmi::build(&keys, 512);
        for probe in [0u64, 1, 3, 4, 79_996, 79_999, 80_000, u64::MAX] {
            assert_eq!(
                rmi.lower_bound(&keys, probe),
                ref_lower_bound(&keys, probe),
                "probe={probe}"
            );
        }
    }

    #[test]
    fn edge_cases() {
        assert_eq!(Rmi::build(&[], 8).lower_bound(&[], 42), 0);
        let one = [7u64];
        let rmi = Rmi::build(&one, 8);
        assert_eq!(rmi.lower_bound(&one, 6), 0);
        assert_eq!(rmi.lower_bound(&one, 7), 0);
        assert_eq!(rmi.lower_bound(&one, 8), 1);
        // All-equal keys (degenerate slope).
        let flat = vec![3u64; 1000];
        let rmi = Rmi::build(&flat, 64);
        assert_eq!(rmi.lower_bound(&flat, 3), 0);
        assert_eq!(rmi.lower_bound(&flat, 4), 1000);
        assert_eq!(rmi.lower_bound(&flat, 2), 0);
    }
}
