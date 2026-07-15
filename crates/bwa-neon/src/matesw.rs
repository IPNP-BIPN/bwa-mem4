//! Lane-batched local Smith-Waterman for **mate rescue** (`kswv` in bwa-mem2).
//!
//! Mate rescue realigns a mate read against an insert-size window when its pairing is missing. Each
//! such alignment is an independent full local SW ([`bwa_extend::ksw_align2`]) returning
//! `{score, qb, qe, tb, te, score2, te2}` (no CIGAR/traceback). bwa-mem2 vectorizes this
//! **inter-sequence**: many rescue jobs packed into SoA lanes (16 x u8 / 8 x i16 on NEON), each lane
//! a different job, length-sorted so lanes finish together. This mirrors [`crate::batched`] (seed
//! extension) but with the local-SW recurrence and the two-phase start recovery (`KSW_XSTART`).
//!
//! [`batched_ksw_align2`] returns one [`KswAlignResult`] per job, each byte-identical to
//! [`bwa_extend::ksw_align2`] on that job. The scalar per-job loop is the portable fallback and the
//! source of truth the NEON kernels are validated against (`matesw_equals_scalar`).

use bwa_extend::{ksw_align2, KswAlignResult};

/// One mate-rescue local-SW job: align `query` against `target` (both `0..=4` codes).
#[derive(Clone, Copy)]
pub struct KswJob<'a> {
    pub query: &'a [u8],
    pub target: &'a [u8],
}

/// Batched local SW: `out[i]` equals [`ksw_align2`] on `jobs[i]`. Dispatches to the NEON kernel where
/// available (length/score-binned into u8 / i16 lanes), else the scalar per-job fallback.
#[allow(clippy::too_many_arguments)]
pub fn batched_ksw_align2(
    jobs: &[KswJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    max_sc: i32,
) -> Vec<KswAlignResult> {
    // NEON kernel dispatch lands here (Stage B); scalar fallback is byte-identical meanwhile.
    batched_ksw_align2_scalar(jobs, m, mat, o_del, e_del, o_ins, e_ins, minsc, max_sc)
}

/// Portable fallback: run [`ksw_align2`] per job. Byte-identical by construction; the source of truth.
#[allow(clippy::too_many_arguments)]
pub fn batched_ksw_align2_scalar(
    jobs: &[KswJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    max_sc: i32,
) -> Vec<KswAlignResult> {
    jobs.iter()
        .map(|j| {
            ksw_align2(
                j.query, j.target, m, mat, o_del, e_del, o_ins, e_ins, minsc, max_sc,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bwa 5x5 score matrix: match `a`, mismatch `-b`, N row/col `-1`.
    fn scmat(a: i8, b: i8) -> Vec<i8> {
        let mut mat = vec![0i8; 25];
        let mut k = 0;
        for i in 0..4 {
            for j in 0..4 {
                mat[k] = if i == j { a } else { -b };
                k += 1;
            }
            mat[k] = -1;
            k += 1;
        }
        for _ in 0..5 {
            mat[k] = -1;
            k += 1;
        }
        mat
    }

    /// Random mate-rescue-shaped jobs (short query, longer window, some shared substring so the SW
    /// finds a real local alignment), then assert the batched kernel matches per-job `ksw_align2` on
    /// every field. This is the byte-identity gate for the NEON kernels.
    #[test]
    fn matesw_equals_scalar() {
        let mat = scmat(1, 4);
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let (minsc, max_sc) = (19, 1);

        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        // Build a pool of jobs and their owned buffers.
        let mut qbufs: Vec<Vec<u8>> = Vec::new();
        let mut tbufs: Vec<Vec<u8>> = Vec::new();
        for _ in 0..400 {
            let qlen = 40 + (next() % 111) as usize; // 40..=150
            let tlen = qlen + (next() % 400) as usize; // window >= query
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
            // Embed a mutated copy of the query into the target so a local alignment exists.
            if next() % 4 != 0 {
                let at = (next() as usize) % (tlen - qlen + 1);
                for k in 0..qlen {
                    t[at + k] = q[k];
                }
                // a couple of substitutions
                for _ in 0..(next() % 4) {
                    let p = (next() as usize) % qlen;
                    q[p] = (next() % 4) as u8;
                }
            }
            qbufs.push(q);
            tbufs.push(t);
        }
        let jobs: Vec<KswJob> = qbufs
            .iter()
            .zip(tbufs.iter())
            .map(|(q, t)| KswJob {
                query: q.as_slice(),
                target: t.as_slice(),
            })
            .collect();

        let batched =
            batched_ksw_align2(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, max_sc);
        for (i, j) in jobs.iter().enumerate() {
            let want = ksw_align2(
                j.query, j.target, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, max_sc,
            );
            assert_eq!(batched[i], want, "job {i} (qlen {})", j.query.len());
        }
    }
}
