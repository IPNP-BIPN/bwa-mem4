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

use bwa_extend::{ksw_local_fwd, KswAlignResult};

/// One mate-rescue local-SW job: align `query` against `target` (both `0..=4` codes).
#[derive(Clone, Copy)]
pub struct KswJob<'a> {
    pub query: &'a [u8],
    pub target: &'a [u8],
}

/// One forward local-SW pass: the target is `target`, the query is `query`, and the pass reports the
/// max score reaching `>= minsc` and stops early once it reaches `endsc`. This is the unit the
/// vectorized kernel batches; [`batched_ksw_align2`] issues one batch for the forward pass and one for
/// the reverse (`KSW_XSTART`) start-recovery pass.
#[derive(Clone, Copy)]
struct FwdJob<'a> {
    query: &'a [u8],
    target: &'a [u8],
    minsc: i32,
    endsc: i32,
}

/// Batched local SW: `out[i]` equals [`bwa_extend::ksw_align2`] on `jobs[i]`. Structured exactly like
/// `ksw_align2`: a forward pass over every job, then a reverse pass over the truncated/reversed
/// prefixes of the qualifying jobs to recover the start coordinates. Both passes go through
/// [`fwd_local_sw_batch`], the single point the NEON kernel plugs into.
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
    // Forward pass over all jobs.
    let fwd: Vec<FwdJob> = jobs
        .iter()
        .map(|j| FwdJob {
            query: j.query,
            target: j.target,
            minsc,
            endsc: i32::MAX,
        })
        .collect();
    let fr = fwd_local_sw_batch(&fwd, m, mat, o_del, e_del, o_ins, e_ins, max_sc);

    let mut out: Vec<KswAlignResult> = fr
        .iter()
        .map(|&(score, te, qe, score2, te2)| KswAlignResult {
            score,
            qb: -1,
            qe,
            tb: -1,
            te,
            score2,
            te2,
        })
        .collect();

    // Reverse pass (KSW_XSTART): for each qualifying job, align the reversed prefixes ending at
    // (qe, te) and stop at `score`; the reversed end offsets give the start coords.
    let mut rbufs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut ridx: Vec<usize> = Vec::new();
    for (i, j) in jobs.iter().enumerate() {
        let (score, te, qe) = (out[i].score, out[i].te, out[i].qe);
        if score >= minsc && qe >= 0 {
            let qrev: Vec<u8> = j.query[..=qe as usize].iter().rev().copied().collect();
            let trev: Vec<u8> = j.target[..=te as usize].iter().rev().copied().collect();
            rbufs.push((qrev, trev));
            ridx.push(i);
        }
    }
    let rjobs: Vec<FwdJob> = rbufs
        .iter()
        .zip(ridx.iter())
        .map(|((q, t), &i)| FwdJob {
            query: q,
            target: t,
            minsc: i32::MAX,
            endsc: out[i].score,
        })
        .collect();
    let rr = fwd_local_sw_batch(&rjobs, m, mat, o_del, e_del, o_ins, e_ins, max_sc);
    for (k, &i) in ridx.iter().enumerate() {
        let (rscore, rte, rqe, _, _) = rr[k];
        if out[i].score == rscore {
            out[i].tb = out[i].te - rte;
            out[i].qb = out[i].qe - rqe;
        }
    }
    out
}

/// Batched forward local-SW pass: `out[i] = (score, te, qe, score2, te2)` for `jobs[i]`, each equal to
/// [`ksw_local_fwd`]. The NEON inter-sequence kernel plugs in here; the scalar per-job loop is the
/// portable fallback and the byte-identity source of truth.
#[allow(clippy::too_many_arguments)]
fn fwd_local_sw_batch(
    jobs: &[FwdJob],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    max_sc: i32,
) -> Vec<(i32, i32, i32, i32, i32)> {
    jobs.iter()
        .map(|j| {
            ksw_local_fwd(
                j.query, j.target, m, mat, o_del, e_del, o_ins, e_ins, j.minsc, j.endsc, max_sc,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_extend::ksw_align2;

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
