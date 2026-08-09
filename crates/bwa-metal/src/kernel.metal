// Mate-rescue forward local Smith-Waterman, one JOB per thread (issue #55).
//
// This file is embedded in the binary as a string and compiled by the Metal driver at run time
// (`newLibraryWithSource:`). That is what lets the shipped binary need no Metal toolchain at build
// time, and it is why there is no `.metallib` anywhere in this repo.
//
// # The mapping, and why it is one job per thread
//
// The CPU kernels are INTER-SEQUENCE: sixteen jobs share one vector, one lane each. A GPU wants the
// same idea taken to its limit, one job per thread, because that keeps every thread's control flow
// independent and needs no cross-lane communication at all. No `simd_shuffle`, no threadgroup
// reduction, nothing that could let one job's data reach another's result. That is not only simpler,
// it is what makes the byte-identity argument short: each thread computes exactly what
// `ksw_local_fwd` computes for its own job, and threads cannot interact.
//
// The cost of that choice is the padding tax the CPU pays disappears (no PAD lanes, no dead cells),
// and in exchange each thread needs its own H/E rails in device memory rather than in registers.
//
// # Arithmetic
//
// Plain 32-bit signed, mirroring `ksw_local_fwd`, NOT the CPU's saturating u8. The u8 form would run
// four cells per lane in a `uchar4` and the ceiling probe measured that at ~330 Gcell/s, but it
// carries the bias, the `U8_SCORE_LIMIT` guard and the saturation edge cases, and none of that is
// worth taking on before a correct backend exists. This is the correct one; the packing is the next
// optimisation, and `scripts/msl_probe.sh` already measured what it is worth.
//
// # What must not drift
//
// - the row argmax uses STRICT `>`, so the FIRST maximum in a row keeps `qe` (`ksw.cpp:216-218`).
//   The extension DP uses the opposite rule; do not copy one into the other.
// - the padded query columns score 0 and therefore carry the diagonal through. They are observable
//   through `score2`, so they are computed, not skipped.
// - `endsc` freezes the job the instant a row max reaches it, and the row that froze it is the last
//   row whose maximum counts.

#include <metal_stdlib>
using namespace metal;

// One job's geometry, filled by the host. Offsets index the single flat sequence buffer, which on
// Apple Silicon is a shared `MTLBuffer` the CPU wrote in place: no copy happens anywhere.
struct Job {
    uint q_off;    // query start in `seqs`
    uint q_len;    // real query bases
    uint q_pad;    // padded column count, ksw's `slen * lanes` (columns q_len..q_pad score 0)
    uint t_off;    // target start in `seqs`
    uint t_len;    // target rows
    int  endsc;    // stop as soon as a row max reaches this; INT_MAX disables
    uint kind;     // 1 = this job's ceiling fits a byte and it belongs to `rescue_fwd_u8`, 0 = it
                   // belongs to `rescue_fwd`. Both kernels run over the whole grid and each returns
                   // immediately for the other's jobs: one wasted launch, and one host code path.
    uint _pad;     // keep the struct 32-byte aligned on both sides of the boundary
};

// What the host gets back per job. `score2`/`te2` are NOT computed here: they come from `rowmax` on
// the CPU, through the same `SuboptimalTracker` the scalar and NEON paths use, so the merge rule and
// the exclusion window cannot drift between three implementations of a subtle rule.
struct Res {
    int score;
    int te;
    int qe;
    int limit;     // last row this job actually processed, inclusive (-1 = none)
};

kernel void rescue_fwd(
    device const uchar *seqs      [[buffer(0)]],
    device const Job   *jobs      [[buffer(1)]],
    device Res         *out       [[buffer(2)]],
    device int         *h_prev    [[buffer(3)]],
    device int         *h_cur     [[buffer(4)]],
    device int         *e_rail    [[buffer(5)]],
    device int         *rowmax    [[buffer(6)]],
    constant int       &mtch      [[buffer(7)]],
    constant int       &mispen    [[buffer(8)]],   // positive magnitude
    constant int       &npen      [[buffer(9)]],   // positive magnitude, bwa's N score is -1
    constant int       &oe_del    [[buffer(10)]],
    constant int       &e_del     [[buffer(11)]],
    constant int       &oe_ins    [[buffer(12)]],
    constant int       &e_ins     [[buffer(13)]],
    constant uint      &rail_qmax [[buffer(14)]],  // unused, kept for binding stability
    constant uint      &rail_tmax [[buffer(15)]],  // unused, kept for binding stability
    constant uint      &n_jobs    [[buffer(16)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_jobs) return;
    Job j = jobs[gid];
    if (j.kind != 0) return;   // this one belongs to the u8 kernel

    // COLUMN-MAJOR rails: element `c` of thread `gid` lives at `c * n_jobs + gid`, not at
    // `gid * qmax + c`. The job-major layout is the obvious one and it is the wrong one: adjacent
    // threads would then be `qmax * 4` bytes apart, so every one of the 32 threads in a SIMD group
    // would touch a different cache line on every access. Column-major puts the group's 32 accesses
    // in 128 consecutive bytes. Measured on this M4 Max, same kernel, same data: **2.30 Gcell/s
    // job-major against 7.66 column-major, 3.3x from the indexing alone**.
    device int *hp = h_prev;
    device int *hc = h_cur;
    device int *ev = e_rail;
    device int *rm = rowmax;
    const ulong stride = (ulong)n_jobs;
    const ulong self = (ulong)gid;

    for (uint c = 0; c < j.q_pad; ++c) {
        ulong o = (ulong)c * stride + self;
        hp[o] = 0; hc[o] = 0; ev[o] = 0;
    }

    int gmax = 0, te = -1, qe = 0;
    int limit = (int)j.t_len - 1;

    for (uint i = 0; i < j.t_len; ++i) {
        uchar t = seqs[j.t_off + i];
        int f = 0, h_diag = 0, imax = 0, imax_col = 0;
        for (uint c = 0; c < j.q_pad; ++c) {
            int diag;
            if (c >= j.q_len) {
                // ksw profile padding: score 0, so the cell carries the diagonal through. Dropping
                // these columns would be the obvious simplification and it changes `score2`.
                diag = h_diag;
            } else {
                uchar q = seqs[j.q_off + c];
                // bwa's matrix collapsed to three numbers: +match on the diagonal, -mismatch off it,
                // and -1 wherever either side is N (code 4), which wins over the equality test.
                int s = (t == 4 || q == 4) ? -npen : (t == q ? mtch : -mispen);
                diag = max(h_diag + s, 0);
            }
            const ulong o = (ulong)c * stride + self;
            int h = max(max(diag, ev[o]), f);
            // STRICT `>`: a tie keeps the earlier column, which is what fixes `qe`. Scanning columns
            // in increasing order, this leaves `imax_col` at the SMALLEST column attaining the row
            // maximum, which is what `ksw_local_fwd` recovers with its post-hoc scan over the saved
            // H row. Same answer, without saving the row.
            if (h > imax) { imax = h; imax_col = (int)c; }
            h_diag = hp[o];
            hc[o] = h;
            // Both gaps open from `h`, exactly as `ksw_local_fwd` does. The CPU vector kernels open
            // F from `mfe` instead, which is a proven-equivalent reassociation that shortens their
            // critical path; there is no such path to shorten here, so this mirrors the reference
            // literally rather than reproducing an optimisation whose proof would have to be
            // re-checked in a second language.
            ev[o] = max(max(ev[o] - e_del, h - oe_del), 0);
            f     = max(max(f - e_ins,     h - oe_ins), 0);
        }
        rm[(ulong)i * stride + self] = imax;
        if (imax > gmax) {
            gmax = imax;
            te = (int)i;
            qe = imax_col;
            if (gmax >= j.endsc) { limit = (int)i; break; }
        }
        // Swap the rails rather than copy them.
        device int *tmp = hp; hp = hc; hc = tmp;
    }

    Res r;
    r.score = gmax;
    r.te = te;
    r.qe = (te >= 0) ? qe : -1;
    r.limit = limit;
    out[gid] = r;
}


// ---------------------------------------------------------------------------------------------
// The same DP with u8 rails.
//
// Identical control flow to `rescue_fwd`, and deliberately so: it is a copy with the rail element
// type narrowed from `int` to `uchar` and the three `max(0, .)` clamps folded into the saturating
// ops MSL provides (`addsat`, `subsat`, verified by `scripts/msl_probe.sh` to clamp at 255 and 0
// exactly). The scores themselves stay in `int` registers, so nothing about `gmax`, `te` or `qe`
// changes.
//
// Why: the 32-bit version moves 3 loads and 2 stores of FOUR bytes per cell through device memory,
// and that traffic is what holds it to 2.3% of the measured register-only ceiling. One byte per
// value is a 4x cut in exactly the resource that binds. Measured on this M4 Max, same probe, same
// data, the whole progression:
//
//     job-major rails, 32-bit      2.30 Gcell/s
//     column-major rails, 32-bit   7.66          (3.3x, from the indexing alone)
//     column-major rails, u8      10.95          (+43%, this kernel)
//     registers only, uchar4     ~330            (the ceiling `scripts/msl_probe.sh` measures)
//
// So 3.3% of the ceiling, and about one CPU thread's worth. The rest of the gap is the packing: four
// cells per lane in a `uchar4`, which is what the ceiling probe actually measures and which this
// kernel does not do.
//
// Precondition, checked by the host and NOT here: every H/E/F value of these jobs must fit a byte,
// i.e. `min(qlen, tlen) * max_sc < U8_SCORE_LIMIT`. The host runs the 32-bit kernel for anything
// else, which is the same binning `fwd_local_sw_batch` does on the CPU. Violating it would silently
// saturate rather than trap, which is why the split lives on the host where it can be asserted.
kernel void rescue_fwd_u8(
    device const uchar *seqs      [[buffer(0)]],
    device const Job   *jobs      [[buffer(1)]],
    device Res         *out       [[buffer(2)]],
    device uchar       *h_prev    [[buffer(3)]],
    device uchar       *h_cur     [[buffer(4)]],
    device uchar       *e_rail    [[buffer(5)]],
    device int         *rowmax    [[buffer(6)]],
    constant int       &mtch      [[buffer(7)]],
    constant int       &mispen    [[buffer(8)]],
    constant int       &npen      [[buffer(9)]],
    constant int       &oe_del    [[buffer(10)]],
    constant int       &e_del     [[buffer(11)]],
    constant int       &oe_ins    [[buffer(12)]],
    constant int       &e_ins     [[buffer(13)]],
    constant uint      &rail_qmax [[buffer(14)]],
    constant uint      &rail_tmax [[buffer(15)]],
    constant uint      &n_jobs    [[buffer(16)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_jobs) return;
    Job j = jobs[gid];
    if (j.kind != 1) return;   // this one belongs to the 32-bit kernel

    device uchar *hp = h_prev;
    device uchar *hc = h_cur;
    device uchar *ev = e_rail;
    device int   *rm = rowmax;
    const ulong stride = (ulong)n_jobs;
    const ulong self = (ulong)gid;

    const uchar mt = (uchar)mtch, mp = (uchar)mispen, np = (uchar)npen;
    const uchar edel = (uchar)e_del, oedel = (uchar)oe_del;
    const uchar eins = (uchar)e_ins, oeins = (uchar)oe_ins;

    for (uint c = 0; c < j.q_pad; ++c) {
        ulong o = (ulong)c * stride + self;
        hp[o] = 0; hc[o] = 0; ev[o] = 0;
    }

    int gmax = 0, te = -1, qe = 0;
    int limit = (int)j.t_len - 1;

    for (uint i = 0; i < j.t_len; ++i) {
        uchar t = seqs[j.t_off + i];
        uchar f = 0, h_diag = 0;
        int imax = 0, imax_col = 0;
        for (uint c = 0; c < j.q_pad; ++c) {
            uchar diag;
            if (c >= j.q_len) {
                diag = h_diag;
            } else {
                uchar q = seqs[j.q_off + c];
                // `subsat` IS the `max(0, .)` the reference writes explicitly, and `addsat` cannot
                // overflow under the host's precondition. Same three cases as the 32-bit kernel.
                diag = (t == 4 || q == 4) ? subsat(h_diag, np)
                     : (t == q          ? addsat(h_diag, mt)
                                        : subsat(h_diag, mp));
            }
            const ulong o = (ulong)c * stride + self;
            uchar h = max(max(diag, ev[o]), f);
            if ((int)h > imax) { imax = (int)h; imax_col = (int)c; }
            h_diag = hp[o];
            hc[o] = h;
            ev[o] = max(subsat(ev[o], edel), subsat(h, oedel));
            f     = max(subsat(f,     eins), subsat(h, oeins));
        }
        rm[(ulong)i * stride + self] = imax;
        if (imax > gmax) {
            gmax = imax;
            te = (int)i;
            qe = imax_col;
            if (gmax >= j.endsc) { limit = (int)i; break; }
        }
        device uchar *tmp = hp; hp = hc; hc = tmp;
    }

    Res r;
    r.score = gmax;
    r.te = te;
    r.qe = (te >= 0) ? qe : -1;
    r.limit = limit;
    out[gid] = r;
}
