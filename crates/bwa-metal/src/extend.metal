// Banded local seed extension, `ksw_extend2`, one JOB per thread (issue #55, the extension seam).
//
// Compiled by the driver at run time from the string this file is embedded as, like the rescue
// kernel beside it. No Metal toolchain at build time.
//
// # Why this stage and not mate rescue
//
// Measured on a production run: extension is 30% of CPU time and submits ~10 800 jobs per call at
// -t4, while mate rescue is 4.5% and submits ~181. The GPU saturates around 32 000 concurrent jobs
// (one job per thread, so the batch size IS the thread count), so extension is a factor of 3 away
// and rescue a factor of 180. Aggregating the threads of one chunk closes the former and nothing
// closes the latter without deferring a stage whose output order is observable.
//
// # The port
//
// This is `ksw_extend2` transliterated, not reimagined. Every quantity below has the same name as in
// the scalar reference, and the loop structure is line for line the same, because the acceptance
// gate is equality of all six output fields against that function, job for job.
//
// Two things are deliberately NOT done here:
//
//   - the band clamp. The reference computes `max_ins`/`max_del` in `f64` and the project's backend
//     contract says a backend must take the clamped `w` from the host rather than re-derive it.
//     There is no floating point anywhere in this file, which is the rule for every DP kernel here.
//   - the query profile. The reference materialises `qp[tc * qlen + j]`; a thread here indexes the
//     `m * m` matrix directly. Same value, one less buffer, and it keeps the kernel general over any
//     matrix rather than assuming the uniform DNA shape the SIMD kernels require.
//
// # What must not drift
//
// - `mj = row_max > h ? mj : j` keeps the LAST column on a tie, the opposite of the rescue kernel's
//   rule. Copying one into the other is the mistake #54's trap 2 exists to catch.
// - `gscore` updates on `end == qlen && gscore <= h1`, non-strict, so the LAST such row wins.
// - the row breaks (`row_max == 0`, and z-drop) leave `beg`/`end` as they are; the band tightening
//   that follows them is skipped, which is why it sits after the breaks and not before.

#include <metal_stdlib>
using namespace metal;

struct EJob {
    uint q_off;      // query start in `seqs`
    uint q_len;
    uint t_off;      // target start in `seqs`
    uint t_len;
    int  h0;         // the seed's earned score, the DP's starting value
    int  w;          // band half-width, ALREADY clamped by the host
    uint rail;       // unused; keeps the struct a multiple of 8 bytes on both sides
    uint _pad;
};

struct ERes {
    int score;
    int qle;
    int tle;
    int gtle;
    int gscore;
    int max_off;
    int _pad0;
    int _pad1;
};

kernel void extend_fwd(
    device const uchar *seqs   [[buffer(0)]],
    device const EJob  *jobs   [[buffer(1)]],
    device ERes        *out    [[buffer(2)]],
    device int         *eh_h   [[buffer(3)]],
    device int         *eh_e   [[buffer(4)]],
    constant int       *mat    [[buffer(5)]],   // m * m, row-major, as the reference reads it
    constant int       &m      [[buffer(6)]],
    constant int       &o_del  [[buffer(7)]],
    constant int       &e_del  [[buffer(8)]],
    constant int       &o_ins  [[buffer(9)]],
    constant int       &e_ins  [[buffer(10)]],
    constant int       &zdrop  [[buffer(11)]],
    constant uint      &n_jobs [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_jobs) return;
    EJob j = jobs[gid];
    const int qlen = (int)j.q_len;
    const int tlen = (int)j.t_len;
    const int oe_del = o_del + e_del;
    const int oe_ins = o_ins + e_ins;

    // COLUMN-MAJOR rails, the layout the rescue kernel measured at 3.3x over the job-major one:
    // element `c` of thread `gid` lives at `c * n_jobs + gid`, so the 32 threads of a SIMD group
    // touch 128 consecutive bytes instead of 32 different cache lines.
    const ulong stride = (ulong)n_jobs;
    const ulong self = (ulong)gid;
#define EH(a, c) a[(ulong)(c) * stride + self]

    // The h0 ladder: H(-1, 0) = h0, then one insertion opened and extended while it stays positive.
    for (int c = 0; c <= qlen; ++c) { EH(eh_h, c) = 0; EH(eh_e, c) = 0; }
    EH(eh_h, 0) = j.h0;
    if (qlen >= 1) EH(eh_h, 1) = (j.h0 > oe_ins) ? (j.h0 - oe_ins) : 0;
    for (int c = 2; c <= qlen; ++c) {
        int prev = EH(eh_h, c - 1);
        if (prev <= e_ins) break;
        EH(eh_h, c) = prev - e_ins;
    }

    int best = j.h0, max_i = -1, max_j = -1, max_ie = -1, gscore = -1, max_off = 0;
    int beg = 0, end = qlen;

    for (int i = 0; i < tlen; ++i) {
        int f = 0, row_max = 0, mj = -1;
        int tc = (int)seqs[j.t_off + i];
        if (beg < i - j.w) beg = i - j.w;
        if (end > i + j.w + 1) end = i + j.w + 1;
        if (end > qlen) end = qlen;
        // Reaching column 0 of this row costs one deletion opened at the start and extended down.
        int h1 = (beg == 0) ? max(j.h0 - (o_del + e_del * (i + 1)), 0) : 0;

        int c = beg;
        for (; c < end; ++c) {
            int big_m = EH(eh_h, c);        // H(i-1, c-1)
            int e = EH(eh_e, c);            // E(i, c)
            EH(eh_h, c) = h1;               // H(i, c-1), for the next row
            int qc = (int)seqs[j.q_off + c];
            // The local restart: a diagonal predecessor of 0 means the alignment starts here, so the
            // substitution score is not added to it.
            big_m = (big_m != 0) ? (big_m + mat[tc * m + qc]) : 0;
            int h = max(max(big_m, e), f);
            h1 = h;
            // NON-strict: on a tie the LATER column wins, which is `ksw.cpp:491` and the opposite of
            // the rescue kernel.
            mj = (row_max > h) ? mj : c;
            row_max = max(row_max, h);
            // Both gaps open from `big_m`, not from `h`: bandedSWA's MAIN_CODE16 does the same.
            EH(eh_e, c) = max(e - e_del, max(big_m - oe_del, 0));
            f = max(f - e_ins, max(big_m - oe_ins, 0));
        }
        EH(eh_h, end) = h1;
        EH(eh_e, end) = 0;
        // `c == qlen` means the row ran to the end of the query, so `h1` is a global-alignment
        // candidate. Non-strict, so the last such row wins.
        if (c == qlen && gscore <= h1) { max_ie = i; gscore = h1; }
        if (row_max == 0) break;
        if (row_max > best) {
            best = row_max;
            max_i = i;
            max_j = mj;
            int off = abs(mj - i);
            if (off > max_off) max_off = off;
        } else if (zdrop > 0) {
            if (i - max_i > mj - max_j) {
                if (best - row_max - ((i - max_i) - (mj - max_j)) * e_del > zdrop) break;
            } else if (best - row_max - ((mj - max_j) - (i - max_i)) * e_ins > zdrop) {
                break;
            }
        }
        // Band tightening: drop the dead cells at both ends of the live range.
        int first_live = beg;
        while (first_live < end && EH(eh_h, first_live) == 0 && EH(eh_e, first_live) == 0) {
            ++first_live;
        }
        beg = first_live;
        int last_live = end;
        while (last_live >= beg && EH(eh_h, last_live) == 0 && EH(eh_e, last_live) == 0) {
            --last_live;
        }
        end = (last_live + 2 < qlen) ? (last_live + 2) : qlen;
    }

    ERes r;
    r.score = best;
    r.qle = max_j + 1;
    r.tle = max_i + 1;
    r.gtle = max_ie + 1;
    r.gscore = gscore;
    r.max_off = max_off;
    r._pad0 = 0;
    r._pad1 = 0;
    out[gid] = r;
#undef EH
}
