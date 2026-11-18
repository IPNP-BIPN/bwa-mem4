// Cooperative banded seed extension (`ksw_extend2`) on Apple GPUs.
//
// One job per 16 lanes (two jobs per 32-wide simdgroup), row-synchronous. Lane `li` owns the
// contiguous query columns [li*C, li*C + C). The arrays `eh.h` / `eh.e` (qlen+1 entries) live in
// those lanes' registers for the whole job, stale entries included, because ksw_extend2 reads stale
// `eh` slots when `end` grows back after shrinking: a sliding window would not be identical.
//
// The only horizontal dependency of a row is F. With f = 0 at `beg` and the recurrence
//     F(j+1) = max(F(j) - e_ins, max(M(j) - oe_ins, 0))
// unrolled, F(j) = max(0, max_{beg<=k<j} (M(k) + k*e_ins) - o_ins - j*e_ins). That is an exclusive
// prefix max, computed per lane then across lanes with 4 shuffles. Exact in int32: no saturation,
// no fallback tier, the same integers as the scalar reference.
//
// Tie rules, each copied from sw.rs (ksw_extend2):
//   row max   : the LAST column reaching it wins (`mj = m > h ? mj : j`),
//   global max: the FIRST row wins (strict `>`), max_off only moves on a strictly better row,
//   gscore    : the LAST row wins (`gscore <= h1`).
//
// The substitution score is computed, not looked up: the host only sends matrices of bwa's uniform
// shape (`a` on the diagonal, `-b` off it, one value for every cell touching N, code 4). The row
// max and its column travel as one key `(H << 8) | j`; the host only sends jobs whose scores fit.
#include <metal_stdlib>
using namespace metal;

struct EJob {
    uint q_off;
    uint q_len;
    uint t_off;
    uint t_len;
    int h0;
    int w;    // band already clamped on the host, in f64 exactly as ksw_extend2 does
    uint out; // index of this job's result slot
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

struct Params {
    int o_del;
    int e_del;
    int o_ins;
    int e_ins;
    int zdrop;
    uint n_jobs;
    int sc_match;    // mat[i][i], i < 4
    int sc_mismatch; // mat[i][j], i != j, both < 4
    int sc_n;        // any cell with an N (code 4) on either side
    int _pad0;
    int _pad1;
    int _pad2;
};

constant int NEG = -0x40000000;

// Segmented 16-lane max: xor masks <= 8 never leave the job's half of the simdgroup.
static inline int seg_max(int x) {
    x = max(x, simd_shuffle_xor(x, ushort(8)));
    x = max(x, simd_shuffle_xor(x, ushort(4)));
    x = max(x, simd_shuffle_xor(x, ushort(2)));
    x = max(x, simd_shuffle_xor(x, ushort(1)));
    return x;
}

// This job's 16 ballot bits: bit k is lane k of the job.
static inline uint half_ballot(bool pred, ushort hbase) {
    return uint(ulong(simd_ballot(pred)) >> hbase) & 0xffffu;
}

template <int C>
static inline void extend_job(device const uchar* seqs, device const EJob& jb, device ERes* res,
                              constant Params& p, ushort lane) {
    const ushort li = lane & 15;
    const ushort hbase = lane & 16;
    const int qlen = int(jb.q_len);
    const int tlen = int(jb.t_len);
    const int h0 = jb.h0;
    const int w = jb.w;
    const int e_del = p.e_del;
    const int e_ins = p.e_ins;
    const int o_ins = p.o_ins;
    const int oe_del = p.o_del + p.e_del;
    const int oe_ins = p.o_ins + p.e_ins;
    const int base = int(li) * C;
    const int jei0 = base * e_ins;

    uchar qb[C];
    int ehh[C];
    int ehe[C];
    int mm[C];
    // First row, closed form of ksw_extend2's fill: eh[0].h = h0, eh[j].h = max(h0 - oe_ins -
    // (j-1)*e_ins, 0) for 1 <= j <= qlen, everything else zero (the C callocs).
    for (int c = 0; c < C; ++c) {
        const int j = base + c;
        qb[c] = j < qlen ? seqs[jb.q_off + uint(j)] : uchar(0);
        const int v = j == 0 ? h0 : max(h0 - oe_ins - (j - 1) * e_ins, 0);
        ehh[c] = j <= qlen ? v : 0;
        ehe[c] = 0;
    }

    int best = h0, max_i = -1, max_j = -1, max_ie = -1, gscore = -1, max_off = 0;
    int beg = 0, end = qlen;
    device const uchar* tg = seqs + jb.t_off;
    // Target bases one row ahead, so the load never sits on a row's critical path.
    int t_next = tlen > 0 ? int(tg[0]) : 0;

    for (int i = 0; i < tlen; ++i) {
        const int t = t_next;
        if (i + 1 < tlen) t_next = int(tg[i + 1]);
        if (beg < i - w) beg = i - w;
        if (end > i + w + 1) end = i + w + 1;
        if (end > qlen) end = qlen;
        const int h1_init = beg == 0 ? max(h0 - (p.o_del + e_del * (i + 1)), 0) : 0;
        const bool ran = beg < end;
        // Score of a query base against this row's target base `t`.
        const int s_eq = t >= 4 ? p.sc_n : p.sc_match;
        const int s_ne = t >= 4 ? p.sc_n : p.sc_mismatch;

        // Pass 1: M of every column, and this lane's max of M(k) + k*e_ins over the active ones.
        // Active columns of this lane, as a range of c: [c_lo, c_hi).
        const int c_lo = clamp(beg - base, 0, C);
        const int c_hi = clamp(end - base, 0, C);
        int lane_g = NEG;
        int je = jei0; // j * e_ins, stepped rather than multiplied
        for (int c = 0; c < C; ++c) {
            const int q = int(qb[c]);
            const int s = q >= 4 ? p.sc_n : (q == t ? s_eq : s_ne);
            const int mr = ehh[c];
            const int M = mr != 0 ? mr + s : 0;
            mm[c] = M;
            if (c >= c_lo && c < c_hi) lane_g = max(lane_g, M + je);
            je += e_ins;
        }
        // Exclusive prefix max across the job's 16 lanes.
        int inc = lane_g;
        for (ushort d = 1; d < 16; d <<= 1) {
            const int y = simd_shuffle_up(inc, d);
            if (li >= d) inc = max(inc, y);
        }
        int run = simd_shuffle_up(inc, ushort(1));
        if (li == 0) run = NEG;

        // Pass 2: F, H, E, the row max key, eh.h shifted by one column (eh[j].h = H(i, j-1)), and
        // the live bits of every slot in [beg, end]. Column 0 of a lane needs the previous lane's
        // last H: patched after the shuffle below.
        int key = -1;  // max of (H << 8) | j: the row max, the last column on ties
        int hprev = 0; // H(i, j-1) inside the lane; at the end, this lane's last active H
        bool fix0 = false;
        int h_end = 0; // H(i, end-1), meaningful in the lane that owns column end-1
        uint live = 0; // bit c: slot base+c, in [beg, end], has h or e non-zero
        je = jei0;
        for (int c = 0; c < C; ++c, je += e_ins) {
            const int j = base + c;
            if (c >= c_lo && c < c_hi) {
                const int M = mm[c];
                const int F = max(0, run - o_ins - je);
                const int e = ehe[c];
                const int H = max(max(M, e), F);
                run = max(run, M + je);
                key = max(key, (H << 8) | j);
                const int ne = max(e - e_del, max(M - oe_del, 0));
                ehe[c] = ne;
                int nh = hprev;
                if (j == beg) {
                    nh = h1_init;
                } else if (c == 0) {
                    fix0 = true;
                }
                ehh[c] = nh;
                if ((nh | ne) != 0) live |= 1u << c;
                hprev = H;
                h_end = H;
            } else if (j == end) {
                // eh[end] = {h1, 0}: h1 is H(i, end-1) when the row ran, else the first column.
                ehe[c] = 0;
                int nh = hprev;
                if (!ran) {
                    nh = h1_init;
                } else if (c == 0) {
                    fix0 = true;
                }
                ehh[c] = nh;
                if (nh != 0) live |= 1u << c;
            }
        }
        const int from_prev = simd_shuffle_up(hprev, ushort(1));
        if (fix0) {
            ehh[0] = from_prev;
            live = (live & ~1u) | ((from_prev | ehe[0]) != 0 ? 1u : 0u);
        }

        const int kmax = seg_max(key);
        const int m_row = ran ? (kmax >> 8) : 0;
        const int mj = ran ? (kmax & 255) : -1;
        int h1_final = h1_init;
        if (ran) h1_final = simd_shuffle(h_end, ushort(hbase + (end - 1) / C));

        const int jfin = ran ? end : beg;
        if (jfin == qlen && gscore <= h1_final) {
            max_ie = i;
            gscore = h1_final;
        }
        if (m_row == 0) break;
        if (m_row > best) {
            best = m_row;
            max_i = i;
            max_j = mj;
            const int off = abs(mj - i);
            if (off > max_off) max_off = off;
        } else if (p.zdrop > 0) {
            if (i - max_i > mj - max_j) {
                if (best - m_row - ((i - max_i) - (mj - max_j)) * e_del > p.zdrop) break;
            } else if (best - m_row - ((mj - max_j) - (i - max_i)) * e_ins > p.zdrop) {
                break;
            }
        }

        // beg: first live slot in [beg, end), else end. end: last live slot in [beg, end] plus 2,
        // "none" being the new beg minus one. Every set bit of `live` is a slot in [beg, end], and
        // nothing below the new beg is live, so both searches can run over [beg, end] at once.
        const uint end_bit = (end >= base && end < base + C) ? (1u << (end - base)) : 0u;
        const uint lo = live & ~end_bit;
        const uint lanes_lo = half_ballot(lo != 0, hbase);
        const uint lanes_hi = half_ballot(live != 0, hbase);
        int nbeg = end;
        if (lanes_lo != 0) {
            const int cand = base + int(ctz(lo));
            nbeg = simd_shuffle(cand, ushort(hbase + ctz(lanes_lo)));
        }
        int ll = nbeg - 1;
        if (lanes_hi != 0) {
            const int cand = base + 31 - int(clz(live | 1u));
            ll = simd_shuffle(cand, ushort(hbase + 31 - clz(lanes_hi)));
        }
        beg = nbeg;
        end = ll + 2 < qlen ? ll + 2 : qlen;
    }

    if (li == 0) {
        device ERes& r = res[jb.out];
        r.score = best;
        r.qle = max_j + 1;
        r.tle = max_i + 1;
        r.gtle = max_ie + 1;
        r.gscore = gscore;
        r.max_off = max_off;
    }
}

#define EXTEND_KERNEL(C)                                                                       \
    kernel void extend16_c##C(device const uchar* seqs [[buffer(0)]],                          \
                              device const EJob* jobs [[buffer(1)]],                           \
                              device ERes* res [[buffer(2)]],                                  \
                              constant Params& p [[buffer(3)]],                                \
                              uint gid [[thread_position_in_grid]],                            \
                              ushort lane [[thread_index_in_simdgroup]]) {                     \
        const uint job = gid >> 4;                                                             \
        if (job >= p.n_jobs) return;                                                           \
        extend_job<C>(seqs, jobs[job], res, p, lane);                                          \
    }

EXTEND_KERNEL(2)
EXTEND_KERNEL(4)
EXTEND_KERNEL(6)
EXTEND_KERNEL(8)
EXTEND_KERNEL(10)
EXTEND_KERNEL(12)
EXTEND_KERNEL(14)
EXTEND_KERNEL(16)
