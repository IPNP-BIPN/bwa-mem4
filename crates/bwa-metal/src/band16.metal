// Mate-rescue band lanes (`bwa_neon::offload::band_reference`) on Apple GPUs.
//
// One band per 16 lanes (two per 32-wide simdgroup), row-synchronous, in query columns: lane `li`
// owns j in [li*C, li*C + C), C = quanta / 16 rounded up to a bucket. The band (diagonals k in
// [0, w), k = r + o - j) only bounds each row's columns, j in [r + o - w + 1, r + o]: a band wider
// than the query costs a row of the query, as on the CPU, whatever its width.
//
// The cell, as the NEON band kernel's: h0 = max(diag + s, E, 0), H = max(h0, F),
// E' = max(h0 - oe_del, E - e_del, 0) feeds (r+1, j), F' = max(h0 - oe_ins, F - e_ins, 0) feeds
// (r, j+1). h0 does not read F, so the row's F is an exact exclusive prefix max over j:
//     F(j) = max(0, max_{lo <= j' < j} (h0(j') + j'*e_ins) - oe_ins - (j-1)*e_ins).
// Cells outside the band or outside [0, quanta) hold 0. Columns len(q) <= j < quanta score 0 and
// count (ksw's pad). The row max and the first column reaching it travel as one key
// (H << 9) | (511 - j).
#include <metal_stdlib>
using namespace metal;

struct BJob {
    uint q_off;
    uint q_len;
    uint t_off;
    uint t_len;   // rows of the band
    int quanta;
    int o;
    int w;
    uint thr;
    uint rows_off; // this band's row maxima in the rows buffer
    uint out;
    uint _pad0;
    uint _pad1;
};

struct BRes {
    int gmax;
    int te;
    int qe;
    int _pad;
};

struct BParams {
    int oe_del;
    int e_del;
    int oe_ins;
    int e_ins;
    int sc_match;
    int sc_mismatch;
    int sc_n;
    uint n_jobs;
};

constant int NEG = -0x40000000;
// Query codes: 0-4 bases, PAD scores 0 and counts, NONE is no column.
constant uchar PAD = 5;
constant uchar NONE = 7;

static inline int seg_max(int x) {
    x = max(x, simd_shuffle_xor(x, ushort(8)));
    x = max(x, simd_shuffle_xor(x, ushort(4)));
    x = max(x, simd_shuffle_xor(x, ushort(2)));
    x = max(x, simd_shuffle_xor(x, ushort(1)));
    return x;
}

template <int C>
static inline void band_job(device const uchar* seqs, device const BJob& jb, device BRes* res,
                            device uchar* rows, constant BParams& p, ushort lane) {
    const ushort li = lane & 15;
    const int base = int(li) * C;
    const int len2 = int(jb.q_len);
    const int quanta = jb.quanta;
    const int o = jb.o;
    const int w = jb.w;
    const int nr = int(jb.t_len);
    const int e_ins = p.e_ins;
    device const uchar* q = seqs + jb.q_off;
    device const uchar* tg = seqs + jb.t_off;
    device uchar* rout = rows + jb.rows_off;

    int hp[C];
    int ep[C];
    uchar qb[C];
    for (int c = 0; c < C; ++c) {
        const int j = base + c;
        hp[c] = 0;
        ep[c] = 0;
        qb[c] = j < len2 ? q[j] : j < quanta ? PAD : NONE;
    }

    int gmax = 0, te = -1, qe = 0;
    int t_next = nr > 0 ? int(tg[0]) : 0;
    for (int r = 0; r < nr; ++r) {
        const int t = t_next;
        if (r + 1 < nr) t_next = int(tg[r + 1]);
        const int s_eq = t >= 4 ? p.sc_n : p.sc_match;
        const int s_ne = t >= 4 ? p.sc_n : p.sc_mismatch;
        // This row's columns, as a range of c.
        const int c_lo = clamp(r + o - w + 1 - base, 0, C);
        const int c_hi = clamp(r + o + 1 - base, 0, C);
        // Diagonal H(r-1, j-1): the previous lane's last column for c = 0.
        const int d_in = simd_shuffle_up(hp[C - 1], ushort(1));

        int h0[C];
        int lane_g = NEG;
        int dprev = li == 0 ? 0 : d_in;
        int je = (base - 1) * e_ins;
        for (int c = 0; c < C; ++c) {
            je += e_ins; // j * e_ins
            const int d = dprev;
            dprev = hp[c];
            const uchar qc = qb[c];
            if (c < c_lo || c >= c_hi || qc == NONE) {
                h0[c] = NEG;
                continue;
            }
            const int s = qc == PAD ? 0 : (qc == 4 ? p.sc_n : (int(qc) == t ? s_eq : s_ne));
            h0[c] = max(max(d + s, 0), ep[c]);
            lane_g = max(lane_g, h0[c] + je);
        }
        // Exclusive prefix max over the job's 16 lanes.
        int inc = lane_g;
        for (ushort dd = 1; dd < 16; dd <<= 1) {
            const int y = simd_shuffle_up(inc, dd);
            if (li >= dd) inc = max(inc, y);
        }
        int run = simd_shuffle_up(inc, ushort(1));
        if (li == 0) run = NEG;

        int key = 0;
        je = (base - 1) * e_ins;
        for (int c = 0; c < C; ++c) {
            je += e_ins;
            if (h0[c] == NEG) {
                hp[c] = 0;
                ep[c] = 0;
                continue;
            }
            const int f = max(0, run - p.oe_ins - je + e_ins);
            run = max(run, h0[c] + je);
            const int h = max(h0[c], f);
            ep[c] = max(max(h0[c] - p.oe_del, ep[c] - p.e_del), 0);
            hp[c] = h;
            key = max(key, (h << 9) | (511 - (base + c)));
        }
        const int kmax = seg_max(key);
        const int rmax = kmax >> 9;
        if (li == 0) rout[r] = uchar(rmax);
        if (rmax > gmax) {
            gmax = rmax;
            te = r;
            qe = 511 - (kmax & 511);
        }
    }
    if (li == 0) {
        device BRes& x = res[jb.out];
        x.gmax = gmax;
        x.te = gmax > 0 ? te : -1;
        x.qe = gmax >= int(jb.thr) ? qe : 0;
    }
}

#define BAND_KERNEL(C)                                                                         \
    kernel void band16_c##C(device const uchar* seqs [[buffer(0)]],                            \
                            device const BJob* jobs [[buffer(1)]],                             \
                            device BRes* res [[buffer(2)]],                                    \
                            device uchar* rows [[buffer(3)]],                                  \
                            constant BParams& p [[buffer(4)]],                                 \
                            uint gid [[thread_position_in_grid]],                              \
                            ushort lane [[thread_index_in_simdgroup]]) {                       \
        const uint job = gid >> 4;                                                             \
        if (job >= p.n_jobs) return;                                                           \
        band_job<C>(seqs, jobs[job], res, rows, p, lane);                                      \
    }

BAND_KERNEL(1)
BAND_KERNEL(2)
BAND_KERNEL(4)
BAND_KERNEL(6)
BAND_KERNEL(8)
BAND_KERNEL(10)
BAND_KERNEL(12)
BAND_KERNEL(16)
