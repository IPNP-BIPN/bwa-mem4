// Banded local seed extension, `ksw_extend2`, one JOB per invocation, in WGSL.
//
// This is the same transliteration as `crates/bwa-metal/src/extend.metal`, and both are checked
// against the scalar reference `bwa_extend::ksw_extend2`, not against each other. WGSL is compiled
// by wgpu to SPIR-V (Vulkan: NVIDIA, AMD, Intel, and Mesa's lavapipe on a GPU-less host), MSL
// (Apple) or HLSL (DX12), so one file covers every discrete GPU a Linux server is likely to have.
//
// # Rules this file keeps, and why
//
// - Integers only. WGSL `i32` wraps on overflow and has no saturation, which is exactly the scalar
//   reference's arithmetic for the magnitudes a 150 bp..10 kb extension reaches.
// - `mj = row_max > h ? mj : c`: on a tie the LATER column wins (`ksw.cpp:491`). Issue #54 trap 2.
// - `gscore` updates on `end == qlen && gscore <= h1`, non-strict, so the LAST such row wins.
// - Both gaps open from `big_m`, not from `h`. The removed 2026-07 Metal shader opened them from
//   `h`, which is the one real bug a GPU port of this recurrence has shipped in this project.
// - The band clamp (`f64` in the reference) is done on the host and arrives already applied in `w`.
//
// # Layout
//
// Sequences are packed four bases per `u32` (WGSL storage has no 8-bit type). The two DP rails are
// column-major, element `c` of job `k` at `c * stride + k`, so neighbouring invocations touch
// neighbouring words: the layout the Metal rescue kernel measured at 3.3x over job-major.

struct EJob {
    q_off: u32,
    q_len: u32,
    t_off: u32,
    t_len: u32,
    h0: i32,
    w: i32,
    _pad0: u32,
    _pad1: u32,
};

struct ERes {
    score: i32,
    qle: i32,
    tle: i32,
    gtle: i32,
    gscore: i32,
    max_off: i32,
    _pad0: i32,
    _pad1: i32,
};

struct Params {
    m: i32,
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    zdrop: i32,
    n_jobs: u32,
    // Invocations per row of the 2-D dispatch grid, for batches past 65 535 workgroups.
    grid_x: u32,
};

@group(0) @binding(0) var<storage, read> seqs: array<u32>;
@group(0) @binding(1) var<storage, read> jobs: array<EJob>;
@group(0) @binding(2) var<storage, read_write> out: array<ERes>;
@group(0) @binding(3) var<storage, read_write> eh_h: array<i32>;
@group(0) @binding(4) var<storage, read_write> eh_e: array<i32>;
@group(0) @binding(5) var<storage, read> mat: array<i32>;
@group(0) @binding(6) var<uniform> p: Params;

fn base_at(i: u32) -> i32 {
    return i32((seqs[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);
}

@compute @workgroup_size(64)
fn extend_fwd(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.y * p.grid_x + gid3.x;
    if (gid >= p.n_jobs) {
        return;
    }
    let j = jobs[gid];
    let qlen = i32(j.q_len);
    let tlen = i32(j.t_len);
    let oe_del = p.o_del + p.e_del;
    let oe_ins = p.o_ins + p.e_ins;
    let stride = p.n_jobs;
    let e_del = p.e_del;
    let e_ins = p.e_ins;

    // The h0 ladder: H(-1, 0) = h0, then one insertion opened and extended while it stays positive.
    for (var c: i32 = 0; c <= qlen; c = c + 1) {
        let ix = u32(c) * stride + gid;
        eh_h[ix] = 0;
        eh_e[ix] = 0;
    }
    eh_h[gid] = j.h0;
    if (qlen >= 1) {
        eh_h[stride + gid] = select(0, j.h0 - oe_ins, j.h0 > oe_ins);
    }
    for (var c: i32 = 2; c <= qlen; c = c + 1) {
        let prev = eh_h[u32(c - 1) * stride + gid];
        if (prev <= e_ins) {
            break;
        }
        eh_h[u32(c) * stride + gid] = prev - e_ins;
    }

    var best: i32 = j.h0;
    var max_i: i32 = -1;
    var max_j: i32 = -1;
    var max_ie: i32 = -1;
    var gscore: i32 = -1;
    var max_off: i32 = 0;
    var beg: i32 = 0;
    var end: i32 = qlen;

    for (var i: i32 = 0; i < tlen; i = i + 1) {
        var f: i32 = 0;
        var row_max: i32 = 0;
        var mj: i32 = -1;
        let tc = base_at(j.t_off + u32(i));
        let mrow = tc * p.m;
        if (beg < i - j.w) {
            beg = i - j.w;
        }
        if (end > i + j.w + 1) {
            end = i + j.w + 1;
        }
        if (end > qlen) {
            end = qlen;
        }
        // Reaching column 0 of this row costs one deletion opened at the start and extended down.
        var h1: i32 = 0;
        if (beg == 0) {
            h1 = max(j.h0 - (p.o_del + e_del * (i + 1)), 0);
        }

        var c: i32 = beg;
        loop {
            if (c >= end) {
                break;
            }
            let ix = u32(c) * stride + gid;
            var big_m = eh_h[ix];      // H(i-1, c-1)
            let e = eh_e[ix];          // E(i, c)
            eh_h[ix] = h1;             // H(i, c-1), for the next row
            let qc = base_at(j.q_off + u32(c));
            // A diagonal predecessor of 0 means the alignment starts here: no substitution added.
            big_m = select(0, big_m + mat[mrow + qc], big_m != 0);
            let h = max(max(big_m, e), f);
            h1 = h;
            // NON-strict: on a tie the LATER column wins.
            mj = select(c, mj, row_max > h);
            row_max = max(row_max, h);
            eh_e[ix] = max(e - e_del, max(big_m - oe_del, 0));
            f = max(f - e_ins, max(big_m - oe_ins, 0));
            c = c + 1;
        }
        let ixe = u32(end) * stride + gid;
        eh_h[ixe] = h1;
        eh_e[ixe] = 0;
        // `c == qlen`: the row reached the end of the query, a global-alignment candidate.
        if (c == qlen && gscore <= h1) {
            max_ie = i;
            gscore = h1;
        }
        if (row_max == 0) {
            break;
        }
        if (row_max > best) {
            best = row_max;
            max_i = i;
            max_j = mj;
            let off = abs(mj - i);
            if (off > max_off) {
                max_off = off;
            }
        } else if (p.zdrop > 0) {
            if (i - max_i > mj - max_j) {
                if (best - row_max - ((i - max_i) - (mj - max_j)) * e_del > p.zdrop) {
                    break;
                }
            } else if (best - row_max - ((mj - max_j) - (i - max_i)) * e_ins > p.zdrop) {
                break;
            }
        }
        // Band tightening: drop the dead cells at both ends of the live range.
        var first_live: i32 = beg;
        loop {
            if (first_live >= end) {
                break;
            }
            let ix = u32(first_live) * stride + gid;
            if (eh_h[ix] != 0 || eh_e[ix] != 0) {
                break;
            }
            first_live = first_live + 1;
        }
        beg = first_live;
        var last_live: i32 = end;
        loop {
            if (last_live < beg) {
                break;
            }
            let ix = u32(last_live) * stride + gid;
            if (eh_h[ix] != 0 || eh_e[ix] != 0) {
                break;
            }
            last_live = last_live - 1;
        }
        end = select(qlen, last_live + 2, last_live + 2 < qlen);
    }

    var r: ERes;
    r.score = best;
    r.qle = max_j + 1;
    r.tle = max_i + 1;
    r.gtle = max_ie + 1;
    r.gscore = gscore;
    r.max_off = max_off;
    r._pad0 = 0;
    r._pad1 = 0;
    out[gid] = r;
}
