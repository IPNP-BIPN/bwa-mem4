//! GPU (Metal) backend for the batched Smith-Waterman seed extension (phase 9b).
//!
//! Apple Silicon has **unified memory**, so buffers are shared CPU/GPU with no PCIe copy. The seed
//! extension is **integer** (scores fit i32), so a Metal compute kernel running the *same* banded
//! recurrence as [`bwa_extend::ksw_extend2`] is deterministic and **byte-identical**, not merely
//! concordant. One GPU thread runs one alignment's full DP (inter-sequence parallelism, as GASAL2 /
//! ADEPT); the GPU wins by massive thread count, not intra-alignment SIMD.
//!
//! [`MetalBackend`] implements [`bwa_extend::SwBackend`]; `extend_batch` packs the jobs into flat
//! shared buffers, dispatches one thread per job, and reads the results straight back. The per-job
//! band width `w` is clamped CPU-side in `f64` (identical to `ksw_extend2`) and passed in, so the
//! kernel never does float math. Falls back to the scalar backend when there is no Metal device or
//! the matrix is not the uniform bwa matrix.

#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
pub use backend::MetalBackend;

#[cfg(target_os = "macos")]
mod metal_ctx {
    use metal::{ComputePipelineState, Device, Function, Library};
    use std::sync::OnceLock;

    pub struct MetalCtx {
        pub device: Device,
        pub queue: metal::CommandQueue,
        pub library: Library,
    }

    // SAFETY: Metal objects are internally reference-counted; each call builds its own command
    // buffer, so the shared context is only read concurrently. Markers let the `OnceLock` be `Sync`.
    unsafe impl Send for MetalCtx {}
    unsafe impl Sync for MetalCtx {}

    static CTX: OnceLock<Option<MetalCtx>> = OnceLock::new();

    impl MetalCtx {
        pub fn get() -> Option<&'static MetalCtx> {
            CTX.get_or_init(|| {
                let device = Device::system_default()?;
                let queue = device.new_command_queue();
                let library = device
                    .new_library_with_source(super::MSL_SRC, &metal::CompileOptions::new())
                    .ok()?;
                Some(MetalCtx {
                    device,
                    queue,
                    library,
                })
            })
            .as_ref()
        }

        pub fn pipeline(&self, name: &str) -> Option<ComputePipelineState> {
            let f: Function = self.library.get_function(name, None).ok()?;
            self.device
                .new_compute_pipeline_state_with_function(&f)
                .ok()
        }
    }
}

/// Metal Shading Language source: the feasibility probe plus the banded SW extension kernel, a
/// faithful port of `bwa_extend::ksw_extend2` (H-based gap opens, adaptive band, z-drop, gscore).
#[cfg(target_os = "macos")]
const MSL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void square_u32(device uint* data [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
    uint v = data[gid];
    data[gid] = v * v;
}

struct Params {
    int a; int mm; int npen;          // uniform matrix: match / mismatch / ambiguous
    int o_del; int e_del; int o_ins; int e_ins;
    int end_bonus; int zdrop;
    int njobs; int stride;            // stride = max_qlen + 1 (per-thread DP slice)
};

// One thread = one seed extension. DP arrays live in device memory, a `stride`-sized slice per
// thread. Byte-identical to ksw_extend2: same recurrence, band shrink, z-drop and tie-breaks.
kernel void sw_extend(device const uchar* qbuf   [[buffer(0)]],
                      device const uchar* tbuf   [[buffer(1)]],
                      device const int*   qoff   [[buffer(2)]],
                      device const int*   qlen_a [[buffer(3)]],
                      device const int*   toff   [[buffer(4)]],
                      device const int*   tlen_a [[buffer(5)]],
                      device const int*   h0buf  [[buffer(6)]],
                      device const int*   wbuf   [[buffer(7)]],
                      device int*         ehh    [[buffer(8)]],
                      device int*         ehe    [[buffer(9)]],
                      constant Params&    p      [[buffer(10)]],
                      device int*         out    [[buffer(11)]],  // 6 ints per job
                      uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.njobs) return;

    const int qlen = qlen_a[gid];
    const int tlen = tlen_a[gid];
    const int h0   = h0buf[gid];
    int w          = wbuf[gid];
    device const uchar* q = qbuf + qoff[gid];
    device const uchar* t = tbuf + toff[gid];
    device int* eh_h = ehh + (uint)gid * (uint)p.stride;
    device int* eh_e = ehe + (uint)gid * (uint)p.stride;

    const int oe_del = p.o_del + p.e_del;
    const int oe_ins = p.o_ins + p.e_ins;

    for (int j = 0; j <= qlen; j++) { eh_h[j] = 0; eh_e[j] = 0; }
    eh_h[0] = h0;
    if (qlen >= 1) eh_h[1] = h0 > oe_ins ? h0 - oe_ins : 0;
    { int j = 2; while (j <= qlen && eh_h[j-1] > p.e_ins) { eh_h[j] = eh_h[j-1] - p.e_ins; j++; } }

    int mx = h0, max_i = -1, max_j = -1, max_ie = -1, gscore = -1, max_off = 0;
    int beg = 0, end = qlen;

    for (int i = 0; i < tlen; i++) {
        int f = 0, row_max = 0, mj = -1;
        int tc = t[i];
        if (beg < i - w) beg = i - w;
        if (end > i + w + 1) end = i + w + 1;
        if (end > qlen) end = qlen;
        int h1 = 0;
        if (beg == 0) { int v = h0 - (p.o_del + p.e_del * (i + 1)); h1 = v > 0 ? v : 0; }
        int j = beg;
        while (j < end) {
            int big_m = eh_h[j];
            int e = eh_e[j];
            eh_h[j] = h1;
            int qc = q[j];
            int sc = (tc >= 4 || qc >= 4) ? p.npen : (tc == qc ? p.a : p.mm);
            big_m = big_m != 0 ? big_m + sc : 0;
            int h = big_m > e ? big_m : e;
            h = h > f ? h : f;
            h1 = h;
            mj = row_max > h ? mj : j;
            row_max = row_max > h ? row_max : h;
            int tt = h - oe_del; if (tt < 0) tt = 0;
            e -= p.e_del; e = e > tt ? e : tt;
            eh_e[j] = e;
            tt = h - oe_ins; if (tt < 0) tt = 0;
            f -= p.e_ins; f = f > tt ? f : tt;
            j++;
        }
        eh_h[end] = h1;
        eh_e[end] = 0;
        if (j == qlen && gscore <= h1) { max_ie = i; gscore = h1; }
        if (row_max == 0) break;
        if (row_max > mx) {
            mx = row_max; max_i = i; max_j = mj;
            int off = mj - i; if (off < 0) off = -off;
            if (off > max_off) max_off = off;
        } else if (p.zdrop > 0) {
            int drop;
            if (i - max_i > mj - max_j) drop = mx - row_max - ((i - max_i) - (mj - max_j)) * p.e_del;
            else                        drop = mx - row_max - ((mj - max_j) - (i - max_i)) * p.e_ins;
            if (drop > p.zdrop) break;
        }
        int jb = beg;
        while (jb < end && eh_h[jb] == 0 && eh_e[jb] == 0) jb++;
        beg = jb;
        int je = end;
        while (je >= beg && eh_h[je] == 0 && eh_e[je] == 0) je--;
        end = (je + 2 < qlen) ? je + 2 : qlen;
    }

    device int* o = out + (uint)gid * 6u;
    o[0] = mx;            // score
    o[1] = max_j + 1;    // qle
    o[2] = max_i + 1;    // tle
    o[3] = max_ie + 1;   // gtle
    o[4] = gscore;       // gscore
    o[5] = max_off;      // max_off
}
"#;

/// Feasibility probe: square `data` on the GPU. Returns `false` if there is no Metal device.
#[cfg(target_os = "macos")]
pub fn square_u32(data: &mut [u32]) -> bool {
    use metal::{MTLResourceOptions, MTLSize};
    let Some(ctx) = metal_ctx::MetalCtx::get() else {
        return false;
    };
    let Some(pso) = ctx.pipeline("square_u32") else {
        return false;
    };
    let n = data.len();
    let buf = ctx.device.new_buffer_with_data(
        data.as_ptr() as *const _,
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&buf), 0);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let out = unsafe { std::slice::from_raw_parts(buf.contents() as *const u32, n) };
    data.copy_from_slice(out);
    true
}

#[cfg(all(test, target_os = "macos"))]
mod backend_tests {
    use super::MetalBackend;
    use bwa_extend::{
        assert_backend_batch_matches_scalar, assert_backend_matches_scalar, ksw_extend2, ExtendJob,
        SwBackend,
    };

    #[test]
    fn metal_matches_scalar_shared_gate() {
        // The shared acceptance gate (random DNA, qlen/tlen <= 80, varied band/gaps/zdrop).
        assert_backend_matches_scalar(&MetalBackend);
        assert_backend_batch_matches_scalar(&MetalBackend);
        assert_eq!(MetalBackend.name(), "metal");
    }

    #[test]
    fn metal_matches_scalar_read_sized() {
        let (a, b) = (1i8, 4i8);
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
        let mut state = 0x6D3A_0000_0000_0001u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        for round in 0..80u32 {
            let batch = *[1usize, 8, 64, 256, 1000]
                .get((next() % 5) as usize)
                .unwrap();
            let w = 50 + (next() % 120) as i32;
            let zdrop = (next() % 200) as i32;
            let end_bonus = (next() % 12) as i32;
            let mut queries: Vec<Vec<u8>> = Vec::new();
            let mut targets: Vec<Vec<u8>> = Vec::new();
            let mut h0s: Vec<i32> = Vec::new();
            for _ in 0..batch {
                let qlen = 40 + (next() % 220) as usize;
                let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                let tlen = qlen + (next() % 40) as usize;
                let mut t: Vec<u8> = Vec::with_capacity(tlen);
                let mut qi = 0usize;
                while t.len() < tlen {
                    if qi < q.len() && next() % 100 >= 5 {
                        t.push(q[qi]);
                        qi += 1;
                    } else {
                        t.push((next() % 4) as u8);
                        if next() % 2 == 0 {
                            qi += 1;
                        }
                    }
                }
                queries.push(q);
                targets.push(t);
                h0s.push(20 + (next() % 30) as i32);
            }
            let jobs: Vec<ExtendJob> = (0..batch)
                .map(|i| ExtendJob {
                    query: &queries[i],
                    target: &targets[i],
                    h0: h0s[i],
                })
                .collect();
            let got = MetalBackend.extend_batch(
                &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
            );
            for (i, g) in got.iter().enumerate() {
                let expected = ksw_extend2(
                    &queries[i],
                    &targets[i],
                    5,
                    &mat,
                    o_del,
                    e_del,
                    o_ins,
                    e_ins,
                    w,
                    end_bonus,
                    zdrop,
                    h0s[i],
                );
                assert_eq!(*g, expected, "Metal diverged round {round} job {i}");
            }
        }
    }
}
