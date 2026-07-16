//! [`MetalBackend`]: [`SwBackend`] over the Metal `sw_extend` compute kernel.

use crate::metal_ctx::MetalCtx;
use bwa_extend::{ExtendJob, ExtendResult, ScalarBackend, SwBackend};
use metal::{MTLResourceOptions, MTLSize};

/// GPU seed-extension backend. Byte-identical to [`bwa_extend::ksw_extend2`] (integer DP on the GPU),
/// falling back to the scalar backend when there is no Metal device or the matrix is non-uniform.
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalBackend;

/// Mirror of the MSL `Params` struct (field order and `i32` layout must match exactly).
#[repr(C)]
struct Params {
    a: i32,
    mm: i32,
    npen: i32,
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    end_bonus: i32,
    zdrop: i32,
    njobs: i32,
    stride: i32,
}

/// Per-job band clamp, identical to `ksw_extend2` / the NEON kernel (computed in `f64` on the CPU so
/// the GPU never does float math).
#[allow(clippy::too_many_arguments)]
fn clamp_band(
    w0: i32,
    qlen: usize,
    max_sc: i32,
    end_bonus: i32,
    o_ins: i32,
    e_ins: i32,
    o_del: i32,
    e_del: i32,
) -> i32 {
    let mut wl = w0;
    let max_ins = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_ins))
        / f64::from(e_ins))
        + 1.0) as i32;
    wl = wl.min(max_ins.max(1));
    let max_del = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_del))
        / f64::from(e_del))
        + 1.0) as i32;
    wl = wl.min(max_del.max(1));
    wl
}

/// `BWA3_GPU_STATS=1` phase counters. End-to-end wall time cannot tell a faster kernel from a slower
/// pack, and those pull in opposite directions for a layout change -- so measure them apart. At `-t1`
/// nothing else holds the queue, so `commit` -> `wait_until_completed` IS the GPU execution span.
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    pub static PACK_NS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_NS: AtomicU64 = AtomicU64::new(0);
    pub static JOBS: AtomicU64 = AtomicU64::new(0);
    pub static CALLS: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA3_GPU_STATS").is_some())
    }

    /// Print the accumulated counters to stderr. Called from the CLI at the end of a run.
    pub fn dump() {
        if !enabled() {
            return;
        }
        let (pack, gpu) = (
            PACK_NS.load(Ordering::Relaxed) as f64 / 1e9,
            GPU_NS.load(Ordering::Relaxed) as f64 / 1e9,
        );
        eprintln!(
            "[gpu-stats] calls={} jobs={} pack={:.3}s gpu_exec={:.3}s (pack is {:.0}% of the two)",
            CALLS.load(Ordering::Relaxed),
            JOBS.load(Ordering::Relaxed),
            pack,
            gpu,
            100.0 * pack / (pack + gpu).max(1e-9),
        );
    }
}

fn is_uniform_dna(mat: &[i8], m: usize) -> bool {
    if m < 2 || mat.len() < m * m {
        return false;
    }
    let (a, mm, npen) = (mat[0], mat[1], mat[m - 1]);
    for i in 0..m {
        for j in 0..m {
            let want = if i == m - 1 || j == m - 1 {
                npen
            } else if i == j {
                a
            } else {
                mm
            };
            if mat[i * m + j] != want {
                return false;
            }
        }
    }
    true
}

impl SwBackend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    #[allow(clippy::too_many_arguments)]
    fn extend(
        &self,
        query: &[u8],
        target: &[u8],
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        w: i32,
        end_bonus: i32,
        zdrop: i32,
        h0: i32,
    ) -> ExtendResult {
        let job = ExtendJob { query, target, h0 };
        self.extend_batch(
            std::slice::from_ref(&job),
            m,
            mat,
            o_del,
            e_del,
            o_ins,
            e_ins,
            w,
            end_bonus,
            zdrop,
        )
        .pop()
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn extend_batch(
        &self,
        jobs: &[ExtendJob],
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        w0: i32,
        end_bonus: i32,
        zdrop: i32,
    ) -> Vec<ExtendResult> {
        // Fall back to scalar for the empty batch, a non-uniform matrix, or no Metal device.
        let ctx = MetalCtx::get();
        if jobs.is_empty() || !is_uniform_dna(mat, m) || ctx.is_none() {
            return ScalarBackend.extend_batch(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            );
        }
        let ctx = ctx.unwrap();
        let t_pack = std::time::Instant::now();
        let pso = match ctx.pipeline("sw_extend") {
            Some(p) => p,
            None => {
                return ScalarBackend.extend_batch(
                    jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
                )
            }
        };

        let n = jobs.len();
        let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;
        let max_q = jobs.iter().map(|j| j.query.len()).max().unwrap_or(0);
        let stride = (max_q + 1) as i32;

        // Pack the jobs into flat buffers: concatenated codes + per-job offset/length/h0/band.
        // NB the AoS `[lane * stride + column]` layout here and in the kernel is deliberate, not an
        // oversight: transposing it to the CPU kernel's `[column * lane]` SoA form was measured
        // 2.2x SLOWER (see docs/optimization-roadmap.md). One GPU thread owns one alignment and
        // walks its row contiguously, so a fetched cache line already serves that thread's next 32
        // columns; SoA amortises the same line across lanes instead, for the same line count, and
        // pays an ~800 KB stride between consecutive columns on top.
        let mut qbuf: Vec<u8> = Vec::new();
        let mut tbuf: Vec<u8> = Vec::new();
        let mut qoff = vec![0i32; n];
        let mut toff = vec![0i32; n];
        let mut qlen = vec![0i32; n];
        let mut tlen = vec![0i32; n];
        let mut h0v = vec![0i32; n];
        let mut wv = vec![0i32; n];
        for (k, j) in jobs.iter().enumerate() {
            qoff[k] = qbuf.len() as i32;
            qbuf.extend_from_slice(j.query);
            toff[k] = tbuf.len() as i32;
            tbuf.extend_from_slice(j.target);
            qlen[k] = j.query.len() as i32;
            tlen[k] = j.target.len() as i32;
            h0v[k] = j.h0;
            wv[k] = clamp_band(
                w0,
                j.query.len(),
                max_sc,
                end_bonus,
                o_ins,
                e_ins,
                o_del,
                e_del,
            );
        }
        // Never hand a zero-length buffer to Metal.
        if qbuf.is_empty() {
            qbuf.push(0);
        }
        if tbuf.is_empty() {
            tbuf.push(0);
        }

        let dev = &ctx.device;
        let shared = MTLResourceOptions::StorageModeShared;
        let mkdata = |ptr: *const u8, len: usize| {
            dev.new_buffer_with_data(ptr as *const _, len as u64, shared)
        };
        let b_q = mkdata(qbuf.as_ptr(), qbuf.len());
        let b_t = mkdata(tbuf.as_ptr(), tbuf.len());
        let b_qoff = mkdata(qoff.as_ptr() as *const u8, qoff.len() * 4);
        let b_qlen = mkdata(qlen.as_ptr() as *const u8, qlen.len() * 4);
        let b_toff = mkdata(toff.as_ptr() as *const u8, toff.len() * 4);
        let b_tlen = mkdata(tlen.as_ptr() as *const u8, tlen.len() * 4);
        let b_h0 = mkdata(h0v.as_ptr() as *const u8, h0v.len() * 4);
        let b_w = mkdata(wv.as_ptr() as *const u8, wv.len() * 4);
        let scratch_len = (n * stride as usize * 4) as u64;
        let b_ehh = dev.new_buffer(scratch_len.max(4), shared);
        let b_ehe = dev.new_buffer(scratch_len.max(4), shared);
        let b_out = dev.new_buffer((n * 6 * 4) as u64, shared);

        let params = Params {
            a: max_sc,
            mm: i32::from(mat[1]),
            npen: i32::from(mat[m - 1]),
            o_del,
            e_del,
            o_ins,
            e_ins,
            end_bonus,
            zdrop,
            njobs: n as i32,
            stride,
        };
        let b_params = dev.new_buffer_with_data(
            (&params as *const Params) as *const _,
            std::mem::size_of::<Params>() as u64,
            shared,
        );

        let pack_ns = t_pack.elapsed().as_nanos() as u64;
        let t_gpu = std::time::Instant::now();

        let cmd = ctx.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for (i, b) in [
            &b_q, &b_t, &b_qoff, &b_qlen, &b_toff, &b_tlen, &b_h0, &b_w, &b_ehh, &b_ehe,
        ]
        .iter()
        .enumerate()
        {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_buffer(10, Some(&b_params), 0);
        enc.set_buffer(11, Some(&b_out), 0);
        let tg = pso.max_total_threads_per_threadgroup().min(64);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        if stats::enabled() {
            use std::sync::atomic::Ordering::Relaxed;
            stats::PACK_NS.fetch_add(pack_ns, Relaxed);
            stats::GPU_NS.fetch_add(t_gpu.elapsed().as_nanos() as u64, Relaxed);
            stats::JOBS.fetch_add(n as u64, Relaxed);
            stats::CALLS.fetch_add(1, Relaxed);
        }

        let raw = unsafe { std::slice::from_raw_parts(b_out.contents() as *const i32, n * 6) };
        (0..n)
            .map(|k| ExtendResult {
                score: raw[k * 6],
                qle: raw[k * 6 + 1],
                tle: raw[k * 6 + 2],
                gtle: raw[k * 6 + 3],
                gscore: raw[k * 6 + 4],
                max_off: raw[k * 6 + 5],
            })
            .collect()
    }
}
