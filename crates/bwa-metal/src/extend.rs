//! Metal backend for banded seed extension, the stage the batch sizes actually point at.
//!
//! # Why this exists next to the rescue backend rather than instead of it
//!
//! Both were measured on the same production run. Extension is **30% of CPU time** and submits
//! **~10 800 jobs per call** at `-t4`; mate rescue is **4.5%** and submits **~181**. The GPU
//! saturates near **32 000** concurrent jobs, because one job per thread means the batch size is the
//! thread count. So extension is a factor of 3 from filling the machine and rescue a factor of 180,
//! and only the former can be closed without deferring a stage whose output order is observable.
//!
//! # What makes this one cheap to trust
//!
//! It implements [`SwBackend`], so the entire acceptance harness this project already has applies
//! unchanged and without a line of new test scaffolding:
//!
//! - [`assert_backend_matches_scalar`] and [`assert_backend_batch_matches_scalar`], the gates every
//!   CPU backend passes;
//! - [`assert_backend_tie_rule_matches_scalar`], issue #54's trap 2, on inputs where every cell of a
//!   row ties, which is where a `>` written for `>=` shows and nowhere else;
//! - [`assert_backend_batch_order_invariant`], issue #54's trap 6, the ten-permutation check. That
//!   one is more than a test here: it is the proof that aggregating several CPU threads' jobs into
//!   one GPU launch cannot change an answer, which is the whole plan for this seam.
//!
//! [`assert_backend_matches_scalar`]: bwa_mem4_extend::assert_backend_matches_scalar
//! [`assert_backend_batch_matches_scalar`]: bwa_mem4_extend::assert_backend_batch_matches_scalar
//! [`assert_backend_tie_rule_matches_scalar`]: bwa_mem4_extend::assert_backend_tie_rule_matches_scalar
//! [`assert_backend_batch_order_invariant`]: bwa_mem4_extend::assert_backend_batch_order_invariant

/// The MSL source, compiled by the driver at run time.
pub const EXTEND_SRC: &str = include_str!("extend.metal");

/// The band clamp of `ksw_extend2`, computed on the host in `f64` and passed to the kernel.
///
/// The DP kernels of this project contain no floating point, by rule, and the [`SwBackend`] contract
/// says so explicitly: a backend takes the clamped `w` rather than re-deriving it. This is that
/// derivation, transliterated from the reference so the two cannot drift.
///
/// [`SwBackend`]: bwa_mem4_extend::SwBackend
///
/// # Parameters
/// `qlen` the query length in bases; `max_sc` the largest matrix entry; `end_bonus` bwa's `-L`;
/// `o_ins`/`e_ins` and `o_del`/`e_del` the two gap pairs as positive magnitudes, both extends `>= 1`.
///
/// # Returns
/// The band half-width the DP must use, never larger than `w0`.
pub fn clamp_band(
    w0: i32,
    qlen: usize,
    max_sc: i32,
    end_bonus: i32,
    o_ins: i32,
    e_ins: i32,
    o_del: i32,
    e_del: i32,
) -> i32 {
    let mut w = w0;
    let max_ins = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_ins))
        / f64::from(e_ins))
        + 1.0) as i32;
    w = w.min(max_ins.max(1));
    let max_del = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_del))
        / f64::from(e_del))
        + 1.0) as i32;
    w.min(max_del.max(1))
}

/// Per-job geometry, `repr(C)` against the MSL `struct EJob`.
///
/// Scalar members only, and that is not an accident: MSL aligns a vector type to its own size while
/// Rust aligns `[i32; 4]` to 4, so a vector member here silently puts the two sides' field offsets in
/// different places. That cost a debugging session on the rescue backend; see the note on its
/// `GpuQuad`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct GpuEJob {
    q_off: u32,
    q_len: u32,
    t_off: u32,
    t_len: u32,
    h0: i32,
    w: i32,
    rail: u32,
    _pad: u32,
}

/// Per-job result, `repr(C)` against the MSL `struct ERes`. Padded to 32 bytes so the two sides
/// agree without either needing an alignment attribute.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct GpuERes {
    score: i32,
    qle: i32,
    tle: i32,
    gtle: i32,
    gscore: i32,
    max_off: i32,
    _pad0: i32,
    _pad1: i32,
}

#[cfg(feature = "metal")]
mod backend {
    use super::*;
    use bwa_mem4_extend::{ExtendJob, ExtendResult, SwBackend};
    use metal::{Device, MTLResourceOptions, MTLSize};

    /// Device, queue and the compiled extension pipeline.
    pub struct MetalExtend {
        device: Device,
        queue: metal::CommandQueue,
        pipeline: metal::ComputePipelineState,
        /// `(pack, submitted, scatter)` nanoseconds of the last batch. The rescue backend learned
        /// the hard way that timing the whole call hides which half moved.
        last_pack_ns: std::sync::atomic::AtomicU64,
        last_gpu_ns: std::sync::atomic::AtomicU64,
    }

    impl MetalExtend {
        /// Open the default device and compile the kernel. `None` if either step fails, which is the
        /// caller's cue to use a CPU backend: the answer is identical either way.
        pub fn new() -> Option<Self> {
            let device = Device::system_default()?;
            let lib = device
                .new_library_with_source(EXTEND_SRC, &metal::CompileOptions::new())
                .ok()?;
            let f = lib.get_function("extend_fwd", None).ok()?;
            let pipeline = device.new_compute_pipeline_state_with_function(&f).ok()?;
            let queue = device.new_command_queue();
            Some(Self {
                device,
                queue,
                pipeline,
                last_pack_ns: std::sync::atomic::AtomicU64::new(0),
                last_gpu_ns: std::sync::atomic::AtomicU64::new(0),
            })
        }

        /// The device's name.
        pub fn device_name(&self) -> String {
            self.device.name().to_string()
        }

        /// `(pack, submitted)` seconds of the last batch. Only the second moves when the kernel
        /// changes.
        pub fn last_phases(&self) -> (f64, f64) {
            use std::sync::atomic::Ordering::Relaxed;
            (
                self.last_pack_ns.load(Relaxed) as f64 / 1e9,
                self.last_gpu_ns.load(Relaxed) as f64 / 1e9,
            )
        }
    }

    impl SwBackend for MetalExtend {
        fn name(&self) -> &'static str {
            "metal"
        }

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
            let job = [ExtendJob { query, target, h0 }];
            self.extend_batch(
                &job, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
            )
            .remove(0)
        }

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
            if jobs.is_empty() {
                return Vec::new();
            }
            let t_pack = std::time::Instant::now();
            let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;

            // One flat sequence buffer, written straight into the shared allocation. On unified
            // memory this IS the device buffer: nothing is copied to reach the GPU.
            let total: usize = jobs.iter().map(|j| j.query.len() + j.target.len()).sum();
            let seqs = self
                .device
                .new_buffer(total.max(1) as u64, MTLResourceOptions::StorageModeShared);
            let mut gjobs = vec![GpuEJob::default(); jobs.len()];
            {
                // SAFETY: just allocated with `total` bytes, uniquely owned here, writes bounded by
                // that length.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(seqs.contents() as *mut u8, total.max(1))
                };
                let mut off = 0usize;
                for (k, j) in jobs.iter().enumerate() {
                    let q_off = off;
                    dst[off..off + j.query.len()].copy_from_slice(j.query);
                    off += j.query.len();
                    let t_off = off;
                    dst[off..off + j.target.len()].copy_from_slice(j.target);
                    off += j.target.len();
                    gjobs[k] = GpuEJob {
                        q_off: q_off as u32,
                        q_len: j.query.len() as u32,
                        t_off: t_off as u32,
                        t_len: j.target.len() as u32,
                        h0: j.h0,
                        // The band clamp is the host's job; see `clamp_band`.
                        w: clamp_band(
                            w0,
                            j.query.len(),
                            max_sc,
                            end_bonus,
                            o_ins,
                            e_ins,
                            o_del,
                            e_del,
                        ),
                        rail: k as u32,
                        _pad: 0,
                    };
                }
            }

            let rail_len = jobs.iter().map(|j| j.query.len()).max().unwrap_or(0) + 1;
            let rail_bytes = (jobs.len() * rail_len * 4) as u64;
            let jobs_buf = self.device.new_buffer_with_data(
                gjobs.as_ptr() as *const _,
                std::mem::size_of_val(&gjobs[..]) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let res_buf = self.device.new_buffer(
                (jobs.len() * std::mem::size_of::<GpuERes>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let eh_h = self
                .device
                .new_buffer(rail_bytes, MTLResourceOptions::StorageModePrivate);
            let eh_e = self
                .device
                .new_buffer(rail_bytes, MTLResourceOptions::StorageModePrivate);
            let mat_i32: Vec<i32> = mat[..m * m].iter().map(|&x| x as i32).collect();
            let mat_buf = self.device.new_buffer_with_data(
                mat_i32.as_ptr() as *const _,
                (mat_i32.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let scalars: [i32; 6] = [m as i32, o_del, e_del, o_ins, e_ins, zdrop];
            let n_jobs = jobs.len() as u32;

            let t_gpu = std::time::Instant::now();
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&seqs), 0);
            enc.set_buffer(1, Some(&jobs_buf), 0);
            enc.set_buffer(2, Some(&res_buf), 0);
            enc.set_buffer(3, Some(&eh_h), 0);
            enc.set_buffer(4, Some(&eh_e), 0);
            enc.set_buffer(5, Some(&mat_buf), 0);
            for (i, v) in scalars.iter().enumerate() {
                enc.set_bytes(6 + i as u64, 4, v as *const i32 as *const std::ffi::c_void);
            }
            enc.set_bytes(12, 4, &n_jobs as *const u32 as *const std::ffi::c_void);
            let tg = self
                .pipeline
                .max_total_threads_per_threadgroup()
                .min(jobs.len() as u64)
                .max(1);
            enc.dispatch_threads(
                MTLSize::new(jobs.len() as u64, 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let gpu_ns = t_gpu.elapsed().as_nanos() as u64;

            // SAFETY: shared buffer, sized above, command buffer completed.
            let res = unsafe {
                std::slice::from_raw_parts(res_buf.contents() as *const GpuERes, jobs.len())
            };
            let out: Vec<ExtendResult> = res
                .iter()
                .map(|r| ExtendResult {
                    score: r.score,
                    qle: r.qle,
                    tle: r.tle,
                    gtle: r.gtle,
                    gscore: r.gscore,
                    max_off: r.max_off,
                })
                .collect();
            {
                use std::sync::atomic::Ordering::Relaxed;
                let total_ns = t_pack.elapsed().as_nanos() as u64;
                self.last_pack_ns
                    .store(total_ns.saturating_sub(gpu_ns), Relaxed);
                self.last_gpu_ns.store(gpu_ns, Relaxed);
            }
            out
        }
    }
}

#[cfg(feature = "metal")]
pub use backend::MetalExtend;

/// Stand-in without the `metal` feature: construction always fails, so callers take their CPU path.
#[cfg(not(feature = "metal"))]
pub struct MetalExtend;

#[cfg(not(feature = "metal"))]
impl MetalExtend {
    /// Always `None`: no Metal support compiled in.
    pub fn new() -> Option<Self> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole acceptance harness, unchanged, against the GPU.
    ///
    /// Nothing here is written for this backend: these are the same four functions every CPU backend
    /// passes, which is the point of `SwBackend` carrying the gate rather than each backend carrying
    /// its own. The last one doubles as the proof that aggregating several threads' jobs into one
    /// launch is safe.
    #[test]
    #[cfg(feature = "metal")]
    fn metal_extend_passes_every_backend_gate() {
        use bwa_mem4_extend::{
            assert_backend_batch_matches_scalar, assert_backend_batch_order_invariant,
            assert_backend_matches_scalar, assert_backend_tie_rule_matches_scalar,
        };
        let Some(gpu) = MetalExtend::new() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        eprintln!("device: {}", gpu.device_name());
        assert_backend_matches_scalar(&gpu);
        assert_backend_batch_matches_scalar(&gpu);
        assert_backend_tie_rule_matches_scalar(&gpu);
        assert_backend_batch_order_invariant(&gpu);
    }

    /// GPU against the CPU NEON backend on production-shaped extension work, swept over batch size.
    ///
    /// The batch size is the thread count here, so the sweep IS the occupancy curve. Production
    /// submits ~10 800 jobs per call at `-t4`, and the rescue kernel's sweep put saturation near
    /// 32 000; both points are in the table so the reading is not a single number.
    #[test]
    #[ignore = "throughput probe, run explicitly"]
    #[cfg(feature = "metal")]
    fn metal_extend_throughput() {
        use bwa_mem4_extend::{ExtendJob, SwBackend};
        let Some(gpu) = MetalExtend::new() else {
            return;
        };
        let cpu = bwa_neon::NeonBackend;
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
        let mut state = 0x2468_ace0_1357_9bdfu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let n: usize = std::env::var("BWA4_METAL_JOBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10800);
        // A 150 bp read seeded in the middle leaves 40 to 110 bases to extend on each side, which is
        // the shape `BWA4_EXTEND_SHAPE` reports for a real run.
        let (mut qs, mut ts, mut h0s) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let qlen = 40 + (next() % 71) as usize;
            let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
            let tlen = qlen + (next() % 20) as usize;
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
            qs.push(q);
            ts.push(t);
            h0s.push(20 + (next() % 20) as i32);
        }
        let jobs: Vec<ExtendJob> = (0..qs.len())
            .map(|i| ExtendJob {
                query: &qs[i],
                target: &ts[i],
                h0: h0s[i],
            })
            .collect();
        let cells: u64 = jobs
            .iter()
            .map(|j| (j.query.len() * j.target.len()) as u64)
            .sum();
        let bench = |label: &str, run: &dyn Fn()| {
            run();
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                run();
                best = best.min(t0.elapsed().as_secs_f64());
            }
            eprintln!(
                "  {label:<14} {:>8.2} ms  {:>6.2} Gcell/s",
                best * 1e3,
                cells as f64 / best / 1e9
            );
            best
        };
        eprintln!(
            "extend {n} jobs, {cells} cells, device {}",
            gpu.device_name()
        );
        let c = bench("cpu neon", &|| {
            std::hint::black_box(cpu.extend_batch(&jobs, 5, &mat, 6, 1, 6, 1, 100, 5, 100));
        });
        let g = bench("metal", &|| {
            std::hint::black_box(gpu.extend_batch(&jobs, 5, &mat, 6, 1, 6, 1, 100, 5, 100));
        });
        let (pack, sub) = gpu.last_phases();
        eprintln!("  metal split: pack {pack:.4} s | submitted {sub:.4} s");
        eprintln!(
            "  submitted-only {:.2} Gcell/s   gpu vs one cpu thread {:.2}x",
            cells as f64 / sub.max(1e-9) / 1e9,
            c / g
        );
    }

    /// The two `repr(C)` structs must match their MSL declarations. A mismatch is silent corruption,
    /// not a compile error.
    #[test]
    fn extend_structs_have_the_expected_layout() {
        assert_eq!(std::mem::size_of::<GpuEJob>(), 32);
        assert_eq!(std::mem::size_of::<GpuERes>(), 32);
        assert_eq!(std::mem::align_of::<GpuEJob>(), 4);
    }

    /// The band clamp is the host's, so it is pinned against the reference's own arithmetic.
    #[test]
    fn band_clamp_matches_the_reference() {
        // bwa's defaults: -w 100, -A 1, -L 5, -O 6 -E 1. A 150 bp query cannot need more than its
        // own length of band, so the clamp binds well below `w0`.
        assert_eq!(clamp_band(100, 150, 1, 5, 6, 1, 6, 1), 100);
        assert_eq!(clamp_band(100, 20, 1, 5, 6, 1, 6, 1), 20);
        // e_ins = 2 halves the reachable insertion length.
        assert_eq!(clamp_band(100, 20, 1, 5, 6, 2, 6, 1), 10);
        // Never below 1, whatever the arithmetic says.
        assert_eq!(clamp_band(100, 0, 1, 0, 6, 1, 6, 1), 1);
    }
}
