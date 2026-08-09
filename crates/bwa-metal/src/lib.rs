//! Metal backend for the mate-rescue forward pass (issue #55).
//!
//! # What this is, and what it is not
//!
//! It is a working, byte-identity-gated GPU implementation of `ksw_local_fwd`, the forward local
//! Smith-Waterman that mate rescue runs. One job per thread, no cross-thread communication, sequence
//! delivered in a **shared** `MTLBuffer` the CPU writes in place, so on Apple Silicon's unified
//! memory nothing is copied to reach the GPU.
//!
//! It is **not** wired into the aligner, and the reason is a measurement rather than laziness: issue
//! #53's step 0 counted **72 jobs per rescue call** at 150 bp. A GPU launch does not amortise over 72
//! jobs, so switching the aligner to this today would lose. What closes that gap is the cross-thread
//! aggregation queue #53 describes, or the extension seam, which submits **5 381 jobs per call**.
//! This crate exists so that when either lands, the GPU half is already written and already proved.
//!
//! # Why it can ship without a Metal toolchain
//!
//! The kernel is embedded as a string and compiled by the driver at run time
//! (`newLibraryWithSource:`). There is no `.metallib`, no build script, and no Xcode requirement,
//! for this crate or for the binary that links it. `scripts/msl_probe.sh` is the standalone
//! demonstration of that route.
//!
//! # Arithmetic
//!
//! Plain 32-bit, mirroring the scalar reference rather than the CPU's saturating u8. The `uchar4`
//! form runs four cells per lane and the probe measured its ceiling at ~330 Gcell/s, but it brings
//! the bias, the `U8_SCORE_LIMIT` guard and the saturation edge cases with it. Correct first; that
//! packing is the next step and its value is already measured.

#![cfg_attr(not(feature = "metal"), allow(dead_code))]

#[cfg(feature = "metal")]
use bwa_mem4_extend::SuboptimalTracker;

pub mod extend;

/// The MSL source, compiled by the driver at run time. See the file for the kernel's contract.
pub const KERNEL_SRC: &str = include_str!("kernel.metal");

/// ksw's padded column count for a query: `slen * lanes`, where bwa picks 16 lanes when
/// `qlen * max_sc < 250` and 8 otherwise (`bwamem_pair.cpp:208`, the `KSW_XBYTE` test).
///
/// This is **observable**: the padding columns score 0, carry the diagonal, and therefore feed
/// `score2`. It is a property of the JOB, not of whatever width the backend happens to use, which is
/// why the GPU has to reproduce it even though it has no lanes at all.
/// Score ceiling under which the u8 rails are exact, mirroring `bwa-neon`'s constant of the same
/// name. A job at or above it takes the 32-bit kernel.
pub const U8_SCORE_LIMIT: i32 = 250;

pub fn padded_qlen(qlen: usize, max_sc: i32) -> usize {
    let lanes = if (qlen as i32) * max_sc < 250 { 16 } else { 8 };
    qlen.div_ceil(lanes) * lanes
}

/// One rescue job: a read against a reference window, plus the two thresholds the pass uses.
#[derive(Clone, Copy, Debug)]
pub struct RescueJob<'a> {
    /// The read, 2-bit codes with 4 for N.
    pub query: &'a [u8],
    /// The reference window, same encoding.
    pub target: &'a [u8],
    /// Row maxima below this are not `score2` candidates. `i32::MAX` suppresses the list.
    pub minsc: i32,
    /// Stop the moment a row max reaches this. `i32::MAX` disables it.
    pub endsc: i32,
}

/// Per-job geometry handed to the kernel. `repr(C)` and padded to match the MSL `struct Job`
/// field for field; a mismatch here is silent corruption, not a compile error, which is why the
/// layout is asserted in a test.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct GpuJob {
    q_off: u32,
    q_len: u32,
    q_pad: u32,
    t_off: u32,
    t_len: u32,
    endsc: i32,
    kind: u32,
    _pad: u32,
}

/// Per-job result coming back. `score2`/`te2` are deliberately absent: they are computed on the CPU
/// from `rowmax` with the same [`SuboptimalTracker`] the scalar and NEON paths use, so the merge
/// rule and the exclusion window exist once rather than three times.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct GpuRes {
    score: i32,
    te: i32,
    qe: i32,
    limit: i32,
}

#[cfg(feature = "metal")]
mod backend {
    use super::*;
    use metal::{Device, MTLResourceOptions, MTLSize};

    /// A ready-to-use Metal device, queue and compiled pipeline.
    ///
    /// Construction can fail for ordinary reasons (no device, driver refuses the source), and every
    /// one of them is a `None` rather than a panic: the caller's contract is to fall back to the CPU
    /// silently, since the answer is identical either way.
    pub struct MetalRescue {
        device: Device,
        queue: metal::CommandQueue,
        /// 32-bit rails, correct for every job.
        pipeline: metal::ComputePipelineState,
        /// u8 rails, for jobs whose score ceiling fits a byte. Same control flow, a quarter of the
        /// device-memory traffic; the host picks between them per job, as `fwd_local_sw_batch` does.
        pipeline_u8: metal::ComputePipelineState,
        /// Nanoseconds of the last `forward_batch`, split three ways: filling the shared buffers,
        /// the submitted work, and the CPU-side `score2` pass.
        ///
        /// This exists because the throughput probe used to time `forward_batch` WHOLE, and that
        /// number is the sum of three unrelated things. A `uchar4` experiment was abandoned on the
        /// strength of a comparison that measurement could not distinguish (see ROADMAP.md), so the
        /// split comes before any further kernel work.
        ///
        /// The middle figure is wall time around `commit`/`wait_until_completed`, not the driver's
        /// own GPU counters: `metal` 0.33 does not expose `GPUStartTime`, and adding an Objective-C
        /// runtime dependency to read them is not worth it for a probe. It therefore includes
        /// submission and scheduling latency, which is the honest reading of "what the GPU arm
        /// costs the caller".
        last_pack_ns: std::sync::atomic::AtomicU64,
        last_gpu_ns: std::sync::atomic::AtomicU64,
        last_post_ns: std::sync::atomic::AtomicU64,
    }

    impl MetalRescue {
        /// Open the default device and compile the kernel. `None` if either step fails.
        pub fn new() -> Option<Self> {
            let device = Device::system_default()?;
            let opts = metal::CompileOptions::new();
            let lib = device.new_library_with_source(KERNEL_SRC, &opts).ok()?;
            let f = lib.get_function("rescue_fwd", None).ok()?;
            let pipeline = device.new_compute_pipeline_state_with_function(&f).ok()?;
            let f8 = lib.get_function("rescue_fwd_u8", None).ok()?;
            let pipeline_u8 = device.new_compute_pipeline_state_with_function(&f8).ok()?;
            let queue = device.new_command_queue();
            Some(Self {
                device,
                queue,
                pipeline,
                pipeline_u8,
                last_pack_ns: std::sync::atomic::AtomicU64::new(0),
                last_gpu_ns: std::sync::atomic::AtomicU64::new(0),
                last_post_ns: std::sync::atomic::AtomicU64::new(0),
            })
        }

        /// The last `forward_batch` split into `(pack, submitted, score2)` seconds.
        ///
        /// Only the middle one is affected by a change to the kernel. The other two are the price of
        /// the seam itself, and knowing them is what stops a host-side regression from being read as
        /// a kernel result, which is exactly what happened to the `uchar4` attempt.
        pub fn last_phases(&self) -> (f64, f64, f64) {
            use std::sync::atomic::Ordering::Relaxed;
            (
                self.last_pack_ns.load(Relaxed) as f64 / 1e9,
                self.last_gpu_ns.load(Relaxed) as f64 / 1e9,
                self.last_post_ns.load(Relaxed) as f64 / 1e9,
            )
        }

        /// The device's name, for a startup line if a caller wants one.
        pub fn device_name(&self) -> String {
            self.device.name().to_string()
        }

        /// Run the forward pass for a whole batch.
        ///
        /// # Parameters
        /// As `ksw_local_fwd`, shared by the batch except `minsc`/`endsc`, which travel per job.
        /// `mat` must be bwa's uniform DNA matrix; the kernel reads three numbers out of it rather
        /// than the matrix itself, exactly as the CPU vector kernels do.
        ///
        /// # Returns
        /// One `(score, te, qe, score2, te2)` per job, in job order, equal to `ksw_local_fwd` on that
        /// job. The equality is the gate in `tests`, not an aspiration.
        #[allow(clippy::too_many_arguments)]
        pub fn forward_batch(
            &self,
            jobs: &[RescueJob],
            m: usize,
            mat: &[i8],
            o_del: i32,
            e_del: i32,
            o_ins: i32,
            e_ins: i32,
            max_sc: i32,
        ) -> Vec<(i32, i32, i32, i32, i32)> {
            if jobs.is_empty() {
                return Vec::new();
            }
            let t_pack = std::time::Instant::now();
            let mtch = mat[0] as i32;
            let mispen = -(mat[1] as i32);
            let npen = -(mat[m - 1] as i32);
            let (oe_del, oe_ins) = (o_del + e_del, o_ins + e_ins);

            // The same score-ceiling test the CPU dispatch uses: a local alignment can match at
            // most `min(qlen, tlen)` bases and only loses score from there, so `min(len) * max_sc`
            // bounds every H/E/F cell. Under `U8_SCORE_LIMIT` the u8 rails are exact; at or above it
            // they would saturate, so those jobs take the 32-bit kernel.
            let fits_u8 = |j: &RescueJob| {
                (j.query.len().min(j.target.len()) as i32) * max_sc < U8_SCORE_LIMIT
            };

            // One flat sequence buffer, written straight into shared memory: the CPU fills the
            // `MTLBuffer`'s own storage, and the GPU reads the same bytes. No staging copy exists.
            let total: usize = jobs.iter().map(|j| j.query.len() + j.target.len()).sum();
            let seqs = self
                .device
                .new_buffer(total.max(1) as u64, MTLResourceOptions::StorageModeShared);
            let rail_qmax = jobs
                .iter()
                .map(|j| padded_qlen(j.query.len(), max_sc))
                .max()
                .unwrap_or(1)
                .max(1);
            let rail_tmax = jobs
                .iter()
                .map(|j| j.target.len())
                .max()
                .unwrap_or(1)
                .max(1);

            let mut gjobs = vec![GpuJob::default(); jobs.len()];
            {
                // SAFETY: the buffer was just allocated with exactly `total` bytes and nothing else
                // holds a reference to it; the writes below are bounded by that length.
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
                    gjobs[k] = GpuJob {
                        q_off: q_off as u32,
                        q_len: j.query.len() as u32,
                        q_pad: padded_qlen(j.query.len(), max_sc) as u32,
                        t_off: t_off as u32,
                        t_len: j.target.len() as u32,
                        endsc: j.endsc,
                        kind: u32::from(fits_u8(j)),
                        _pad: 0,
                    };
                }
            }

            let jobs_buf = self.device.new_buffer_with_data(
                gjobs.as_ptr() as *const _,
                std::mem::size_of_val(&gjobs[..]) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let res_buf = self.device.new_buffer(
                (jobs.len() * std::mem::size_of::<GpuRes>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let rail_bytes = (jobs.len() * rail_qmax * 4) as u64;
            let rail_bytes_u8 = (jobs.len() * rail_qmax) as u64;
            let h_prev = self
                .device
                .new_buffer(rail_bytes, MTLResourceOptions::StorageModePrivate);
            let h_cur = self
                .device
                .new_buffer(rail_bytes, MTLResourceOptions::StorageModePrivate);
            let e_rail = self
                .device
                .new_buffer(rail_bytes, MTLResourceOptions::StorageModePrivate);
            // The u8 kernel's rails: a quarter of the bytes, which is the point of it.
            let h_prev8 = self
                .device
                .new_buffer(rail_bytes_u8, MTLResourceOptions::StorageModePrivate);
            let h_cur8 = self
                .device
                .new_buffer(rail_bytes_u8, MTLResourceOptions::StorageModePrivate);
            let e_rail8 = self
                .device
                .new_buffer(rail_bytes_u8, MTLResourceOptions::StorageModePrivate);
            let rowmax = self.device.new_buffer(
                (jobs.len() * rail_tmax * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let scalars: [i32; 7] = [mtch, mispen, npen, oe_del, e_del, oe_ins, e_ins];
            let dims: [u32; 3] = [rail_qmax as u32, rail_tmax as u32, jobs.len() as u32];

            let cb = self.queue.new_command_buffer();
            // Two dispatches in one command buffer, one per rail width. Each kernel returns
            // immediately for the jobs that are not its kind, so the split costs one extra launch
            // and no host bookkeeping, and the two write disjoint entries of `res_buf`.
            for (pipe, hp, hc, ev) in [
                (&self.pipeline, &h_prev, &h_cur, &e_rail),
                (&self.pipeline_u8, &h_prev8, &h_cur8, &e_rail8),
            ] {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(&seqs), 0);
                enc.set_buffer(1, Some(&jobs_buf), 0);
                enc.set_buffer(2, Some(&res_buf), 0);
                enc.set_buffer(3, Some(hp), 0);
                enc.set_buffer(4, Some(hc), 0);
                enc.set_buffer(5, Some(ev), 0);
                enc.set_buffer(6, Some(&rowmax), 0);
                for (i, v) in scalars.iter().enumerate() {
                    enc.set_bytes(7 + i as u64, 4, v as *const i32 as *const std::ffi::c_void);
                }
                for (i, v) in dims.iter().enumerate() {
                    enc.set_bytes(14 + i as u64, 4, v as *const u32 as *const std::ffi::c_void);
                }
                let tg = pipe
                    .max_total_threads_per_threadgroup()
                    .min(jobs.len() as u64)
                    .max(1);
                enc.dispatch_threads(
                    MTLSize::new(jobs.len() as u64, 1, 1),
                    MTLSize::new(tg, 1, 1),
                );
                enc.end_encoding();
            }
            let t_gpu = std::time::Instant::now();
            cb.commit();
            cb.wait_until_completed();
            let gpu_ns = t_gpu.elapsed().as_nanos() as u64;
            let t_post = std::time::Instant::now();

            // SAFETY: both buffers are shared, sized above, and the command buffer has completed, so
            // the GPU is no longer writing them.
            let res = unsafe {
                std::slice::from_raw_parts(res_buf.contents() as *const GpuRes, jobs.len())
            };
            let rows = unsafe {
                std::slice::from_raw_parts(rowmax.contents() as *const i32, jobs.len() * rail_tmax)
            };

            // `score2` on the CPU, through the shared tracker. The GPU produced the row maxima; the
            // subtle part (merging consecutive rows, then excluding everything within
            // `ceil(score / max_sc)` of `te`) stays in the one implementation all three backends use.
            let out: Vec<(i32, i32, i32, i32, i32)> = jobs
                .iter()
                .enumerate()
                .map(|(k, j)| {
                    let r = res[k];
                    let mut b = SuboptimalTracker::new();
                    if r.limit >= 0 {
                        for i in 0..=r.limit {
                            // Column-major, matching the kernel: row `i` of job `k` is at
                            // `i * n_jobs + k`.
                            b.push_row(i, rows[i as usize * jobs.len() + k], j.minsc);
                        }
                    }
                    let (score2, te2) = b.finish(r.score, r.te, max_sc);
                    (r.score, r.te, r.qe, score2, te2)
                })
                .collect();
            {
                use std::sync::atomic::Ordering::Relaxed;
                // `t_pack` covers everything before the submission, so the pack figure is its
                // elapsed time minus the two later phases.
                let total = t_pack.elapsed().as_nanos() as u64;
                let post = t_post.elapsed().as_nanos() as u64;
                self.last_pack_ns
                    .store(total.saturating_sub(gpu_ns + post), Relaxed);
                self.last_gpu_ns.store(gpu_ns, Relaxed);
                self.last_post_ns.store(post, Relaxed);
            }
            out
        }
    }
}

#[cfg(feature = "metal")]
pub use backend::MetalRescue;

/// Stand-in when the crate is built without the `metal` feature: construction always fails, so the
/// caller takes its CPU path. Present so callers compile identically on every platform and the
/// fallback needs no `cfg` of its own.
#[cfg(not(feature = "metal"))]
pub struct MetalRescue;

#[cfg(not(feature = "metal"))]
impl MetalRescue {
    /// Always `None`: this build has no Metal support compiled in.
    pub fn new() -> Option<Self> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two `repr(C)` structs must match the MSL declarations field for field. A mismatch is
    /// silent corruption rather than a compile error, so it is asserted rather than assumed.
    #[test]
    fn gpu_structs_have_the_expected_layout() {
        assert_eq!(std::mem::size_of::<GpuJob>(), 32);
        assert_eq!(std::mem::size_of::<GpuRes>(), 16);
        assert_eq!(std::mem::align_of::<GpuJob>(), 4);
    }

    /// The byte-identity gate: the Metal kernel against `ksw_local_fwd`, field for field, on
    /// mate-rescue-shaped jobs.
    ///
    /// This is the same generator shape the NEON gate uses, and it draws bases `% 5` rather than
    /// `% 4` so N appears on both sides. Every job also carries a finite `endsc` on half the batch,
    /// so the freeze path runs and `limit` is exercised rather than always being `tlen - 1`.
    ///
    /// Skipped, not failed, when the crate is built without the `metal` feature or on a machine with
    /// no Metal device: that is the fallback contract, and a test that failed there would make the
    /// fallback unusable in CI.
    #[test]
    #[cfg(feature = "metal")]
    fn metal_matches_scalar() {
        use bwa_mem4_extend::ksw_local_fwd;
        let Some(gpu) = MetalRescue::new() else {
            eprintln!("skipping metal_matches_scalar: no Metal device");
            return;
        };
        eprintln!("device: {}", gpu.device_name());

        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
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
        let max_sc = a as i32;

        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (mut qs, mut ts) = (Vec::new(), Vec::new());
        for _ in 0..2000 {
            let qlen = 5 + (next() % 146) as usize;
            let tlen = qlen + (next() % 500) as usize;
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
            let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
            if next() % 5 != 0 {
                for _ in 0..(1 + next() % 2) {
                    if tlen > qlen {
                        let at = (next() as usize) % (tlen - qlen + 1);
                        t[at..at + qlen].copy_from_slice(&q);
                    }
                }
                for _ in 0..(next() % 4) {
                    let p = (next() as usize) % qlen;
                    q[p] = (next() % 4) as u8;
                }
            }
            // N on both sides of every job, plus a both-N cell: the same rule the CPU generators
            // follow since #43.
            q[(next() as usize) % qlen] = 4;
            t[(next() as usize) % tlen] = 4;
            let at = (next() as usize) % qlen;
            q[at] = 4;
            t[at.min(tlen - 1)] = 4;
            qs.push(q);
            ts.push(t);
        }

        let minsc = 19;
        let jobs: Vec<RescueJob> = (0..qs.len())
            .map(|i| RescueJob {
                query: &qs[i],
                target: &ts[i],
                minsc,
                // Half the batch freezes early, half runs to the end of its window.
                endsc: if i % 2 == 0 { i32::MAX } else { 30 },
            })
            .collect();

        let got = gpu.forward_batch(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, max_sc);
        assert_eq!(got.len(), jobs.len());
        for (i, j) in jobs.iter().enumerate() {
            let lanes = if j.query.len() as i32 * max_sc < 250 {
                16
            } else {
                8
            };
            let want = ksw_local_fwd(
                j.query, j.target, 5, &mat, o_del, e_del, o_ins, e_ins, j.minsc, j.endsc, max_sc,
                lanes,
            );
            assert_eq!(
                got[i],
                want,
                "job {i} (qlen {}, tlen {}, endsc {})",
                j.query.len(),
                j.target.len(),
                j.endsc
            );
        }
    }

    /// Issue #54 trap 2, applied to the GPU: the argmax tie rule, on inputs where ties are the norm.
    ///
    /// Periodic targets against in-phase queries, so every cell of a row ties. A kernel that wrote
    /// `>=` instead of `>` returns the right `score` and a different `qe`, and only here. Period 7
    /// shares no factor with any width in this project, so a tie cannot coincide with a boundary by
    /// luck.
    #[test]
    #[cfg(feature = "metal")]
    fn metal_tie_rule_matches_scalar() {
        use bwa_mem4_extend::ksw_local_fwd;
        let Some(gpu) = MetalRescue::new() else {
            return;
        };
        let mat = super::tests::bwa_matrix(1, 4);
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        let (mut qs, mut ts) = (Vec::new(), Vec::new());
        for &period in &[1usize, 2, 3, 4, 7] {
            for &qlen in &[9usize, 32, 63, 100, 149] {
                for &tlen in &[qlen, qlen + 1, qlen * 3, qlen * 3 + 5] {
                    qs.push(
                        (0..qlen)
                            .map(|i| (i % period) as u8 % 4)
                            .collect::<Vec<u8>>(),
                    );
                    ts.push(
                        (0..tlen)
                            .map(|i| (i % period) as u8 % 4)
                            .collect::<Vec<u8>>(),
                    );
                }
            }
        }
        for &minsc in &[1i32, 19, 100] {
            let jobs: Vec<RescueJob> = (0..qs.len())
                .map(|i| RescueJob {
                    query: &qs[i],
                    target: &ts[i],
                    minsc,
                    endsc: i32::MAX,
                })
                .collect();
            let got = gpu.forward_batch(&jobs, 5, &mat, o_del, e_del, o_ins, e_ins, 1);
            for (i, j) in jobs.iter().enumerate() {
                let want = ksw_local_fwd(
                    j.query,
                    j.target,
                    5,
                    &mat,
                    o_del,
                    e_del,
                    o_ins,
                    e_ins,
                    minsc,
                    i32::MAX,
                    1,
                    16,
                );
                assert_eq!(got[i], want, "tie job {i} (period pattern, minsc {minsc})");
            }
        }
    }

    /// Issue #54 trap 6, applied to the GPU: batching must not make a job's result depend on the
    /// company it keeps.
    ///
    /// The same jobs are submitted in ten random permutations and each must give its own result
    /// every time. On this backend that also checks the rail indexing: every thread addresses its
    /// H/E rails through its own `rail` slot, and an off-by-one there would show up only when the
    /// neighbours change.
    #[test]
    #[cfg(feature = "metal")]
    fn metal_batch_order_invariant() {
        let Some(gpu) = MetalRescue::new() else {
            return;
        };
        let mat = super::tests::bwa_matrix(1, 4);
        let mut state = 0x0ddb_a11c_0ffe_e511u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (mut qs, mut ts) = (Vec::new(), Vec::new());
        for _ in 0..150 {
            let qlen = 5 + (next() % 146) as usize;
            let tlen = qlen + (next() % 300) as usize;
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
            let q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
            let at = (next() as usize) % (tlen - qlen + 1);
            t[at..at + qlen].copy_from_slice(&q);
            qs.push(q);
            ts.push(t);
        }
        let n = qs.len();
        let mk = |i: usize| RescueJob {
            query: &qs[i],
            target: &ts[i],
            minsc: 19,
            endsc: if i % 3 == 0 { 40 } else { i32::MAX },
        };
        let reference =
            gpu.forward_batch(&(0..n).map(mk).collect::<Vec<_>>(), 5, &mat, 6, 1, 6, 1, 1);
        for round in 0..10 {
            let mut ord: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let s = (next() as usize) % (i + 1);
                ord.swap(i, s);
            }
            let jobs: Vec<RescueJob> = ord.iter().map(|&i| mk(i)).collect();
            let got = gpu.forward_batch(&jobs, 5, &mat, 6, 1, 6, 1, 1);
            for (slot, &i) in ord.iter().enumerate() {
                assert_eq!(got[slot], reference[i], "round {round}, job {i}");
            }
        }
    }

    /// Throughput on mate-rescue-shaped work, for the record. Ignored by default: it is a
    /// measurement, not a gate.
    #[test]
    #[ignore = "throughput probe, run explicitly"]
    #[cfg(feature = "metal")]
    fn metal_throughput() {
        let Some(gpu) = MetalRescue::new() else {
            return;
        };
        let mat = super::tests::bwa_matrix(1, 4);
        let mut state = 0xfeed_face_1234_5678u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        // The shape the probe in ROADMAP.md reports for a 150 bp run: 150 bp reads against ~620 bp
        // windows.
        let (mut qs, mut ts) = (Vec::new(), Vec::new());
        let n: usize = std::env::var("BWA4_METAL_JOBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        for _ in 0..n {
            let qlen = 150usize;
            let tlen = 600 + (next() % 50) as usize;
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
            let at = (next() as usize) % (tlen - qlen + 1);
            t[at..at + qlen].copy_from_slice(&q);
            qs.push(q);
            ts.push(t);
        }
        let jobs: Vec<RescueJob> = (0..qs.len())
            .map(|i| RescueJob {
                query: &qs[i],
                target: &ts[i],
                minsc: 19,
                endsc: i32::MAX,
            })
            .collect();
        // Cells as this project counts them: real query x real target, summed over jobs.
        let cells: u64 = jobs
            .iter()
            .map(|j| (j.query.len() * j.target.len()) as u64)
            .sum();
        let _ = gpu.forward_batch(&jobs, 5, &mat, 6, 1, 6, 1, 1);
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            let out = gpu.forward_batch(&jobs, 5, &mat, 6, 1, 6, 1, 1);
            let dt = t0.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            best = best.min(dt);
        }
        let (pack, sub, post) = gpu.last_phases();
        eprintln!(
            "metal rescue_fwd: {} jobs, {cells} cells, device {}",
            jobs.len(),
            gpu.device_name()
        );
        eprintln!(
            "  whole call   {:>8.4} s   {:>6.2} Gcell/s",
            best,
            cells as f64 / best / 1e9
        );
        // The split that the uchar4 experiment lacked: only the middle line moves when the kernel
        // changes. Reported from the last rep, which is the one the best-of picked often enough for
        // the shares to be representative; the absolute times of the three add up to that rep, not
        // to `best`.
        eprintln!(
            "  of which     pack {:.4} s | submitted {:.4} s | score2 {:.4} s",
            pack, sub, post
        );
        eprintln!(
            "  submitted-only throughput: {:.2} Gcell/s",
            cells as f64 / sub.max(1e-9) / 1e9
        );
    }

    /// bwa's 5x5 DNA matrix, as `bwa_fill_scmat` builds it.
    pub(super) fn bwa_matrix(a: i8, b: i8) -> Vec<i8> {
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

    /// The padding rule is bwa's, and it is observable through `score2`. Pinned at the 250 boundary
    /// the `KSW_XBYTE` test turns on.
    #[test]
    fn padded_qlen_follows_bwa() {
        assert_eq!(padded_qlen(150, 1), 160); // 16 lanes: 150*1 < 250
        assert_eq!(padded_qlen(160, 1), 160);
        assert_eq!(padded_qlen(249, 1), 256); // still 16 lanes at the boundary
        assert_eq!(padded_qlen(250, 1), 256); // 250*1 == 250 -> 8 lanes, 250 -> 256
        assert_eq!(padded_qlen(100, 3), 104); // 100*3 >= 250 -> 8 lanes
    }
}
