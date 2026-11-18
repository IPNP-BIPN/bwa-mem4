//! Metal backend for the banded seed extension (`ksw_extend2`), byte-identical to the scalar kernel,
//! and for the mate-rescue band pass (`band16.metal`, through `bwa_neon::offload`), identical to
//! the NEON band kernel lane for lane.
//!
//! # Layout
//!
//! The two earlier Metal kernels gave each alignment one thread. That layout keeps the DP rails in
//! device memory and needs ~32 000 jobs in flight to hide the latency, while a production call
//! carries 1 000 to 7 000. This one gives each job **16 lanes** that run its rows in lockstep, with
//! the whole `eh` rail in registers (`extend16.metal`). The only horizontal dependency of a row, F,
//! is an exact prefix max, so the kernel computes the scalar's integers in int32 with no
//! saturation tier and no fallback for large scores.
//!
//! # What is offloaded, what is not
//!
//! Queries up to [`MAX_QLEN`] bases run on the GPU, longer ones on the CPU with [`ksw_extend2`]
//! while the GPU works. The answer is the same either way, which is what the [`SwBackend`] gates
//! check (`cargo test -p bwa-mem4-metal --features metal`).
//!
//! [`SwBackend`]: bwa_extend::SwBackend
//! [`ksw_extend2`]: bwa_extend::ksw_extend2

/// The MSL source, compiled by the driver at run time (no Xcode, no `metallib` in the tree).
pub const EXTEND16_SRC: &str = include_str!("extend16.metal");

/// The mate-rescue band kernels (`bwa_neon::offload`), compiled the same way.
pub const BAND16_SRC: &str = include_str!("band16.metal");

/// Query columns per lane for a band whose query pads to `quanta` columns, one of the compiled
/// buckets, or `None` past 256 columns. The band's width does not matter: the kernel runs each row
/// over the query columns the band holds.
pub fn band_columns(quanta: i32) -> Option<usize> {
    if !(1..=256).contains(&quanta) {
        return None;
    }
    let c = (quanta as usize).div_ceil(16);
    BAND_BUCKETS.iter().copied().find(|&b| b >= c)
}

/// The compiled band buckets, in pipeline order.
pub const BAND_BUCKETS: [usize; 8] = [1, 2, 4, 6, 8, 10, 12, 16];

/// Longest query the GPU kernel takes: 16 lanes times at most 16 columns hold `qlen + 1` slots.
pub const MAX_QLEN: usize = 16 * 16 - 1;

/// Columns per lane for a query of `qlen` bases, one of the eight compiled buckets (2, 4, .., 16),
/// or `None` when the query is too long for the GPU.
pub fn columns_per_lane(qlen: usize) -> Option<usize> {
    if qlen > MAX_QLEN {
        return None;
    }
    // ceil((qlen + 1) / 16), rounded up to an even count so only eight kernels are compiled.
    Some((qlen + 1).div_ceil(16).div_ceil(2) * 2)
}

/// Scores the kernel computes instead of looking up: `(match, mismatch, n)` when `mat` has bwa's
/// uniform DNA shape (`m == 5`, one value on the diagonal, one off it, one for every cell that
/// touches N), else `None` and the batch stays on the CPU.
pub fn uniform_scores(m: usize, mat: &[i8]) -> Option<(i32, i32, i32)> {
    if m != 5 || mat.len() < 25 {
        return None;
    }
    let (a, b, n) = (mat[0], mat[1], mat[4]);
    let ok = (0..5).all(|t| {
        (0..5).all(|q| {
            let want = if t == 4 || q == 4 {
                n
            } else if t == q {
                a
            } else {
                b
            };
            mat[t * 5 + q] == want
        })
    });
    ok.then(|| (i32::from(a), i32::from(b), i32::from(n)))
}

/// Whether every H of this job fits the kernel's `(H << 8) | j` row-max key: H never exceeds
/// `max(h0, 0) + qlen * max_sc`, since every other term of the recurrence is a penalty.
pub fn scores_fit_key(h0: i32, qlen: usize, max_sc: i32) -> bool {
    i64::from(h0.max(0)) + qlen as i64 * i64::from(max_sc.max(0)) < 1 << 22
}

/// The band clamp of `ksw_extend2`, done on the host in `f64` exactly as the reference does, so the
/// kernel holds no floating point.
#[allow(clippy::too_many_arguments)]
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
    let reach = |o: i32, e: i32| {
        let v = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o))
            / f64::from(e))
            + 1.0) as i32;
        v.max(1)
    };
    w0.min(reach(o_ins, e_ins)).min(reach(o_del, e_del))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod backend {
    use super::{
        band_columns, clamp_band, columns_per_lane, scores_fit_key, uniform_scores, BAND16_SRC,
        BAND_BUCKETS, EXTEND16_SRC,
    };
    use bwa_extend::{ksw_extend2, ExtendJob, ExtendResult, SwBackend};
    use bwa_neon::offload::{BandOffload, BandOut, BandPending, BandScoring, BandTask};
    use metal::objc::rc::autoreleasepool;
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{
        Buffer, CommandBuffer, CommandQueue, CompileOptions, ComputePipelineState, Device,
        MTLCommandBufferStatus, MTLDispatchType, MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::{Arc, Mutex};

    /// `repr(C)` mirror of the MSL `struct EJob`. Scalars only: MSL aligns vector types to their
    /// size and Rust does not, which silently shifts field offsets.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct GpuJob {
        q_off: u32,
        q_len: u32,
        t_off: u32,
        t_len: u32,
        h0: i32,
        w: i32,
        out: u32,
        _pad: u32,
    }

    /// `repr(C)` mirror of the MSL `struct ERes`, padded to 32 bytes.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct GpuRes {
        score: i32,
        qle: i32,
        tle: i32,
        gtle: i32,
        gscore: i32,
        max_off: i32,
        _pad0: i32,
        _pad1: i32,
    }

    /// `repr(C)` mirror of the MSL `struct Params`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Params {
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        zdrop: i32,
        n_jobs: u32,
        sc_match: i32,
        sc_mismatch: i32,
        sc_n: i32,
        _pad: [i32; 3],
    }

    /// Lanes per job, and threads per threadgroup (4 jobs): 32 to 128 measured alike, 256 and up
    /// lose at production batch sizes (a threadgroup frees its slot only when its longest job ends).
    const LANES: u64 = 16;
    const TG: u64 = 64;
    /// Command queues, about one per worker.
    const QUEUES: usize = 16;

    /// The scoring of one [`SwBackend::extend_batch`] call, as the trait passes it.
    #[derive(Clone, Copy)]
    pub struct Scoring<'a> {
        pub m: usize,
        pub mat: &'a [i8],
        pub o_del: i32,
        pub e_del: i32,
        pub o_ins: i32,
        pub e_ins: i32,
        pub w: i32,
        pub end_bonus: i32,
        pub zdrop: i32,
    }

    impl Scoring<'_> {
        fn max_sc(&self) -> i32 {
            i32::from(
                self.mat[..self.m * self.m]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
            )
        }

        /// The scalar reference on one job.
        pub fn scalar(&self, j: &ExtendJob) -> ExtendResult {
            ksw_extend2(
                j.query,
                j.target,
                self.m,
                self.mat,
                self.o_del,
                self.e_del,
                self.o_ins,
                self.e_ins,
                self.w,
                self.end_bonus,
                self.zdrop,
                j.h0,
            )
        }

        /// `backend.extend_batch` on `jobs` with this scoring.
        pub fn run<B: SwBackend>(&self, backend: &B, jobs: &[ExtendJob]) -> Vec<ExtendResult> {
            backend.extend_batch(
                jobs,
                self.m,
                self.mat,
                self.o_del,
                self.e_del,
                self.o_ins,
                self.e_ins,
                self.w,
                self.end_bonus,
                self.zdrop,
            )
        }
    }

    /// Device, queue and the eight compiled pipelines. Every Metal object here is `Send + Sync`,
    /// and a command queue takes command buffers from several threads, so one instance serves all
    /// workers.
    pub struct MetalExtend {
        device: Device,
        /// Command buffers of one queue run nearly one after another (8 batches of 256 bands in
        /// flight: 8.2 ms against 1.8 ms each); calls take these in turn so theirs overlap.
        queues: Vec<CommandQueue>,
        next_queue: AtomicU64,
        /// One pipeline per bucket of [`columns_per_lane`]: 2, 4, .., 16 columns per lane.
        pipes: Vec<ComputePipelineState>,
        /// One pipeline per entry of [`BAND_BUCKETS`].
        band_pipes: Vec<ComputePipelineState>,
        /// GPU execution time of the last batch (`GPUEndTime - GPUStartTime`), nanoseconds.
        last_kernel_ns: AtomicU64,
        /// Shared buffers the GPU is done with, for the next call. A fresh buffer per call is fresh
        /// pages, zeroed by the kernel: +1 to +1.7 s of system time on 2M GIAB pairs.
        pool: Arc<Mutex<Vec<Buffer>>>,
    }

    /// A batch on the GPU. Dropping it unwaited is fine: the command buffer keeps its buffers alive
    /// until it completes, and nobody reads the results.
    pub struct Pending<'a> {
        owner: &'a MetalExtend,
        cb: CommandBuffer,
        /// The call's one buffer (jobs, results, sequences); `None` once handed back to the pool.
        buf: Option<Buffer>,
        /// Byte offset of the results in `buf`.
        res_off: usize,
        n: usize,
    }

    /// Back to the pool only once the GPU is done with it: reused early, a buffer still being read
    /// would be overwritten. Dropped unfinished (the CPU took every job), it is simply released, and
    /// the command buffer keeps it alive until the kernel ends.
    impl Drop for Pending<'_> {
        fn drop(&mut self) {
            if let Some(b) = self.buf.take() {
                if self.is_done() {
                    self.owner.give_buffer(b);
                }
            }
        }
    }

    impl Pending<'_> {
        /// True once the GPU has finished (or failed): [`Pending::wait`] then returns at once.
        pub fn is_done(&self) -> bool {
            matches!(
                self.cb.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
        }

        /// The results, in submission order, or `None` if the GPU reported an error, in which case
        /// the caller recomputes on the CPU: a failed launch must never become a wrong answer.
        // objc's `sel_impl!` tests `feature = "cargo-clippy"`, which this crate does not declare.
        #[allow(unexpected_cfgs)]
        pub fn wait(mut self) -> Option<Vec<ExtendResult>> {
            self.cb.wait_until_completed();
            if self.cb.status() != MTLCommandBufferStatus::Completed {
                return None;
            }
            // SAFETY: both selectors exist on every MTLCommandBuffer and return a double.
            let (t0, t1): (f64, f64) = unsafe {
                (
                    msg_send![self.cb.as_ref(), GPUStartTime],
                    msg_send![self.cb.as_ref(), GPUEndTime],
                )
            };
            self.owner
                .last_kernel_ns
                .store(((t1 - t0).max(0.0) * 1e9) as u64, Relaxed);
            let buf = self.buf.as_ref()?;
            // SAFETY: `n` results at `res_off` in a shared buffer sized for them, every slot written
            // by the completed kernel.
            let r = unsafe {
                std::slice::from_raw_parts(
                    (buf.contents() as *const u8).add(self.res_off) as *const GpuRes,
                    self.n,
                )
            };
            let out = Some(
                r.iter()
                    .map(|g| ExtendResult {
                        score: g.score,
                        qle: g.qle,
                        tle: g.tle,
                        gtle: g.gtle,
                        gscore: g.gscore,
                        max_off: g.max_off,
                    })
                    .collect(),
            );
            if let Some(b) = self.buf.take() {
                self.owner.give_buffer(b);
            }
            out
        }
    }

    impl MetalExtend {
        /// Open the default device and compile the kernels. `None` if either step fails, which is
        /// the caller's cue to stay on the CPU: the answer is identical either way.
        pub fn new() -> Option<Self> {
            autoreleasepool(|| {
                let device = Device::system_default()?;
                let lib = device
                    .new_library_with_source(EXTEND16_SRC, &CompileOptions::new())
                    .ok()?;
                let pipe = |c: usize| {
                    let f = lib.get_function(&format!("extend16_c{c}"), None).ok()?;
                    device.new_compute_pipeline_state_with_function(&f).ok()
                };
                let pipes = (1..=8).map(|k| pipe(2 * k)).collect::<Option<Vec<_>>>()?;
                let blib = device
                    .new_library_with_source(BAND16_SRC, &CompileOptions::new())
                    .ok()?;
                let band_pipes = BAND_BUCKETS
                    .iter()
                    .map(|c| {
                        let f = blib.get_function(&format!("band16_c{c}"), None).ok()?;
                        device.new_compute_pipeline_state_with_function(&f).ok()
                    })
                    .collect::<Option<Vec<_>>>()?;
                let queues = (0..QUEUES).map(|_| device.new_command_queue()).collect();
                Some(Self {
                    device,
                    queues,
                    next_queue: AtomicU64::new(0),
                    pipes,
                    band_pipes,
                    last_kernel_ns: AtomicU64::new(0),
                    pool: Arc::new(Mutex::new(Vec::new())),
                })
            })
        }

        /// GPU execution time of the last batch waited on, seconds: the kernel alone, without
        /// packing, submission or the wait.
        pub fn last_kernel_seconds(&self) -> f64 {
            self.last_kernel_ns.load(Relaxed) as f64 / 1e9
        }

        /// The next command queue, round robin.
        fn queue(&self) -> &CommandQueue {
            let i = self.next_queue.fetch_add(1, Relaxed) as usize;
            &self.queues[i % self.queues.len()]
        }

        /// A shared buffer of at least `len` bytes, recycled when one is free.
        fn take_buffer(&self, len: usize) -> Buffer {
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(i) = pool.iter().position(|b| b.length() as usize >= len) {
                return pool.swap_remove(i);
            }
            drop(pool);
            self.device.new_buffer(
                len.next_power_of_two() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        /// Return a buffer the GPU no longer uses. The pool holds at most one per worker or so.
        fn give_buffer(&self, b: Buffer) {
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            if pool.len() < 64 {
                pool.push(b);
            }
        }

        /// The GPU's name, for logs.
        pub fn device_name(&self) -> String {
            self.device.name().to_string()
        }

        /// Whether the kernel takes this scoring at all (bwa's uniform matrix shape).
        pub fn takes_scoring(sc: &Scoring) -> bool {
            uniform_scores(sc.m, sc.mat).is_some()
        }

        /// Whether the kernel takes this job under a scoring it takes: query short enough for the
        /// registers, scores small enough for the row-max key.
        pub fn takes_job(sc: &Scoring, j: &ExtendJob) -> bool {
            columns_per_lane(j.query.len()).is_some()
                && scores_fit_key(j.h0, j.query.len(), sc.max_sc())
        }

        /// Pack `jobs` and commit them to the GPU without waiting. `None` if the scoring or any job
        /// is one the kernel does not take (see [`Self::takes_job`]), or the batch is too large to
        /// address.
        pub fn submit(&self, jobs: &[ExtendJob], sc: &Scoring) -> Option<Pending<'_>> {
            let (sc_match, sc_mismatch, sc_n) = uniform_scores(sc.m, sc.mat)?;
            let total: usize = jobs.iter().map(|j| j.query.len() + j.target.len()).sum();
            if jobs.is_empty()
                || total > u32::MAX as usize
                || !jobs.iter().all(|j| Self::takes_job(sc, j))
            {
                return None;
            }
            let max_sc = sc.max_sc();
            autoreleasepool(|| {
                let mut buckets: [Vec<GpuJob>; 8] = Default::default();
                let mut off = 0usize;
                for (k, j) in jobs.iter().enumerate() {
                    let c = columns_per_lane(j.query.len())?;
                    let q_off = off;
                    off += j.query.len();
                    let t_off = off;
                    off += j.target.len();
                    buckets[c / 2 - 1].push(GpuJob {
                        q_off: q_off as u32,
                        q_len: j.query.len() as u32,
                        t_off: t_off as u32,
                        t_len: j.target.len() as u32,
                        h0: j.h0,
                        w: clamp_band(
                            sc.w,
                            j.query.len(),
                            max_sc,
                            sc.end_bonus,
                            sc.o_ins,
                            sc.e_ins,
                            sc.o_del,
                            sc.e_del,
                        ),
                        out: k as u32,
                        _pad: 0,
                    });
                }
                // Longest targets first: the two jobs of a simdgroup, and the simdgroups of a
                // threadgroup, then run about as many rows.
                for list in &mut buckets {
                    list.sort_unstable_by_key(|g| std::cmp::Reverse(g.t_len));
                }
                // One buffer per call: the eight job lists, then the results, then the sequences,
                // each section 256-byte aligned.
                let align = |x: usize| x.div_ceil(256) * 256;
                let mut job_off = [0usize; 8];
                let mut at = 0usize;
                for (b, list) in buckets.iter().enumerate() {
                    job_off[b] = at;
                    at = align(at + std::mem::size_of_val(&list[..]));
                }
                let res_off = at;
                let seq_off = align(res_off + jobs.len() * std::mem::size_of::<GpuRes>());
                let buf = self.take_buffer(seq_off + total.max(1));
                // SAFETY: `buf` holds at least `seq_off + total` bytes, comes from a fresh
                // allocation or from a command buffer that has completed, and is not on the GPU
                // until the commit below.
                let base = buf.contents() as *mut u8;
                unsafe {
                    for (b, list) in buckets.iter().enumerate() {
                        std::ptr::copy_nonoverlapping(
                            list.as_ptr() as *const u8,
                            base.add(job_off[b]),
                            std::mem::size_of_val(&list[..]),
                        );
                    }
                    let mut p = base.add(seq_off);
                    for j in jobs {
                        std::ptr::copy_nonoverlapping(j.query.as_ptr(), p, j.query.len());
                        p = p.add(j.query.len());
                        std::ptr::copy_nonoverlapping(j.target.as_ptr(), p, j.target.len());
                        p = p.add(j.target.len());
                    }
                }
                let cb = self.queue().new_command_buffer().to_owned();
                // Concurrent: the buckets write disjoint result slots, and a serial encoder would
                // make each one wait for the previous bucket's longest job.
                let enc =
                    cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
                enc.set_buffer(0, Some(&buf), seq_off as u64);
                enc.set_buffer(2, Some(&buf), res_off as u64);
                for (b, list) in buckets.iter().enumerate() {
                    if list.is_empty() {
                        continue;
                    }
                    let pipe = &self.pipes[b];
                    let params = Params {
                        o_del: sc.o_del,
                        e_del: sc.e_del,
                        o_ins: sc.o_ins,
                        e_ins: sc.e_ins,
                        zdrop: sc.zdrop,
                        n_jobs: list.len() as u32,
                        sc_match,
                        sc_mismatch,
                        sc_n,
                        _pad: [0; 3],
                    };
                    enc.set_compute_pipeline_state(pipe);
                    enc.set_buffer(1, Some(&buf), job_off[b] as u64);
                    enc.set_bytes(
                        3,
                        std::mem::size_of::<Params>() as u64,
                        &params as *const Params as *const c_void,
                    );
                    // A whole number of simdgroups per threadgroup, so a job's 16 lanes never
                    // straddle two simdgroups.
                    let tg = TG.min(pipe.max_total_threads_per_threadgroup()) / 32 * 32;
                    let threads = list.len() as u64 * LANES;
                    enc.dispatch_thread_groups(
                        MTLSize::new(threads.div_ceil(tg), 1, 1),
                        MTLSize::new(tg, 1, 1),
                    );
                }
                enc.end_encoding();
                cb.commit();
                Some(Pending {
                    owner: self,
                    cb,
                    buf: Some(buf),
                    res_off,
                    n: jobs.len(),
                })
            })
        }
    }

    /// `repr(C)` mirror of the MSL `struct BJob`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct GpuBand {
        q_off: u32,
        q_len: u32,
        t_off: u32,
        t_len: u32,
        quanta: i32,
        o: i32,
        w: i32,
        thr: u32,
        rows_off: u32,
        out: u32,
        _pad: [u32; 2],
    }

    /// `repr(C)` mirror of the MSL `struct BRes`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct GpuBandRes {
        gmax: i32,
        te: i32,
        qe: i32,
        _pad: i32,
    }

    /// `repr(C)` mirror of the MSL `struct BParams`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct BandParams {
        oe_del: i32,
        e_del: i32,
        oe_ins: i32,
        e_ins: i32,
        sc_match: i32,
        sc_mismatch: i32,
        sc_n: i32,
        n_jobs: u32,
    }

    /// Bands packed for the GPU: job records with offsets into `seqs`, row maxima and results
    /// counted from 0. Batches concatenate these, rebasing the offsets.
    struct Packed {
        jobs: Vec<GpuBand>,
        seqs: Vec<u8>,
        n_rows: usize,
        sc: BandScoring,
    }

    impl Packed {
        fn new(tasks: &[BandTask], sc: &BandScoring) -> Option<Self> {
            if tasks.is_empty() || !tasks.iter().all(|t| MetalExtend::takes_band(t, sc)) {
                return None;
            }
            let total: usize = tasks.iter().map(|t| t.q.len() + t.t.len()).sum();
            let mut seqs = Vec::with_capacity(total);
            let mut jobs = Vec::with_capacity(tasks.len());
            let mut roff = 0usize;
            for (k, t) in tasks.iter().enumerate() {
                let off = seqs.len();
                seqs.extend_from_slice(t.q);
                seqs.extend_from_slice(t.t);
                jobs.push(GpuBand {
                    q_off: off as u32,
                    q_len: t.q.len() as u32,
                    t_off: (off + t.q.len()) as u32,
                    t_len: t.t.len() as u32,
                    quanta: t.quanta,
                    o: t.o,
                    w: t.w,
                    thr: u32::from(t.thr),
                    rows_off: roff as u32,
                    out: k as u32,
                    _pad: [0; 2],
                });
                roff += t.t.len();
            }
            Some(Self {
                jobs,
                seqs,
                n_rows: roff,
                sc: *sc,
            })
        }
    }

    fn same_scoring(a: &BandScoring, b: &BandScoring) -> bool {
        (a.a, a.b, a.n, a.oe_del, a.e_del, a.oe_ins, a.e_ins)
            == (b.a, b.b, b.n, b.oe_del, b.e_del, b.oe_ins, b.e_ins)
    }

    /// One command buffer of bands, possibly from several calls. Its buffer goes back to the pool
    /// once the GPU is done and nobody holds the batch.
    struct Batch {
        cb: CommandBuffer,
        buf: Option<Buffer>,
        res_off: usize,
        rows_off: usize,
        pool: Arc<Mutex<Vec<Buffer>>>,
    }

    impl Batch {
        fn done(&self) -> bool {
            matches!(
                self.cb.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
        }

        /// `n` results from `res0` and `n_rows` row maxima from `rows0`, once completed.
        fn read(
            &self,
            res0: usize,
            n: usize,
            rows0: usize,
            n_rows: usize,
        ) -> Option<(Vec<BandOut>, Vec<u8>)> {
            if self.cb.status() != MTLCommandBufferStatus::Completed {
                return None;
            }
            let base = self.buf.as_ref()?.contents() as *const u8;
            // SAFETY: the batch's results and rows sections hold these ranges, written by the
            // completed kernel; the buffer lives as long as `self`.
            let (res, rows) = unsafe {
                (
                    std::slice::from_raw_parts(
                        (base.add(self.res_off) as *const GpuBandRes).add(res0),
                        n,
                    ),
                    std::slice::from_raw_parts(base.add(self.rows_off + rows0), n_rows),
                )
            };
            let outs = res
                .iter()
                .map(|g| BandOut {
                    gmax: g.gmax as u8,
                    te: g.te,
                    qe: g.qe as u8,
                })
                .collect();
            Some((outs, rows.to_vec()))
        }
    }

    impl Drop for Batch {
        fn drop(&mut self) {
            if let Some(b) = self.buf.take() {
                if self.done() {
                    let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
                    if pool.len() < 64 {
                        pool.push(b);
                    }
                }
            }
        }
    }

    /// A batch of one call's bands, sent directly: for tests and probes.
    pub struct BandsPending {
        batch: Batch,
        n: usize,
        n_rows: usize,
    }

    impl BandPending for BandsPending {
        fn is_done(&self) -> bool {
            self.batch.done()
        }

        // objc's `sel_impl!` tests `feature = "cargo-clippy"`, which this crate does not declare.
        #[allow(unexpected_cfgs)]
        fn wait(self: Box<Self>) -> Option<(Vec<BandOut>, Vec<u8>)> {
            let cb = &self.batch.cb;
            cb.wait_until_completed();
            if cb.status() == MTLCommandBufferStatus::Completed {
                // SAFETY: both selectors exist on every MTLCommandBuffer and return a double.
                let (t0, t1): (f64, f64) = unsafe {
                    (
                        msg_send![cb.as_ref(), GPUStartTime],
                        msg_send![cb.as_ref(), GPUEndTime],
                    )
                };
                LAST_BAND_NS.store(((t1 - t0).max(0.0) * 1e9) as u64, Relaxed);
            }
            self.batch.read(0, self.n, 0, self.n_rows)
        }
    }

    /// GPU time of the last direct band batch waited on, nanoseconds.
    static LAST_BAND_NS: AtomicU64 = AtomicU64::new(0);

    impl MetalExtend {
        /// Whether the band kernels take this band: a query of at most 256 padded columns, and
        /// every score under 256 as in the u8 CPU kernel (`a * len(q)` bounds H, the pad scoring 0).
        pub fn takes_band(t: &BandTask, sc: &BandScoring) -> bool {
            band_columns(t.quanta).is_some()
                && t.w >= 1
                && i64::from(sc.a.max(0)) * t.q.len() as i64 <= 255
        }

        /// GPU execution time of the last [`Self::submit_bands`] batch waited on, seconds.
        pub fn last_band_seconds(&self) -> f64 {
            LAST_BAND_NS.load(Relaxed) as f64 / 1e9
        }

        /// Pack `tasks` into a batch of their own and commit it; results in task order.
        pub fn submit_bands(&self, tasks: &[BandTask], sc: &BandScoring) -> Option<BandsPending> {
            let p = Packed::new(tasks, sc)?;
            let (n, n_rows) = (p.jobs.len(), p.n_rows);
            let batch = self.encode_bands(&[&p])?;
            Some(BandsPending { batch, n, n_rows })
        }

        /// One command buffer for `parts` (one scoring), results and rows in part order.
        fn encode_bands(&self, parts: &[&Packed]) -> Option<Batch> {
            let sc = parts.first()?.sc;
            let n: usize = parts.iter().map(|p| p.jobs.len()).sum();
            let n_rows: usize = parts.iter().map(|p| p.n_rows).sum();
            let total: usize = parts.iter().map(|p| p.seqs.len()).sum();
            if n == 0 || total > u32::MAX as usize || n_rows > u32::MAX as usize {
                return None;
            }
            autoreleasepool(|| {
                let mut buckets: [Vec<GpuBand>; 8] = Default::default();
                let (mut sbase, mut rbase, mut obase) = (0u32, 0u32, 0u32);
                for p in parts {
                    for g in &p.jobs {
                        let c = band_columns(g.quanta)?;
                        let b = BAND_BUCKETS.iter().position(|&x| x == c)?;
                        let mut g = *g;
                        g.q_off += sbase;
                        g.t_off += sbase;
                        g.rows_off += rbase;
                        g.out += obase;
                        buckets[b].push(g);
                    }
                    sbase += p.seqs.len() as u32;
                    rbase += p.n_rows as u32;
                    obase += p.jobs.len() as u32;
                }
                for list in &mut buckets {
                    list.sort_unstable_by_key(|g| std::cmp::Reverse(g.t_len));
                }
                let align = |x: usize| x.div_ceil(256) * 256;
                let mut job_off = [0usize; 8];
                let mut at = 0usize;
                for (b, list) in buckets.iter().enumerate() {
                    job_off[b] = at;
                    at = align(at + std::mem::size_of_val(&list[..]));
                }
                let res_off = at;
                let rows_off = align(res_off + n * std::mem::size_of::<GpuBandRes>());
                let seq_off = align(rows_off + n_rows);
                let buf = self.take_buffer(seq_off + total.max(1));
                let base = buf.contents() as *mut u8;
                // SAFETY: as in `submit`.
                unsafe {
                    for (b, list) in buckets.iter().enumerate() {
                        std::ptr::copy_nonoverlapping(
                            list.as_ptr() as *const u8,
                            base.add(job_off[b]),
                            std::mem::size_of_val(&list[..]),
                        );
                    }
                    let mut q = base.add(seq_off);
                    for p in parts {
                        std::ptr::copy_nonoverlapping(p.seqs.as_ptr(), q, p.seqs.len());
                        q = q.add(p.seqs.len());
                    }
                }
                let cb = self.queue().new_command_buffer().to_owned();
                let enc =
                    cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
                enc.set_buffer(0, Some(&buf), seq_off as u64);
                enc.set_buffer(2, Some(&buf), res_off as u64);
                enc.set_buffer(3, Some(&buf), rows_off as u64);
                for (b, list) in buckets.iter().enumerate() {
                    if list.is_empty() {
                        continue;
                    }
                    let pipe = &self.band_pipes[b];
                    let params = BandParams {
                        oe_del: sc.oe_del,
                        e_del: sc.e_del,
                        oe_ins: sc.oe_ins,
                        e_ins: sc.e_ins,
                        sc_match: sc.a,
                        sc_mismatch: sc.b,
                        sc_n: sc.n,
                        n_jobs: list.len() as u32,
                    };
                    enc.set_compute_pipeline_state(pipe);
                    enc.set_buffer(1, Some(&buf), job_off[b] as u64);
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<BandParams>() as u64,
                        &params as *const BandParams as *const c_void,
                    );
                    let tg = TG.min(pipe.max_total_threads_per_threadgroup()) / 32 * 32;
                    let threads = list.len() as u64 * LANES;
                    enc.dispatch_thread_groups(
                        MTLSize::new(threads.div_ceil(tg), 1, 1),
                        MTLSize::new(tg, 1, 1),
                    );
                }
                enc.end_encoding();
                cb.commit();
                Some(Batch {
                    cb,
                    buf: Some(buf),
                    res_off,
                    rows_off,
                    pool: self.pool.clone(),
                })
            })
        }
    }

    /// Where one call's bands are: waiting to be batched, given up by their caller, or in a batch
    /// at these result and row offsets.
    enum Slot {
        Staged,
        Cancelled,
        Flushed {
            batch: Arc<Batch>,
            res0: usize,
            rows0: usize,
        },
    }

    type SlotRef = Arc<Mutex<Slot>>;

    fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The workers' bands, batched. Command buffers barely overlap on the GPU, even from separate
    /// queues (8 batches of 256 bands in flight: 8.0 ms, against 1.8 ms for one), while one batch
    /// of 4096 runs at 126 Gcell/s: so each call's bands are staged here, and whatever is staged
    /// goes out as one command buffer whenever fewer than [`MAX_BATCHES`] are on the GPU.
    #[derive(Default)]
    struct Agg {
        state: Mutex<AggState>,
    }

    #[derive(Default)]
    struct AggState {
        staged: Vec<(SlotRef, Packed)>,
        inflight: Vec<Arc<Batch>>,
    }

    /// Band batches on the GPU at once.
    const MAX_BATCHES: usize = 2;

    impl Agg {
        /// Drop finished batches; send what is staged when there is room, or with `force`.
        fn pump(st: &mut AggState, gpu: &MetalExtend, force: bool) {
            st.inflight.retain(|b| !b.done());
            if st.staged.is_empty() || (!force && st.inflight.len() >= MAX_BATCHES) {
                return;
            }
            let staged = std::mem::take(&mut st.staged);
            let live: Vec<(SlotRef, Packed)> = staged
                .into_iter()
                .filter(|(slot, _)| !matches!(*lock(slot), Slot::Cancelled))
                .collect();
            // One scoring per batch: the first one's; the rest wait for the next.
            let Some(sc0) = live.first().map(|(_, p)| p.sc) else {
                return;
            };
            let (take, rest): (Vec<_>, Vec<_>) = live
                .into_iter()
                .partition(|(_, p)| same_scoring(&p.sc, &sc0));
            st.staged = rest;
            let parts: Vec<&Packed> = take.iter().map(|(_, p)| p).collect();
            let Some(batch) = gpu.encode_bands(&parts) else {
                // The callers compute these on the CPU.
                for (slot, _) in &take {
                    *lock(slot) = Slot::Cancelled;
                }
                return;
            };
            let batch = Arc::new(batch);
            let (mut res0, mut rows0) = (0usize, 0usize);
            for (slot, p) in &take {
                let mut s = lock(slot);
                // Cancelled since the filter: its results are simply not read.
                if matches!(*s, Slot::Staged) {
                    *s = Slot::Flushed {
                        batch: batch.clone(),
                        res0,
                        rows0,
                    };
                }
                res0 += p.jobs.len();
                rows0 += p.n_rows;
            }
            st.inflight.push(batch);
        }
    }

    /// One call's bands in the aggregator.
    struct AggPending<'a> {
        agg: &'a Agg,
        gpu: &'a MetalExtend,
        slot: SlotRef,
        n: usize,
        n_rows: usize,
    }

    impl Drop for AggPending<'_> {
        fn drop(&mut self) {
            let mut s = lock(&self.slot);
            if matches!(*s, Slot::Staged) {
                *s = Slot::Cancelled;
            }
        }
    }

    impl BandPending for AggPending<'_> {
        fn is_done(&self) -> bool {
            match &*lock(&self.slot) {
                Slot::Flushed { batch, .. } => return batch.done(),
                Slot::Cancelled => return true,
                Slot::Staged => {}
            }
            // Still staged: send it if there is room, without making a worker wait on the lock.
            if let Ok(mut st) = self.agg.state.try_lock() {
                Agg::pump(&mut st, self.gpu, false);
            }
            false
        }

        fn wait(self: Box<Self>) -> Option<(Vec<BandOut>, Vec<u8>)> {
            if matches!(*lock(&self.slot), Slot::Staged) {
                Agg::pump(&mut lock(&self.agg.state), self.gpu, true);
            }
            let (batch, res0, rows0) = match &*lock(&self.slot) {
                Slot::Flushed { batch, res0, rows0 } => (batch.clone(), *res0, *rows0),
                _ => return None,
            };
            batch.cb.wait_until_completed();
            batch.read(res0, self.n, rows0, self.n_rows)
        }
    }

    /// The GPU's share of the mate-rescue bands, through `bwa_neon::offload`.
    impl<B: Sync> BandOffload for Hybrid<B> {
        fn takes(&self, t: &BandTask, sc: &BandScoring) -> bool {
            MetalExtend::takes_band(t, sc)
        }

        fn band_share(&self) -> f64 {
            self.band_share
        }

        fn submit<'s>(
            &'s self,
            tasks: &[BandTask],
            sc: &BandScoring,
        ) -> Option<Box<dyn BandPending + 's>> {
            let p = Packed::new(tasks, sc)?;
            let (n, n_rows) = (p.jobs.len(), p.n_rows);
            let slot: SlotRef = Arc::new(Mutex::new(Slot::Staged));
            {
                let mut st = lock(&self.bands.state);
                st.staged.push((slot.clone(), p));
                Agg::pump(&mut st, &self.gpu, false);
            }
            self.band_tasks.fetch_add(n as u64, Relaxed);
            let pending = Box::new(AggPending {
                agg: &self.bands,
                gpu: &self.gpu,
                slot,
                n,
                n_rows,
            });
            if !self.verify {
                return Some(pending);
            }
            // Verify: every GPU lane against the scalar reference, here and now.
            let (outs, rows) = pending.wait()?;
            let mut off = 0;
            for (t, o) in tasks.iter().zip(&outs) {
                let (want, wrows) = bwa_neon::offload::band_reference(t, sc);
                assert_eq!(*o, want, "GPU band diverges from the reference");
                assert_eq!(
                    &rows[off..off + t.t.len()],
                    &wrows[..],
                    "GPU band row maxima"
                );
                off += t.t.len();
            }
            Some(Box::new(Done(Some((outs, rows)))))
        }
    }

    /// A result already in hand (verify mode).
    struct Done(Option<(Vec<BandOut>, Vec<u8>)>);

    impl BandPending for Done {
        fn is_done(&self) -> bool {
            true
        }
        fn wait(mut self: Box<Self>) -> Option<(Vec<BandOut>, Vec<u8>)> {
            self.0.take()
        }
    }

    /// Scatter `results` (one per entry of `idx`) into `out` at those indices.
    fn scatter(out: &mut [ExtendResult], idx: &[usize], results: &[ExtendResult]) {
        for (&k, r) in idx.iter().zip(results) {
            out[k] = *r;
        }
    }

    const ZERO: ExtendResult = ExtendResult {
        score: 0,
        qle: 0,
        tle: 0,
        gtle: 0,
        gscore: 0,
        max_off: 0,
    };

    /// GPU-only backend: what the kernel takes goes to the GPU, the rest to `ksw_extend2` while the
    /// GPU works. The gates of `bwa-extend` run against this.
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
            // One job is not worth a launch.
            ksw_extend2(
                query, target, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
            )
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
            w: i32,
            end_bonus: i32,
            zdrop: i32,
        ) -> Vec<ExtendResult> {
            let sc = Scoring {
                m,
                mat,
                o_del,
                e_del,
                o_ins,
                e_ins,
                w,
                end_bonus,
                zdrop,
            };
            if jobs.len() < 2 || !Self::takes_scoring(&sc) {
                return jobs.iter().map(|j| sc.scalar(j)).collect();
            }
            let (on_gpu, on_cpu): (Vec<usize>, Vec<usize>) =
                (0..jobs.len()).partition(|&k| Self::takes_job(&sc, &jobs[k]));
            let gathered: Vec<ExtendJob> = on_gpu.iter().map(|&k| jobs[k]).collect();
            let pending = self.submit(&gathered, &sc);
            let mut out = vec![ZERO; jobs.len()];
            for &k in &on_cpu {
                out[k] = sc.scalar(&jobs[k]);
            }
            match pending.and_then(Pending::wait) {
                Some(r) => scatter(&mut out, &on_gpu, &r),
                None => {
                    for &k in &on_gpu {
                        out[k] = sc.scalar(&jobs[k]);
                    }
                }
            }
            out
        }
    }

    /// The GPU as a co-processor next to a CPU backend, for the aligner's workers.
    ///
    /// Each call sends the GPU its longest `share` of the jobs it takes and runs the rest on `cpu`.
    /// When the CPU is done before the GPU, it does not wait: it takes the GPU's jobs from the far
    /// end, in chunks, until the GPU finishes, and the GPU's results are used only for what the CPU
    /// has not reached. Every job's result comes from one backend or the other, both of which the
    /// identity gates hold to `ksw_extend2`, so where the split falls cannot move a byte of output.
    pub struct Hybrid<B> {
        gpu: MetalExtend,
        cpu: B,
        share: f64,
        verify: bool,
        /// Share of each rescue call's band cells for the GPU (`bwa_neon::offload`).
        band_share: f64,
        /// Bands submitted to the GPU so far.
        band_tasks: AtomicU64,
        /// The workers' bands, batched.
        bands: Agg,
    }

    /// Jobs per CPU chunk while racing the GPU: enough for the batched kernel's lanes, small enough
    /// that the CPU notices the GPU finishing soon after it does.
    const STEAL_CHUNK: usize = 64;
    /// Below this many GPU jobs a launch costs more than it saves.
    const MIN_GPU_JOBS: usize = 64;

    impl<B> Hybrid<B> {
        /// `gpu` next to `cpu`, the GPU taking `share` (0..1) of each call's jobs. With `verify`,
        /// every result the GPU returns is recomputed by `ksw_extend2` and a mismatch panics:
        /// a run that would have written a wrong byte stops instead.
        pub fn new(gpu: MetalExtend, cpu: B, share: f64, verify: bool) -> Self {
            Self {
                gpu,
                cpu,
                share: share.clamp(0.0, 1.0),
                verify,
                band_share: 0.0,
                band_tasks: AtomicU64::new(0),
                bands: Agg::default(),
            }
        }

        /// Bands submitted to the GPU so far (whether or not the CPU took them back).
        pub fn band_tasks(&self) -> u64 {
            self.band_tasks.load(Relaxed)
        }

        /// Give the GPU `share` (0..1) of each mate-rescue call's band cells once this is
        /// registered with `bwa_neon::offload::set_band_offload`. 0 by default.
        pub fn with_band_share(mut self, share: f64) -> Self {
            self.band_share = share.clamp(0.0, 1.0);
            self
        }

        /// The GPU half.
        pub fn gpu(&self) -> &MetalExtend {
            &self.gpu
        }
    }

    impl<B: SwBackend> SwBackend for Hybrid<B> {
        fn name(&self) -> &'static str {
            "metal+cpu"
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
            self.cpu.extend(
                query, target, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0,
            )
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
            w: i32,
            end_bonus: i32,
            zdrop: i32,
        ) -> Vec<ExtendResult> {
            let sc = Scoring {
                m,
                mat,
                o_del,
                e_del,
                o_ins,
                e_ins,
                w,
                end_bonus,
                zdrop,
            };
            let want = (jobs.len() as f64 * self.share) as usize;
            if want < MIN_GPU_JOBS || !MetalExtend::takes_scoring(&sc) {
                return sc.run(&self.cpu, jobs);
            }
            // The GPU's share: the longest queries it takes (most columns per launch), so the CPU
            // keeps the short ones. Ties keep input order.
            let mut eligible: Vec<usize> = (0..jobs.len())
                .filter(|&k| MetalExtend::takes_job(&sc, &jobs[k]))
                .collect();
            eligible.sort_by_key(|&k| std::cmp::Reverse(jobs[k].query.len()));
            eligible.truncate(want);
            if eligible.len() < MIN_GPU_JOBS {
                return sc.run(&self.cpu, jobs);
            }
            let mut is_gpu = vec![false; jobs.len()];
            for &k in &eligible {
                is_gpu[k] = true;
            }
            let on_gpu = eligible;
            let on_cpu: Vec<usize> = (0..jobs.len()).filter(|&k| !is_gpu[k]).collect();
            let gathered: Vec<ExtendJob> = on_gpu.iter().map(|&k| jobs[k]).collect();
            let Some(pending) = self.gpu.submit(&gathered, &sc) else {
                return sc.run(&self.cpu, jobs);
            };

            let mut out = vec![ZERO; jobs.len()];
            let cpu_jobs: Vec<ExtendJob> = on_cpu.iter().map(|&k| jobs[k]).collect();
            scatter(&mut out, &on_cpu, &sc.run(&self.cpu, &cpu_jobs));
            // Endgame: race the GPU from the far end of its share.
            let mut lo = gathered.len();
            while lo > 0 && !pending.is_done() {
                let from = lo.saturating_sub(STEAL_CHUNK);
                scatter(
                    &mut out,
                    &on_gpu[from..lo],
                    &sc.run(&self.cpu, &gathered[from..lo]),
                );
                lo = from;
            }
            if lo > 0 {
                match pending.wait() {
                    Some(r) => {
                        if self.verify {
                            for (j, got) in gathered[..lo].iter().zip(&r) {
                                let want = sc.scalar(j);
                                assert_eq!(
                                    *got,
                                    want,
                                    "BWA4_GPU_VERIFY: GPU extension diverged from ksw_extend2 \
                                     (qlen {} tlen {} h0 {} w {} zdrop {} end_bonus {})",
                                    j.query.len(),
                                    j.target.len(),
                                    j.h0,
                                    sc.w,
                                    sc.zdrop,
                                    sc.end_bonus
                                );
                            }
                        }
                        scatter(&mut out, &on_gpu[..lo], &r[..lo]);
                    }
                    None => scatter(&mut out, &on_gpu[..lo], &sc.run(&self.cpu, &gathered[..lo])),
                }
            }
            out
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use backend::{BandsPending, Hybrid, MetalExtend, Pending, Scoring};

/// Stand-in when the `metal` feature is off or the target is not macOS: never constructed.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub struct MetalExtend(());

#[cfg(not(all(feature = "metal", target_os = "macos")))]
impl MetalExtend {
    /// Always `None`: there is no GPU backend in this build.
    pub fn new() -> Option<Self> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_cover_qlen_plus_one() {
        for qlen in 0..=MAX_QLEN {
            let c = columns_per_lane(qlen).unwrap();
            assert!(
                16 * c > qlen && c.is_multiple_of(2) && c <= 16,
                "qlen {qlen} -> {c}"
            );
            assert!(
                qlen < 16 || 16 * (c - 2) <= qlen,
                "qlen {qlen} -> {c} wastes a bucket"
            );
        }
        assert_eq!(columns_per_lane(MAX_QLEN + 1), None);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    mod gpu {
        use super::super::{Hybrid, MetalExtend};
        use bwa_extend::{ksw_extend2, ExtendJob, ScalarBackend, SwBackend};

        fn backend() -> MetalExtend {
            MetalExtend::new().expect("no Metal device")
        }

        #[test]
        fn passes_every_backend_gate() {
            let b = backend();
            bwa_extend::assert_backend_matches_scalar(&b);
            bwa_extend::assert_backend_batch_matches_scalar(&b);
            bwa_extend::assert_backend_tie_rule_matches_scalar(&b);
            bwa_extend::assert_backend_saturation_boundary_matches_scalar(&b);
            bwa_extend::assert_backend_batch_order_invariant(&b);
        }

        /// Throughput against the NEON backend at production batch sizes (an extension call
        /// carries ~1 000 to 7 300 jobs). `cargo test --release -p bwa-mem4-metal --features metal
        /// -- --ignored --nocapture throughput`. Cells are `qlen * tlen`, as in the earlier probes.
        #[test]
        #[ignore = "throughput probe, run explicitly"]
        fn throughput() {
            let gpu = backend();
            let cpu = bwa_neon::NeonBackend;
            let mut mat = vec![-1i8; 25];
            for i in 0..4 {
                for j in 0..4 {
                    mat[i * 5 + j] = if i == j { 1 } else { -4 };
                }
            }
            let mut s = 0x2468_ace0_1357_9bdfu64;
            let mut rnd = move || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s >> 33
            };
            let max_n = 32_768;
            let mut seqs = Vec::with_capacity(max_n);
            for _ in 0..max_n {
                // A 150 bp read cut at a seed: what is left of it on one side, against that much
                // reference plus the band.
                let qlen = 1 + (rnd() % 150) as usize;
                let q: Vec<u8> = (0..qlen).map(|_| (rnd() % 4) as u8).collect();
                let tlen = qlen + 20 + (rnd() % 80) as usize;
                let t: Vec<u8> = (0..tlen)
                    .map(|i| {
                        if i < qlen && rnd() % 100 >= 3 {
                            q[i]
                        } else {
                            (rnd() % 4) as u8
                        }
                    })
                    .collect();
                seqs.push((q, t, 19 + (rnd() % 40) as i32));
            }
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            eprintln!("device {}, {threads} CPU threads", gpu.device_name());
            for n in [1024usize, 4096, 8192, 32_768] {
                let jobs: Vec<ExtendJob> = seqs[..n]
                    .iter()
                    .map(|(q, t, h0)| ExtendJob {
                        query: q,
                        target: t,
                        h0: *h0,
                    })
                    .collect();
                let cells: u64 = jobs
                    .iter()
                    .map(|j| (j.query.len() * j.target.len()) as u64)
                    .sum();
                let best = |run: &dyn Fn()| {
                    run();
                    (0..7)
                        .map(|_| {
                            let t0 = std::time::Instant::now();
                            run();
                            t0.elapsed().as_secs_f64()
                        })
                        .fold(f64::INFINITY, f64::min)
                };
                let one = best(&|| {
                    std::hint::black_box(cpu.extend_batch(&jobs, 5, &mat, 6, 1, 6, 1, 100, 5, 100));
                });
                let all = best(&|| {
                    let mat = &mat;
                    std::thread::scope(|sc| {
                        for chunk in jobs.chunks(n.div_ceil(threads)) {
                            sc.spawn(move || {
                                std::hint::black_box(
                                    cpu.extend_batch(chunk, 5, mat, 6, 1, 6, 1, 100, 5, 100),
                                )
                            });
                        }
                    });
                });
                // The GPU clocks down between bursts: warm it, then take medians of wall and of
                // kernel time over a sustained run.
                let mut walls = Vec::new();
                let mut kernels = Vec::new();
                let start = std::time::Instant::now();
                while start.elapsed().as_secs_f64() < 0.6 {
                    let t0 = std::time::Instant::now();
                    std::hint::black_box(gpu.extend_batch(&jobs, 5, &mat, 6, 1, 6, 1, 100, 5, 100));
                    if start.elapsed().as_secs_f64() > 0.3 {
                        walls.push(t0.elapsed().as_secs_f64());
                        kernels.push(gpu.last_kernel_seconds());
                    }
                }
                let median = |v: &mut Vec<f64>| {
                    v.sort_by(f64::total_cmp);
                    v[v.len() / 2]
                };
                let g = median(&mut walls);
                let kernel = median(&mut kernels);
                // Production shape: every worker submits its own share at the same time.
                let gpu_all = best(&|| {
                    let (mat, gpu) = (&mat, &gpu);
                    std::thread::scope(|sc| {
                        for chunk in jobs.chunks(n.div_ceil(threads)) {
                            sc.spawn(move || {
                                std::hint::black_box(
                                    gpu.extend_batch(chunk, 5, mat, 6, 1, 6, 1, 100, 5, 100),
                                )
                            });
                        }
                    });
                });
                let gc = |t: f64| cells as f64 / t / 1e9;
                eprintln!(
                    "{n:>6} jobs: metal {:>7.3} ms {:>6.1} Gcell/s (kernel {:>6.3} ms {:>6.1}) | metal x{threads} {:>6.1} | neon x1 {:>6.1} | neon x{threads} {:>6.1} | metal/x1 {:>5.2}x",
                    g * 1e3,
                    gc(g),
                    kernel * 1e3,
                    gc(kernel),
                    gc(gpu_all),
                    gc(one),
                    gc(all),
                    one / g
                );
                // Against the scalar reference, not NEON: NEON's tight band legitimately moves
                // `gscore`/`gtle` inside the window the caller never reads, this kernel does not.
                let got = gpu.extend_batch(&jobs, 5, &mat, 6, 1, 6, 1, 100, 5, 100);
                for (k, j) in jobs.iter().enumerate() {
                    let scalar =
                        ksw_extend2(j.query, j.target, 5, &mat, 6, 1, 6, 1, 100, 5, 100, j.h0);
                    assert_eq!(
                        got[k],
                        scalar,
                        "batch {n} job {k} qlen={} tlen={} h0={}",
                        j.query.len(),
                        j.target.len(),
                        j.h0
                    );
                }
            }
        }

        /// The gates draw queries up to 60 bases. This covers every bucket, long targets, narrow
        /// bands, z-drop, `h0 <= 0` (`--meth`), mutated copies (long diagonals, the realistic case)
        /// and the CPU fallback past `MAX_QLEN`, all in single large batches.
        #[test]
        fn random_batches_match_scalar() {
            random_batches(&backend());
        }

        /// The hybrid on the same batches, with a scalar CPU half so every field must match. A
        /// small GPU share leaves the GPU idle at the end, a large one makes the CPU take most of
        /// the GPU's jobs back from the far end: both orders of finishing are exercised.
        #[test]
        fn hybrid_matches_scalar_at_any_share() {
            for share in [0.3, 0.97] {
                let h = Hybrid::new(backend(), ScalarBackend, share, true);
                random_batches(&h);
                bwa_extend::assert_backend_batch_order_invariant(&h);
            }
        }

        /// The GPU band lanes against `bwa_neon::offload::band_reference`, field for field and
        /// row for row, on random bands of every bucket: planted diagonals, N, pads, `thr`, bands
        /// reaching past either end of the query.
        #[test]
        fn band_lanes_match_reference() {
            use bwa_neon::offload::{band_reference, BandScoring, BandTask};
            let gpu = backend();
            let mut st = 0x2545_f491_4f6c_dd1du64;
            let mut next = move || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                st as usize
            };
            for round in 0..12 {
                let (a, b) = (1 + round % 2, -(1 + (next() % 5) as i32));
                let sc = BandScoring {
                    a,
                    b,
                    n: -1,
                    oe_del: 1 + (next() % 8) as i32,
                    e_del: 1 + (next() % 2) as i32,
                    oe_ins: 1 + (next() % 8) as i32,
                    e_ins: 1 + (next() % 2) as i32,
                };
                let mut data = Vec::new();
                for k in 0..700 {
                    let qlen = 1 + next() % if a == 1 { 250 } else { 127 };
                    let quanta = qlen.div_ceil(16) as i32 * 16;
                    let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                    if k % 13 == 0 {
                        q[next() % qlen] = 4;
                    }
                    let w = 1 + (next() % 1600) as i32;
                    let o = (next() % (quanta as usize + w as usize + 20)) as i32 - w - 10;
                    let nr = 1 + next() % 900;
                    let mut t: Vec<u8> = (0..nr).map(|_| (next() % 4) as u8).collect();
                    let k0 = (next() % w as usize) as i32;
                    for (r, x) in t.iter_mut().enumerate() {
                        let j = r as i32 + o - k0;
                        if (0..qlen as i32).contains(&j) && next() % 12 != 0 {
                            *x = q[j as usize];
                        }
                    }
                    if k % 29 == 0 {
                        t[next() % nr] = 4;
                    }
                    data.push((q, t, quanta, o, w, (next() % 60) as u8));
                }
                let tasks: Vec<BandTask> = data
                    .iter()
                    .map(|(q, t, quanta, o, w, thr)| BandTask {
                        q,
                        t,
                        quanta: *quanta,
                        o: *o,
                        w: *w,
                        thr: *thr,
                    })
                    .collect();
                let p = gpu.submit_bands(&tasks, &sc).expect("submit");
                let (outs, rows) = bwa_neon::offload::BandPending::wait(Box::new(p)).expect("gpu");
                let mut off = 0;
                for (i, t) in tasks.iter().enumerate() {
                    let (want, wrows) = band_reference(t, &sc);
                    assert_eq!(
                        outs[i],
                        want,
                        "round {round} band {i}: w {} o {} rows {}",
                        t.w,
                        t.o,
                        t.t.len()
                    );
                    assert_eq!(
                        &rows[off..off + t.t.len()],
                        &wrows[..],
                        "round {round} band {i}: rows"
                    );
                    off += t.t.len();
                }
            }
        }

        /// The whole mate rescue with the GPU taking most band cells: `batched_mate_rescue` (and
        /// its lazy form) against `ksw_align2` on the full window, on generated windows with planted
        /// copies, repeats and indels. Every consumed result must be identical.
        #[test]
        fn rescue_with_gpu_bands_matches_ksw_align2() {
            use bwa_extend::{ksw_align2, KswAlignResult};
            use bwa_neon::{batched_mate_rescue, batched_mate_rescue_lazy, KswJob, SCORE2_PENDING};
            let h: &'static Hybrid<ScalarBackend> = Box::leak(Box::new(
                Hybrid::new(backend(), ScalarBackend, 0.25, true).with_band_share(0.8),
            ));
            assert!(bwa_neon::offload::set_band_offload(h));
            let mut state = 0x7a3d_19c4_e5b2_0f61u64;
            let mut next = move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as usize
            };
            let mut gen: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            for k in 0..6000 {
                let qlen = if k % 50 == 0 { 250 } else { 20 + next() % 160 };
                let tlen = qlen + next() % 1400;
                let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
                if next() % 6 == 0 {
                    let period = 1 + next() % 4;
                    let at = next() % tlen;
                    let n = (next() % 300).min(tlen - at);
                    for x in 0..n {
                        t[at + x] = t[at + x % period];
                    }
                }
                let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                for _ in 0..next() % 4 {
                    let l = (10 + next() % qlen.min(80)).min(qlen);
                    let (qs, at) = (next() % (qlen - l + 1), next() % (tlen - l + 1));
                    t[at..at + l].copy_from_slice(&q[qs..qs + l]);
                    for _ in 0..next() % 4 {
                        t[at + next() % l] = (next() % 4) as u8;
                    }
                    if next() % 3 == 0 && l > 8 {
                        let p = at + 2 + next() % (l - 4);
                        if next() % 2 == 0 {
                            t.remove(p);
                            t.push((next() % 4) as u8);
                        } else {
                            t.insert(p, (next() % 4) as u8);
                            t.pop();
                        }
                    }
                }
                gen.push((t, q));
            }
            let mut mat = vec![-4i8; 25];
            for i in 0..5 {
                for j in 0..5 {
                    if i == 4 || j == 4 {
                        mat[i * 5 + j] = -1;
                    } else if i == j {
                        mat[i * 5 + j] = 1;
                    }
                }
            }
            let minsc = 19;
            let mut consumed_n = 0;
            for chunk in gen.chunks(1000) {
                let jobs: Vec<KswJob> = chunk
                    .iter()
                    .map(|(t, q)| KswJob {
                        query: q,
                        target: t,
                    })
                    .collect();
                let got = batched_mate_rescue(&jobs, 5, &mat, 6, 1, 6, 1, minsc, 1);
                let lazy = batched_mate_rescue_lazy(&jobs, 5, &mat, 6, 1, 6, 1, minsc, 1);
                for (i, (t, q)) in chunk.iter().enumerate() {
                    let full = ksw_align2(q, t, 5, &mat, 6, 1, 6, 1, minsc, 1, 16);
                    let consumed = |r: &KswAlignResult| r.score >= minsc;
                    if consumed(&full) || consumed(&got[i]) {
                        consumed_n += 1;
                        assert_eq!(got[i], full, "job {i}");
                    }
                    let mut l = lazy[i];
                    if l.score2 == SCORE2_PENDING {
                        (l.score2, l.te2) = (full.score2, full.te2);
                    }
                    if consumed(&full) || consumed(&l) {
                        assert_eq!(l, full, "lazy job {i}");
                    }
                }
            }
            assert!(consumed_n > 1000, "too few consumed rescues: {consumed_n}");
            eprintln!(
                "{consumed_n} consumed rescues, {} bands on the GPU",
                h.band_tasks()
            );
            assert!(
                h.band_tasks() > 1000,
                "too few GPU bands: {}",
                h.band_tasks()
            );
        }

        /// Band kernel throughput on rescue-like shapes: whole-window bands (w = rows + quanta),
        /// 148 bp mates, ~1400 rows, in batches of 16 to 1024. Cells are live cells,
        /// rows x min(w, quanta), as the CPU probe counts them.
        #[test]
        #[ignore = "throughput probe, run explicitly"]
        fn band_throughput() {
            use bwa_neon::offload::{BandPending, BandScoring, BandTask};
            let gpu = backend();
            let sc = BandScoring {
                a: 1,
                b: -4,
                n: -1,
                oe_del: 7,
                e_del: 1,
                oe_ins: 7,
                e_ins: 1,
            };
            let mut st = 0x1234_5678_9abc_def1u64;
            let mut next = move || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                st as usize
            };
            let data: Vec<(Vec<u8>, Vec<u8>)> = (0..4096)
                .map(|_| {
                    let q: Vec<u8> = (0..148).map(|_| (next() % 4) as u8).collect();
                    let t: Vec<u8> = (0..1200 + next() % 400)
                        .map(|_| (next() % 4) as u8)
                        .collect();
                    (q, t)
                })
                .collect();
            for n in [16usize, 64, 256, 1024, 4096] {
                let tasks: Vec<BandTask> = data[..n]
                    .iter()
                    .map(|(q, t)| BandTask {
                        q,
                        t,
                        quanta: 160,
                        o: 159,
                        w: t.len() as i32 + 160,
                        thr: 19,
                    })
                    .collect();
                let cells: f64 = tasks.iter().map(|t| t.t.len() as f64 * 160.0).sum();
                let mut ks = Vec::new();
                // Warm: the GPU clocks down between bursts.
                let warm = std::time::Instant::now();
                while warm.elapsed().as_secs_f64() < 0.3 {
                    Box::new(gpu.submit_bands(&tasks, &sc).unwrap())
                        .wait()
                        .unwrap();
                }
                for _ in 0..7 {
                    let p = gpu.submit_bands(&tasks, &sc).unwrap();
                    Box::new(p).wait().unwrap();
                    ks.push(gpu.last_band_seconds());
                }
                ks.sort_by(f64::total_cmp);
                if n == 256 {
                    // Eight batches in flight at once, as eight workers would submit them.
                    let t = std::time::Instant::now();
                    let ps: Vec<_> = (0..8)
                        .map(|_| gpu.submit_bands(&tasks, &sc).unwrap())
                        .collect();
                    for p in ps {
                        Box::new(p).wait().unwrap();
                    }
                    eprintln!(
                        "8 x 256 bands in flight: {:.3} ms wall",
                        t.elapsed().as_secs_f64() * 1e3
                    );
                }
                eprintln!(
                    "bands {n:5}: kernel {:.3} ms, {:.1} Gcell/s",
                    ks[3] * 1e3,
                    cells / ks[3] / 1e9
                );
            }
        }

        fn random_batches<B: SwBackend>(b: &B) {
            let mut s = 0x9e37_79b9_7f4a_7c15u64;
            let mut rnd = move || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            for round in 0..24 {
                let (a, mm) = (1 + (rnd() % 3) as i8, 1 + (rnd() % 6) as i8);
                let mut mat = vec![0i8; 25];
                for i in 0..5 {
                    for j in 0..5 {
                        mat[i * 5 + j] = if i == 4 || j == 4 {
                            -1
                        } else if i == j {
                            a
                        } else {
                            -mm
                        };
                    }
                }
                let o_del = 1 + (rnd() % 8) as i32;
                let e_del = 1 + (rnd() % 3) as i32;
                let o_ins = 1 + (rnd() % 8) as i32;
                let e_ins = 1 + (rnd() % 3) as i32;
                let w = [1, 3, 10, 33, 100][(rnd() % 5) as usize];
                let zdrop = [0, 5, 40, 100][(rnd() % 4) as usize];
                let end_bonus = (rnd() % 8) as i32;
                let n = 500 + (rnd() % 1500) as usize;
                let mut seqs = Vec::with_capacity(n);
                for _ in 0..n {
                    let qlen = match rnd() % 10 {
                        0 => 1 + (rnd() % 8) as usize,
                        1 => 256 + (rnd() % 64) as usize,
                        _ => 1 + (rnd() % 255) as usize,
                    };
                    let query: Vec<u8> = (0..qlen)
                        .map(|_| {
                            if rnd() % 50 == 0 {
                                4
                            } else {
                                (rnd() % 4) as u8
                            }
                        })
                        .collect();
                    let tlen = (rnd() % (qlen as u64 + 120)) as usize;
                    let target: Vec<u8> = if round % 2 == 0 {
                        (0..tlen)
                            .map(|t| {
                                if t < qlen && rnd() % 12 != 0 {
                                    query[t]
                                } else {
                                    (rnd() % 4) as u8
                                }
                            })
                            .collect()
                    } else {
                        (0..tlen).map(|_| (rnd() % 4) as u8).collect()
                    };
                    let h0 = match rnd() % 8 {
                        0 => -((rnd() % 10) as i32),
                        1 => 0,
                        _ => 1 + (rnd() % 120) as i32,
                    };
                    seqs.push((query, target, h0));
                }
                let jobs: Vec<ExtendJob> = seqs
                    .iter()
                    .map(|(q, t, h0)| ExtendJob {
                        query: q,
                        target: t,
                        h0: *h0,
                    })
                    .collect();
                let got = b.extend_batch(
                    &jobs, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                );
                for (k, j) in jobs.iter().enumerate() {
                    let want = ksw_extend2(
                        j.query, j.target, 5, &mat, o_del, e_del, o_ins, e_ins, w, end_bonus,
                        zdrop, j.h0,
                    );
                    assert_eq!(
                        got[k],
                        want,
                        "round {round} job {k}: qlen={} tlen={} h0={} w={w} zdrop={zdrop} \
                         end_bonus={end_bonus} O={o_del},{o_ins} E={e_del},{e_ins} a={a} b={mm}",
                        j.query.len(),
                        j.target.len(),
                        j.h0
                    );
                }
            }
        }
    }
}
