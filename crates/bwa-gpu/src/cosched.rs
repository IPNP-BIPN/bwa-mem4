//! CPU+GPU co-scheduling of seed extension (issue #57, levels 1 and 2), and the cross-thread
//! aggregation the 2026-08-09 measurements named as the condition for a GPU to pay past `-t4`.
//!
//! # The three pieces
//!
//! - [`DeviceKernel`]: what a GPU backend has to provide. One method, run a pre-flattened batch.
//!   Nothing else about the device leaks out, so this module is tested with a CPU stand-in.
//! - [`GpuService`]: one thread owning the device. Every aligner worker sends its GPU share to it;
//!   while the device is busy with launch N, the requests of all workers pile up in the channel, and
//!   launch N+1 takes **all of them at once**. That is the aggregation: without it each worker's
//!   ~2 700..10 800 jobs is its own launch and the kernel runs at a quarter of its saturated rate
//!   (22.4 against 9.08 Gcell/s on the M4 Max, ROADMAP 2026-08-09).
//! - [`CoSched`]: a [`SwBackend`] that splits every `extend_batch` call. A fraction `f` of the jobs
//!   goes to the service, the calling thread runs the rest on its CPU backend **while** the GPU works,
//!   then the two halves are stitched back in job order. The CPU never sleeps while the GPU computes,
//!   which is the difference between "the GPU makes a stage faster" and "the machine gets bigger".
//!
//! # Why none of this can change a SAM byte
//!
//! Every extension job is a pure function of its own `(query, target, h0, w)` and the shared scoring
//! (the invariant `across.rs` rests on, and what `assert_backend_batch_order_invariant` checks). So
//! which device runs a job, which launch it shares, and what `f` is, are scheduling. `f` may even
//! change on every call, which is what the adaptive mode does. The tests below run the whole
//! acceptance harness through [`CoSched`] at several fixed splits and in adaptive mode.
//!
//! # The adaptive split
//!
//! After each call the caller knows how long its CPU share took and how long the GPU share took from
//! submit to reply, **queueing behind other workers included**. That is the effective GPU rate as
//! this worker experiences it, which is exactly the rate the split has to balance against. The target
//! is `f* = r_gpu / (r_gpu + r_cpu)` (both finish together), smoothed by an exponential moving average
//! shared by all workers, and kept inside `[F_MIN, F_MAX]` so neither side stops being measured.

use bwa_mem4_extend::{ExtendJob, ExtendResult, SwBackend};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

/// The band clamp of `ksw_extend2`, on the host, in `f64`, exactly as the reference computes it.
///
/// A device kernel receives each job's `w` already clamped: the project rule is that no DP kernel
/// contains floating point, and the [`SwBackend`] contract says a backend takes the clamped `w` from
/// the host rather than re-deriving it.
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
    let max_ins = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_ins))
        / f64::from(e_ins))
        + 1.0) as i32;
    let w = w0.min(max_ins.max(1));
    let max_del = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_del))
        / f64::from(e_del))
        + 1.0) as i32;
    w.min(max_del.max(1))
}

/// The scoring shared by every job of a launch. Two requests can share a launch only if these are
/// equal; in the aligner they always are, since they come from the options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scoring {
    /// Alphabet size, 5 in bwa.
    pub m: usize,
    /// `m * m` row-major substitution matrix.
    pub mat: Vec<i8>,
    /// Gap penalties, positive magnitudes.
    pub o_del: i32,
    pub e_del: i32,
    pub o_ins: i32,
    pub e_ins: i32,
    /// Z-drop, `<= 0` disables.
    pub zdrop: i32,
}

/// One job of a [`FlatBatch`]: offsets into its byte buffer plus the per-job DP inputs. `w` is
/// already clamped (see [`clamp_band`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlatJob {
    pub q_off: u32,
    pub q_len: u32,
    pub t_off: u32,
    pub t_len: u32,
    pub h0: i32,
    pub w: i32,
}

/// Extension jobs flattened into one owned byte buffer, the shape a device buffer wants.
///
/// Owned, unlike [`ExtendJob`], because it crosses to the service thread: the caller's slices are
/// only borrowed for the duration of `extend_batch`.
#[derive(Debug, Clone, Default)]
pub struct FlatBatch {
    /// Every query then its target, back to back, in job order; padded to a multiple of 4 so a
    /// device can read it as `u32` words.
    pub bytes: Vec<u8>,
    /// One entry per job.
    pub jobs: Vec<FlatJob>,
    /// Longest query, which sizes the device's DP rails.
    pub max_qlen: usize,
    /// DP cells (`qlen * tlen` summed), the unit the adaptive split measures rates in.
    pub cells: u64,
}

impl FlatBatch {
    /// Flatten `jobs`, clamping each job's band from `w0`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_jobs(jobs: &[ExtendJob], s: &Scoring, w0: i32, end_bonus: i32) -> Self {
        let max_sc = s.mat[..s.m * s.m].iter().copied().max().unwrap_or(0) as i32;
        let total: usize = jobs.iter().map(|j| j.query.len() + j.target.len()).sum();
        let mut b = FlatBatch {
            bytes: Vec::with_capacity(total + 4),
            jobs: Vec::with_capacity(jobs.len()),
            max_qlen: 0,
            cells: 0,
        };
        for j in jobs {
            let q_off = b.bytes.len();
            b.bytes.extend_from_slice(j.query);
            let t_off = b.bytes.len();
            b.bytes.extend_from_slice(j.target);
            assert!(
                b.bytes.len() <= u32::MAX as usize,
                "batch exceeds 4 GB of sequence; offsets are u32"
            );
            b.jobs.push(FlatJob {
                q_off: q_off as u32,
                q_len: j.query.len() as u32,
                t_off: t_off as u32,
                t_len: j.target.len() as u32,
                h0: j.h0,
                w: clamp_band(
                    w0,
                    j.query.len(),
                    max_sc,
                    end_bonus,
                    s.o_ins,
                    s.e_ins,
                    s.o_del,
                    s.e_del,
                ),
            });
            b.max_qlen = b.max_qlen.max(j.query.len());
            b.cells += (j.query.len() * j.target.len()) as u64;
        }
        b.pad();
        b
    }

    fn pad(&mut self) {
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
    }

    /// Append another batch, rebasing its offsets. How the service merges the requests of several
    /// workers into one launch.
    pub fn append(&mut self, other: &FlatBatch) {
        let base = self.bytes.len() as u32;
        self.bytes.extend_from_slice(&other.bytes);
        assert!(
            self.bytes.len() <= u32::MAX as usize,
            "merged batch exceeds 4 GB"
        );
        self.jobs.extend(other.jobs.iter().map(|j| FlatJob {
            q_off: j.q_off + base,
            t_off: j.t_off + base,
            ..*j
        }));
        self.max_qlen = self.max_qlen.max(other.max_qlen);
        self.cells += other.cells;
        self.pad();
    }

    /// Jobs in the batch.
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Whether the batch holds no job.
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Job `k`'s query and target.
    pub fn seqs(&self, k: usize) -> (&[u8], &[u8]) {
        let j = &self.jobs[k];
        (
            &self.bytes[j.q_off as usize..][..j.q_len as usize],
            &self.bytes[j.t_off as usize..][..j.t_len as usize],
        )
    }
}

/// What a device backend provides: run a flattened batch.
pub trait DeviceKernel: Send + Sync {
    /// Short name, e.g. `"wgpu"`.
    fn name(&self) -> &'static str;
    /// The device's own description, for the log line.
    fn device_name(&self) -> String;
    /// One result per job, in job order, integer-identical to `ksw_extend2` with each job's own
    /// (already clamped) `w`. `None` if the device failed; the caller then runs the batch on the CPU,
    /// which produces the same bytes.
    fn run(&self, batch: &FlatBatch, s: &Scoring) -> Option<Vec<ExtendResult>>;
}

/// A CPU stand-in for [`DeviceKernel`], the scalar reference behind the device interface. Lets the
/// service and the co-scheduler be tested, and their byte-identity proved, on any machine.
pub struct CpuDevice;

impl DeviceKernel for CpuDevice {
    fn name(&self) -> &'static str {
        "cpu-device"
    }
    fn device_name(&self) -> String {
        "scalar reference".into()
    }
    fn run(&self, b: &FlatBatch, s: &Scoring) -> Option<Vec<ExtendResult>> {
        Some(
            (0..b.len())
                .map(|k| {
                    let (q, t) = b.seqs(k);
                    let j = &b.jobs[k];
                    // `w` is already clamped by the host. The reference clamps again, and a large
                    // `end_bonus` makes that second clamp a no-op (it only ever lowers `w`, and a
                    // bigger bonus loosens it), so the host's clamp is the one that binds, exactly
                    // as it is for a real device.
                    bwa_mem4_extend::ksw_extend2(
                        q,
                        t,
                        s.m,
                        &s.mat,
                        s.o_del,
                        s.e_del,
                        s.o_ins,
                        s.e_ins,
                        j.w,
                        1 << 20,
                        s.zdrop,
                        j.h0,
                    )
                })
                .collect(),
        )
    }
}

/// Counters the service keeps, for the end-of-run line and the probes.
#[derive(Debug, Default)]
pub struct ServiceStats {
    /// Device launches.
    pub launches: AtomicU64,
    /// Requests (one per worker call) served.
    pub requests: AtomicU64,
    /// Jobs run on the device.
    pub jobs: AtomicU64,
    /// Cells run on the device.
    pub cells: AtomicU64,
    /// Nanoseconds inside `DeviceKernel::run`.
    pub busy_ns: AtomicU64,
    /// Requests the device failed, rerun on the CPU by their caller.
    pub failures: AtomicU64,
}

struct Request {
    batch: FlatBatch,
    scoring: Arc<Scoring>,
    reply: mpsc::SyncSender<Option<Vec<ExtendResult>>>,
}

/// The single thread that owns the device and merges concurrent requests into one launch.
pub struct GpuService {
    tx: Mutex<Option<mpsc::Sender<Request>>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    stats: Arc<ServiceStats>,
    name: &'static str,
    device_name: String,
}

/// Upper bound on the jobs merged into one launch. Past saturation (~32 k jobs on the M4 Max, the
/// same knee on every GPU measured) a bigger launch buys nothing and delays every waiting worker.
const MAX_MERGED_JOBS: usize = 1 << 18;

impl GpuService {
    /// Start the service thread for `kernel`.
    pub fn start(kernel: Arc<dyn DeviceKernel>) -> Arc<Self> {
        let (tx, rx) = mpsc::channel::<Request>();
        let stats = Arc::new(ServiceStats::default());
        let st = Arc::clone(&stats);
        let name = kernel.name();
        let device_name = kernel.device_name();
        let worker = std::thread::Builder::new()
            .name("bwa-gpu".into())
            .spawn(move || serve(kernel.as_ref(), rx, &st))
            .expect("spawn GPU service thread");
        Arc::new(Self {
            tx: Mutex::new(Some(tx)),
            worker: Mutex::new(Some(worker)),
            stats,
            name,
            device_name,
        })
    }

    /// Backend short name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The device, as the driver describes it.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// The service's counters.
    pub fn stats(&self) -> &ServiceStats {
        &self.stats
    }

    /// Submit a batch; the returned receiver yields its results (or `None` on device failure).
    fn submit(
        &self,
        batch: FlatBatch,
        scoring: Arc<Scoring>,
    ) -> Option<mpsc::Receiver<Option<Vec<ExtendResult>>>> {
        let (reply, rx) = mpsc::sync_channel(1);
        let tx = self.tx.lock().ok()?.as_ref()?.clone();
        tx.send(Request {
            batch,
            scoring,
            reply,
        })
        .ok()?;
        Some(rx)
    }

    /// Stop the thread after it drains what is queued. Idempotent.
    pub fn shutdown(&self) {
        if let Ok(mut g) = self.tx.lock() {
            g.take();
        }
        if let Some(h) = self.worker.lock().ok().and_then(|mut g| g.take()) {
            let _ = h.join();
        }
    }
}

impl Drop for GpuService {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn serve(kernel: &dyn DeviceKernel, rx: mpsc::Receiver<Request>, st: &ServiceStats) {
    // A request whose scoring differed from the group being built; it heads the next group.
    let mut carry: Option<Request> = None;
    loop {
        let first = match carry.take() {
            Some(r) => r,
            None => match rx.recv() {
                Ok(r) => r,
                Err(_) => return,
            },
        };
        // Everything that queued while the previous launch ran joins this one. No timer: the
        // device's own busy time is the batching window, which is the classic adaptive batcher.
        let mut group = vec![first];
        let mut n_jobs = group[0].batch.len();
        while n_jobs < MAX_MERGED_JOBS {
            match rx.try_recv() {
                Ok(r) if r.scoring == group[0].scoring => {
                    n_jobs += r.batch.len();
                    group.push(r);
                }
                Ok(r) => {
                    carry = Some(r);
                    break;
                }
                Err(_) => break,
            }
        }
        let scoring = Arc::clone(&group[0].scoring);
        let t = Instant::now();
        let results = if group.len() == 1 {
            kernel.run(&group[0].batch, &scoring)
        } else {
            let mut merged = FlatBatch::default();
            for r in &group {
                merged.append(&r.batch);
            }
            kernel.run(&merged, &scoring)
        };
        st.busy_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        st.launches.fetch_add(1, Ordering::Relaxed);
        st.requests.fetch_add(group.len() as u64, Ordering::Relaxed);
        match results {
            Some(all) if all.len() == n_jobs => {
                st.jobs.fetch_add(n_jobs as u64, Ordering::Relaxed);
                st.cells.fetch_add(
                    group.iter().map(|r| r.batch.cells).sum::<u64>(),
                    Ordering::Relaxed,
                );
                let mut it = all.into_iter();
                for r in group {
                    let part: Vec<ExtendResult> = it.by_ref().take(r.batch.len()).collect();
                    let _ = r.reply.send(Some(part));
                }
            }
            _ => {
                st.failures.fetch_add(group.len() as u64, Ordering::Relaxed);
                for r in group {
                    let _ = r.reply.send(None);
                }
            }
        }
    }
}

/// How the jobs of a call are divided between the device and the calling thread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Split {
    /// A fixed fraction of jobs to the device, `0.0..=1.0`. `BWA4_GPU_SPLIT=<f>`.
    Fixed(f32),
    /// Balanced from measured rates (issue #57 level 2). The default.
    Adaptive,
}

impl Split {
    /// Parse `BWA4_GPU_SPLIT`: a number in `[0, 1]`, or anything else (including unset) for
    /// adaptive. Like `BWA4_GPU`, a bad value is not an error: the output is identical at any split.
    pub fn from_env() -> Self {
        std::env::var("BWA4_GPU_SPLIT")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|f| (0.0..=1.0).contains(f))
            .map_or(Split::Adaptive, Split::Fixed)
    }
}

/// Adaptive split bounds: neither side is ever starved of work, so both rates stay measured.
const F_MIN: f32 = 0.05;
const F_MAX: f32 = 0.98;
/// Where the adaptive split starts. A discrete GPU is many cores' worth; the controller converges
/// within a few dozen calls either way.
const F_START: f32 = 0.5;
/// Below this many jobs a call stays entirely on the CPU: a round trip to the service costs tens of
/// microseconds, which is more than a handful of extensions.
const MIN_DEVICE_JOBS: usize = 64;

/// A [`SwBackend`] that shares every batch between a CPU backend and a [`GpuService`].
pub struct CoSched<C: SwBackend> {
    cpu: C,
    service: Arc<GpuService>,
    split: Split,
    /// Current adaptive fraction, `f32` bits.
    frac: AtomicU32,
    /// Cells executed on each side, for the end-of-run line.
    cpu_cells: AtomicU64,
    dev_cells: AtomicU64,
    /// Scoring interned once: it is identical for every call of a run, and the service compares it.
    scoring: Mutex<Option<Arc<Scoring>>>,
    /// Calls with fewer jobs than this stay on the CPU.
    min_device_jobs: usize,
}

impl<C: SwBackend> CoSched<C> {
    /// Share work between `cpu` and `service` according to `split`.
    pub fn new(cpu: C, service: Arc<GpuService>, split: Split) -> Self {
        let f0 = match split {
            Split::Fixed(f) => f,
            Split::Adaptive => F_START,
        };
        Self {
            cpu,
            service,
            split,
            frac: AtomicU32::new(f0.to_bits()),
            cpu_cells: AtomicU64::new(0),
            dev_cells: AtomicU64::new(0),
            scoring: Mutex::new(None),
            min_device_jobs: MIN_DEVICE_JOBS,
        }
    }

    /// Override the small-call threshold. Tests set 0 so the acceptance harness's small batches
    /// reach the device; production keeps the default.
    pub fn with_min_device_jobs(mut self, n: usize) -> Self {
        self.min_device_jobs = n;
        self
    }

    /// The service this scheduler submits to.
    pub fn service(&self) -> &Arc<GpuService> {
        &self.service
    }

    /// The fraction of jobs currently sent to the device.
    pub fn fraction(&self) -> f32 {
        f32::from_bits(self.frac.load(Ordering::Relaxed))
    }

    /// `(cpu cells, device cells)` so far.
    pub fn cells(&self) -> (u64, u64) {
        (
            self.cpu_cells.load(Ordering::Relaxed),
            self.dev_cells.load(Ordering::Relaxed),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scoring(
        &self,
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        zdrop: i32,
    ) -> Arc<Scoring> {
        let want = Scoring {
            m,
            mat: mat[..m * m].to_vec(),
            o_del,
            e_del,
            o_ins,
            e_ins,
            zdrop,
        };
        let mut g = self.scoring.lock().expect("scoring lock");
        match g.as_ref() {
            Some(s) if **s == want => Arc::clone(s),
            _ => {
                let s = Arc::new(want);
                *g = Some(Arc::clone(&s));
                s
            }
        }
    }

    fn update(&self, cpu_cells: u64, cpu_s: f64, dev_cells: u64, dev_s: f64) {
        if self.split != Split::Adaptive || cpu_cells == 0 || dev_cells == 0 {
            return;
        }
        let rc = cpu_cells as f64 / cpu_s.max(1e-7);
        let rg = dev_cells as f64 / dev_s.max(1e-7);
        let target = (rg / (rg + rc)) as f32;
        // EMA with alpha 1/8. Racy read-modify-write across workers is fine: any interleaving
        // leaves a value inside the bounds, and the value never reaches the output.
        let cur = self.fraction();
        let next = (cur + (target - cur) * 0.125).clamp(F_MIN, F_MAX);
        self.frac.store(next.to_bits(), Ordering::Relaxed);
    }
}

fn cells_of(jobs: &[ExtendJob]) -> u64 {
    jobs.iter()
        .map(|j| (j.query.len() * j.target.len()) as u64)
        .sum()
}

impl<C: SwBackend> SwBackend for CoSched<C> {
    fn name(&self) -> &'static str {
        "cosched"
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
        let n = jobs.len();
        let k = if n < self.min_device_jobs.max(1) {
            0
        } else {
            ((self.fraction() * n as f32).round() as usize).min(n)
        };
        if k == 0 {
            self.cpu_cells.fetch_add(cells_of(jobs), Ordering::Relaxed);
            return self.cpu.extend_batch(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
            );
        }
        // The device takes the FIRST k jobs. The aligner sorts a round's jobs longest first, so the
        // device gets the long ones, where a GPU thread's fixed cost is best amortised.
        let (dev_jobs, cpu_jobs) = jobs.split_at(k);
        let scoring = self.scoring(m, mat, o_del, e_del, o_ins, e_ins, zdrop);
        let t0 = Instant::now();
        let batch = FlatBatch::from_jobs(dev_jobs, &scoring, w, end_bonus);
        let dev_cells = batch.cells;
        let pending = self.service.submit(batch, scoring);

        let cpu_cells = cells_of(cpu_jobs);
        let t_cpu = Instant::now();
        let mut out = Vec::with_capacity(n);
        let cpu_res = self.cpu.extend_batch(
            cpu_jobs, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
        );
        let cpu_s = t_cpu.elapsed().as_secs_f64();

        let dev_res = pending.and_then(|rx| rx.recv().ok().flatten());
        let dev_s = t0.elapsed().as_secs_f64();
        match dev_res {
            Some(r) if r.len() == k => {
                out.extend(r);
                self.dev_cells.fetch_add(dev_cells, Ordering::Relaxed);
                self.update(cpu_cells, cpu_s, dev_cells, dev_s);
            }
            _ => {
                // Device failed or the service is gone: the CPU computes the same answer.
                out.extend(self.cpu.extend_batch(
                    dev_jobs, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
                ));
                self.cpu_cells.fetch_add(dev_cells, Ordering::Relaxed);
            }
        }
        self.cpu_cells.fetch_add(cpu_cells, Ordering::Relaxed);
        out.extend(cpu_res);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_mem4_extend::{
        assert_backend_batch_matches_scalar, assert_backend_batch_order_invariant,
        assert_backend_matches_scalar, assert_backend_tie_rule_matches_scalar, ScalarBackend,
    };

    /// A device that always fails, to prove the fallback path is byte-identical too.
    struct BrokenDevice;
    impl DeviceKernel for BrokenDevice {
        fn name(&self) -> &'static str {
            "broken"
        }
        fn device_name(&self) -> String {
            "always fails".into()
        }
        fn run(&self, _: &FlatBatch, _: &Scoring) -> Option<Vec<ExtendResult>> {
            None
        }
    }

    fn gates<B: SwBackend>(b: &B) {
        assert_backend_matches_scalar(b);
        assert_backend_batch_matches_scalar(b);
        assert_backend_tie_rule_matches_scalar(b);
        assert_backend_batch_order_invariant(b);
    }

    /// Issue #54 trap 7, at unit level: every split, fixed or adaptive, passes the whole harness.
    #[test]
    fn every_split_passes_every_gate() {
        let svc = GpuService::start(Arc::new(CpuDevice));
        for split in [
            Split::Fixed(0.0),
            Split::Fixed(0.25),
            Split::Fixed(0.5),
            Split::Fixed(0.75),
            Split::Fixed(1.0),
            Split::Adaptive,
        ] {
            let cs = CoSched::new(ScalarBackend, Arc::clone(&svc), split).with_min_device_jobs(0);
            gates(&cs);
        }
        assert!(svc.stats().launches.load(Ordering::Relaxed) > 0);
        assert_eq!(svc.stats().failures.load(Ordering::Relaxed), 0);
    }

    /// A failing device degrades to the CPU, silently and identically.
    #[test]
    fn device_failure_falls_back_identically() {
        let svc = GpuService::start(Arc::new(BrokenDevice));
        let cs = CoSched::new(ScalarBackend, Arc::clone(&svc), Split::Fixed(1.0))
            .with_min_device_jobs(0);
        gates(&cs);
        assert!(svc.stats().failures.load(Ordering::Relaxed) > 0);
    }

    /// Concurrent callers get merged into shared launches and each still gets its own results.
    #[test]
    fn concurrent_workers_are_merged_and_unmixed() {
        let svc = GpuService::start(Arc::new(CpuDevice));
        let cs = CoSched::new(ScalarBackend, Arc::clone(&svc), Split::Fixed(1.0))
            .with_min_device_jobs(0);
        std::thread::scope(|sc| {
            for _ in 0..8 {
                sc.spawn(|| assert_backend_batch_matches_scalar(&cs));
            }
        });
        let st = svc.stats();
        assert!(st.launches.load(Ordering::Relaxed) > 0);
        assert!(st.requests.load(Ordering::Relaxed) >= st.launches.load(Ordering::Relaxed));
    }

    /// Merging rebases offsets: job k of the second batch reads back its own bytes.
    #[test]
    fn append_rebases_offsets() {
        let s = Scoring {
            m: 5,
            mat: vec![1; 25],
            o_del: 6,
            e_del: 1,
            o_ins: 6,
            e_ins: 1,
            zdrop: 100,
        };
        let (q1, t1, q2, t2) = (vec![0u8, 1, 2], vec![3u8; 5], vec![2u8; 7], vec![1u8, 0]);
        let a = FlatBatch::from_jobs(
            &[ExtendJob {
                query: &q1,
                target: &t1,
                h0: 3,
            }],
            &s,
            100,
            5,
        );
        let b = FlatBatch::from_jobs(
            &[ExtendJob {
                query: &q2,
                target: &t2,
                h0: 4,
            }],
            &s,
            100,
            5,
        );
        let mut m = a.clone();
        m.append(&b);
        assert_eq!(m.len(), 2);
        assert_eq!(m.seqs(0), (&q1[..], &t1[..]));
        assert_eq!(m.seqs(1), (&q2[..], &t2[..]));
        assert!(m.bytes.len().is_multiple_of(4));
        assert_eq!(m.max_qlen, 7);
    }

    #[test]
    fn split_env_parses_and_defaults() {
        let saved = std::env::var("BWA4_GPU_SPLIT").ok();
        for (v, want) in [
            (Some("0.25"), Split::Fixed(0.25)),
            (Some("1"), Split::Fixed(1.0)),
            (Some("auto"), Split::Adaptive),
            (Some("7"), Split::Adaptive),
            (None, Split::Adaptive),
        ] {
            match v {
                Some(x) => std::env::set_var("BWA4_GPU_SPLIT", x),
                None => std::env::remove_var("BWA4_GPU_SPLIT"),
            }
            assert_eq!(Split::from_env(), want, "{v:?}");
        }
        match saved {
            Some(x) => std::env::set_var("BWA4_GPU_SPLIT", x),
            None => std::env::remove_var("BWA4_GPU_SPLIT"),
        }
    }
}
