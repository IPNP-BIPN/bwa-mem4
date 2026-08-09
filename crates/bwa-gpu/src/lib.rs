//! The GPU seam: everything a GPU backend needs that is not GPU code (issue #53, step 2).
//!
//! # What is here, and why it is here rather than in a backend
//!
//! Three things are common to every GPU target and none of them involve a GPU:
//!
//! 1. **Flattening.** [`ExtendJob`] carries `&[u8]` into the caller's buffers. A GPU wants one flat
//!    allocation with offsets. [`JobArena`] does that conversion, reuses its allocation across
//!    batches, and grows to the largest batch it has seen. Doing it once here means neither backend
//!    reimplements it and neither one leaks the requirement into the aligner.
//! 2. **Selection.** `BWA4_GPU=metal|cuda|off` picks a target at run time, with a **silent CPU
//!    fallback** when the requested backend is not compiled in or not usable. Silent because a run
//!    that quietly produces the right answer on the CPU is better than one that fails on a laptop
//!    without a GPU, and the SAM is identical either way.
//! 3. **The queue.** How many batches to keep in flight, which is [`SwBackendAsync::queue_depth`].
//!
//! # What is deliberately NOT here
//!
//! Any dependency on CUDA or Metal. `cargo build` with no features adds nothing to the dependency
//! graph, which is the first acceptance criterion of issue #53. The concrete backends will live in
//! their own crates behind the `metal` and `cuda` features; until they exist, asking for one gets
//! the fallback, and that path is tested rather than assumed.
//!
//! # Why the seam is worth having before a single GPU kernel
//!
//! Issue #53's step 0 measured the batch sizes this seam has to carry, and the answer split:
//! extension submits **5 381 jobs per call** at 150 bp, comfortably over the threshold where one
//! kernel launch per call amortises, while mate rescue submits **72**. So the extension side can use
//! this seam as it stands, and the rescue side needs a cross-thread aggregation queue that does not
//! exist yet. That is why [`JobArena`] is written against extension jobs and why `queue_depth` is a
//! backend property rather than a constant.

use bwa_mem4_extend::{ExtendJob, ExtendResult, ScalarBackend, SwBackendAsync, SyncAsAsync};

/// One batch of [`ExtendJob`]s copied into a single flat allocation, with offsets.
///
/// # Layout
///
/// `bytes` holds every job's query then its target, back to back, in job order. `spans[k]` is job
/// `k`'s `(query offset, query length, target offset, target length)`, and `h0[k]` its seed score.
/// One allocation and one offset table is what a GPU kernel wants: it can be handed to the device as
/// a single buffer, and on unified memory (Apple) it can BE the device buffer, written in place by
/// the CPU with no copy at all. That is the structural advantage issue #55 rests on, and it is the
/// reason the arena owns its storage rather than borrowing the caller's.
///
/// # Reuse
///
/// [`fill`] clears and refills without freeing, so a backend that keeps one arena per queue slot
/// allocates only while batches are still growing. Allocating and freeing per batch would cost more
/// than the launch it is meant to amortise, which is the third implementation point of issue #53.
///
/// [`fill`]: JobArena::fill
#[derive(Debug, Default, Clone)]
pub struct JobArena {
    /// Every query and target, concatenated in job order.
    bytes: Vec<u8>,
    /// Per job: `(q_off, q_len, t_off, t_len)` into [`bytes`](JobArena::bytes).
    spans: Vec<(u32, u32, u32, u32)>,
    /// Per job: the seed score the DP starts from.
    h0: Vec<i32>,
}

impl JobArena {
    /// An empty arena. Call [`fill`](JobArena::fill) to load a batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Copy `jobs` in, replacing whatever was there, keeping the allocation.
    ///
    /// # Panics
    /// If a single batch exceeds 4 GB of sequence, which no realistic batch does: offsets are `u32`
    /// because doubling the offset table for a bound nothing approaches would be a real cost on the
    /// device side. The panic is the honest failure for an impossible input rather than a silent
    /// truncation.
    pub fn fill(&mut self, jobs: &[ExtendJob]) {
        self.bytes.clear();
        self.spans.clear();
        self.h0.clear();
        self.spans.reserve(jobs.len());
        self.h0.reserve(jobs.len());
        let total: usize = jobs.iter().map(|j| j.query.len() + j.target.len()).sum();
        self.bytes.reserve(total);
        for j in jobs {
            let q_off = self.bytes.len();
            self.bytes.extend_from_slice(j.query);
            let t_off = self.bytes.len();
            self.bytes.extend_from_slice(j.target);
            assert!(
                self.bytes.len() <= u32::MAX as usize,
                "batch exceeds 4 GB of sequence; offsets are u32"
            );
            self.spans.push((
                q_off as u32,
                j.query.len() as u32,
                t_off as u32,
                j.target.len() as u32,
            ));
            self.h0.push(j.h0);
        }
    }

    /// Jobs currently loaded.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether no job is loaded.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The flat sequence buffer, the thing a device gets a pointer to.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Per-job `(q_off, q_len, t_off, t_len)`.
    pub fn spans(&self) -> &[(u32, u32, u32, u32)] {
        &self.spans
    }

    /// Per-job seed scores.
    pub fn h0(&self) -> &[i32] {
        &self.h0
    }

    /// Job `k`'s query and target, read back out of the arena.
    ///
    /// Exists so the round trip is testable without a device: an arena that cannot give back what it
    /// was given is a bug that would otherwise only surface as wrong alignments on hardware nobody
    /// here has.
    pub fn job(&self, k: usize) -> (&[u8], &[u8], i32) {
        let (qo, ql, to, tl) = self.spans[k];
        (
            &self.bytes[qo as usize..(qo + ql) as usize],
            &self.bytes[to as usize..(to + tl) as usize],
            self.h0[k],
        )
    }
}

/// What `BWA4_GPU` asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRequest {
    /// No GPU: use the CPU backend. The default, and what an unset or unrecognised value means.
    Off,
    /// Apple Metal (issue #55).
    Metal,
    /// NVIDIA CUDA (issue #56).
    Cuda,
}

/// Parse `BWA4_GPU`. Unset, empty, `off`, or anything unrecognised gives [`GpuRequest::Off`].
///
/// Unrecognised values are NOT an error on purpose: this variable will be set in job scripts that
/// outlive any one version of this binary, and a typo that silently produces correct output on the
/// CPU is a better failure than a run that dies at hour six of a WGS.
pub fn requested() -> GpuRequest {
    match std::env::var("BWA4_GPU").ok().as_deref() {
        Some("metal") => GpuRequest::Metal,
        Some("cuda") => GpuRequest::Cuda,
        _ => GpuRequest::Off,
    }
}

/// Why a requested backend was not used. Returned rather than logged so the caller decides whether
/// to say anything; the aligner's own policy is to stay silent, since the output is identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// Nothing was requested.
    NotRequested,
    /// Requested, but this binary was built without that feature.
    NotCompiledIn,
    /// Requested and compiled in, but no usable device was found at run time.
    NoDevice,
}

/// The backend the process should use, and why it is that one.
///
/// Today the answer is always the CPU: no GPU backend exists yet. The shape is what matters, because
/// it is what lets #55 and #56 be a change of ONE match arm rather than a change of the aligner.
pub struct Selection {
    /// The chosen backend, already wearing the async seam.
    pub backend: SyncAsAsync<ScalarBackend>,
    /// What was asked for.
    pub requested: GpuRequest,
    /// Why the request was not honoured, if it was not.
    pub fallback: Fallback,
}

/// Choose a backend from the environment, falling back to the CPU without complaint.
///
/// # Returns
/// A [`Selection`] whose `fallback` says what happened. On a machine with no GPU, or a build with no
/// GPU feature, this is the CPU backend and the run is bit-for-bit what it would have been with
/// `BWA4_GPU` unset. That equality is the point: it is what makes the variable safe to leave in a
/// pipeline script.
pub fn select() -> Selection {
    let requested = requested();
    // Each arm becomes `try_metal()` / `try_cuda()` when those crates land. The `cfg` is written now
    // so that `--features metal` on a GPU-less host compiles and takes the fallback, which is issue
    // #53's second acceptance criterion and is tested below.
    let fallback = match requested {
        GpuRequest::Off => Fallback::NotRequested,
        GpuRequest::Metal => {
            if cfg!(feature = "metal") {
                Fallback::NoDevice
            } else {
                Fallback::NotCompiledIn
            }
        }
        GpuRequest::Cuda => {
            if cfg!(feature = "cuda") {
                Fallback::NoDevice
            } else {
                Fallback::NotCompiledIn
            }
        }
    };
    Selection {
        backend: SyncAsAsync::new(ScalarBackend),
        requested,
        fallback,
    }
}

/// Run a batch through the async seam, the way a pipelining caller would.
///
/// One function so the seam has a single tested entry point while no caller uses it yet: submitting
/// and collecting in the same breath is the degenerate case of pipelining, and it must equal
/// `extend_batch` exactly. That equality is what [`SwBackendAsync`] promises and what the test below
/// checks against the scalar reference.
#[allow(clippy::too_many_arguments)]
pub fn round_trip<B: SwBackendAsync>(
    backend: &B,
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
    let ticket = backend.submit(
        jobs, m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop,
    );
    backend.collect(ticket)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_mem4_extend::SwBackend;

    /// bwa's default DNA matrix, as `bwa_fill_scmat` builds it.
    fn scoring() -> Vec<i8> {
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
        mat
    }

    /// Deterministic jobs of ragged lengths, including an empty-ish one, so the arena's offsets are
    /// exercised rather than the one length that happens to work.
    fn jobs_storage(n: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<i32>) {
        let mut state = 0x51ed_1234_9876_abcdu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (mut q, mut t, mut h) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..n {
            let qlen = 1 + (next() % 150) as usize;
            let tlen = qlen + (next() % 40) as usize;
            // `% 5`, not `% 4`: code 4 is N. The arena is byte transport, so it must carry N as
            // faithfully as anything else, and the project's generators all draw this way.
            q.push((0..qlen).map(|_| (next() % 5) as u8).collect::<Vec<u8>>());
            t.push((0..tlen).map(|_| (next() % 5) as u8).collect::<Vec<u8>>());
            h.push(1 + (k % 17) as i32);
        }
        (q, t, h)
    }

    /// The arena gives back exactly what it was given, for every job, including after a refill with
    /// a smaller batch (which is where a length-vs-capacity confusion would show).
    #[test]
    fn arena_round_trips() {
        let (qs, ts, hs) = jobs_storage(64);
        let jobs: Vec<ExtendJob> = (0..qs.len())
            .map(|i| ExtendJob {
                query: &qs[i],
                target: &ts[i],
                h0: hs[i],
            })
            .collect();
        let mut arena = JobArena::new();
        arena.fill(&jobs);
        assert_eq!(arena.len(), jobs.len());
        for (k, j) in jobs.iter().enumerate() {
            let (q, t, h0) = arena.job(k);
            assert_eq!(q, j.query, "job {k} query");
            assert_eq!(t, j.target, "job {k} target");
            assert_eq!(h0, j.h0, "job {k} h0");
        }
        // Refill smaller: the allocation is kept, the contents must not be.
        arena.fill(&jobs[..7]);
        assert_eq!(arena.len(), 7);
        for (k, j) in jobs[..7].iter().enumerate() {
            let (q, t, _) = arena.job(k);
            assert_eq!(q, j.query);
            assert_eq!(t, j.target);
        }
        arena.fill(&[]);
        assert!(arena.is_empty());
    }

    /// `collect(submit(..))` equals `extend_batch`, which is the whole contract of the async seam.
    #[test]
    fn async_seam_equals_blocking() {
        let mat = scoring();
        let (qs, ts, hs) = jobs_storage(97);
        let jobs: Vec<ExtendJob> = (0..qs.len())
            .map(|i| ExtendJob {
                query: &qs[i],
                target: &ts[i],
                h0: hs[i],
            })
            .collect();
        let backend = SyncAsAsync::new(ScalarBackend);
        let blocking = backend.extend_batch(&jobs, 5, &mat, 6, 1, 6, 1, 100, 5, 100);
        let seam = round_trip(&backend, &jobs, 5, &mat, 6, 1, 6, 1, 100, 5, 100);
        assert_eq!(seam, blocking);
    }

    /// Tickets may be collected out of order, and each batch keeps its own results. A backend that
    /// mixed two in-flight batches would pass every single-batch test and corrupt a pipelined run.
    #[test]
    fn tickets_are_independent_and_order_free() {
        let mat = scoring();
        let (qs, ts, hs) = jobs_storage(40);
        let mk = |r: std::ops::Range<usize>| -> Vec<ExtendJob> {
            r.map(|i| ExtendJob {
                query: &qs[i],
                target: &ts[i],
                h0: hs[i],
            })
            .collect()
        };
        let (a, b) = (mk(0..20), mk(20..40));
        let backend = SyncAsAsync::new(ScalarBackend);
        let ta = backend.submit(&a, 5, &mat, 6, 1, 6, 1, 100, 5, 100);
        let tb = backend.submit(&b, 5, &mat, 6, 1, 6, 1, 100, 5, 100);
        // Collected in the reverse of the submitted order, on purpose.
        let got_b = backend.collect(tb);
        let got_a = backend.collect(ta);
        assert_eq!(
            got_a,
            backend.extend_batch(&a, 5, &mat, 6, 1, 6, 1, 100, 5, 100)
        );
        assert_eq!(
            got_b,
            backend.extend_batch(&b, 5, &mat, 6, 1, 6, 1, 100, 5, 100)
        );
    }

    /// A CPU backend must not claim an overlap it does not have.
    #[test]
    fn cpu_queue_depth_is_one() {
        assert_eq!(SyncAsAsync::new(ScalarBackend).queue_depth(), 1);
    }

    /// Selection falls back without complaint, and says why. Run for each request value rather than
    /// only the default, because the case that matters is `BWA4_GPU=metal` on a machine that cannot
    /// honour it: issue #53's acceptance criterion is that this is a silent CPU run, not an error.
    #[test]
    fn selection_falls_back_and_explains() {
        // The env var is process-global, so this test sets and restores it rather than relying on
        // the ambient value, and it is the only test here that touches it.
        let saved = std::env::var("BWA4_GPU").ok();
        for (val, want_req, want_fb) in [
            (None, GpuRequest::Off, Fallback::NotRequested),
            (Some("off"), GpuRequest::Off, Fallback::NotRequested),
            (Some("nonsense"), GpuRequest::Off, Fallback::NotRequested),
            (
                Some("metal"),
                GpuRequest::Metal,
                if cfg!(feature = "metal") {
                    Fallback::NoDevice
                } else {
                    Fallback::NotCompiledIn
                },
            ),
            (
                Some("cuda"),
                GpuRequest::Cuda,
                if cfg!(feature = "cuda") {
                    Fallback::NoDevice
                } else {
                    Fallback::NotCompiledIn
                },
            ),
        ] {
            match val {
                Some(v) => std::env::set_var("BWA4_GPU", v),
                None => std::env::remove_var("BWA4_GPU"),
            }
            let sel = select();
            assert_eq!(sel.requested, want_req, "request for {val:?}");
            assert_eq!(sel.fallback, want_fb, "fallback for {val:?}");
            assert_eq!(sel.backend.name(), "scalar");
        }
        match saved {
            Some(v) => std::env::set_var("BWA4_GPU", v),
            None => std::env::remove_var("BWA4_GPU"),
        }
    }
}
