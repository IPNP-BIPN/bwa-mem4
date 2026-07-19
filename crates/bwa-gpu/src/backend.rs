//! [`MetalBackend`]: [`SwBackend`] over the Metal `sw_extend` compute kernel.
//!
//! # What is here
//!
//! The Rust half of the GPU backend. All the alignment arithmetic lives in the shader string in
//! `lib.rs`; this file only marshals data across the boundary. Read it in this order:
//!
//! 1. [`Params`], the CPU mirror of the shader's `Params` struct. Nothing checks that the two agree,
//!    so it is the first place to look when GPU results go quietly wrong.
//! 2. `MetalBackend::extend_batch`, the whole flow: bail out to scalar, pack, dispatch, unpack.
//! 3. [`clamp_band`] and [`is_uniform_dna`], the two pieces of work deliberately kept on the CPU
//!    (float rounding we cannot control, and a matrix shape the shader assumes).
//!
//! `MetalBackend::extend` exists only for the trait and for the acceptance gate: a batch of one
//! pays a full command-buffer round trip.

use crate::metal_ctx::MetalCtx;
use bwa_extend::{ExtendJob, ExtendResult, ScalarBackend, SwBackend};
use metal::{MTLResourceOptions, MTLSize};

/// GPU seed-extension backend. Byte-identical to [`bwa_extend::ksw_extend2`] (integer DP on the GPU),
/// falling back to the scalar backend when there is no Metal device or the matrix is non-uniform.
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalBackend;

/// Mirror of the MSL `Params` struct in `lib.rs` (field order and `i32` layout must match exactly).
/// There is no compile-time link between the two: the shader is a runtime-compiled string, so a
/// field reordered on one side and not the other produces silently wrong scores, not an error.
/// All fields are `i32` so the layout is trivially 4-byte-aligned and identical to MSL's `int`.
#[repr(C)]
struct Params {
    /// Match score, positive (bwa's `-A`). Set from the scoring matrix maximum, which for a bwa
    /// matrix *is* the match score; the same value is passed to [`clamp_band`] as `max_sc`, so the
    /// band clamp and the kernel cannot disagree about it. Read by the shader's `sc` expression.
    a: i32,
    /// Mismatch score, negative (the negation of bwa's `-B`). Taken from `mat[1]`, the first
    /// off-diagonal entry. Applied when both bases are concrete (`code < 4`) and different.
    mm: i32,
    /// Score when either base is ambiguous (code >= 4, i.e. `N`). Taken from `mat[m-1]`, the last
    /// entry of the matrix's first row, which for bwa is `-1`. Wins over `a`/`mm`.
    npen: i32,
    /// Deletion (gap in the query, CIGAR `D`) gap-open penalty, positive, subtracted by the kernel.
    o_del: i32,
    /// Deletion gap-extend penalty per base, positive. Must be >= 1: [`clamp_band`] divides by it.
    e_del: i32,
    /// Insertion (gap in the target, CIGAR `I`) gap-open penalty, positive.
    o_ins: i32,
    /// Insertion gap-extend penalty per base, positive. Must be >= 1, same reason as `e_del`.
    e_ins: i32,
    /// Bonus for reaching the query end. Uploaded only so the two structs stay field-for-field
    /// identical: the shader never reads it, because in `ksw_extend2` it only ever widens the band
    /// clamp, and [`clamp_band`] has already consumed it on this side.
    end_bonus: i32,
    /// Z-drop: how far a row's best score may fall below the running best (after discounting the
    /// gap implied by the drift off the diagonal) before the kernel abandons that job. `<= 0`
    /// disables the test entirely.
    zdrop: i32,
    /// Number of alignments in this dispatch, i.e. `jobs.len()`. The kernel compares its thread id
    /// against this to retire the tail threads that `dispatch_threads` rounds the grid up to.
    njobs: i32,
    /// Width, in `i32` elements, of one thread's DP slice inside the `ehh`/`ehe` scratch buffers:
    /// the longest query in the batch plus one sentinel column. Uniform across threads, so the
    /// kernel addresses its slice with a single multiply.
    stride: i32,
}

/// Per-job band clamp: narrow the requested half-band `w0` to the widest gap that could ever pay
/// for itself, identical to `ksw_extend2` (`ksw.cpp:456-461`) and to the NEON kernel.
///
/// A gap of length L costs `o + L*e`, and an alignment can earn at most `qlen * max_sc` (plus
/// `end_bonus`), so bands beyond `(qlen*max_sc + end_bonus - o)/e + 1` are provably unreachable.
/// Computed here on the CPU, in `f64`, for two reasons: the C does the division in `double` and a
/// truncating integer division would give a different `w` on some inputs, and doing it GPU-side
/// would put floating point (whose rounding mode and precision we do not control across devices)
/// on the byte-identity path. The kernel receives the finished integer per job in `wbuf`.
///
/// `max_sc` is the largest entry of the scoring matrix (bwa's `-A`); every other argument has the
/// same meaning and units as in [`bwa_extend::ksw_extend2`]. `e_ins`/`e_del` must be non-zero.
///
/// # Parameters
///
/// * `w0`: the caller's requested half band width, in query columns. Comes from the aligner's
///   `-w` option by way of `extend_batch`, and is shared by the whole batch.
/// * `qlen`: this one job's query length in bases. The only per-job input, which is why the clamp
///   has to be recomputed for every job rather than once per batch.
/// * `max_sc`: the largest entry of the scoring matrix, i.e. the match score. Bounds the score an
///   alignment could possibly earn per query base.
/// * `end_bonus`: bonus for reaching the query end. Enters only here, never a DP score.
/// * `o_ins`, `e_ins`, `o_del`, `e_del`: the affine gap penalties as positive numbers. The extends
///   are divisors and must be non-zero; zero would produce an infinity and then a garbage `as i32`.
///
/// # Returns
///
/// The clamped half band width, always >= 1 (a zero or negative band would evaluate no cells at
/// all), and always <= `w0`. Uploaded per job in the `wbuf` buffer at binding 7.
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
    let mut clamped = w0;
    // Widest insertion that could still pay for itself: beyond this, the gap penalty exceeds the
    // most the whole query could earn. The `as i32` truncates toward zero, matching the C cast.
    let max_ins = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_ins))
        / f64::from(e_ins))
        + 1.0) as i32;
    clamped = clamped.min(max_ins.max(1));
    // Same bound for deletions. `ksw.cpp:461` tags this one "TODO: is this necessary?"; it stays,
    // because dropping it would change `w` and therefore the alignment.
    let max_del = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_del))
        / f64::from(e_del))
        + 1.0) as i32;
    clamped = clamped.min(max_del.max(1));
    clamped
}

/// `BWA3_GPU_STATS=1` phase counters. End-to-end wall time cannot tell a faster kernel from a slower
/// pack, and those pull in opposite directions for a layout change -- so measure them apart. At `-t1`
/// nothing else holds the queue, so `commit` -> `wait_until_completed` IS the GPU execution span.
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    /// Nanoseconds spent CPU-side marshalling jobs into buffers: everything from the pipeline
    /// lookup to the `Params` upload, i.e. the cost the GPU path adds over just doing the DP.
    /// Summed across all aligner threads; only accumulated when [`enabled`].
    pub static PACK_NS: AtomicU64 = AtomicU64::new(0);
    /// Nanoseconds spent between `commit` and `wait_until_completed` returning. At `-t1` nothing
    /// else holds the queue, so this is the GPU execution span; at higher thread counts it also
    /// includes time queued behind other threads' dispatches.
    pub static GPU_NS: AtomicU64 = AtomicU64::new(0);
    /// Total alignments dispatched, summed over every batch. Divide the timings by this for a
    /// per-alignment cost.
    pub static JOBS: AtomicU64 = AtomicU64::new(0);
    /// Number of `extend_batch` calls that actually reached the GPU. `JOBS / CALLS` is the mean
    /// batch size, which is the number that decides whether the fixed dispatch cost is amortised.
    pub static CALLS: AtomicU64 = AtomicU64::new(0);

    /// Whether `BWA3_GPU_STATS` was set in the environment.
    ///
    /// # Returns
    ///
    /// `true` if the variable is present with any value (including empty). Read once into a
    /// `OnceLock` and cached, so this is a cheap test on the per-batch hot path and cannot change
    /// mid-run.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA3_GPU_STATS").is_some())
    }

    /// Print the accumulated counters to stderr. Called from the CLI at the end of a run.
    ///
    /// # Parameters
    ///
    /// None: everything comes from the module's atomics. Silently does nothing unless
    /// [`enabled`], so it is safe to call unconditionally.
    pub fn dump() {
        if !enabled() {
            return;
        }
        // Both counters converted from nanoseconds to seconds for the report.
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

/// Does `mat` have bwa's uniform shape, i.e. is it fully described by three numbers (match on the
/// diagonal, one mismatch value off it, one ambiguity value on the whole last row and column)?
///
/// This is the precondition for the shader's inlined `sc` expression, which reconstructs the score
/// arithmetically instead of loading the matrix. Anything else (a user-supplied non-uniform matrix
/// via `-A`-style scoring files, or a different alphabet) makes `extend_batch` route the whole
/// batch to the scalar backend. So this is a *domain guard*, not an approximation: the GPU path is
/// only ever taken where the two formulations are exactly equal.
///
/// Note it derives the three values from `mat[0]`, `mat[1]` and `mat[m-1]` and then verifies every
/// cell against them, so a matrix that merely looks uniform in its first row is still rejected.
///
/// # Parameters
///
/// * `mat`: the substitution matrix, row-major and at least `m*m` long, so `mat[t*m + q]` scores
///   target base `t` against query base `q`. Supplied by the aligner (`bwa_fill_scmat`, or a user
///   scoring file). A longer slice is accepted; only the first `m*m` entries are inspected.
/// * `m`: the alphabet size, 5 for DNA (`A,C,G,T,N`). `m < 2` is rejected outright, since the
///   "last row and column are the ambiguity penalty" shape is meaningless below that.
///
/// # Returns
///
/// `true` if the whole matrix is reproduced by the three derived values, which is exactly the
/// condition under which the shader's arithmetic `sc` expression equals a matrix lookup. `false`
/// sends the entire batch to [`ScalarBackend`].
fn is_uniform_dna(mat: &[i8], m: usize) -> bool {
    if m < 2 || mat.len() < m * m {
        return false;
    }
    // The three values a uniform matrix would be built from: the diagonal (match), the first
    // off-diagonal entry (mismatch), and the last entry of row 0 (the N column, i.e. ambiguity).
    let (a, mm, npen) = (mat[0], mat[1], mat[m - 1]);
    for i in 0..m {
        for j in 0..m {
            // What a uniform matrix must hold at (i, j): the N row/column dominates, then the
            // diagonal, then everything else is the mismatch value.
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

    /// One seed extension, byte-identical to [`bwa_extend::ksw_extend2`].
    ///
    /// # Parameters
    ///
    /// * `query`: the read bases to extend over, one byte per base as 2-bit codes
    ///   (`0=A 1=C 2=G 3=T`, `4=N`), not ASCII. May be empty.
    /// * `target`: the reference stretch to align the query against, same encoding.
    /// * `m`: alphabet size, 5 for DNA.
    /// * `mat`: the `m*m` row-major substitution matrix, `mat[t*m + q]`. If it is not uniform in
    ///   the sense of [`is_uniform_dna`], this call transparently runs on the CPU instead.
    /// * `o_del`, `e_del`: deletion open and per-base extend penalties, positive; `e_del >= 1`.
    /// * `o_ins`, `e_ins`: insertion open and per-base extend penalties, positive; `e_ins >= 1`.
    /// * `w`: requested half band width in query columns, clamped internally by [`clamp_band`].
    /// * `end_bonus`: bonus for reaching the query end. Only reaches the band clamp, never a score.
    /// * `zdrop`: give-up threshold; `<= 0` disables the z-drop test.
    /// * `h0`: the score the seed already earned, the DP's starting value. Must be `> 0`.
    ///
    /// # Returns
    ///
    /// The single [`ExtendResult`] for this alignment, exactly what `ksw_extend2` would return.
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
        // A batch of one. Correct but pathologically slow (a full command-buffer round trip per
        // alignment), so this exists for the acceptance gate and for API completeness; production
        // code goes through `extend_batch` with thousands of jobs.
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

    /// The production entry point: run every job in `jobs` as one GPU dispatch, one thread each.
    ///
    /// # Parameters
    ///
    /// * `jobs`: the alignments to run. Each carries its own `query`, `target` and `h0`; those are
    ///   the only inputs that vary across the batch. Any length, including empty (which falls back
    ///   to the scalar backend rather than dispatching nothing).
    /// * `m`, `mat`, `o_del`, `e_del`, `o_ins`, `e_ins`, `end_bonus`, `zdrop`: the scoring scheme,
    ///   shared by every job. Same meaning and units as in [`MetalBackend::extend`].
    /// * `w0`: the requested half band width before clamping, shared by every job. It is clamped
    ///   *per job* below, because the clamp depends on that job's query length.
    ///
    /// # Returns
    ///
    /// One [`ExtendResult`] per job, in the same order, each equal to what
    /// [`bwa_extend::ksw_extend2`] would return for that job. The length always equals
    /// `jobs.len()`.
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
        // ---- Step 1: decide whether the GPU can take this batch at all --------------------
        // Fall back to scalar for the empty batch, a non-uniform matrix, or no Metal device.
        let ctx = MetalCtx::get();
        if jobs.is_empty() || !is_uniform_dna(mat, m) || ctx.is_none() {
            return ScalarBackend.extend_batch(
                jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
            );
        }
        let ctx = ctx.unwrap();
        // Start of the "pack" phase for BWA3_GPU_STATS: everything from here to the dispatch is
        // marshalling cost, i.e. the overhead the GPU route adds over simply doing the DP.
        let t_pack = std::time::Instant::now();
        // The cached pipeline state object. `None` means the MSL failed to compile at startup.
        let pso = match ctx.pipeline("sw_extend") {
            Some(p) => p,
            None => {
                return ScalarBackend.extend_batch(
                    jobs, m, mat, o_del, e_del, o_ins, e_ins, w0, end_bonus, zdrop,
                )
            }
        };

        // Job count: also the thread count, the length of every per-job array below, and `njobs`.
        let n = jobs.len();
        // Largest entry of the scoring matrix. For a bwa matrix this is the match score `a`, and
        // it is both the kernel's `Params::a` and the band clamp's score-per-base bound.
        let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;
        // One DP slice per thread, all the same width, sized by the longest query in the batch;
        // `+ 1` for the sentinel column ksw_extend2 writes at `end` (the C's `calloc(qlen + 1)`).
        // Uniform stride wastes memory on short jobs but keeps the addressing a single multiply,
        // and the scratch is transient anyway.
        let max_q = jobs.iter().map(|j| j.query.len()).max().unwrap_or(0);
        let stride = (max_q + 1) as i32;

        // ---- Step 2: pack the jobs into flat, GPU-visible buffers -------------------------
        // Pack the jobs into flat buffers: concatenated codes + per-job offset/length/h0/band.
        // NB the AoS `[lane * stride + column]` layout here and in the kernel is deliberate, not an
        // oversight: transposing it to the CPU kernel's `[column * lane]` SoA form was measured
        // 2.2x SLOWER (see docs/optimization-roadmap.md). One GPU thread owns one alignment and
        // walks its row contiguously, so a fetched cache line already serves that thread's next 32
        // columns; SoA amortises the same line across lanes instead, for the same line count, and
        // pays an ~800 KB stride between consecutive columns on top.
        // qbuf/tbuf: every job's query and target bases concatenated, one byte per base as 2-bit
        // codes. qoff/toff give each job's starting byte index into them; qlen/tlen its length in
        // bases; h0v its seed score; wv its band width AFTER the per-job f64 clamp. All eight
        // become buffers 0..=7, in that order.
        let mut qbuf: Vec<u8> = Vec::new();
        let mut tbuf: Vec<u8> = Vec::new();
        let mut qoff = vec![0i32; n];
        let mut toff = vec![0i32; n];
        let mut qlen = vec![0i32; n];
        let mut tlen = vec![0i32; n];
        let mut h0v = vec![0i32; n];
        let mut wv = vec![0i32; n];
        for (k, job) in jobs.iter().enumerate() {
            qoff[k] = qbuf.len() as i32;
            qbuf.extend_from_slice(job.query);
            toff[k] = tbuf.len() as i32;
            tbuf.extend_from_slice(job.target);
            qlen[k] = job.query.len() as i32;
            tlen[k] = job.target.len() as i32;
            h0v[k] = job.h0;
            wv[k] = clamp_band(
                w0,
                job.query.len(),
                max_sc,
                end_bonus,
                o_ins,
                e_ins,
                o_del,
                e_del,
            );
        }
        // Never hand a zero-length buffer to Metal: `new_buffer_with_data` with length 0 is invalid.
        // Reachable because a job may legitimately carry an empty query or target (an extension that
        // has nothing left to extend into). The pushed byte is never read, since qlen/tlen are 0.
        if qbuf.is_empty() {
            qbuf.push(0);
        }
        if tbuf.is_empty() {
            tbuf.push(0);
        }

        // The shared Metal device; every buffer below is allocated from it.
        let dev = &ctx.device;
        // Unified memory: `StorageModeShared` means CPU and GPU address the same pages, so
        // `new_buffer_with_data` is a single memcpy and reading `b_out.contents()` afterwards needs
        // no copy back. On a discrete GPU this would be two PCIe transfers instead.
        let shared = MTLResourceOptions::StorageModeShared;
        // Copy `len` bytes starting at `ptr` into a fresh shared buffer. `len` is in BYTES, hence
        // the `* 4` on every i32 array below; passing 0 would be invalid, which is what the
        // `qbuf`/`tbuf` filler pushes above prevent.
        let new_shared_buffer = |ptr: *const u8, len: usize| {
            dev.new_buffer_with_data(ptr as *const _, len as u64, shared)
        };
        // Bindings 0..=7, in the order the kernel declares them.
        let b_q = new_shared_buffer(qbuf.as_ptr(), qbuf.len());
        let b_t = new_shared_buffer(tbuf.as_ptr(), tbuf.len());
        let b_qoff = new_shared_buffer(qoff.as_ptr() as *const u8, qoff.len() * 4);
        let b_qlen = new_shared_buffer(qlen.as_ptr() as *const u8, qlen.len() * 4);
        let b_toff = new_shared_buffer(toff.as_ptr() as *const u8, toff.len() * 4);
        let b_tlen = new_shared_buffer(tlen.as_ptr() as *const u8, tlen.len() * 4);
        let b_h0 = new_shared_buffer(h0v.as_ptr() as *const u8, h0v.len() * 4);
        let b_w = new_shared_buffer(wv.as_ptr() as *const u8, wv.len() * 4);
        // DP scratch: n threads * stride cells * 4 bytes, for each of H and E. Left uninitialised;
        // the kernel zeroes its own slice, since only [0, qlen] of it can ever be touched.
        // `.max(4)` guards the degenerate all-empty-query batch (stride would be 1 and n >= 1, so
        // this is belt and braces rather than a live case).
        // Byte size of one scratch buffer: n threads * stride cells * 4 bytes per i32.
        let scratch_len = (n * stride as usize * 4) as u64;
        let b_ehh = dev.new_buffer(scratch_len.max(4), shared);
        let b_ehe = dev.new_buffer(scratch_len.max(4), shared);
        // 6 i32 per job, matching the `o[0..6]` writes at the end of the shader.
        let b_out = dev.new_buffer((n * 6 * 4) as u64, shared);

        // The one shared scoring/geometry record, uploaded to binding 10. Field order here must
        // stay identical to the MSL `Params`; nothing checks it.
        let params = Params {
            // `a` is the matrix maximum, which for a bwa matrix is the match score. Same value fed
            // to `clamp_band` as `max_sc`, so the band clamp and the kernel cannot disagree.
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

        // Packing is done: everything is in GPU-visible memory. `pack_ns` is held rather than
        // added to the counter straight away so the atomic write itself is outside both spans.
        let pack_ns = t_pack.elapsed().as_nanos() as u64;
        // Start of the GPU span: encode, commit and block until the dispatch has finished.
        let t_gpu = std::time::Instant::now();

        // ---- Step 3: bind the buffers, dispatch one thread per job, wait ------------------
        // A fresh command buffer and encoder per call: they are single-use, and making them per
        // call (rather than sharing) is what lets several aligner threads submit concurrently to
        // the one shared queue.
        let cmd = ctx.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for (buffer_index, buffer) in [
            &b_q, &b_t, &b_qoff, &b_qlen, &b_toff, &b_tlen, &b_h0, &b_w, &b_ehh, &b_ehe,
        ]
        .iter()
        .enumerate()
        {
            enc.set_buffer(buffer_index as u64, Some(buffer), 0);
        }
        // Indices 10 and 11 must match the `[[buffer(N)]]` attributes in the MSL signature; the
        // loop above fills 0..=9 in the same order the kernel declares them.
        enc.set_buffer(10, Some(&b_params), 0);
        enc.set_buffer(11, Some(&b_out), 0);
        // Threads per threadgroup we ask for. Threads here are wholly independent (no barriers,
        // no threadgroup memory), so this only trades occupancy against scheduling granularity,
        // never correctness: changing it cannot change a single output byte.
        //
        // UNVERIFIED: whether 64 is tuned or simply a safe default. Nothing in the tree records a
        // measurement behind it, and it matches the threadgroup size the `square_u32` probe uses.
        const PREFERRED_THREADS_PER_GROUP: u64 = 64;
        // Lowered to whatever this pipeline actually permits (a kernel using many registers can be
        // capped below 64 by the driver), so the dispatch below can never be rejected.
        let threads_per_group = pso
            .max_total_threads_per_threadgroup()
            .min(PREFERRED_THREADS_PER_GROUP);
        // `dispatch_threads` (not `dispatch_thread_groups`) lets Metal handle a grid that is not a
        // multiple of the threadgroup size; the kernel's `gid >= njobs` early-return covers the tail.
        enc.dispatch_threads(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
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

        // SAFETY: `wait_until_completed` above has returned, so the GPU is done writing; the buffer
        // is shared-storage and was allocated with exactly n*6 i32, so the length and alignment are
        // correct and it outlives the slice. Field order below must match the shader's `o[0..6]`.
        // ---- Step 4: unpack the 6-int-per-job result block ------------------------------
        // Ints the kernel writes per job: score, qle, tle, gtle, gscore, max_off. This is a
        // contract with the shader's `o[0..6]` writes and with `ExtendResult`'s field order;
        // changing any one of the three without the others silently scrambles every result.
        const INTS_PER_RESULT: usize = 6;
        // The whole result block as one flat i32 slice, job k occupying [k*6, k*6 + 6).
        let out_ints = unsafe {
            std::slice::from_raw_parts(b_out.contents() as *const i32, n * INTS_PER_RESULT)
        };
        (0..n)
            .map(|k| {
                // First int of job k's 6-int slot.
                let base = k * INTS_PER_RESULT;
                ExtendResult {
                    score: out_ints[base],
                    qle: out_ints[base + 1],
                    tle: out_ints[base + 2],
                    gtle: out_ints[base + 3],
                    gscore: out_ints[base + 4],
                    max_off: out_ints[base + 5],
                }
            })
            .collect()
    }
}
