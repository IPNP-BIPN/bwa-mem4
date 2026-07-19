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
//!
//! # Order to read this crate in
//!
//! 1. This file's [`MSL_SRC`] string, specifically the `sw_extend` kernel inside it. That is the
//!    actual algorithm, and it is a line-by-line transcription of `bwa_extend::ksw_extend2`.
//! 2. `backend.rs`: `MetalBackend::extend_batch`, which packs jobs into buffers, dispatches, and
//!    unpacks. Its `Params` struct must stay field-for-field in step with the shader's.
//! 3. `metal_ctx`, the one-time device/library/pipeline setup, and the `kernel_probe` test that
//!    stops a broken shader from silently degrading into a scalar fallback.
//!
//! # Glossary
//!
//! The shader keeps the C's short names so it can be diffed against `ksw.cpp` and against the Rust
//! port: `i`/`j` are the target row and query column, `h` is the best score ending at a cell,
//! `big_m` the best score ending there *on the diagonal* (an aligned pair), `e` the best score with
//! a **deletion** open (gap in the query), `f` with an **insertion** open (gap in the target),
//! `h1` the `h` of the cell to the left carried rightwards, `h0` the seed's own score, `mj` the
//! column of the row's best cell, `beg`/`end` the live column window, `zdrop` the give-up
//! threshold, and `qle`/`tle`/`gtle`/`gscore` the reported lengths and scores. The full table is in
//! the `bwa_extend::sw` module header. `mx` is the shader's spelling of the C's `max`, renamed only
//! because `max` is a builtin in MSL.
//!
//! Nothing inside the shader string may be renamed: the acceptance harness matches its identifiers,
//! and the `[[buffer(N)]]` indices are a contract with `backend.rs`.

#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
pub use backend::MetalBackend;

/// Print `BWA3_GPU_STATS` phase counters (no-op unless the env var is set, or off macOS).
///
/// # Parameters
///
/// None. The counters are process-global atomics in [`backend::stats`], accumulated by every
/// `extend_batch` call on every thread. Call this once, from the CLI, after alignment finishes.
pub fn dump_stats() {
    #[cfg(target_os = "macos")]
    backend::stats::dump();
}

#[cfg(target_os = "macos")]
mod metal_ctx {
    use metal::{ComputePipelineState, Device, Function, Library};
    use std::sync::OnceLock;

    /// Process-wide Metal state, built once. Device creation, MSL compilation and pipeline linking
    /// each cost milliseconds, which would dwarf a batch that runs in microseconds.
    pub struct MetalCtx {
        /// The system default GPU (the only one on Apple Silicon). Owns every buffer allocation and
        /// every pipeline built below; read by `extend_batch` to allocate the shared buffers.
        pub device: Device,
        /// The single command queue every dispatch is submitted to. Metal command queues are
        /// thread-safe, so all aligner threads share this one rather than each creating its own
        /// (queue creation is a per-object cost that would repeat per batch).
        pub queue: metal::CommandQueue,
        /// The compiled [`super::MSL_SRC`] library, i.e. every kernel in this crate. Kept so
        /// [`MetalCtx::pipeline`] can look up a function by name without recompiling the source.
        pub library: Library,
        /// The `sw_extend` pipeline state, built once (compiling a PSO per `extend_batch` call was
        /// pure launch overhead on the hot path).
        pub sw_extend: Option<ComputePipelineState>,
    }

    // SAFETY: Metal objects are internally reference-counted; each call builds its own command
    // buffer, so the shared context is only read concurrently. Markers let the `OnceLock` be `Sync`.
    unsafe impl Send for MetalCtx {}
    unsafe impl Sync for MetalCtx {}

    /// The one process-wide context, built on first [`MetalCtx::get`] and never dropped. The inner
    /// `Option` distinguishes "initialised, and there is no usable Metal device" (`Some(None)`)
    /// from "not yet initialised" (`OnceLock` empty), so a machine without a GPU pays the probe
    /// once rather than on every batch.
    static CTX: OnceLock<Option<MetalCtx>> = OnceLock::new();

    impl MetalCtx {
        /// `None` on a machine with no Metal device. Note the `?`/`.ok()?` chain makes *every*
        /// failure (no device, MSL compile error, missing function) collapse to `None`, and callers
        /// then fall back to scalar. That is the right production behaviour but it is also a silent
        /// one, which is why `kernel_probe::sw_extend_pso_builds` asserts the pipeline really built.
        ///
        /// # Parameters
        ///
        /// None. Everything comes from the system (the default device) and from the compile-time
        /// [`super::MSL_SRC`] string.
        ///
        /// # Returns
        ///
        /// `Some` borrow of the process-wide context, valid for the rest of the program, or `None`
        /// if the device, the MSL compile or the `sw_extend` lookup failed. `Some` does *not*
        /// promise `sw_extend` built: check [`MetalCtx::sw_extend`] for that.
        pub fn get() -> Option<&'static MetalCtx> {
            CTX.get_or_init(|| {
                // The integrated GPU on Apple Silicon; `None` on a headless/virtualised host.
                let device = Device::system_default()?;
                let queue = device.new_command_queue();
                // Runtime compile of the whole shader string, using default `CompileOptions`
                // (default MSL language version). This is where an MSL syntax error surfaces, as
                // an `Err` that the `.ok()?` below turns into a scalar fallback.
                let library = device
                    .new_library_with_source(super::MSL_SRC, &metal::CompileOptions::new())
                    .ok()?;
                // Look the kernel up by its MSL name and link it into a pipeline state object once.
                // `None` here (missing function, or linking failure) is not fatal: it leaves
                // `sw_extend: None` and every batch falls back to the scalar backend.
                let sw_extend = library
                    .get_function("sw_extend", None)
                    .ok()
                    .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok());
                Some(MetalCtx {
                    device,
                    queue,
                    library,
                    sw_extend,
                })
            })
            .as_ref()
        }

        /// The cached `sw_extend` pipeline; any other kernel is built on demand (test-only path).
        ///
        /// # Parameters
        ///
        /// * `name`: the MSL kernel's function name exactly as spelled in [`super::MSL_SRC`], e.g.
        ///   `"sw_extend"` or `"square_u32"`. Supplied by the caller as a literal; an unknown name
        ///   is not an error, it just yields `None`.
        ///
        /// # Returns
        ///
        /// A clone of the pipeline state object (Metal ref-counts it, so cloning is cheap and the
        /// clone shares the compiled machine code), or `None` if the function does not exist or the
        /// pipeline would not link.
        pub fn pipeline(&self, name: &str) -> Option<ComputePipelineState> {
            if name == "sw_extend" {
                return self.sw_extend.clone();
            }
            let f: Function = self.library.get_function(name, None).ok()?;
            self.device
                .new_compute_pipeline_state_with_function(&f)
                .ok()
        }
    }
}

/// Metal Shading Language source: the feasibility probe plus the banded SW extension kernel, a
/// faithful port of `bwa_extend::ksw_extend2` (M-based gap opens, adaptive band, z-drop, gscore).
///
/// The shader is a Rust string literal compiled at runtime by `new_library_with_source`, so nothing
/// here is checked by `cargo build`: a syntax error in this string only shows up as
/// `sw_extend` failing to build a pipeline, which the `kernel_probe` test below exists to catch.
///
/// Keep this kernel a line-by-line transcription of `ksw_extend2`. It is not a place to be clever:
/// every restructuring risk lands directly on SAM byte-identity, and there is no compiler to tell
/// you when it lands. See the gap-open comment inside `sw_extend` for what happened the one time
/// this file drifted from the CPU paths.
#[cfg(target_os = "macos")]
const MSL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

/* Feasibility probe only: proves a device exists, a library compiles and a dispatch round-trips.
   Nothing in the aligner calls it. */
/* Parameters:
   *   data [[buffer(0)]] - in/out array of `n` uint, shared storage, written in place. `n` is the
   *                        grid size the host passed to dispatch_threads; the host is the Rust
   *                        `square_u32` in lib.rs, which sizes the buffer from the caller's slice.
   *                        No bounds check is needed: dispatch_threads with a 1-D grid of exactly
   *                        `n` launches exactly `n` threads, so gid < n always.
   *   gid                - this thread's linear index in the grid, supplied by the hardware.
   *                        Thread `gid` owns element `gid` and no other, so there is no race. */
kernel void square_u32(device uint* data [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
    /* v: this thread's own element, read once before it is overwritten below. */
    uint v = data[gid];
    data[gid] = v * v;
}

/* Scoring shared by every job in a dispatch. Layout must match `Params` in backend.rs field for
   field (all `int`, no padding, `#[repr(C)]` on the Rust side).

   The full 5x5 matrix is NOT uploaded: the CPU side checks with `is_uniform_dna` that the matrix
   has bwa's uniform shape and then sends only its three distinct values, so the shader scores a
   cell with two comparisons instead of a dependent memory load. A non-uniform matrix makes the
   whole batch fall back to the scalar backend, so this is a narrowing of the input domain, not an
   approximation. `a` = match, `mm` = mismatch (negative), `npen` = the penalty for either base
   being ambiguous (code >= 4). */
/* Every field is written once by the host (backend.rs, `let params = Params {...}`) and only
   * read by the kernel; the whole struct arrives as one `constant` buffer at index 10.
   *
   * Fields:
   *   a        - match score, > 0. bwa's -A. Also the matrix maximum, and the same value the host
   *              fed to clamp_band as max_sc, so band and kernel cannot disagree.
   *   mm       - mismatch score, negative (bwa's -B negated). Applied when both bases are concrete
   *              (code < 4) and differ.
   *   npen     - score when either base is ambiguous (code >= 4, i.e. N). -1 for a bwa matrix.
   *              Overrides `a`/`mm`, so an N-vs-N pair scores npen, not a match.
   *   o_del    - deletion gap-OPEN penalty as a positive number, subtracted. bwa's -O, first value.
   *   e_del    - deletion gap-EXTEND penalty per base, positive, subtracted. bwa's -E. Must be
   *              >= 1: the host's band clamp divides by it.
   *   o_ins    - insertion gap-open penalty, positive. Second -O value.
   *   e_ins    - insertion gap-extend penalty per base, positive, >= 1. Second -E value.
   *   end_bonus- uploaded for layout/parity with the CPU signature but never read below:
   *              ksw_extend2 only uses it inside the band clamp, which the CPU already did (see
   *              `wbuf`). Removing the field would also mean removing it from the Rust mirror.
   *   zdrop    - give-up threshold: abandon a job's DP once its row best has fallen this far below
   *              the running best, after discounting the implied gap. <= 0 disables the test.
   *   njobs    - number of alignments in this dispatch, i.e. jobs.len() on the host. Threads with
   *              gid >= njobs retire immediately (dispatch rounds the grid up to a whole
   *              threadgroup).
   *   stride   - width in ints of one thread's DP slice in `ehh`/`ehe`: max query length over the
   *              whole batch, + 1 for the sentinel column. Uniform across threads so addressing is
   *              a single multiply. Every job satisfies qlen < stride. */
struct Params {
    int a; int mm; int npen;          // uniform matrix: match / mismatch / ambiguous
    int o_del; int e_del; int o_ins; int e_ins;
    /* end_bonus is uploaded for layout/parity with the CPU signature but is never read below:
       ksw_extend2 only uses it inside the band clamp, which the CPU already did (see `wbuf`). */
    int end_bonus; int zdrop;
    int njobs; int stride;            // stride = max_qlen + 1 (per-thread DP slice)
};


/* One thread = one seed extension (inter-sequence parallelism): thread `gid` runs the entire DP
   for job `gid` and touches no other thread's state, so there are no barriers and no threadgroup
   memory. Threads diverge freely (different tlen, different early exits); that costs SIMD-group
   occupancy but never correctness.

   DP arrays live in device memory as a `stride`-sized slice per thread, because qlen is only known
   at runtime and can exceed anything that would fit in registers or threadgroup memory.

   Byte-identical to `bwa_extend::ksw_extend2` (`reference/bwa-mem2/src/ksw.cpp:432-533`): same
   recurrence, same adaptive band shrink, same z-drop, same tie-breaks. Read that function's Rust
   port alongside this; the comments there explain the *why* of each step and are not repeated in
   full here. Only the GPU-specific deviations are annotated below.

   Buffers 0-7 are per-job inputs (concatenated sequences plus per-job offsets, lengths, h0 and the
   already-clamped band), 8-9 are scratch, 10 is the shared scoring, 11 is the output. */
/* Parameters (all buffers are StorageModeShared, allocated and filled by `extend_batch` in
   * backend.rs; "per job" means the array is indexed by gid and has exactly njobs elements):
   *
   *   qbuf   [[buffer(0)]] - every job's QUERY (the read) concatenated back to back, one byte per
   *                          base holding a 2-bit code: 0=A 1=C 2=G 3=T, 4=N. Not ASCII. Host
   *                          writes, kernel reads. Length = sum of all query lengths, or 1 filler
   *                          byte if that sum is 0, which no thread reads since its qlen is 0.
   *   tbuf   [[buffer(1)]] - same, for every job's TARGET (the reference stretch).
   *   qoff   [[buffer(2)]] - per job: byte index where this job's query starts in qbuf.
   *   qlen_a [[buffer(3)]] - per job: query length in bases, >= 0. Bounds the column loop.
   *   toff   [[buffer(4)]] - per job: byte index where this job's target starts in tbuf.
   *   tlen_a [[buffer(5)]] - per job: target length in bases, >= 0. Bounds the row loop.
   *   h0buf  [[buffer(6)]] - per job: h0, the score the seed already earned, i.e. the DP's
   *                          starting value at the origin. Must be > 0 (0 means unreachable).
   *   wbuf   [[buffer(7)]] - per job: the half band width in query columns, ALREADY clamped on the
   *                          CPU in f64 by `clamp_band`. The kernel must not re-derive it.
   *   ehh    [[buffer(8)]] - DP scratch for H, njobs * stride ints. Thread gid owns the slice
   *                          [gid*stride, gid*stride + stride). Uninitialised on entry (a reused
   *                          allocation, so it holds a previous dispatch's values); each thread
   *                          zeroes its own [0, qlen] before use. Never read by the host.
   *   ehe    [[buffer(9)]] - DP scratch for E, same shape and ownership rule as ehh.
   *   p      [[buffer(10)]] - the shared scoring/geometry struct above, one instance for the whole
   *                          dispatch, in `constant` address space (broadcast, cached).
   *   out    [[buffer(11)]] - results: 6 ints per job at offset gid*6, in ExtendResult field order
   *                          (score, qle, tle, gtle, gscore, max_off). Kernel writes, host reads
   *                          after wait_until_completed.
   *   gid                  - this thread's linear grid index = the job index it aligns. Supplied
   *                          by the hardware; may exceed njobs on the dispatch tail. */
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
    /* ---------------------------------------------------------------------------------------
       Setup 1/3: unpack this thread's job out of the flat buffers
       --------------------------------------------------------------------------------------- */
    /* dispatch_threads rounds the grid up to whole threadgroups, so the tail threads must retire
       before touching any buffer. */
    if ((int)gid >= p.njobs) return;

    /* This thread's job geometry: qlen/tlen are the DP's column and row counts (the query and
     * target lengths in bases), h0 is the seed score the DP starts from at the origin. */
    const int qlen = qlen_a[gid];
    const int tlen = tlen_a[gid];
    const int h0   = h0buf[gid];
    /* Already clamped on the CPU by `clamp_band`, which does the f64 division that ksw_extend2 does
       (ksw.cpp:456-461). Doing it here would mean float math on the GPU, whose rounding we do not
       control and cannot afford to differ by one. */
    int w          = wbuf[gid];
    /* q[j] and t[i] are this job's own bases (2-bit codes), reached by adding the job's offset to
     * the concatenated buffer. Valid for j in [0, qlen) and i in [0, tlen). */
    device const uchar* q = qbuf + qoff[gid];
    device const uchar* t = tbuf + toff[gid];
    /* This thread's private DP rows, carved out of the shared scratch. eh_h[j] holds H of the
     * PREVIOUS row at column j-1 (the diagonal predecessor the inner loop is about to consume);
     * eh_e[j] holds E at column j, carried down from the previous row. Both are valid for
     * j in [0, qlen]; index qlen is the sentinel column. No other thread touches this slice. */
    device int* eh_h = ehh + (uint)gid * (uint)p.stride;
    device int* eh_e = ehe + (uint)gid * (uint)p.stride;

    /* Total cost of opening a one-base gap: the open penalty plus the first base's extend penalty.
     * Precomputed because the inner loop subtracts it twice per cell. */
    const int oe_del = p.o_del + p.e_del;
    const int oe_ins = p.o_ins + p.e_ins;

    /* ---------------------------------------------------------------------------------------
       Setup 2/3: clear the DP arrays and lay down row -1 (the leading-insertion run)
       --------------------------------------------------------------------------------------- */
    /* The C gets its zeroed score arrays from calloc (ksw.cpp:441); a reused device buffer holds the
       previous dispatch's garbage, so zero it explicitly. Only [0, qlen] is cleared, because that is
       all this job's rows can reach even though the slice is `stride` wide. Zero is the "no
       alignment reaches this cell" sentinel, not a score. */
    for (int j = 0; j <= qlen; j++) { eh_h[j] = 0; eh_e[j] = 0; }
    /* Row -1: score h0 at the origin, then a run of leading insertions (ksw.cpp:449-451). The
       `qlen >= 1` guard is the one addition over the C, which indexes eh[1] unconditionally because
       its caller never passes qlen == 0. */
    eh_h[0] = h0;
    if (qlen >= 1) eh_h[1] = h0 > oe_ins ? h0 - oe_ins : 0;
    /* Columns 2.. of row -1: each is one more inserted base, so subtract e_ins again. The test is
     * `> e_ins` and not `>= e_ins`, stopping one step before the value would hit 0, because 0 is
     * the unreachable sentinel rather than a score of zero. Columns past the stop stay 0. */
    { int j = 2; while (j <= qlen && eh_h[j-1] > p.e_ins) { eh_h[j] = eh_h[j-1] - p.e_ins; j++; } }

    /* ---------------------------------------------------------------------------------------
       Setup 3/3: result trackers and the adaptive band window
       --------------------------------------------------------------------------------------- */
    /* `mx` is the C's `max`, renamed only because `max` is a builtin in MSL. Starts at h0, so the
       seed is the baseline; the -1 sentinels are what make qle/tle/gtle come out 0 and gscore -1
       when nothing beats it. */
    /* mx     = best H seen anywhere so far (the reported `score`), starts at h0.
     * max_i  = target row of that best cell, 0-based, -1 = nothing beat h0. Reported as tle - 1.
     * max_j  = query column of that best cell, 0-based, -1 = none. Reported as qle - 1.
     * max_ie = target row of the best alignment that consumed the WHOLE query, -1 = no row ever
     *          reached the query end. Reported as gtle - 1.
     * gscore = that whole-query alignment's score, -1 = never happened (and -1 is written out as
     *          is, not incremented).
     * max_off= largest |column - row| distance any row's best cell strayed from the main diagonal;
     *          the caller uses it to decide whether to retry with a wider band. */
    int mx = h0, max_i = -1, max_j = -1, max_ie = -1, gscore = -1, max_off = 0;
    /* Live column window for the row about to be computed, half-open [beg, end). Invariant at the
     * top of each row: columns outside it are known unreachable (H and E both 0), so they need not
     * be evaluated. `beg` only ever grows and `end` only ever shrinks. */
    int beg = 0, end = qlen;   /* adaptive live window, narrows across rows */

    /* =========================================================================================
       Main DP: one iteration per target base (i = row), inner loop over query columns (j).
       Reminder: h = best score ending at this cell, big_m = ending on the diagonal (aligned pair),
       e = with a deletion (gap in the query) open, f = with an insertion (gap in the target) open.
       ========================================================================================= */
    for (int i = 0; i < tlen; i++) {
        /* --- row prologue: intersect the band, seed the left-edge cell --- */
        /* f       = F at the cell about to be computed, i.e. the best score reaching it with an
         *           insertion (gap in the target) already open. 0 at the row's left edge means no
         *           insertion is open there. Flows rightwards only, so it lives in a register.
         * row_max = best H found so far in THIS row; 0 until a reachable cell appears.
         * mj      = the column achieving row_max, -1 while row_max is still 0. */
        int f = 0, row_max = 0, mj = -1;
        /* This row's target base code (0..4). Selects which row of the substitution matrix the
         * inner loop scores against; here it just feeds the inlined `sc` expression. */
        int tc = t[i];
        if (beg < i - w) beg = i - w;
        if (end > i + w + 1) end = i + w + 1;
        if (end > qlen) end = qlen;
        /* H(i, beg-1): reachable from the seed origin only by a deletion of length i+1, and only
           while the window still touches column 0 (ksw.cpp:474-477). */
        int h1 = 0;
        /* v: the score of that pure-deletion path before flooring; <= 0 means the deletion has
         * already cost more than the seed earned, so the cell is unreachable (0). */
        if (beg == 0) { int v = h0 - (p.o_del + p.e_del * (i + 1)); h1 = v > 0 ? v : 0; }
        /* --- inner loop: one cell per query column ---
         * Invariant at the top of each iteration: h1 holds H(i, j-1), the cell immediately to the
         * left, already final. eh_h[j] still holds H(i-1, j-1) from the previous row and eh_e[j]
         * still holds E(i, j) carried down; both are read before being overwritten. `f` holds
         * F(i, j). After the body, h1 holds H(i, j) and eh_h[j] holds H(i, j-1), which is exactly
         * what row i+1 will want as its diagonal predecessor at column j. */
        int j = beg;
        while (j < end) {
            int big_m = eh_h[j];   /* H(i-1, j-1), the diagonal predecessor */
            int e = eh_e[j];       /* E(i, j), carried from the previous row */
            eh_h[j] = h1;          /* publish H(i, j-1) for the next row to read as its diagonal */
            /* qc: this column's query base code (0..4). */
            int qc = q[j];
            /* Query profile inlined as arithmetic. Equivalent to mat[tc*5 + qc] for the uniform
               matrix this kernel is gated on: the N row/column wins over the diagonal, exactly as
               bwa_fill_scmat lays it out (row 4 and column 4 are all -1 whatever a and b are). */
            int sc = (tc >= 4 || qc >= 4) ? p.npen : (tc == qc ? p.a : p.mm);
            /* M(i,j), but a zero diagonal stays zero: 0 means unreachable and must not be extended
               by a positive substitution score. This is what makes the surface local (ksw.cpp:487). */
            big_m = big_m != 0 ? big_m + sc : 0;
            /* h: H(i, j), this cell's final best score. Assigned into h1 straight after, so the
             * next iteration sees it as its left neighbour. */
            int h = big_m > e ? big_m : e;
            h = h > f ? h : f;     /* H(i,j) = max(M, E, F); >= 0 since E, F are floored below */
            h1 = h;
            /* Ties go to the later column (`>` on the incumbent, per ksw.cpp:491-492). */
            mj = row_max > h ? mj : j;
            row_max = row_max > h ? row_max : h;
            /* THE BUG THAT LIVED HERE. Gaps open from M (the diagonal score), NOT from
               h = max(M, E, F). Same rule as ksw_extend2 (ksw.cpp:493 `t = M - oe_del` and
               ksw.cpp:498 `t = M - oe_ins`) and as bwa-mem2's vectorized bandedSWA, whose
               MAIN_CODE16 subtracts oe_del256/oe_ins256 from `m11`, not `h11`
               (bandedSWA.cpp:327-342). Opening from H would allow a CIGAR like 100M3I3D20M, which
               ksw.cpp:487 says in so many words it is separating H and M to prevent.

               This shader kept `h - oe_*` after the CPU paths were corrected, and nothing caught it:
               on ordinary extensions h == M at every cell on the optimal path, so the two agree.
               The difference only surfaces under asymmetric or non-default gap penalties and inside
               repeats, and the acceptance sweep in bwa-extend pinned scoring to bwa's defaults at
               the time, making it structurally incapable of generating such a case. Widening that
               sweep over (a, b, o_del, e_del, o_ins, e_ins) exposed this immediately. Do not narrow
               it again, and do not "simplify" these two lines back to `h`.

               E(i+1,j) = max(0, E(i,j) - e_del, M(i,j) - o_del - e_del). */
            /* tt: scratch for the "open a fresh gap here" candidate, floored at 0 (a negative
             * score means unreachable, not a real alternative). Reused for F below. */
            int tt = big_m - oe_del; if (tt < 0) tt = 0;
            e -= p.e_del; e = e > tt ? e : tt;
            eh_e[j] = e;
            /* F(i,j+1) = max(0, F(i,j) - e_ins, M(i,j) - o_ins - e_ins). F stays in a register: it
               only ever flows rightward along a row, hence no eh_f array. */
            tt = big_m - oe_ins; if (tt < 0) tt = 0;
            f -= p.e_ins; f = f > tt ? f : tt;
            j++;
        }
        /* --- row epilogue: sentinel column, to-end score, termination tests, band tightening --- */
        /* Sentinel column so row i+1 sees a well-defined diagonal at `end` (ksw.cpp:503). */
        eh_h[end] = h1;
        eh_e[end] = 0;
        /* The row ran to the query end, so h1 is this row's to-end score. `<=` keeps the LATER
           target row on a tie, matching ksw.cpp:504-507; that choice sets gtle. */
        if (j == qlen && gscore <= h1) { max_ie = i; gscore = h1; }
        if (row_max == 0) break;   /* whole row unreachable: nothing below it can be reachable */
        if (row_max > mx) {
            mx = row_max; max_i = i; max_j = mj;
            int off = mj - i; if (off < 0) off = -off;   /* abs(): distance off the main diagonal */
            if (off > max_off) max_off = off;
        } else if (p.zdrop > 0) {
            /* Z-drop, restructured into a single `drop` variable purely for readability; the two
               branches are the C's two branches (ksw.cpp:512-517) unchanged. The drop is discounted
               by the gap that the drift from (max_i,max_j) to (i,mj) implies, charged at the extend
               rate of whichever side is longer, so a long legitimate indel does not trip it. */
            /* drop: how far this row's best has fallen below the running best `mx`, already
             * discounted by the implied gap. Compared against p.zdrop, in score units. */
            int drop;
            if (i - max_i > mj - max_j) drop = mx - row_max - ((i - max_i) - (mj - max_j)) * p.e_del;
            else                        drop = mx - row_max - ((mj - max_j) - (i - max_i)) * p.e_ins;
            if (drop > p.zdrop) break;
        }
        /* Shrink the live window to the columns still reachable (H and E both non-zero), the
           adaptive band of ksw.cpp:520-523. `je + 2` because je is the last live column, je+1 is
           what it can reach next row, and `end` is exclusive. */
        /* jb: scans up from the old `beg` to the first still-reachable column, becoming the next
         * row's `beg`. If the whole window is dead it ends at `end`. */
        int jb = beg;
        while (jb < end && eh_h[jb] == 0 && eh_e[jb] == 0) jb++;
        beg = jb;
        /* je: scans down from `end` to the LAST still-reachable column. Starts at `end` rather
         * than end-1 because the sentinel written just above is a genuine cell of this row. */
        int je = end;
        while (je >= beg && eh_h[je] == 0 && eh_e[je] == 0) je--;
        end = (je + 2 < qlen) ? je + 2 : qlen;
    }

    /* =========================================================================================
       Write back this thread's result.
       ========================================================================================= */
    /* 6 ints per job, in ExtendResult field order; backend.rs unpacks them positionally, so this
       layout and that struct must be changed together. The `+ 1`s convert a 0-based best-cell index
       into a consumed length (and yield 0 from the -1 sentinels). */
    /* o: this thread's own 6-int output slot. Disjoint from every other thread's, so the writes
     * below need no synchronisation. */
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
///
/// Nothing in the aligner calls this; it exists to prove end to end that a device, a compiled
/// library and a dispatch round trip all work on this machine.
///
/// # Parameters
///
/// * `data`: squared in place. Any length including 0 is accepted (a 0-length dispatch is a no-op).
///   Values wrap on overflow, as `uint` multiplication does on the GPU.
///
/// # Returns
///
/// `true` if the kernel ran and `data` now holds the squares; `false` if there is no Metal device
/// or the `square_u32` pipeline would not build, in which case `data` is left untouched.
#[cfg(target_os = "macos")]
pub fn square_u32(data: &mut [u32]) -> bool {
    use metal::{MTLResourceOptions, MTLSize};
    let Some(ctx) = metal_ctx::MetalCtx::get() else {
        return false;
    };
    let Some(pso) = ctx.pipeline("square_u32") else {
        return false;
    };
    // Element count, which is also the grid size: one thread per element.
    let n = data.len();
    // Shared storage, so this is one memcpy in and zero copies out on unified memory.
    let buf = ctx.device.new_buffer_with_data(
        data.as_ptr() as *const _,
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&buf), 0);
    // Grid of exactly n threads in threadgroups of 64. 64 is a probe-only choice with no
    // correctness role: `dispatch_threads` handles a grid that is not a multiple of it.
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    // SAFETY: the GPU has finished (wait_until_completed returned), the buffer is shared storage
    // holding exactly n u32, and it outlives this borrow.
    let out = unsafe { std::slice::from_raw_parts(buf.contents() as *const u32, n) };
    data.copy_from_slice(out);
    true
}

/// True when a Metal device with the `sw_extend` pipeline is available (so `--gpu` can route here).
///
/// # Parameters
///
/// None. Reads the process-wide context, building it on the first call.
///
/// # Returns
///
/// `true` only if both the device exists and the kernel linked. `false` means every `extend_batch`
/// would silently fall back to the scalar backend, so the caller can warn instead of running slow.
#[cfg(target_os = "macos")]
pub fn metal_available() -> bool {
    metal_ctx::MetalCtx::get()
        .map(|c| c.sw_extend.is_some())
        .unwrap_or(false)
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

    // The shared gate uses uniform random sequences of at most 80 bases, which is neither the
    // length nor the *similarity* profile of a real seed extension. This one builds targets by
    // copying the query with roughly 5% substitutions and occasional skips, at 40-260 bases and in
    // batches up to 1000: high-identity alignments that run far down the diagonal, which is what
    // actually exercises the adaptive band shrink, the z-drop and the to-end/gscore path. Keep both
    // tests; neither subsumes the other.
    #[test]
    fn metal_matches_scalar_read_sized() {
        // bwa's default scoring: match +1 (-A), mismatch -4 (-B). `mat` is the 5x5 row-major
        // substitution matrix over A,C,G,T,N built the way bwa_fill_scmat does, so mat[t*5 + q]
        // scores target base t against query base q; `k` is just the fill cursor.
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
        // Fixed-seed LCG (the MMIX constants), so the 80 rounds below are the same on every run
        // and on every machine: a divergence is always reproducible. `next` returns the top 31
        // bits, which are the well-mixed ones in an LCG.
        let mut state = 0x6D3A_0000_0000_0001u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let (o_del, e_del, o_ins, e_ins) = (6, 1, 6, 1);
        for round in 0..80u32 {
            // Batch sizes spanning one job up to a realistic dispatch; 1000 is not a multiple of
            // the 64-thread threadgroup, so it also exercises the kernel's `gid >= njobs` tail.
            let batch = *[1usize, 8, 64, 256, 1000]
                .get((next() % 5) as usize)
                .unwrap();
            // Half band width 50..169 columns: wide enough not to clip a high-identity diagonal,
            // narrow enough that the adaptive shrink still runs. zdrop 0..199, with 0 meaning the
            // z-drop test is disabled, so both the early-exit and run-to-completion paths appear.
            // end_bonus only reaches the band clamp, never a score.
            let w = 50 + (next() % 120) as i32;
            let zdrop = (next() % 200) as i32;
            let end_bonus = (next() % 12) as i32;
            let mut queries: Vec<Vec<u8>> = Vec::new();
            let mut targets: Vec<Vec<u8>> = Vec::new();
            let mut h0s: Vec<i32> = Vec::new();
            for _ in 0..batch {
                // Read-sized query, 40..259 bases of uniform random A/C/G/T (codes 0..3, so N
                // never appears; see the coverage note on `fill_scmat_vec`).
                let qlen = 40 + (next() % 220) as usize;
                let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
                // Target is the query plus 0..39 extra bases, so there is always something left
                // to extend into. `qi` is the read cursor into `q` while `t` is built: usually it
                // copies the next query base (95% of the time), otherwise it substitutes a random
                // base and half the time also skips a query base, which is how indels appear.
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
                // Seed score 20..49. Must be > 0: 0 is the DP's "unreachable" sentinel.
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

#[cfg(all(test, target_os = "macos"))]
mod kernel_probe {
    /// Guards the silent-fallback trap: if the MSL fails to compile, `pipeline()` returns `None`,
    /// `extend_batch` quietly falls back to the scalar backend, and every byte-identity gate in this
    /// crate passes *without the GPU ever running*. Assert the kernel actually built, so a broken
    /// shader fails loudly here instead of masquerading as a green suite.
    #[test]
    fn sw_extend_pso_builds() {
        let ctx = crate::metal_ctx::MetalCtx::get().expect("no Metal device");
        assert!(
            ctx.sw_extend.is_some(),
            "sw_extend PSO failed to build -- MSL compile error; the gates would be vacuous"
        );
    }
}
