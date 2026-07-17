# CPU optimization roadmap (bwa-mem3-rs)

> ## ⚠️ 2026-07-16: this document's central premise is WRONG. Read this box first.
>
> Everything below is framed on "**SW extension is ~85% of cycles**" (and the later profile's
> ~40-43%). **Both numbers were measured on `work/region.fa`**, the 2 Mbp test index, where the BWT
> is cache-resident and seeding is therefore nearly free. That inflates extension's share by
> construction.
>
> **Measured on the real genome index** (`work/genome.fa`, 500k reads, `-t1`, quiet host, interleaved,
> instrumented via `BWA3_GPU_STATS`):
>
> | | wall |
> |---|---|
> | CPU `-t1` | 19.73s |
> | GPU `-t1` (extension offloaded) | 15.91s |
> | of which `extend_batch` (pack 0.182 + `gpu_exec` 0.332) | **0.514s** |
>
> Since the non-extension work is identical in both arms, `E_cpu = 19.73 - 15.91 + 0.514 = 4.33s`.
>
> **SW extension is ~22% of `-t1` wall (~24% of compute) at genome scale, not 85% and not 40-43%.**
> Seeding + `get_sa` + chaining are the other ~78%. This matches the profile recorded in
> `lisa-learned-index-dead-end` (seeding ~69%, extension ~31%) and not this document's. It is also
> the *expected* result: `5af30cd`, `06b0225` and `7be53c2` all removed DP work, and the ungapped
> fast path now answers most diagonals in closed form.
>
> **Consequences.** Every kernel lever below is capped by Amdahl at `1/(1-0.22) = 1.28x`, not the
> ~7x the "85%" framing implies. Levers 1-5 are all extension-side and are therefore worth ~4x less
> than this document claims. **Do not size any extension work off this document. Re-measure on
> `work/genome.fa`.**

Synthesis of a deep-research pass (2026-07) into CPU-optimization state of the art for a
byte-identical bwa-mem2 reimplementation. Grounded in our profile (**see the correction above: this
profile is region-index-inflated and wrong**): Smith-Waterman extension is ~85% of cycles,
seeding is already lockstep+prefetch batched, NEON 128-bit/16-lane is the width
ceiling on Apple Silicon (SME/SVE rejected: max-plus DP is not multiply-accumulate), and the
learned index (LISA/BWA-MEME) was measured ~5x slower here (see `lisa-learned-index-dead-end`).

## The reframing

Per-cell *cost* is already at the ceiling: inter-sequence 16-lane u8 / 8-lane i16 (SWIPE-style) is
the optimal SIMD layout for batched short reads. Striped/Farrar/lazy-F/prefix-scan solve an
intra-sequence dependency we don't have; difference-recurrence (KSW2) is intra-sequence; Block
Aligner / adaptive banding are heuristic (break byte-identity). So the wins are **not "faster
cells"** — they are: fill the idle pipes, compute fewer cells, skip full DP.

## Prioritized levers

| # | Lever | Mechanism | Gain | Byte-identity | Effort |
|---|---|---|---|---|---|
| 1 | ~~**Fill the 4 NEON pipes**~~ **(TESTED — no gain on NEON, register-bound)** | The `for j` inner loop carries `f_v`/`h1_v` — a serial chain (`f_v→h_v→f_new`, ~3-4 dependent ops, each 2-3cy latency). M-series retires 4 vec ops/cy across 4 pipes → we run at ~25% of peak. Interleave 3-4 **independent alignment groups'** row-sweeps so the scheduler has ready uops. | **Predicted ~3-4x; measured ~0%** | Identical (proven, gated) | Medium-high (restructure `batched.rs` macro) |

### Lever #1 result (2026-07): measured, negative — abandoned on NEON

Built and validated a 2-group ILP-interleaved u8 kernel (`define_sw_kernel_ilp!`, byte-identical to
scalar across 400 randomized rounds straddling the group boundary). **It gave no speedup.**

- **Latency headroom is real.** Breaking the `f_v` loop-carried dependency (same op count, disjoint
  probe) sped the single-group kernel up **~1.55x** (batch 16: 77.9 → 50.7 ms). So the recurrence *is*
  latency-bound with ~3 of 4 NEON pipes idle, exactly as predicted.
- **But it can't be captured on NEON.** Interleaving a second independent group needs a second copy of
  the live vector state. Register accounting: **2 groups × 9 live vectors (f/h1/rowmax/mj + beg/end/
  active/t/t_is_n) + 9 loop-invariant constants + `j_v` = 28 live before ~15 temporaries**, vs only
  **32 NEON registers**. Disassembly of the x2 kernel: **102 vector spills** in the function. The spill
  traffic exactly cancels the ILP.
- **Measured (uniform lengths, both groups equal work, batch 32/64):** single-group 92.4 / 93.8 ms vs
  x2 interleave 91.5 / 91.6 ms → **parity (<1%)**. With random lengths, group-length mismatch makes x2
  slightly *worse*.

Conclusion: the SW kernel is at the practical byte-identical ceiling on Apple Silicon — 16-lane NEON
saturates the register file, and the only ILP source (a 2nd independent alignment group) doesn't fit.
This confirms the systems-agent caveat and the `apple-silicon-simd-width-ceiling` finding. The x2 code
was reverted (not shipped); a native-NEON byte-identity gate (`neon_verify`) was kept.

**x86 caveat:** AVX2 has only **16** vector registers (worse — would spill harder), but **AVX-512 has
32 zmm**. A 2-group interleave *might* pay off on AVX-512 (more registers + the recurrence is the same
latency-bound shape). Untestable on Apple; revisit only when the Intel/AVX-512 path gets CI. The
`define_sw_kernel_ilp!` approach (in git history on `feat/sw-kernel-ilp`) is the starting point.

### Idea "port getScores16's exact band" — refuted by reading the reference (2026-07)

Studied bwa-mem2's actual SIMD kernel (`reference/bwa-mem2/src/bandedSWA.cpp`,
`smithWaterman256_8` / `MAIN_CODE8`). It is **structurally identical to ours**, so porting it changes
nothing on NEON:
- Same column-carried serial chain: bwa-mem2 carries `e11` (E-gap) across columns in a register,
  reset to 0 per row (line 805) — the exact analog of our `f_v` (E↔F swapped by convention). Same
  `e→h→e` latency chain.
- Same memory-streaming: H/E/F live in arrays (`H_h`, `F`), loaded/stored every cell — not a
  register-resident band. So there is **no "fixed band frees registers"** to exploit; if anything
  bwa-mem2 keeps *more* live vectors (`head/tail/myband/qlen/mlen/i256/i1_256` + `MAIN_CODE` temps).
- Its only extra is `#pragma unroll(4)` on the column loop, which unrolls the *same* serial chain — a
  true dependency it cannot break (consistent with the f_v-break proof that the chain is the limiter).

bwa-mem2 is not faster *per cell* than us on the same ISA; its absolute speed is **AVX-512 width
(64 u8 lanes vs our 16)** — the Apple width ceiling again. **Do not port getScores16 expecting a NEON
win.** The kernel is at the byte-identical ceiling on Apple Silicon.

### ~~PGO reassessed — low value here (~2% overall)~~ — WRONG, refuted by measurement (2026-07-16)

> **This section is a casualty of the profile error in the box at the top, and it argued against a
> lever that was already built and measured.** Its reasoning: "PGO/BOLT only helps branchy code
> (driver, dispatch, seeding ≈ 15% of runtime); the 85% is hand-written branchless NEON PGO can't
> improve, so the realistic overall gain is ~2%."
>
> Both premises are false. Seeding is **~78%** of genome-scale runtime, not 15%, and extension is
> ~22%, not 85% — so PGO applies to the *large* share, not the small one. And `feat/pgo` (`26e58e1`)
> **measured SE +3.1% / PE +4.5%**, parity clean, with a complete 3-stage `cargo-pgo` harness in
> `scripts/pgo.sh`. The "version friction" caveat is also stale: `cargo-pgo` is installed and the
> script pins `PATH=/opt/homebrew/opt/llvm/bin` itself. (`llvm-bolt` genuinely is absent, so BOLT
> remains untested.)
>
> **Lesson: this file talked a measured, merge-ready win out of existing for months on the strength
> of a bad profile.** Prefer measuring to reasoning about percentages.

### The `-t8` ceiling is the memory system, and it eats per-thread levers whole (2026-07-16)

**This is the most important thing on this page.** Two independent levers were measured this session,
and both have the *same shape*: a large win at `-t1` that is worth **zero** at `-t8`.

| lever | `-t1` | `-t8` |
|---|---|---|
| GPU extension offload (`--gpu`) | **1.25x** | 1.07x |
| Fully-sampled (flat) suffix array | **1.12x** | **1.00x** |

The flat-SA experiment shows the mechanism directly. Per SA lookup:

| | `-t1` | `-t8` | scales? |
|---|---|---|---|
| sampled SA (1-in-8, W=32 lockstep + prefetch) | 177 ns | **169 ns** | **yes** |
| flat SA (one direct load, 49.6 GB) | 71 ns | **134 ns** | **no — 1.9x worse** |

The sampled path is **latency**-bound and the lockstep window hides it, so 8 threads barely disturb
it. The flat path degrades ~1.9x under contention.

> **RETRACTION (2026-07-16, same session).** An earlier version of this section claimed the flat path
> was *TLB-bound* and that **"`prfm` cannot help because ARM drops a prefetch on a TLB miss rather
> than walking the tables"**. **That claim was not supported and is withdrawn.** It was inferred from
> a null result (adding a prefetch changed 71 -> 71 ns), and a null result does not establish a
> mechanism. Direct measurement (`bench/uarch/`) shows the honest picture:
>
> | access pattern, 16 GB span | 1 thread | 8 threads |
> |---|---|---|
> | independent random gathers (MLP exploited) | **3.5 ns** | **4.3 ns** |
> | serial dependent pointer chase (no MLP) | ~108 ns | ~128 ns |
>
> With a proper warm-up, an explicit `prfm` adds only **1.02-1.08x** over the no-prefetch loop. That
> does **not** mean the prefetch is dropped: it means it is **redundant**, because the out-of-order
> engine already extracts the ~28 concurrent misses the core supports. The experiment cannot
> distinguish "dropped" from "unnecessary", so no mechanism claim is made either way.
>
> **Two of this session's own microbenchmark results were cold-start artifacts** and are also
> withdrawn: apparent prefetch speedups of 13.89x and 20.15x vanished to ~1.02x once a warm-up pass
> ran before timing. The un-prefetched arm always ran first on a freshly faulted mapping and was
> charged every one-off cost (frequency ramp, page-table population, compressor settling). **Warm up
> before timing, and interleave the arms — including in a 60-line C microbenchmark.**

So at `-t8` the aligner is bound by a **shared** resource (DRAM + TLB + page walkers), and any lever
that only reduces *per-thread* work stops paying once the threads contend. This explains, with one
mechanism, three separate observations: the GPU's `-t1`-only win, the flat SA's `-t1`-only win, and
why our ratio vs bwa-mem2 sags from ~2.6x at `-t1` to ~2.0x at `-t8`. **Quote `-t8`, always: a `-t1`
number measures a machine nobody runs.**

Two corollaries for anyone benchmarking here:
- **29% of the 500k-read `-t8` benchmark is fixed index load** (1.04s of 3.59s; measured by aligning
  a single read). It amortises away on real WGS, so the `-t8` *compute* ratio is better than this
  benchmark shows, and any lever's share is correspondingly *understated* by it.
- The `-t1` vs `-t8` gap is not noise or thermals. It is the signal.

### Flat / denser suffix array: `-t1` +12%, `-t8` +0%. Not worth the RAM. (2026-07-16, measured)

`get_sa` is the single largest cost at genome scale (**18.8%** of `-t1` wall, 21.3M lookups at
177-193 ns; see `BWA3_CHAIN_TIME`), because the SA is sampled 1-in-8 and 7 lookups in 8 walk ~7 LF
steps, each a DRAM miss. Densifying the SA removes the walk **by construction**.

Measured with the real thing, not a model: the 49.6 GB flat i64 SA extracted during the LISA work
(`work/genome.lisa_sa`) was wired into `get_sa_batch` behind an env var. **Byte-identical** (500k
records `cmp`-clean; `sa[j] == get_sa(j)` confirmed at genome scale). `get_sa_batch` 3.775s -> 1.508s.
End-to-end: **`-t1` 1.109/1.125/1.116x, `-t8` 1.002/1.000/1.000x.**

**Verdict: do not build the Packed40 SA sidecar.** It would cost +27 GB of RAM (index ~20 GB -> ~47 GB,
against bwa-mem2's ~16 GB) to buy ~2.6% at the operating point people actually use, and that ~2.6% is
under this host's ~2.4% noise floor. Packed40 at 31 GB is only 1.6x fewer pages than the 49.6 GB
spike, so it cannot escape the TLB wall that caused the collapse. The salvaged LISA pieces
(`bwa_index::packed::Packed40`, `bwa_core::sysram`) stay unused — this was their last plausible
customer.

### GPU line: at its ceiling, and the ceiling is 1.28x (2026-07-16, measured)

Two results, both from a quiet host, `work/genome.fa`, 500k reads, interleaved, `BWA3_GPU_STATS`:

**(a) The Metal kernel is ~8.4x a CPU core and costs 3% of the run.** `E_cpu = 4.33s` vs
`extend_batch = 0.514s` at `-t1`. `--gpu` delivers 19.73 → 15.91s = **1.25x**, against an Amdahl
ceiling of 1.28x from extension's 22% share. **There is essentially nothing left to win here** — a
kernel infinitely fast would add 3%. At `-t8` it is worth only ~1.07x, because `E_cpu` divides by 8
while the shared Metal queue does not. Keep `--gpu` as the low-`-t` win it is; stop optimizing it.

**(b) SoA rails/query codes: measured 2.2x SLOWER. Do not retry.** The obvious port of the CPU
kernel's inter-sequence `[column * LANES + lane]` layout to MSL (`ehh[j*njobs+gid]`, coalescing the
32 lanes of a SIMD group into one transaction) **regressed `gpu_exec` 0.332 → 0.730s** (pack also
0.182 → 0.368s, a strided-scatter artifact, but the *kernel* is the point). Byte-identity held
throughout; this is a pure speed negative.

Why, and why the existing AoS layout is correct rather than an oversight: **1 GPU thread = 1
alignment**, and that thread walks `eh_h[j]` for increasing `j` **contiguously**, so one fetched
128 B line already serves its next 32 columns. AoS costs 32 lines per SIMD group but each serves 32
cells; SoA costs 1 line per group per column but that line serves 32 cells *once*. **Both amortise
to one line per 32 cells — coalescing buys nothing here** — and SoA additionally puts consecutive
columns `njobs*4` ≈ 800 KB apart, destroying page/TLB locality. The CPU kernel needs SoA because
there the lanes *are* the vector and cannot walk independently; a GPU thread can, and does. The
memory-transaction-amplification hypothesis is therefore **refuted**: the kernel was never
transaction-bound.
| 2 | ✅ **Shorten the carried chain** (Suzuki-Kasahara idea, our-layout form) — **DONE, ~8%, byte-identical** | The column-carried recurrence was `f→h→f_new` (~4 ops). Since `h=max(A,f)` with `A=max(M,E)` independent of the carry and `oe_ins≥e_ins`, it's exactly `f_new=max(f-e_ins, max(A-oe_ins,0))` — the `C=max(A-oe_ins,0)` term is off-path, so the carried chain is 2 ops (sub,max). No anti-diagonal rewrite, no extra registers, byte-identical. | **~8% on kernel** (measured; ~6% end-to-end) | **Identical (proven u8+i16, gated)** | Low (done: `06b0225`) |
| 3 | **Ungapped-first / skip-DP** (minibwa q-mer prefilter) | Most 150bp extensions are gapless. Cheap ungapped score first; full gap-affine DP only when a gap could improve. minibwa: q=7, run SW only if max q-mer count ≥ 10. | Removes a slice of the 85% | Exact only if the skip provably can't change the winner (else a validated divergence) | Low-medium |
| 4 | **WFA for hard extensions** (hybrid) | O(n·s): ~10x fewer cells for low-error reads (2-6x vs KSW2). Keep SIMD banded DP for tiny bands, dispatch WFA only for wide bands. | High for the hard subset | Risky: WFA's M>X>D>I CIGAR tie-break ≠ bwa's; needs canonicalization | High (spike) |
| 5 | **Micro-opts** | `vmax` over `cmp+bsl`; plain `vadd` where diffs permit (vs `vqadd` lat 3); horizontal reductions out of the inner loop; register-tile the band, cut `eh_h`/`eh_e` traffic; LTO + `target-cpu=apple-m1` + PGO (escape-branch layout). | 5-15% each | Identical | Low |

**Redundant-work audit (free, identity-required):** confirm contained-seed drop, chain filtering,
Z-drop cell, and band width match bwa-mem2 exactly — any deviation is both a correctness bug and
wasted SW. Highest ROI, zero risk.

## Do NOT
- **Port Zhang's BWT-region binning. Measured dead (2026-07-17) before building it.** Zhang et al.
  (CCGrid'13 §IV) bin occurrence computations by BWT region so a round's accesses land in a window
  whose pages fit the TLB, reporting 1.43-1.54x single-thread / **1.21-1.36x multithreaded** -- which
  made it the one lever that looked like it survived thread contention, since it cuts shared
  page-walk traffic instead of hiding per-thread latency.
  `examples/bwt_binning.rs` measures the mechanism directly (both arms lockstep W=16 + prefetch, arm
  B paying its counting sort, warm, interleaved, min-of-3):

  | batch N | acc/bin (2^24) | random | binned | speedup |
  |---|---|---|---|---|
  | 4 096 | 11 | 1.9 ns | 2.8 ns | **0.69x** |
  | 65 536 | 177 | 3.1 ns | 3.1 ns | 1.00x |
  | 1 048 576 | 2 834 | 7.8 ns | 7.7 ns | 1.01x |
  | 16 777 216 | 45 344 | 8.3 ns | 7.0 ns | **1.18x** |

  **The knee is at ~16.7M** -- exactly Zhang's batch, which is not a coincidence: binning needs page
  REUSE, and a 2^24-row bin is 1024 pages, so you need tens of thousands of accesses per bin before a
  page is touched twice. Dead twice over: (a) **infeasible** -- our slots carry a 3.6 KB `prev[]`, so
  16.7M of them is ~60 GB *per thread*, and in the reachable range (N <= 1M) binning returns exactly
  1.00-1.01x; (b) **capped** -- even at N=16.7M, 1.18x on seeding's ~41% is **+6.7% end-to-end**, not
  Zhang's 1.21-1.36x.
  Why the gap to their paper: `random` costs **1.9 ns/backward_ext at N=4096**. A random access into
  5.8 GB amortised to 1.9 ns is our W=16 lockstep + prefetch doing its job; binning cannot improve on
  1.9 ns, it can only add sort cost. Their baseline was neither lockstepped nor prefetched.
- **Build a STAR-style "exact prefix table + binary search over a flat SA" seeder. Measured dead
  (2026-07-17), and the number that kills it is cheap to re-derive.** STAR replaces BWA-MEME's
  learned RMI with an exact table (`genomeSAindexNbases 14`: *"length (bases) of the SA pre-indexing
  string ... allow faster searches"*), which looks like a simpler way to reach LISA's design. The
  model that motivates it: resolving a 32-mer costs FM ~32 dependent rounds (one per base) but STAR
  only 1 table lookup + ~2*log2(23) probes, since the mean SA range after 14 bases is 23 rows.
  **The mean is a lie.** Queries come FROM the genome, so a K-mer occurring t times is t times
  likelier to be hit: the distribution a real read sees is **occurrence-weighted**, and repeats
  dominate it. Measured over 200k real 14-mers (`examples/star_vs_fm.rs`):

  | | rows | probes |
  |---|---|---|
  | table average | 23.1 | 9.1 |
  | **occurrence-weighted mean** | **9638.4** | **26.5** |
  | median | 67 | |
  | p90 / p99 | 2985 / 192334 | |
  | max | 2891724 | 42.9 |

  At 26.5 probes x 2 dependent accesses each (read `sa[mid]`, then the reference at that position),
  STAR-style needs **~53 dependent rounds against FM's ~32**. It is ~1.7x *worse*, not 3x better.
  The arms were verified to agree exactly (200000/200000 identical SA intervals), so this is the idea
  losing, not the implementation.
  **Corollary, and it vindicates BWA-MEME's design:** the RMI is not over-engineering of STAR's table.
  A table returns a *range*; the RMI predicts a *position within* the range plus a short last-mile.
  That is precisely the fix for the repeat-weighted range, and an exact table cannot do it.
- **Prefetch the SA arrays in `get_sa_batch`. Built it, measured 0%, reverted (2026-07-17).**
  `sa_ms_byte` and `sa_ls_word` are separate arrays, so every SA read costs two cache lines and two
  TLB entries, and `prefetch_cp` never touched them: the read fires in the same round the walk lands
  on a sampled row. Restructured the lockstep to prefetch both lines and defer the read one full
  round. Byte-identical (SE 500k + PE 1M `cmp`-clean; `get_sa_batch == get_sa` on 2M genome
  positions). **139 -> 138 ns/lookup.** The 32 concurrently-walking slots already give the
  out-of-order engine everything it needs; see `bench/uarch/mlp_probe.c`, where an explicit `prfm`
  adds 1.02x over a plain loop for exactly this reason. **Fourth lever killed by this mechanism**
  (LISA, flat SA, the 10-mer cache, this).
  BWA-MEME co-locates position+key in one 5-byte record instead, making it a single access; that is a
  real layout advantage, but interleaving ours costs ~0.8s of load time to rewrite 3.9 GB against a
  1.04s load, and the measurement above says the second access is free anyway.
- **Port minibwa's 10-mer cache. Built it, measured 0%, reverted (2026-07-17).** Li & Homer
  (arXiv:2606.15357 §2.5) precompute the ds-intervals of all 10-mers "to reduce the number calls to
  BackStep()" and measure *"10% faster to find (19,1)-SMEMs"*. **Their own sentence says "on real
  short reads WITHOUT BATCHING"** — and that qualifier is the whole story. Our round 3
  (`bwt_seed_strategy`) is lockstep-batched (`BwtSeedSlot`, N=16) with per-step `cp_occ` prefetch, so
  the 10 forward extensions the cache removes were **already latency-hidden and nearly free**; the
  cache replaces them with one random miss into a 25 MB table that does not fit the 16 MB L2.
  Measured, byte-identical throughout (SE 500k + PE 1M `cmp`-clean, oracle 5000/5000): SE -t1
  0.959/1.018/1.014, **SE -t8 0.994/0.994/0.997**, PE -t8 0.998/0.994/1.000. Note it can only ever
  apply to round 3 anyway: rounds 1/2 write `prev[num_prev] = smem` at every distinct-`s` step and
  `prev[]` is the entire input to their backward phase, so a LUT would skip exactly the steps whose
  outputs are needed.
  **This is the third lever killed by the same mechanism** (LISA's learned index, the flat SA, this):
  our lockstep+prefetch batching has already eaten the DRAM latency each of them was designed to hide.
  **Before porting any "fewer memory accesses" idea, check whether the accesses it removes are ones we
  already overlap.**
- **Try to engage Apple's DMP** (the pointer-chasing prefetcher). It is built for exactly our shape of
  problem -- Apple's patent (US 9,886,385) names "pointer chasing" with "poor locality"; Augury
  (IEEE S&P'22) measures **3-8x** on array-of-pointers traversal, vanishing on DMP-less Icestorm
  cores; GoFetch (USENIX Sec'24 §4.4) shows it even walks page tables and fills the TLB itself. **It
  still cannot help an FM-index.** The DMP dereferences 64-bit values *stored* in a filled L1 line
  whose bits [55:32] match that line's own address. Our LF walk **computes** its next address
  (`sp = count[b] + occ`: popcount + add) and `cp_occ` holds counts and bitvectors, not addresses.
  There is no array of pointers to dereference, and no fix short of storing real 64-bit VAs -- which
  doubles an array whose +27 GB variant already measured **+0% at -t8**. Also: P-core only, L2 only,
  64 KiB max distance, a ~128-entry history filter that prefetches each pointer once, and **M4 was
  never tested** by either paper.
- **Try to disable/tune the prefetchers from the aligner.** The chicken bits are real
  (`HID5[44]`/`[45]` = disable HW prefetcher load/store, `HID11[30]` = disable DMP; verbatim in
  m1n1's `src/cpu_regs.h`) but they are **EL1** registers. Userspace cannot write them and macOS
  exposes no interface.
- **Assume anything about `PRFM` on a TLB miss.** The ARM ARM (DDI 0487A.a §C2.2 p.C2-138)
  permits treating any prefetch as a NOP *and* explicitly permits it affecting "caches and
  translation lookaside buffers". It mandates neither, and no source documents Apple's choice.
  Measure, do not reason -- and see the retraction above for what happens when you don't.
- Switch to striped/lazy-F/prefix-scan (wrong toolbox for inter-sequence).
- Adopt Block Aligner / adaptive banding / strobemer seeding (heuristic → breaks byte-identity).
- Chase AMX/SME/wider lanes (MAC-only, no max-plus; 16-lane NEON is the hard ceiling).
- Resurrect LISA on Apple Silicon.
- SIMD-vectorize short-read chaining (negligible `n`, breaks identity) or ERT (60GB RAM, redundant
  with our lockstep hiding).

## Recommended sequence (revised after lever #1 tested negative)
1. ~~Spike lever #1~~ — **done, no gain on NEON (register-bound, see above).** The latency headroom
   exists but the register file can't hold a 2nd interleaved group. Do not retry on NEON/AVX2; only
   reconsider on AVX-512 (32 zmm) when the Intel path gets CI.
2. **#2 (difference recurrence, Suzuki-Kasahara)** is now the only remaining kernel-side lever that
   attacks the *same* latency chain without adding registers: it *shortens* the critical path (~8→4
   ops/cell) rather than parallelizing it, so it sidesteps the register wall. Highest-value next spike,
   but verify the CIGAR tie-break stays byte-identical.
3. **#3 (ungapped skip)** — independent of the kernel, removes a slice of the 85% outright. Low-risk;
   pairs with the redundant-work audit. Do this regardless.
4. **Safe, identity-preserving wins that don't need a kernel breakthrough** (land these now): PGO/BOLT
   (`cargo-pgo`, ~10-15% on the driver's branchy dispatch), kill the per-batch `regs_all[i].clone()`
   allocation, per-thread scratch reuse. These are the realistic near-term gains.
5. **#4 (WFA)** only as a gated spike once #2-3 land and CIGAR byte-identity is proven.

## Closest analog
**minibwa** (Heng Li, 2026): same lineage, same NEON target, ~2.5x over bwa-mem2 — its wins are the
skip-DP / prefilter / prefetch class (byte-safe-friendly), not a magic SW kernel. Confirms the
above. Strobealign/Accel-Align win by *fuzzy seeding* (fewer SW calls) but change output.

Sources: Suzuki-Kasahara 2018; Rognes SWIPE 2011; Marco-Sola WFA 2021 / BiWFA 2023; Dougall Johnson
M1 Firestorm SIMD tables; Hajime Suzuki M1 notes; minibwa 2026; ERT ISCA 2021; strobealign 2022;
Accel-Align 2021; mm2-fast 2022.
