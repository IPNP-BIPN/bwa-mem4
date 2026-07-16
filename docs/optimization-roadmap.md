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
> the *expected* result: `5af30cd`, `05f4458` and `0980a03` all removed DP work, and the ungapped
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
> ~22%, not 85% — so PGO applies to the *large* share, not the small one. And `feat/pgo` (`58e7006`)
> **measured SE +3.1% / PE +4.5%**, parity clean, with a complete 3-stage `cargo-pgo` harness in
> `scripts/pgo.sh`. The "version friction" caveat is also stale: `cargo-pgo` is installed and the
> script pins `PATH=/opt/homebrew/opt/llvm/bin` itself. (`llvm-bolt` genuinely is absent, so BOLT
> remains untested.)
>
> **Lesson: this file talked a measured, merge-ready win out of existing for months on the strength
> of a bad profile.** Prefer measuring to reasoning about percentages.

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
| 2 | ✅ **Shorten the carried chain** (Suzuki-Kasahara idea, our-layout form) — **DONE, ~8%, byte-identical** | The column-carried recurrence was `f→h→f_new` (~4 ops). Since `h=max(A,f)` with `A=max(M,E)` independent of the carry and `oe_ins≥e_ins`, it's exactly `f_new=max(f-e_ins, max(A-oe_ins,0))` — the `C=max(A-oe_ins,0)` term is off-path, so the carried chain is 2 ops (sub,max). No anti-diagonal rewrite, no extra registers, byte-identical. | **~8% on kernel** (measured; ~6% end-to-end) | **Identical (proven u8+i16, gated)** | Low (done: `05f4458`) |
| 3 | **Ungapped-first / skip-DP** (minibwa q-mer prefilter) | Most 150bp extensions are gapless. Cheap ungapped score first; full gap-affine DP only when a gap could improve. minibwa: q=7, run SW only if max q-mer count ≥ 10. | Removes a slice of the 85% | Exact only if the skip provably can't change the winner (else a validated divergence) | Low-medium |
| 4 | **WFA for hard extensions** (hybrid) | O(n·s): ~10x fewer cells for low-error reads (2-6x vs KSW2). Keep SIMD banded DP for tiny bands, dispatch WFA only for wide bands. | High for the hard subset | Risky: WFA's M>X>D>I CIGAR tie-break ≠ bwa's; needs canonicalization | High (spike) |
| 5 | **Micro-opts** | `vmax` over `cmp+bsl`; plain `vadd` where diffs permit (vs `vqadd` lat 3); horizontal reductions out of the inner loop; register-tile the band, cut `eh_h`/`eh_e` traffic; LTO + `target-cpu=apple-m1` + PGO (escape-branch layout). | 5-15% each | Identical | Low |

**Redundant-work audit (free, identity-required):** confirm contained-seed drop, chain filtering,
Z-drop cell, and band width match bwa-mem2 exactly — any deviation is both a correctness bug and
wasted SW. Highest ROI, zero risk.

## Do NOT
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
