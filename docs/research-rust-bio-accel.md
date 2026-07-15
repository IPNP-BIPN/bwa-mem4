# Deep-research scan — Rust bio ecosystem + short-read acceleration (2018–2026)

Three-way-scan (WHY/HOW/WHAT + concordance verdict) for a **bwa-mem2-concordant** Rust aligner.
Concordance bar: same reads → same RNAME/POS/CIGAR/MAPQ (optional-tag diffs tolerated). The
deciding components are (a) FM-index **SMEM** seeding and (b) **`ksw_extend2`** local banded+z-drop
extension with bwa's tie-breaks + MAPQ. Any crate/technique that changes seeds, scores, banding, or
traceback tie-breaks changes *which alignment wins* → breaks concordance by construction. I/O is
always concordance-neutral.

## A. Rust crates

### Usable (concordance-neutral or reference)
| Crate | Use | Verdict |
|---|---|---|
| **noodles** (pure Rust) | SAM/BAM/CRAM/BGZF/FASTQ, MIT | ✅ I/O, pure-Rust default |
| **rust-htslib** + hts-sys (C htslib) | BAM/CRAM, htslib parity | ✅ I/O — **now wired for BAM output (`-o out.bam`)** |
| **needletail / seq_io** | fast FASTQ input (seq_io has parallel records) | ✅ input |
| **pulp / multiversion / wide / std::simd** | build the SIMD kernel (runtime dispatch: pulp/multiversion; stable single-target: wide; nightly: std::simd) | ✅ tools; correctness = ours |
| **rust-bio `FMDIndex`** | Li-2012 FMD/SMEM — the exact concept bwa seeds with | ⚠️ *reference only*: scalar, can't read bwa's index; wrap in bwa's seeding heuristics to be concordant |

### Breaks concordance by construction (do NOT use as the deciding component)
- **block-aligner** — heuristic adaptive blocks, <3% error, X-drop≠z-drop → not optimal/identical.
- **parasail-rs / parasailors** — exact but own gap model, no z-drop, own tie-breaks.
- **rust-bio pairwise/banded** — scalar, different band + tie-break; not `ksw_extend2`.
- **minimizer-iter / strobemers-rs** — minimizer/strobemer seeding ≠ bwa SMEMs → different seeds.
- **minimap2-rs** — a different aligner entirely.
- **awry** — fast SIMD FM-index but its own index format + count/locate, not bwa SMEM extension.

**Conclusion (crates):** the two concordance-critical pieces (FMD-SMEM seeder, `ksw_extend2`
extension) have **no drop-in Rust crate** — must be ported from bwa-mem2 (we did). Everything else
(I/O, FASTQ, SIMD tooling) is safely delegated.

## B. Acceleration techniques, ranked by (speedup × concordance-safety)

**Tier 1 — output-preserving CPU levers (pursue):**
1. **Learned-index seeding, BWA-MEME style** (RMI over suffix array) — up to **3.45× seeding, ~1.4×
   overall**, and **provably identical SAM** to bwa-mem2 (Oxford Bioinf. 2022, kaist-ina/BWA-MEME).
   The single best speed/safety trade *in principle*. **Caveat for us:** our own Rust LISA/BWA-MEME
   port measured **~50× slower** than FM ([[lisa-learned-index-dead-end]]) — the idea is sound, our
   implementation wasn't; only worth revisiting if we can match their SIMD/RMI engineering.
2. **Ungapped / skip-DP prefilter, restricted to accept-identical** — our mismatch-tolerant HIT:
   **SE +24% / PE +21%, byte-identical** (DP is ~35% of WGS CPU per minibwa). Highest-value *shipped*
   concordant lever. Keep the strict "only skip when SW would agree" gate.
3. **Algebraic SW-kernel restructuring** — the f-recurrence chain-shortening (~8%, byte-identical),
   already merged. The *safe slice* of the difference-recurrence idea. NEON 16-lane is the width ceiling.

**Tier 2 — adopt only with a shim:** Sapling-style learned SA lookup (>2×, position-preserving) but
bwa uses FM-index not a plain SA → non-trivial integration; secondary to BWA-MEME.

**Dead ends for a concordant CPU port:**
- **Difference recurrence (full semi-global)** — **CONFIRMED changes output**: it is *semi-global*,
  KSW2 README says "no local alignment yet"; bwa extension is *local* (SW restart + X-drop). Different
  objective → different winning traceback. Independently corroborates our measured **17.5% divergence**
  (spike `ee33f29`). Only the arithmetic-restructuring slice (Tier 1.3) is safe.
- **WFA / BiWFA** — optimal gap-affine but different objective/tie-break vs bwa; not a drop-in.
- **minibwa** (Heng Li 2026, ~4× vs BWA-MEM / ~2× vs BWA-MEM2) — author states **not bit-identical**;
  mine its *ideas* (ungapped-first, q-mer prefilter) not its default thresholds.
- **strobealign (4–6×) / Accel-Align (up to 9×)** — different seeding/objective, ~0.2% accuracy delta;
  measured *against* bwa, not equal to it.
- **GPU (Parabricks/SaLoBa) / FPGA** — need non-CPU hardware; out of scope.

## Bottom line for this project
- **I/O**: done — BAM via rust-htslib; noodles is the pure-Rust alternative if we want to drop the C dep.
- **Already captured the safe CPU levers**: skip-DP HIT (+24/21%), kernel chain-shortening (+8%).
- **The one remaining output-preserving lever with real headroom is learned-index seeding (BWA-MEME,
  ~1.4× overall, identical SAM)** — but blocked by our own 50×-slower implementation, not the idea.
- Everything faster than that (strobealign, minibwa defaults, WFA, diff-recurrence) **changes output**
  and is off-limits under concordance.

Sources: BWA-MEME (Oxford Bioinf. 2022, kaist-ina/BWA-MEME); Suzuki–Kasahara 2018 (PMC5836832) + KSW2
README ("no local alignment yet"); minibwa (arXiv:2606.15357); LISA (biorxiv 2020.12.22.423964);
Sapling (Bioinf. 2021); strobealign (Genome Biol. 2022); Accel-Align (BMC Bioinf. 2021); WFA/BiWFA
(Bioinf. 2021/2023); bwa-mem2 (arXiv:1907.12931); noodles / rust-htslib / rust-bio (GitHub); block-aligner
(Bioinf. btad487); awry (Alg. Mol. Biol. 2021); "State of SIMD in Rust 2025".
