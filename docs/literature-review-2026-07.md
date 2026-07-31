# Literature review: what is left to take, under byte-identity (2026-07-29)

Scope: published work on accelerating BWA-MEM-class short-read alignment, read against **our
constraint that the SAM output stays byte-identical to bwa-mem2 2.3**. That constraint is what makes
this review different from a normal survey: most of the published speed comes from changing the
answer, and this document is mostly about separating the two.

Profile this review is anchored on (200k real GIAB pairs, PE `-t8`, `sample`, top-of-stack, after
the rescue-kernel column split landed):

| symbol | share of busy |
|---|---|
| `matesw::fwd_local_sw_batch` (mate-rescue DP) | **37%** |
| `batched::batched_extend_neon_u8` (seed extension DP) | **18%** |
| rayon `batched_regs` closure (seeding + chaining + glue, inlined into the plumbing symbol) | 14% |
| `primary::mem_sort_dedup_patch` | 13% |
| `fmindex::get_sa_batch` | 5% |
| `build_chains_from_resolved` | 3% |

## 1. The paper that matters: minibwa (Li & Homer, arXiv 2606.15357, June 2026)

Heng Li and **Nils Homer** published minibwa six weeks ago. It is the successor line to both
BWA-MEM and bwa-mem3, and it is the frontier we are actually measuring against, so it deserves the
detail.

**What it is**: a hybrid of BWA-MEM seeding, ropebwt3 SMEM search, and minimap2 chaining and base
alignment. Reported at **~4x BWA-MEM and >2x BWA-MEM2**, <20 GB peak, with BWA-MEME at 130 GB in the
same figure. There is also `minibwa-rs`, a Rust rewrite by henriksson-lab, within ~20% of minibwa.

**What it does, item by item, and whether we can take it:**

| minibwa technique | byte-identical for us? |
|---|---|
| Batched SMEM search with a prefetch queue (Alg. 3), "~4.6x BWA-MEM for (19,2)-SMEMs" | **already ours** (`bwa-seed`, round-robin lockstep + `prefetch_occ`) |
| Batched SA locate with prefetch (Alg. 2), "over 4x LocateSlow, identical output" | **already ours** (`get_sa_batch`, W=32 window, prefetch-then-work) |
| 10-mer ds-interval precompute | **already tried here and dead** (see `docs/optimization-roadmap.md`, "minibwa's 10-mer cache") |
| minimap2 variable-length seeding + chaining | **no**: different seeds, different chains, different SAM |
| Suzuki-Kasahara dual-gap DP | **no**: different DP semantics from bwa's `ksw_extend` |
| **q-mer prefilter before pairing SW (q=7, run SW only if `max_t M_t >= 10`)** | **no**, and this is the important one, see below |
| "mate rescue to fewer candidates", "reduced effort in centromeres" | **no**: deliberate recall changes |

**The q-mer prefilter is the single most valuable idea in the paper for our hottest stage, and it is
provably incompatible with byte-identity.** Minibwa counts 7-mer matches per diagonal `t` between the
read `P` and the window `R`, takes `max_t M_t` as an approximation of the best ungapped score, and
skips Smith-Waterman entirely below 10. Our rescue accepts a hit at `minsc = min_seed_len * a = 19`
with `a=1, b=4`. An alignment scoring exactly 19 can be, for instance, 39 matches with 5 mismatches
spread every 6 bases, which leaves **zero** clean 7-mers on any diagonal. So no q-mer threshold above
0 is a sound filter at our acceptance score: the filter necessarily drops alignments bwa would have
kept. That is precisely why minibwa is not bit-identical to BWA-MEM, and the authors say so.

Corollary worth stating plainly: **the gap between us and minibwa is not implementation quality, it
is the constraint.** They buy their speed by declining to reproduce bwa's answer; we sell the answer.

## 2. The other accelerators, and why each is closed

- **BWA-MEME** (Jung & Han, Bioinformatics 2022): learned index (P-RMI) over the suffix array,
  **3.45x seeding, 1.42x overall, identical SAM output**. On paper the perfect lever. **Already
  implemented here and measured ~5x SLOWER** (`crates/bwa-seed/src/lisa_seed.rs`,
  `crates/bwa-index/src/lisa.rs`; see `lisa-learned-index-dead-end`). It also costs ~130 GB peak in
  minibwa's own figure against our 11 GB, which would forfeit the RSS column we currently win.
- **ERT** (Subramaniyan et al., ISCA 2021; `bwa-mem2` `ert` branch, "BWA-Mich"): enumerated radix
  tree, 1.6-2.1x seeding, **identical output**, but a **~60 GB index**. Same RSS objection, and our
  seeding is already prefetch-batched, which is where most of ERT's win comes from.
- **BWA-MEM-SCALE** (ICPP 2022): Exact Match Filter + FM-index Accelerator, 3.19-3.32x. **Not
  identical**: MAPQ, `XS` and `XA` differ on 3.14% of reads in their own measurement. The EMF idea
  (bypass the pipeline for full-length exact matches) fails for exactly our reason: MAPQ needs the
  suboptimal score, which needs the work you just skipped.
- **WFA / BiWFA** (Marco-Sola et al.): optimal gap-affine in O(ns) time, O(s) memory, and genuinely
  beautiful for high-identity pairs. Closed for us on two counts: it is global/ends-free rather than
  local-with-soft-clipping, and byte-identity needs bwa's exact tie-breaking on `qe`/`te` plus
  `score2`, which is a property of the DP matrix, not of the optimal alignment.
- **Sentieon BWA / Parabricks**: proprietary or GPU; Parabricks explicitly does not promise identical
  output. Our own GPU line was measured at a 1.28x ceiling and retired.
- **Zhang et al. BWT region binning** (CCGrid'13): measured dead here before being built (needs a
  ~16.7M-element batch to pay; at reachable batch sizes it is 1.00-1.01x).

## 3. What survives, ranked

Everything below is byte-identical by construction. This is the whole remaining list.

1. **Micro-architectural work on the rescue DP (37% of busy).** The column split landed today
   (-14.2% kernel CPU). The next item is **two-row blocking**: process target rows `i` and `i+1` in
   one pass so row `i`'s H stays in a register for row `i+1`'s diagonal and F, and the E carry is
   chained through both. That removes roughly 2 of the 5 memory ops per cell and interleaves two
   independent H chains. Structural and risky; nothing else on the list is bigger.
2. **AVX-512 seed-extension kernel (18% of busy, x86 only).** Still absent: only mate rescue has an
   AVX-512 path. Needs `avx512bw` hardware to measure, which we do not have locally.
3. **The `mem_sort_dedup_patch` residue (13%).** Three attacks have now failed or landed below the
   floor. What remains needs an incremental structure reproducing klib's exact tie order.
4. **Nothing on the seeding side.** Both of the literature's identical-output seeding levers are
   already in the tree, and the two index-fattening ones (BWA-MEME, ERT) trade away the RSS column.

## 4. Two ideas killed by measurement today

Both came out of this review, both are the same shape ("the kernel runs each 16-lane chunk to the
longest lane, so sort by length first"), and both are worth writing down so nobody proposes them
again.

- **Rescue job length sorting: 0.0% saved.** `EXEC` 168.2G vs `CELLS` 154.3G nominal (1.09x), and the
  length-sorted counterfactual executes *exactly the same* count. Rescue windows are all
  `2 * max_dist` wide except at contig edges, so there is no length spread to exploit; the entire
  1.09x is query padding (160 padded / 148 real). Probe: `BWA4_MATESW_TIME=1`.
- **Extension job length sorting: 0.4% saved.** 6,031,959 jobs, 57.6G nominal cells, 60.2G executed
  (1.04x), sorted 59.9G. This one looked much more promising, since a seed can sit anywhere in the
  read and extension lengths genuinely vary, but consecutive jobs turn out to be similar enough that
  chunk maxima barely exceed the mean. Probe: `BWA4_EXTEND_SHAPE=1` (new,
  `bwa-neon/src/batched.rs`).

Also measured and negative in the same pass: **duplicate rescue jobs**, 19 of 739,868 (0.0%).

## 5. Sources

- Li, H. and Homer, N. (2026). Fast genomic read alignment with minibwa. arXiv:2606.15357.
- Jung, Y. and Han, D. (2022). BWA-MEME: BWA-MEM emulated with a machine learning approach.
  *Bioinformatics* 38:2404-2413.
- Subramaniyan, A. et al. (2021). Accelerated seeding for genome sequence alignment with enumerated
  radix trees. *ISCA 2021*.
- Kim, C. et al. (2022). BWA-MEM-SCALE: accelerating genome sequence mapping on commodity servers.
  *ICPP 2022*.
- Marco-Sola, S. et al. (2023). Optimal gap-affine alignment in O(s) space. *Bioinformatics*
  39:btad074.
- Suzuki, H. and Kasahara, M. (2018). Introducing difference recurrence relations for faster
  semi-global alignment of long sequences. *BMC Bioinformatics* 19:45.
- Rognes, T. (2011). Faster Smith-Waterman database searches with inter-sequence SIMD
  parallelisation. *BMC Bioinformatics* 12:221.
- Prousalis, K. et al. (2025). A survey on sequence alignment algorithms and state-of-the-art
  aligners. *ACM Computing Surveys* 58(3).
