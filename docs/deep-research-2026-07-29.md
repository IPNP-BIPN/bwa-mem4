# Deep research: is there an unexplored exact speedup? (2026-07-29)

A fan-out web harness ran 102 agents over five angles: exact per-row maxima faster than full DP,
bit-parallelism with affine gaps, exact mate-rescue acceleration, 2024-2026 SIMD kernel state of the
art, and unconventional exact approaches (memoization, four-Russians, sub-matrix reuse).

**Status of this document.** The harness hit the account's session limit partway: 46 of 102 agents
finished, 56 errored, and the synthesis step never ran. 25 candidate claims with sources survived; 3
were adversarially confirmed, 4 refuted, 18 were extracted from sources but never voted on. Claims
below are labelled accordingly. Two of the most promising were then settled here by our own
measurement rather than by a vote, which is the stronger evidence anyway.

## The one lead worth building, and why it died

**ALAE's gap-free region collapse** (arXiv:1208.0274, extracted claim, unvoted). The observation is
exact and applies directly to us: in local SW with affine gaps and a 0 floor,

```
E(i,j) = max(0, E(i,j-1) - e_del, H(i,j-1) - oe_del)
F(i,j) = max(0, F(i,j-1) - e_ins, H(i,j-1) - oe_ins)
```

so wherever every H seen so far in the row is `<= oe` (7 for bwa's `o=6, e=1`) and E/F entered the
row at 0, **E and F provably stay 0** and the six operations that maintain them are dead. That would
cut our fast body from ~22 to ~16 ops per cell with no change to any output, including per-row
maxima. Nothing else found in this review touches the 37% mate-rescue block without breaking
byte-identity.

**Measured, and it does not pay.** The predicate has to hold for ALL 16 lanes at once, and each
lane's high-scoring cells sit on its own alignment diagonal, at a different column of the row.
Instrumented on 40k real GIAB pairs (`BWA4_QUIET_PROBE`, since removed):

```
2,074,715,348 fast-body cells
  820,138,941 all-lane quiet (39.5%)
  367,886,116 in a row-leading quiet run (17.7%)
```

A branch-free kernel can only exploit a contiguous run, i.e. **17.7%**: saving 6 of 22 ops there is
4.8% of the kernel, under 2% of busy. Chasing the full 39.5% needs a per-cell all-lane test (a
horizontal max, 2-3 ops) which eats half of what it saves, landing at the same ~2%. Both are under
the 3% floor. **Killed by measurement, cheaply, before writing the kernel.** The 39.5%/17.7% gap is
the whole story: inter-sequence SIMD is exactly the wrong layout for a predicate that is per-lane
sparse.

## Everything else, and why it is closed

**Sub-quadratic exact per-row maxima: does not exist for our problem.** Two confirmed claims settle
this. The Monge/seaweed framework (WABI 2019) defines a scoring scheme as a single triple with one
flat per-gap-character penalty, so it has **no gap-open/gap-extend distinction at all** and cannot
express bwa's `o + e*l`. And its complete-approximate-matching result -- the closest analogue of mate
rescue, one read against every substring -- is `O(mn)`, exactly the 1980 Sellers DP bound, because
kernel construction dominates. Its speedup is confined to *length-bounded* local alignment, a problem
bwa does not solve. Schmidt's all-highest-scoring-paths structure (SIAM J. Comput.) was refuted as a
lead: it costs `O(mn log m)` to build, a log factor *more* than one full DP pass, and its
`{0,-1,1}`-weight fast variant does not cover bwa's scoring.

**Bit-parallelism: structurally incompatible.** Confirmed: BitPAl does not implement affine gaps; its
scoring model is a single per-character indel weight and gap-initiation/extension is listed only as
future work. Unvoted but consistent: the BGSA family (Bioinformatics 2019) is restricted to
`M>=0, I<0, G<0, I>=2G`, i.e. linear gaps, as a hard limit of the bit-vector encoding; it implements
global, semi-global and edit distance but **not local alignment**, and returns one score per pair
rather than per-row maxima. The 7-25x BitPAl headline is against a scalar Needleman-Wunsch, not
against an inter-sequence SIMD kernel.

**Suzuki-Kasahara difference recurrences: already banked.** The exact algebraic content is that the
four difference matrices are bounded by the scoring constants, not by sequence length, so affine-gap
DP fits in 8-bit lanes. **We already run 8-bit lanes** (u8 saturating with a bias), so there is no
lane-count gain left to take. The published 4.15 GCUPS for libgaba comes bundled with an adaptive
band of width 32 and an X-drop, with a measured 2.6% rate of not reproducing BWA-MEM's alignment --
the band and X-drop are exactly what byte-identity forbids, and the 2.1x figure is differences over
an *equally banded* baseline, not over a full matrix. For calibration, our u8 rescue kernel is at
**9.5 Gcell/s per thread**, above every affine-gap number in that paper and within 1.4x of edlib's
bit-parallel *edit-distance* throughput.

**SeedEx (HPCA 2021): proven, byte-identical, and worth 2.8% in software.** This is the one published
certificate for affine-gap SW: Theorem 1 gives closed-form thresholds such that a narrow-band score
above `S2` provably equals the full-matrix optimum, and the authors validated exact SAM equality
against BWA-MEM and BWA-MEM2 across all 787,265,109 reads of ERR194147_1. But their own
pure-software implementation of speculate-and-certify yields **14% on the banded-SW kernel and 2.8%
end to end**, which is why the paper is an FPGA paper. Two further objections for us: it certifies
the *score*, not the per-row maxima that feed `score2`, and it targets seed extension, whose band is
already bwa's own.

**Exact mate-rescue acceleration: nothing published.** The ERT paper (the one BWA-MEM2 acceleration
paper that touches mate rescue) reports only the removal of sorting overhead there, with no
mate-rescue-specific speedup number anywhere. Its zigzag SMEM pruning is real and output-preserving,
but it is a seeding lever, and our seeding is already batched and prefetched.

**Cross-job memoization: the measured reuse is in the wrong regime.** ALAE's Theorem 5 / Lemma 3 do
give exact sub-block copying when two DP forks share a common prefix, with a measured 16.2-31.5%
reuse fraction -- but that fraction *rises with query length* and was reported as negligible at ~1000
characters. Our queries are 150 bp. Their exactness is also stated only with respect to reporting
alignments above a score threshold; the paper never computes per-row maxima or any second-best
statistic, so the theorem as published does not certify our `score2`.

## The conclusion the review keeps arriving at

The binding constraint is not implementation quality, it is the output contract. Every large exact
win in the literature is either (a) for a scoring model without affine gaps, (b) for global or
semi-global rather than local alignment, (c) conditioned on not needing per-row maxima, or (d) an
FPGA/GPU result whose software form is worth single-digit percent. Heng Li reaches the same place
from the other side in the minibwa paper: further speedup required breaking the output contract, and
he broke it.

That leaves the ranked list unchanged from `docs/literature-review-2026-07.md`, minus the row-pairing
item now delivered:

1. AVX-512 seed-extension kernel (18% of busy, x86 only, needs `avx512bw` hardware to measure).
2. The `mem_sort_dedup_patch` residue (13%), needing an incremental structure that reproduces klib's
   tie order.
3. Micro-architectural work on the rescue kernel, now at 9.5 Gcell/s per thread.

## Sources

- Tiskin, A. et al. (2019). Efficient algorithms for local alignment search. *WABI 2019*, LIPIcs 143.
- Schmidt, J.P. (1998). All highest scoring paths in weighted grid graphs. *SIAM J. Comput.* 27(4).
- Loving, J., Hernandez, Y., Benson, G. (2014). BitPAl: a bit-parallel, general integer-scoring
  sequence alignment algorithm. *Bioinformatics* 30(22):3166.
- Ren, S. et al. (2019). BGSA: a bit-parallel global sequence alignment toolkit. *Bioinformatics*
  35(13):2306.
- Suzuki, H., Kasahara, M. (2018). Introducing difference recurrence relations for faster semi-global
  alignment of long sequences. *BMC Bioinformatics* 19:45 (libgaba).
- Fujiki, D. et al. (2021). SeedEx: a genome sequencing accelerator for optimal alignments in
  subminimal space. *MICRO/HPCA*.
- Subramaniyan, A. et al. (2021). Accelerating maximal-exact-match seeding with enumerated radix
  trees. bioRxiv 2020.03.23.003897 (zigzag seeding).
- Xu, B. et al. (2012). ALAE: accelerating local alignment with affine gap exactly. arXiv:1208.0274.
- Li, H., Homer, N. (2026). Fast genomic read alignment with minibwa. arXiv:2606.15357.

---

# Round 2: newest-first sweep, and the STAR / Salmon question (same day)

Run directly rather than through the fan-out harness, which was still inside its session quota.

## Salmon is not an aligner, and that is the whole answer

Salmon's speed comes from **not doing the work**. Quasi-mapping "estimates where the reads best map
to [...] by identifying where informative sequences within the read map to instead of performing
base-by-base alignment", and selective alignment is explicitly framed as bridging "fast mapping" and
"traditional alignment", keeping "much of the efficiency of fast mapping". Its output is transcript
abundance, not SAM: no CIGAR, no MAPQ, no `XS`, no second-best score. There is no version of "adopt
Salmon's approach" that leaves a byte-identical SAM behind, because the thing Salmon skips is exactly
the thing our output contract requires.

## STAR is fast for reasons we are forbidden from copying

STAR's mechanism is (a) an **uncompressed suffix array** searched by binary search for **maximal
mappable prefixes**, then (b) **seed clustering and stitching**, with base-level DP only in the gaps
between stitched seeds. Both halves are closed for us:

- MMPs are not SMEMs. Different seeds, different chains, different SAM. bwa's seed definition is part
  of the output contract.
- The uncompressed SA is the memory trade we already measured and rejected: our flat-SA experiment
  gave `-t1` +12% and **`-t8` +0%**, because the sampled path is latency-bound and the lockstep
  window already hides it while the flat path degrades under thread contention. STAR's index is also
  ~30 GB against our 11 GB peak, and RSS is a column we currently win against the fork.

The published STAR throughput ("550 million 2 x 76 bp paired-end reads per hour on a modest 12-core
server", 2013) cannot be turned into a per-core ratio against us: it is ambiguous between reads and
pairs (a factor of two), it is 2013 hardware, and STAR solves the spliced-RNA problem, which has no
mate rescue over a 1400 bp window. What is decidable is the mechanism, and the mechanism is "do less
DP".

The 2023 review *Performance optimization in DNA short-read alignment* (PMC10060706) catalogues the
same idea as a first-class technique: a "high-speed search for perfect and nearly perfect mappings"
before gapped alignment, "100 or more times faster" for the ~50% of reads that qualify. **We already
have it, structurally, because bwa does**: extension DP is skipped entirely when a seed reaches the
read edge (`across.rs:938` guards the left extension on `seed.qbeg > 0`, `:978` guards the right one
on `seed.qbeg + seed.len != l_query`), and mate rescue never runs for a properly placed pair. The
fast path is not missing; it is why our remaining 55% of DP is concentrated on the reads that
genuinely need it.

## Closing the q-gram door with a derivation instead of an assertion

Round 1 asserted that minibwa's q-gram prefilter (q = 7, run SW only if a diagonal carries >= 10
matches) is incompatible with byte-identity. Here is the actual bound, both ways.

For an ungapped alignment with `M` matches and `X` mismatches under `a = 1, b = 4`, reaching our
acceptance threshold `score >= 19` requires `M >= 19 + 4X`. The number of clean q-grams on that
diagonal is at least `M - qX - (q - 1)`. Substituting:

| q | clean q-grams guaranteed | first X with zero guarantee |
|---|---|---|
| 7 | `13 - 2X` | **X = 7** (M = 47, span 54, score 19) |
| 5 | `15 - X` | X = 15 |
| 3 | `17 + 2X` | never |

So at q = 7 a perfectly legal score-19 alignment can carry **zero** clean 7-mers, and any threshold
above 0 drops alignments bwa keeps: the filter is unsound for us, by construction, not by bad luck.
Push q down to 3 and the filter becomes sound and simultaneously useless: a 150 bp read against a
1400 bp window has ~3,300 chance 3-mer matches, far above any threshold that would exclude anything.
**Sound or useful, never both.** That is the door, closed with a proof.

## The frontier, newest first

Nothing newer than minibwa (arXiv 2606.15357, 13 June 2026) on the exact-DNA-alignment side, and no
2026 paper on SIMD Smith-Waterman kernels at all; the searchable record still ends at Rognes-style
inter-sequence SIMD and the 2018 difference recurrences. Fulcrum's own write-up of minibwa reports
**~100-135 Gbp/hr, ~8 GB peak, only strobealign faster on WGS**, and states plainly: "It does not try
to be a bit-identical replacement for BWA-MEM."

Ordering the field by what it gives up:

| tool | speed vs bwa-mem2 | gives up |
|---|---|---|
| strobealign | fastest on WGS | its own seeds and scoring entirely |
| minibwa | ~2.7x | bit-identity (seeds, chaining, DP, mate-rescue heuristics) |
| bwa-meme | ~1.4x | 130 GB peak memory |
| bwa-mem3 (fork) | ~1.3x | correctness vs bwa-mem2 beyond `HN:i` |
| **bwa-mem4 (us)** | **~2.0x** | **nothing** |

That last row is the whole product: on this machine we are 2x bwa-mem2 and 7% ahead of the fork while
being the only one whose SAM still matches bwa-mem2 byte for byte.
