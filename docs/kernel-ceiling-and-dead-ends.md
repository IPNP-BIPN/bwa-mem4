# The SIMD DP kernels: what is reachable, and what is already ruled out

Two documents in one file, both from the 2026-08-08 multi-agent kernel research (32 agents, 24
levers, 8 survivors of an adversarial review whose default verdict was "refuted"). They lived as
GitHub issues #51 and #49 until 2026-08-18, which is the wrong place for a reference: an issue is
read once by whoever is subscribed, a file in the tree is read by whoever is about to redo the work.

Read part one for the ceiling and the surviving plan, part two before proposing any kernel change:
sixteen of the levers below have already been tried on paper and killed, several of them by two
independent proposers in the same session, and each entry says what would have to be new for the
idea to be worth a second look.


## Part one: the ceiling per CPU class, and the plan (issue #51)

Index and conclusions of the 2026-08-08 multi-agent research into the SIMD DP kernels. 32 agents, 24 levers examined, 8 survived adversarial review whose default verdict was "refuted".

### First, the number that was wrong

`BWA4_MATESW_TIME` printed `ISA ceiling: 16 u8 lanes x ~3.5 GHz = ~56 Gcell/s if 1 cell/lane/cycle` for months. **That is a width bound, not an achievable one**, and it made the kernel look five times worse than it is. One vector operation advances 16 cells, because the layout is inter-sequence and sixteen lanes are sixteen different jobs; a row costs ~15 operations and covers 16 cells, so the real figure is **~0.98 operations per cell**, not one cell per lane per cycle. No affine-gap Smith-Waterman runs in one operation per cell. Corrected in 5368661.

### The ceiling per CPU class

| class | achievable Gcell/s/core, our op mix | where we stand | how derived |
|---|---|---|---|
| **Apple M4 Max** | **16.0** | **10.4, 64%** | measured: peak NEON 16.63 G op/s (3.8/cycle), the kernel's own op sequence in registers 16.04 Gcell/s (93% of peak) |
| **x86 AVX2 only** (Haswell..Comet Lake, Zen 2/3) | port-LP bound, 12-15 cycles per 64-cell pair-column after #43 | unmeasured, but the kernel is missing the score table entirely | uops.info port model |
| **x86 AVX-512** (SKX, ICL-SP, SPR/EMR, Zen 4/5) | p0-bound at 12 cycles/row, 8.5 after #44 | measured 10.39 on a Xeon 8573C | uops.info + our own CI draw |
| **ARM server** (Neoverse N1/V1/N2/V2, Graviton 2-4) | vector-issue bound, ~34-36 V-uops per 16 cells in `batched.rs` | never measured on hardware | Arm SWOGs, LLVM sched models |

**The AVX-512 result deserves its own line.** On the Xeon Platinum 8573C we measured AVX-512 at 10.39 Gcell/s against AVX2's 8.29 — only **1.25x** for double the width. The research explains it: at 512 bits every saturating-integer and max op collapses onto **port 0 alone**, so width buys nothing and the kernel becomes p0-bound. That same Xeon does 8.29 in AVX2 where an AMD EPYC 7763 does 9.78-10.09, so the Intel part's AVX-512 barely reaches what the AMD gets from AVX2 alone. **512 bits is not 2x here and never was.**

### The plan

**Do now, on this machine (NEON):**
- #45 — `USQADD` fusion (+7.5%, measured in hand-written asm) and shared `qsub(h, oe)` (+6%)
- #46 — vector-gate `finish_row`, vector minsc pre-filter in `extract_group`
- #47 — the 12 padded query columns

**Do now, verifiable under Rosetta, measurable only on x86:**
- #43 — the XOR score table on AVX2/SSE4.1 (**+18% to +50%, the single biggest lever**)
- #44 — the same on AVX-512, plus gated port rebalancing
- #48 — `batched.rs`, which has had no ISA tuning at all and costs 2.0 ops/cell against the rescue kernel's 0.95

**Measure before building:**
- #50 — the alignment-span lemma: bound the reverse pass, and share the DP across overlapping rescue windows (the only lever that attacks the cell count itself)

**Do not retry:**
- #49 — sixteen dead ends with reasons

### The rule

**The per-cell arithmetic is finished.** The quad fast body is 71 instructions for 64 cells, 63 of them vector, 0.98 ops per cell against a machine ceiling of 0.97, running at 93% of peak issue rate. Everything left is off the critical path (the scalar epilogues), off the ISA (the x86 score table), or algorithmic (#50). Anyone who proposes shortening the recurrence should be pointed at #49 first.

### Caveat on completeness

Seven of the 32 agents hit the session limit, including the final synthesis and the adversarial verifiers for the Apple, state-of-the-art and algorithmic areas. So #45's measurements are the agent's own and were not independently attacked, and **#50 has never been through a hostile reader at all**. #43, #44, #46, #47 and #48 did survive verification.

The state-of-the-art survey also did not complete. The open question it was asked and did not finish answering: **has anyone published a per-core figure beating 10.4 Gcell/s on 128-bit lanes for affine-gap local SW with a second-best score?**

## Part two: the dead ends, and why (issue #49)

Everything the 2026-08-08 kernel research **killed**, with the reason, so nobody spends a day rediscovering it. Sixteen of the 24 levers examined did not survive adversarial review.

### Byte-identity fatal

**Row pairing in `batched.rs`.** Proposed independently by two research agents, ruled fatal by the verifier. The row epilogue carries the z-drop arithmetic, the row-all-zero termination and the `beg`/`end` retightening, and **row i's retightened band is row i+1's input**. A paired body would have to compute row i+1's cells before its own band exists. Designs exist that compute over a union band and mask afterwards, but they must keep each row's `rowmax_v`/`mj_v` separate and run both epilogues strictly sequentially, at which point most of the win is gone. `matesw` gets away with pairing only because it has no band retightening. Do not retry without a new argument.

### Refuted on the hardware evidence

**A 4-row body on AVX2.** It needs 24 architectural ymm values live before `q_v`, three interior diagonals, three interior E carries and scratch — about 33 values in **16 registers**. It would spill ~17 per iteration, i.e. ~34 extra memory ops per 128 cells against the 5 it saves. **3 rows does not fit either** (20 values). 2 rows is the correct depth on x86, and that is exactly why the +2.6% quad win on NEON (32 registers) does not port. Verdict: refuted at +2%.

**Replacing the argmax `vpblendvb` with `vpandn` + `vpmaxub`.** Corrected gain: **0%**. It is a wash on Haswell (both forms leave 29 uops on the {p1,p5} pair) and an outright **regression on Skylake through Tiger Lake**, where blendv is p015 and can use the otherwise-idle p5 while `vpmaxub` is p01-only: p01 per pair-column goes 24 -> 26 and the bound goes 12 -> 13 cycles. It is a win only on Gracemont (blendv = 8 microcoded uops), Zen 2 (blendv is FP0-only) and Alder Lake-P (3 uops). If ever revisited, gate it through the existing `extend_tier()` timing calibration rather than CPUID matching.

**Rebalancing three of the five `vpmaxub` per row into compare+mask-blend, unconditionally.** Corrected to **0%** as a general lever: it helps on Golden Cove where p0 binds, and hurts on Zen 5, where all four FP pipes take `vpmaxub` at TP 0.25 while `vpcmpub` costs latency 6 / TP 0.50. Kept in #44 only as an explicitly gated, Intel-only variant.

**Porting the u8 `rowmax` buffer and one-store row publish to x86, as a standalone change.** Corrected from +7% to +1.5% on its own. It is still worth doing, but as part of the `finish_row` work in #46, not as a separate item.

**Making the AVX-512 `rowmax` u8 with a single vector store**, standalone: corrected from +9% to +1.2%. Same note.

**Making `extract_group`'s scan row-major, or folding the suboptimal tracker into `finish_row`**: corrected from +2% to +0.3%. The gain is in the vector minsc pre-filter (#46), not in the loop order.

**Vectorising `extract_group`'s minsc filter and skipping the score2 tracker on the reverse pass**: the skip half corrected to +0.1%. The filter half survives and is in #46.

**Carrying the unsigned-saturating rewrite into the i16 kernels**: corrected to +0%. With bwa's default match score a=1 and 150 bp reads, `score_ceiling(j) < U8_SCORE_LIMIT` holds for **every** job, so the i16 tier handles nothing in practice. The honest number is zero until someone runs `-A > 1`.

**Row pairing in the NEON i16 rescue kernel**: survives on the merits but is worth **0%** for the same reason.

### Already in the tree

**Deleting the redundant `max(., 0)` in `batched.rs`'s gap-open paths** was reported as already done. It is not — it is still there, and it is item A of #48. The verifier was looking at the wrong site.

### Previously measured losses, restated

From earlier sessions, still valid and still not to be retried:

| idea | measured |
|---|---|
| lazy argmax column recovery | **18% slower** |
| a loop-invariant branch to skip the N repair (LLVM did not unswitch) | **7% slower** |
| length-sorting the job groups | **0.0%** |
| striped / intra-sequence kernels | hyalite 0.2 at 2.18 Gcell/s against our 10.59 |
| duplicate-job dedup | 0.0% (35 duplicates in 1.84M jobs) |

### The rule the research confirmed

**Do not try to shorten the per-cell recurrence.** The quad fast body is at 0.98 vector operations per cell against a machine ceiling of 0.97, measured at 93% of peak issue rate. Every remaining lever is either off the critical path (the scalar epilogues), off the ISA (the x86 score table), or algorithmic (#50). The arithmetic is finished.

---
Index of the surviving work: #43 #44 #45 #46 #47 #48 #49
