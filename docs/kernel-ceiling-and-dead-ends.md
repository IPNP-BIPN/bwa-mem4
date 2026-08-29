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

## Part three: dead ends found after the file was written

### The two `max(., 0)` clamps in the extension kernel's gap opens (2026-08-19)

The inner loop reads, for both the deletion and the insertion open:

```rust
let open_del_v = $max($sub(bigm_v, oe_del_v), zero_v);
```

and its own comment says the clamp "is redundant for the u8 kernel (saturating `sub` already floors
at 0) but free". Two vector operations of roughly sixteen in the body, on the kernel that is half of
`align`'s CPU, looked like it was worth deleting on the u8 path. It was implemented behind a macro
parameter, so each instantiation kept only what its subtraction type needs.

**Measured: exactly nothing.** Five interleaved repetitions of the `align` stage, minima 24.516 s
against 24.521 s. The reason is that the comment was literally true: `max(x, 0)` on an UNSIGNED
vector is the identity, LLVM knows it, and it had already been folding both of them away. The
"optimisation" removed source, not instructions.

Worth knowing generally, and the reason this entry exists: an op count read off the source is not an
op count. Before deleting arithmetic that the type system makes trivial, check whether the compiler
has already deleted it.

### An `x86-64-v4` release binary, the AVX-512 tier build (2026-08-27)

The x86 tarballs ship two binaries: the baseline one, which runs anywhere, and an `x86-64-v3` one
(AVX2) worth 5.6-6.8% on Zen 3. The residual gap against fg-labs/bwa-mem3 on simulated reads is
attributed to their per-tier builds of the whole SCALAR GLUE, not of the kernels, which we already
dispatch at run time. The obvious next move is therefore an `x86-64-v4` binary: AVX-512F/BW/VL/DQ
for the compiler, everywhere, not just in the hand-written kernels.

**Measured on the part it was for, and it loses to the tier we already ship.** Intel Xeon Platinum
8573C (Emerald Rapids), GRCh38 chr21, 1 M simulated pairs, `-t8`, interleaved, best of 2, all
binaries md5-identical to each other and to bwa-mem2 (`0feca445b3a990adaae368d4a38ed90b`):

| binary | best wall | vs the baseline build |
|---|---|---|
| bwa-mem4 baseline | 93.17 s | 1.000x |
| bwa-mem4 `x86-64-v3` | 88.28 s | 1.055x |
| bwa-mem4 `x86-64-v4` | 89.71 s | 1.039x |

`x86-64-v4` against `x86-64-v3` is **0.984x**: slower, by more than the two reps' own spread
(88.28/88.98 against 89.71/89.88, no overlap). So a third binary in the tarball would cost every
user a decision and a download to give the AVX-512 owners among them a small loss.

Why, when the AVX-512 KERNELS are worth having on the same host? Because they are already selected
at run time: the same run's tier sweep shows `BWA4_RESCUE_TIER=avx512` at 92.51 s against `avx2` at
97.14 s, so the 512-bit rescue kernel is worth about 4.8% end to end **and the baseline binary
already gets it**. What `-C target-cpu=x86-64-v4` adds on top is auto-vectorised glue, and there it
is a wash at best: 512-bit code in cold or branchy paths costs frequency and port pressure it does
not repay. The AVX2 tier build, by contrast, wins because it lets the compiler use AVX2 in the glue
without any of that.

**The rule:** a wider tier for the compiler is not the same lever as a wider kernel for the hot
loop. The kernels are dispatched, so the tier build only ever buys the glue, and the glue does not
want 512 bits.

### Part B of #44: the port arbitrage on the AVX-512 leaf maxes (2026-08-26)

The issue's part B is a port-pressure argument, and a good one on paper. On Golden Cove every
512-bit saturating-integer and max op is single-ported: uops.info measures `VPMAXUB` ZMM at 1 uop,
**p0 only, throughput 1.00**, because at 512 bits port 1's vector ALU folds into p0. `VPCMPUB` is
p5-only and `VPBLENDMB` is p05 at 2/cycle, so replacing a leaf `max_epu8` with compare-then-blend
pays one extra instruction to move work off the port that binds. The port LP said 10 cycles per row
of 64 cells becomes 8.5, i.e. **+18%**, on top of part A.

It was implemented properly: the kernel generic over `LEAF_CMP_BLEND`, the swap applied to the
twelve LEAF maxes only (the E and F updates, never `max(diag, e)` or `max(mfe, f)`, which sit on the
cross-row latency chain), and the spelling chosen per process by a timed calibration in the shape of
`extend_tier`'s, because the same rewrite is a known loss on Zen 5. Both spellings were verified
byte-identical against the scalar reference under Intel SDE, on `-skx` and `-spr` (commit 4c5675e).

**Measured: nothing, on both vendors.** Same runner, same process, best of 5, 8192 jobs of 614.4 M
DP cells:

| host | `vpmaxub` | `vpcmpub` + blend | ratio |
|---|---|---|---|
| Intel Xeon Platinum 8573C (Emerald Rapids, Golden Cove) | 59.39 ms, 10.345 Gcell/s | 59.84 ms, 10.267 Gcell/s | **0.99x** |
| AMD EPYC 9V74 (Zen 4) | 53.24 ms, 11.541 Gcell/s | 52.18 ms, 11.775 Gcell/s | 1.02x |

The Intel number is the one that matters, because Emerald Rapids IS the part the port analysis was
written against, and there the rewrite is a wash at best. Both are inside the calibration's 3%
margin, so on every machine measured it picked `vpmaxub` and part B was doing nothing but costing a
few milliseconds of calibration and a second instantiation of a 400-line kernel. The code was
reverted; part A, the XOR score table, stays and is worth **+11.4%** on the same Intel host (62.99
-> 56.55 ms, 9.753 -> 10.865 Gcell/s, run 33013582129).

Why the LP was wrong is worth stating, since the same reasoning will be tempting again. A port LP
prices instructions against ports and assumes the ports are the binding constraint. This body also
carries loads, stores and a loop-carried `h0 -> e_mid -> mfe1 -> h1` dependence, and the free half of
part B (reusing the `col` mask for `imax`) had already taken the cheapest p0 work away. What is left
is not p0-bound, so moving two more ops off p0 buys nothing and the extra `vpcmpub` uop cancels what
little it does buy.

**The rule:** a port-pressure LP is a hypothesis about the binding constraint, not a measurement of
it. Before implementing one, get the kernel onto the machine class it describes and check that the
port it names is actually the ceiling.

### The mismatch-only shortcut for the CIGAR's global DP (2026-08-20)

At equal query and reference lengths, a global alignment's indels must cancel, so any gapped path
opens two runs and costs at least `2*(o_min+e_min)`; when that strictly exceeds the exact
substitution loss `sum(a - mat[cell])` of the diagonal, all-M is the unique optimum and the DP is
skippable, provably and tie-free. Implemented, byte-identical, and it fired **6 times in 1.46M
calls**. The reason is `infer_bw`: the band the caller hands this DP is already derived from the
KNOWN region score, and it is already `0` for every one- and two-mismatch record, which the existing
`w_ == 0` fast path absorbs. Everything that still reaches the DP has three or more mismatches,
where the shortcut's premise fails. Reverted, since it costs a diagonal scan per call to catch
nothing; the general lesson is that a bound derived from the known score was already in the C, and
checking what `infer_bw` leaves behind should precede any scheme that re-derives it.

## Three x86 hypotheses, measured: one structural bug, one null, one that is their divergence (2026-08-29)

The side-by-side profile (`docs/x86-side-by-side-profile.md`) produced three hypotheses about the
wgsim deficit. Measuring them is what this entry records, because two of the three are now closed.

### The memory-level-parallelism knobs are already right on x86. NULL.

`DEFAULT_LOCKSTEP_WIDTH` is 32 on aarch64 and 16 on x86, with a comment saying no x86 measurement
existed to move it. `BWA4_SA_WINDOW` was swept flat from 16 to 128 on three machines, all of them
Apple Silicon or hosted ARM. `BWA4_SEED_PREFETCH` likewise. All three trade instructions for
memory-level parallelism, so the obvious hypothesis was that they pay more where a DRAM miss costs
more, and the profile put `get_sa_batch` at 4.1 units against the fork's 0.9.

Swept on an EPYC 7763, 1M chr21 pairs, each arm interleaved with a control at the defaults:

| arm | wall | against its own control |
|---|---|---|
| `BWA4_LOCKSTEP_N=16` (the x86 default) | 90.46 s | +0.5% |
| `BWA4_LOCKSTEP_N=32` (the arm64 default) | 91.99 s | -1.5% |
| `BWA4_LOCKSTEP_N=64` | 94.98 s | -2.9% |
| `BWA4_SA_WINDOW=32` | 92.07 s | -0.3% |
| `BWA4_SA_WINDOW=64` | 93.69 s | -0.8% |
| `BWA4_SA_WINDOW=256` | 93.88 s | -0.9% |
| `BWA4_SEED_PREFETCH=0` | 92.26 s | +0.7% |
| `BWA4_SEED_PREFETCH=16` | 92.48 s | -0.3% |
| `BWA4_SEED_PREFETCH=32` | 92.03 s | +0.2% |

Every arm is identical to the control's md5, and every one except `LOCKSTEP_N=64` is inside the
control's own drift over the run (90.6 to 93.1 s). The defaults are right, on x86 as on arm64, and
widening the arm64 value onto x86 would have cost 1.5%. The hypothesis was mine and it is dead.

Note what this does NOT close: the profile's 4.5x on SA resolution is real, and it is not prefetch
tuning. It is either more walks or a slower walk, and the grouping that produced "4.5x" put the
fork's `ls_advance_*_step` in seeding rather than in SA resolution, which is a judgement call that
could move the number.

### The reference unpack had no x86 path at all. A REAL BUG, fixed.

`unpack_pac_fwd` unpacked sixteen bases per iteration under `cfg(target_arch = "aarch64")` and fell
through to a per-base scalar loop everywhere else. That is the whole of the 3.50%-against-1.39% in
the profile. An SSSE3 path, selected at runtime, is in. Not a dead end: recorded here because the
SHAPE of the bug is worth remembering, a vector path written on the development machine and never
mirrored on the deployment one.

### Their tighter band is their divergence, not a free lunch. CANNOT BE TAKEN.

`ungapped_analyze` (`bwamem.cpp:4241`) derives the extension band from a mismatch bitmap and says of
it: "the optimal ungapped score over all prefix lengths in [0, N], with no floor (score is allowed
to dip and recover). It is >= the walk's max_sc, so substituting it into the band proof yields a
strictly tighter (still safe) bound."

Our `tight_band_bound` says the opposite, and says it from experience: "The first version of this
walk floored at zero and kept going, which OVERSTATES S whenever the diagonal dips and recovers,
which narrows the band below what the lemma licenses, which is how it moved XS on 1.1% of chr21
records."

Both cannot be right, and the lemma settles it. The band proof is "no alignment outside the band can
beat S", so S has to be a score the band-limited DP genuinely ATTAINS. A no-floor prefix maximum is
not attainable by a local DP that dies at zero, so a band derived from it can exclude an offset that
would have won. The fork can afford that because it lists score2 and MAPQ convergence among its
accepted differences; we cannot, and we already paid once to learn it.

So part of the extension-kernel row in the side-by-side table is the fork computing FEWER CELLS than
byte-identity permits, rather than computing the same cells faster. That is not a lever, it is a
different acceptance criterion, and it should stop being read as a gap to close.

## The ungapped fast path: priced, then blocked by `max_off` (2026-08-29)

Following the entry above, the one piece of `ungapped_analyze` that IS compatible with byte-identity
is `FP_STATUS_HIT`: skip the banded DP outright when the ungapped diagonal is the unique optimum. The
strictness is what makes it sound. An ungapped walk carrying `X` mismatches scores `X * (a + b)`
below the all-match diagonal; the cheapest gapped alternative pays at least `o_min + e_min` and can
at best convert every one of those mismatches back into a match. So while `X * (a + b) < o_min +
e_min`, no gapped alignment can tie, let alone win. At bwa's defaults that is `X <= 1`.

**Priced first.** `BWA4_UNGAPPED_FP` surveys every extension job. On 1M chr21 wgsim pairs:
7,172,498 of 35,324,706 jobs qualify, **20.3%**, and they carry **7.3% of the DP cells**. The job
count matters as much as the cells, because a skipped job also skips its two buffer allocations, its
two reversals, its lane in the batch and its extraction. Rough value: about 3% of the run.

**Then blocked, by a field that is not a score.** `ksw_extend2` returns `max_off`, the furthest the
best cell of any row strayed from the main diagonal, and the caller's acceptance test reads it:
a job is accepted only when `max_off < (w >> 1) + (w >> 2)`, otherwise it is REQUEUED with a doubled
band, and the band it finally settles on is stored in `MemAlnReg.w` and reaches `gen_cigar2`.

A fast path can predict the score, and cannot predict `max_off`. The uniqueness lemma bounds every
GAPPED alignment below the ungapped optimum, which pins the final `max_i`/`max_j` to the diagonal;
but `max_off` accumulates over every row that IMPROVED on the running maximum, and an intermediate
improving row's best cell may sit off the diagonal while still scoring below the global optimum.
Returning `max_off = 0` would therefore accept jobs the DP would have requeued, and the requeue is
not a no-op: it changes the recorded band.

So the fast path would have to predict a control-flow decision, not just an alignment. The fork can
skip that question because it accepts the divergence; we cannot. Recorded rather than attempted: the
20.3% is real and someone will find it again, and the blocker is one field, not the idea.

What would unblock it: a proof that on a qualifying job no improving row can be off-diagonal, or an
acceptance test that does not consult `max_off` when the ungapped optimum is proven. Both are
research, not implementation.

## Issue #48 lever A is already done, by LLVM, and the disassembly says so (2026-08-29)

#48's first lever looked like the safest few percent in the codebase: the E and F recurrences open a
gap with `max($sub(bigm_v, oe_v), zero_v)`, and for every u8 instantiation `$sub` is a SATURATING
UNSIGNED subtract whose result is already in `[0, 255]`. The issue called it "2 of ~33 vector ALU ops
in the inner loop", "provable, not empirical", and estimated +1.1%.

It was implemented: a `clamp0` slot in the kernel macro, bound to the real `max` for the four i16
instantiations where `$sub` wraps and the clamp is load-bearing, and to a two-argument identity for
the four u8 ones. Byte-identity held (NEON and AVX2 kernel tests green, full SAM md5 unchanged).

Then it was measured, and it is worth nothing:

| batch=64 kernel | before | after |
|---|---|---|
| aarch64 NEON | 68.46 / 68.69 / 67.48 ms | 68.46 / 68.13 / 68.46 ms |
| x86-64-v3 AVX2 | 147.57 / 148.51 / 147.67 ms | 148.16 / 146.82 / 148.41 ms |

A timer that says "no difference" is weak evidence, so the binaries were disassembled instead:

    before: vpmaxub=100  vpsubusb=312
    after:  vpmaxub=100  vpsubusb=312

Identical, and the two executables are the same size to the byte. LLVM already knows that
`max(saturating_sub_u8(a, b), 0)` is `saturating_sub_u8(a, b)` and had removed the op before the
source change asked it to. The change was reverted: it adds a macro slot and a helper for machine
code that does not move.

The general lesson is the one this file keeps learning from the other direction. An op count read off
the SOURCE is not an op count. #43 and #48 both reasoned about "ops per cell" from the Rust, and the
compiler had already collected the easy half. Count instructions in the binary before believing a
source-level tally, especially for anything a peephole optimiser could see.

## Issue #48 is finished, and the extension kernel's remaining gap is its recurrence (2026-08-29)

The extension kernel costs 1.00 vector instruction per cell against the rescue kernel's 0.69,
counted in the shipped binary. That 0.31 is five vector instructions per column on the line that is
24% of the x86 profile, so it was worth finding out what they are. The answer closes the issue.

**Lever A: already done by LLVM.** `max(saturating_sub_u8(a, b), 0)` is `saturating_sub_u8(a, b)`,
and the compiler knew. Implemented, measured at nothing on either ISA, and proven by disassembly
(`vpmaxub=100` and `vpsubusb=312` on both sides of the change, executables the same size). Reverted.

**Lever D: also already done by LLVM.** The issue says `j` is broadcast twice per column, once for
the band compare and once for the `mj` update. On NEON `band_bias` is 0 so the two are the same
value and CSE collapses them; on AVX2 the bias is 0x80 and they are different values, which is where
the lever was supposed to pay. The AVX2 column loop contains **one** `vpbroadcast`. The compiler had
hoisted the other one anyway.

**Levers B and C: already implemented, and on by default.** `inline_sbt_enabled` folds the
substitution pre-pass into the column loop, and `band_bias` builds the band mask in signed space.
The binary agrees: the DP loop has three loads and **two** stores, with no third store for a
`sbt_buf` round trip.

**So what are the five instructions?** They are the recurrence, not slack. Side by side, the NEON
column loops:

| | rescue | extension |
|---|---|---|
| instructions | 17 | 25 |
| vector | 11 | 16 |
| loads / stores | 2 / 2 | 3 / 2 |

The extension kernel carries `cmeq` plus `bic` to implement "H(i-1,j-1) == 0 means unreachable",
which the rescue recurrence does not have; it computes its substitution score inline where the
rescue kernel's is already folded into its loads; and it maintains a band mask per column because
its band is per lane. Every one of those is a term the extension DP has and the rescue DP does not.

There is no slack left in this kernel for issue #48 to remove. Anyone returning here should start
from the recurrence, not from the instruction count, and should note that three of the four levers
in that issue were rewrites the compiler had already performed. A count of operations read off the
source is not a count of operations.

## The rescue kernel's row epilogue is cheaper than a vector guard over it (2026-08-29)

`matesw`'s own note lists what the remaining 36% of that kernel is: "the 1.09x lane-divergence tax,
the row epilogue's scalar sixteen-lane loop, the padded tail columns and the group pack/extract".
Measured today, the arithmetic is worse than that reads:

    shipped kernel            9.91 Gcell/s/thread
    register-only ceiling    16.04
    divergence + padding      1.08x   (BWA4_MATESW_TIME's own accounting)

1.62x of gap, of which the tax explains 1.08x, so **1.50x sits outside the DP body**. That is half
the kernel, and the row epilogue was the obvious suspect: `finish_row!` walks sixteen lanes scalar-
wise once per ROW, which is 9,600 lane-steps for a 600-row job and about 1.3 billion for a 1M-pair
run, while a lane's maximum improves a few dozen times in those 600 rows.

So the loop was guarded: a `gmax_guard` vector holding 255 for lanes that are frozen or past their
target, and `vmaxvq_u8(vcgtq_u8(imax, guard))` to ask "can any lane improve?" in three instructions
before entering it. The two stores into the scalar mirror arrays moved inside the guard as well,
since only the slow path reads them. Byte-identical, provably: the loop is skipped exactly when no
lane satisfies the condition of the only branch that does anything.

Interleaved, three repetitions, alternating arms:

| | base | guarded |
|---|---|---|
| rep 1 | 9.91 | 9.81 |
| rep 2 | 9.92 | 9.81 |
| rep 3 | 9.90 | 9.79 |

**1.1% slower, reproducibly.** `vmaxvq_u8` is a horizontal reduction: it collapses a vector into a
general register, which on these cores costs several cycles of cross-domain latency and puts a
scalar dependency on the row's critical path. The sixteen-iteration loop it was meant to avoid is
predictable, mostly early-exits, and LLVM was already doing better with it than the guard does
without it. Reverted.

Two things to carry forward. The row epilogue is NOT where the 1.50x is, so the note in `matesw.rs`
should not send the next person there first. And a horizontal reduction is not a cheap way to ask a
question about a vector: it costs more than sixteen well-predicted scalar iterations.
