# Catch fg-labs/bwa-mem3 0.16, then pass it

## Where we stand (6 Nov 2026)

Same records as bwa-mem2 on every run below (`--compat=bwa-mem2` for the fork).

| | bwa-mem4 | bwa-mem3 0.16.0 | gap |
|---|---|---|---|
| M4 Max, GRCh38, 2M HG002 pairs PE, `-t12 -K 1e8` | 29.4-32.2 s | 15.6-16.3 s | 1.9x |
| same, SE (R1) | 7.6-8.1 s | 4.7-4.9 s | 1.6x |
| EPYC 9V74 (Zen 4), chr21, real GIAB slice PE, `-t4` | 81.3 s | 44.6 s | 1.8x |
| same, 2M simulated pairs | 125.8 s | 55.8 s | 2.25x |

Where the M4 PE time goes (12 threads, wall):

| | bwa-mem4 | bwa-mem3 |
|---|---|---|
| seed + SA + chain + extend | 15.6 s (`align`) | 8.6 s (SMEM 1.65, chain 1.50, SAL 1.50, BSW 4.93) |
| pairing, rescue, SAM | 13.1 s (rescue 12.0, emit 1.0) | 5.9 s (`WORKER_SAM`) |

SE, sampled with `sample(1)`, share of busy samples: extension 35%, SMEM 20%, chaining
(`build_chains_from_resolved`, `mem_chain_flt`, `discard_contained`) 19%, `get_sa_batch` 7%. Against
the fork's own stage times, our extension costs about what theirs does; SMEM and chaining cost
about 1.8-1.9x theirs.

## Lever 1: banded rescue (the fork's `rescue_band`), priced GO

The fork runs the pass-0 rescue DP only inside the diagonal bands of the K-mer components at
`T1 = max(minsc, ub2 / 2)` (`ub2` = second-largest component bound), proves round 1 exact when
`S >= T1` and (`score2 >= T1` or the hull lies in `[te - S, te + S]`), and otherwise runs round 2 at
`minsc`. Argument: `reference/bwa-mem3-cpp/src/rescue_band.h`, `docs/src/developer-guide/rescue-banding.md`.

Priced offline on 148k real rescue jobs (every 5th of the HG002 dump; `band_t1_bench` in
`crates/bwa-neon/src/rescue_prune.rs`), cells as a share of every job's full window:

| | cells run |
|---|---|
| today: hull, hit gate 400, else full window | 83.1% |
| hull, no hit gate | 71.5% |
| T1 bands, round 2 where needed, no hit gate | **42.9%** |

Round 1 was final for 17,219 of the 17,993 jobs checked with the scalar DP (95.7%). Our 5 Nov no-go
priced bands at `minsc` and inside the hit gate, where 80% of windows never reach the filter.
Band cells are counted as width x query length, without the kernel's lane padding.

Expected: rescue kernel CPU down by about 45%, M4 PE from ~29 s to ~24 s.

Phases, each gated on byte-identity before the next:

1. Scalar reference: components at a threshold, banded zero-state DP returning per-row maxima and
   their first column, the kswv epilogue (`te`, `qe`, `score2`, `te2`) rebuilt from row maxima,
   round 1 / round 2. Gate: equal to `ksw_align2` on every dumped job and on generated adversarial
   jobs (near-ties, two copies, 8-bit ceiling).
2. NEON band kernel, 16 bands per vector, grouped by width class. Gate: equal to the scalar band.
3. Wire into `batched_align` behind a cost model (band only when cheaper than the hull); revisit
   the hit gate, which the bands make cheaper to drop. Gates: `oracle_diff.sh`, `opt_parity.sh`,
   full 2M-pair PE md5.
4. AVX2 kernel (32 lanes) for x86; until then x86 keeps the hull path.

## Lever 2: seeding and chaining (about 1.8x the fork's cost)

Not priced yet. The fork's 0.15/0.16 notes list: unique matches checked against the text with their
SA position kept, the bwtSeed pass reusing the SMEM pass's first walk, jump-table first steps,
backward-step invariants in registers, a packed (pos, chain) key array for chaining, seed pools per
read. Next step: a per-function profile of both on the same reads, then pick.

## Lever 3: idle samples

27% of SE samples are threads waiting. Part of that is the reader and writer threads, which wait
by design; a per-thread split is needed before calling it imbalance.
