# Design: beating `fg-labs/bwa-mem3` on speed and RAM

Date: 2026-07-21. Status: approved (section 1 explicitly, the rest under a grant of autonomy).

## The goal, stated so it can be falsified

Beat @nh13's C++ fork `fg-labs/bwa-mem3` on **wall time and peak RSS**, measured **end-to-end, `-t8`,
on a real WGS**, while the SAM stays **byte-identical to bwa-mem2 2.3**.

Three decisions fix the shape of everything below:

- **The benchmark is `-t8` end-to-end on real data**, not `-t1` align-only. The only number we have
  against the fork is `-t1` align-only (1.29x behind, phase 9a-era), and that regime excludes the two
  things we do better than a bwa-mem2 derivative: mmap instead of a 16 GB memcpy at load, and a
  reader/writer pipeline. The fork has never been measured in the regime a user actually runs.
- **Byte-identity is non-negotiable.** This is the project's acceptance criterion and 4.0.0 just
  shipped on it publicly. It also caps the campaign: it locks us into exact FM-index SMEM seeding,
  which is ~78% of the genome-scale profile. Aligners that go 4.5-6x faster (strobealign) get there
  by abandoning it. Our remaining wins are latency and bookkeeping, worth 5-15% each, not 2x.
- **RAM is opportunistic, on both binaries.** No numeric target. `mem` peaks at ~20.2 GB, `index` at
  ~75 GB. We take what the known levers give and we do not design toward a round number.

## Why this plan does not start with the algorithm

Two facts from this project's own measurement history constrain what is worth trying.

The genome-scale profile is `build_chains_from_smems` 37.6%, `LsSlot::step` 21.1%, `backward_ext`
20.4%, `mem_collect_smem_batched` 5.5%, **the SW kernel 4.1%**. Every kernel lever this project ever
chased (SIMD width, ILP, SME, the f-recurrence, the whole GPU line) was optimising 4% of the time.
The `~85% SW extension` and `~40-43% extension` figures in older docs were measured on `work/region.fa`,
whose 2 Mbp BWT is cache-resident and hides seeding entirely. **No number measured on `region.fa` is
admissible in this campaign.**

And the seeding side is already saturated: five separately published "fewer memory accesses"
techniques (LISA/BWA-MEME, flat 49.6 GB SA, minibwa's 10-mer cache, SA-line prefetch, Zhang BWT
binning) each measured ~0 here, because our W=16 lockstep plus prefetch already hides the latency
they target. A random `cp_occ` access into 5.8 GB costs 1.9 ns under it; nothing improves on 1.9 ns.
All three seeding rounds are lockstepped (`smem_round_2_batched` included, N=16). The obvious lever
is spent.

So the plan attacks **frames nobody has opened** and **costs that are not per-core algorithms**,
and it only returns to the algorithm if a fresh measurement says we are actually behind.

## Phase 0 — the baseline we are allowed to trust

Deliverable: a table, not a speedup. It answers Nils and it decides whether phase B happens.

Three arms in one harness: `bwa-mem2` 2.3 (the oracle and historical denominator),
`reference/bwa-mem3-cpp/bwa-mem3.arm64`, and us. The fork joins `scripts/giab30x_bench.sh` as a third
arm rather than getting a new script, so it inherits the per-pass md5 identity check and the
"a pass this fast means the aligner crashed" guard.

Two additions to the harness:

- **Peak RSS per arm** (`/usr/bin/time -l`). RAM is an objective here, so it is measured on every
  pass, not estimated once.
- **The batch count, printed on every run.** `-K 100000000` on 500k x 150bp is 0.8 batches, which
  makes the reader/writer pipeline structurally inert and silently removes 8-9% from our arm. That
  mistake has been made three times in this project. A benchmark that touches I/O without printing
  its batch count is measuring a configuration with half the work disabled.

Two scales, for cost: iteration runs on 1-4M reads (minutes, and the ratios are depth-invariant), the
full 30x validates only the number we publish, because SE+PE across three arms at that depth is a day
of machine time.

Method rules, each one paid for by a past error in this repo:

- Arms interleaved within a pass. Runs taken minutes apart are worthless; repeated identical runs
  spread ~2.4%. **Nothing under 3% is called a gain.**
- Every binary and every mapping warmed before timing. Cold-start artifacts have produced a "13.89x
  prefetch speedup" (real value 1.02x) and an "11.76x binning win" (cache warming) in this project.
- Genome index only.
- Our arm is the output of `scripts/pgo.sh`. A `cargo build --release` is ~15% slower and is not what
  we ship, so timing it would flatter the fork by 15%.

Phase 0 has three admissible outcomes and all three are accepted in advance: we already win (phase B
is cancelled, the campaign becomes consolidate-and-publish), we lose (phase B is justified and
targeted by the profile), or we win on time and lose on RSS (effort redirects to memory).

## Phase A — the frames nobody has opened

**A1. Re-derive the profile before touching anything.** Including the claim that motivates A2: the
"`mem_sort_dedup_patch` is 11% of PE" figure in ROADMAP.md may itself be region-inflated. Profile with
`sample` on the genome index, and because LTO inlines aggressively, add explicit `Instant` counters
(the `BWA4_CHAIN_TIME` pattern) to split anything a sampler attributes to one giant frame. A1's output
is a share table; if it disagrees with ROADMAP.md, ROADMAP.md is wrong and gets corrected.

**A2. `mem_sort_dedup_patch`.** The one hot frame that has never been optimised, and the only item on
"what remains" that is pure compute. Scope is set by A1: if it is not >=5% of the genome-scale PE
profile, it is dropped without ceremony.

**A3. The fixed index load, which is the same lever as the RAM.** At `-t8` the load is a fixed cost
both tools pay and it was 29% of the 500k benchmark; on a real WGS it amortises, which is precisely
why it must be measured at 30x rather than argued about. Three candidates, in decreasing confidence:

- *Parallel array copy at load.* The BWT arrays are bulk-copied out of the mapping one at a time.
  Copying them in parallel is a pure win with no identity risk.
- *Reconstruct `.0123` from `.pac` (#177).* Drops a 6.2 GB file from the working set at the cost of
  unpacking 2-bit bases on each reference fetch. This is a **trade**, not a win, and it will be
  measured as one: RSS saved against extension slowdown, with the right to decline it.
- *Zero-copy `cp_occ`.* Explicitly expected to be **declined**, and written down so it is not
  rediscovered: the file offset 48 is not 64-byte aligned, so mapping it in place forfeits the
  one-cache-line-per-lookup guarantee that `#[repr(C, align(64))]` buys, and that guarantee measured
  ~8% of seeding. Trading 8% of the dominant frame for 10 GB is the wrong direction for a speed
  campaign. It is listed here only to close it.

**A4. `index` peak RSS (~75 GB) and time.** The i64 suffix array dominates both. SA-IS is still
single-threaded (Tier B of phase 8c, deferred). This is a self-contained project with its own gate
(`scripts/index_diff.sh`, byte-identical index files), and it is sequenced last within A because it
helps a cost paid once rather than per run.

## Phase B — the fork's seeding, conditional on phase 0

Only if phase 0 shows we are behind in the `-t8` regime. Read the fork's seeding path against ours
line by line and port what is portable, byte-identically.

Scoped by what is already known to be exhausted: its SW kernel fast-paths were audited line by line
and closed (its kernel speed is x86 SIMD width, not a portable technique; our AVX2 path is already at
width parity on x86, NEON caps at 16 lanes in hardware), and `FP_STATUS_TIGHT` was declined with a
reason (our masked SIMD kernel walks every column, so a tighter band shortens nothing).

Phase B carries a real risk of a sixth honest negative. That is an acceptable outcome and it gets
recorded in ROADMAP.md the same way the other five were, so it is never re-instructed.

## Phase C — making `-t8` end-to-end structurally cheaper

Not algorithms: housekeeping that the chosen benchmark rewards. `-K` defaults against real batch
counts, BGZF output, and thread behaviour above `-t8`, which has never been measured on this machine's
16 cores. Cheapest wins per hour, weakest claim, hence last.

## Gates, identical for every change in every phase

- `scripts/oracle_diff.sh`: SE 5000/5000 and PE 10000/10000.
- Real-data parity: 1M reads SE and 1M pairs PE, record counts asserted first, then `cmp`.
- Thread-invariance: `-t1` and `-t8` produce the same bytes.
- Interleaved A/B, >=3% or it did not happen.
- Every negative result written into ROADMAP.md with its measurement. A phase that only records wins
  is a marketing document, and this project's main asset is its list of dead ends.

## Explicit non-goals

GPU/Metal (closed: the kernel is 4% of the time, there is nothing to offload), wider SIMD (NEON caps
at 16 lanes; AVX-512 needs real x86 CI that Rosetta cannot provide), LISA/learned indexes (measured
0.47x here), minimizers or strobes (they break byte-identity, which is the project), and DRAGEN
comparisons (proprietary FPGA hash seeding, not bwa, and not reachable from byte-identity).
