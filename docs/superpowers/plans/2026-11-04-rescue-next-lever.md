# Mate rescue: the next lever, priced before it is built

**Goal:** cut the CPU the paired-end path spends on mate rescue, byte-identical to bwa-mem2.

**Why now.** Publication run of 2026-11-04 (2M real HG002 pairs, GRCh38, `-t12`, non-PGO, median of
5, `work/giab/publication_20261104_211657/runs.tsv`):

| | SE wall / user | PE wall / user | PE minus 2 x SE (user) |
|---|---|---|---|
| bwa-mem2 | 29.0 s / 313 s | 100.4 s / 1168 s | **542 CPU-s** |
| bwa-mem4 | 7.4 s / 82 s | 27.0 s / 312 s | **148 CPU-s (47% of PE CPU)** |

PE user / wall = 11.6 of 12 cores busy, so **this is CPU work, not thread scaling**. The 45% rescue
scaling figure in `docs/perf-levers.md` was taken on wgsim and does not apply here.

## What is already closed (do not retry)

From `docs/perf-levers.md` and `ROADMAP.md`, each measured:

- Kernel arithmetic: 0.98 vector op/cell against a 0.97 machine ceiling. Four-row body, XOR score
  table with N slot, padding-free columns, USQADD, row maxima in one store: all banked.
- Lazy argmax (+18%), skipping the N repair (-7%), row-major `score2` (+7.3%): reverted.
- Length sorting (0.0%), duplicate jobs (0.0%), flattening rounds (lanes already full), chunk size
  (null on real data), window union across anchors (1.03x, 0 groups nested), reverse-pass bound
  (-0.8%), dedup re-sorts (insert_into_settled, banked in 4.4.0).
- K-mer pruning (`rescue_prune.rs`, gate 400 hits): 13.6% of jobs proven failing, -0.3% to -1.6%.
  Raising the gate costs more filter than it saves DP.

## Where the cells still are (hypothesis to verify first)

`rescue_prune.rs:325` says ~80% of real windows are OVER the hit gate and run `Full`. The hull only
narrows the remaining ~20%, and keeps ~80% of their rows. So most of the ~760 G cells per 1M pairs
sit in dense (repetitive) windows that nothing touches today.

## Phase 0: re-measure on the current tree (~20 min, quiet host)

On `work/giab/g2m_{1,2}.fq`, PE `-t12 -K 100000000`, current HEAD:

1. `BWA4_MATESW_TIME=1 BWA4_STAGE_TIME=1`: jobs, cells, Gcell/s, prune `fail / hull / full` with
   rows kept, and the rescue stage wall.
2. `BWA4_NO_RESCUE=1`: the rescue cost measured directly (output changes, measurement only).
3. `sample` on a live run: self-time split kernel / filter / `extract_group` / `bns_fetch_seq` /
   dedup / pairing.

Output: one table splitting the 148 CPU-s. **Stop if the kernel's share of `Full` cells is not the
majority**: the lever is then elsewhere and this plan is re-aimed at it.

## Phase 1: price the component bands offline (no kernel change)

Fork issue #535, named as the remaining lever in `ROADMAP.md`: instead of ONE hull `[hb, he]`, run
the DP on each component band (rows whose row max can reach `minsc`), each zero-started.

Exactness argument already in `rescue_prune.rs` (Decision::Hull): with `KSW_XSUBO`, ksw records a row
max only when it is `>= minsc`, so `score2` reads only such rows; rows between bands have row max
`< minsc` and the prefix before a band nets `<= 0`. Per band, the hull lemma applies unchanged.

Probe (test-only, `prune_bench` style, on a `BWA4_RESCUE_DUMP` capture of real jobs):

- rows kept by bands vs by the hull, on `Hull` jobs;
- cells of `Full` jobs as a share of all cells;
- for `Full` jobs, the same two numbers if the filter ran with no gate (filter cost in us/job too).

**Go / no-go:** bands must remove >= 10% of ALL rescue cells (that is ~3% of PE wall, the project's
noise floor). Below that, stop and record the null in `docs/perf-levers.md`.

## Phase 2: build it (only on a go)

- `rescue_prune.rs`: `Decision::Bands(SmallVec<[(hb, he); 4]>)` alongside `Hull`.
- `matesw.rs`: each band becomes its own lane job in the batch (no kernel change); merge back per
  original job: `score` = max with the earliest row winning ties (ksw keeps strict `>`), `qe` and
  `te` from that band shifted by `hb`, `score2`/`te2` recomputed from the union of the bands' row
  maxima with ksw's exact exclusion rule.
- Behind `BWA4_RESCUE_BANDS=0|1` so A/B runs one binary.

## Phase 3: gates and measurement

- `cargo test -p bwa-mem4-neon`: extend `decisions_reproduce_ksw_align2` and
  `batched_mate_rescue_keeps_every_consumed_field` to `Bands`; plus a mutation check (shrink a band
  by one row: must fail).
- `./scripts/check.sh`, `./scripts/oracle_diff.sh`, x86 kernels via
  `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo test --workspace --target x86_64-apple-darwin --release`.
- Real data: 2M HG002 PE record md5 must stay `42e1a795a5bfa57d744bae5d4f7ec083` (bwa-mem2's, today).
- Speed: kernel CPU via `BWA4_MATESW_TIME` (quiet instrument) and wall, `BWA4_RESCUE_BANDS` 0/1
  alternated, >= 4 reps, PGO binary (what ships on macOS arm64). Record the result, null or not.
