# Methylation mode (`--meth`)

bwa-mem4 aligns directional bisulfite, EM-seq, TAPS and Illumina 5-base reads with the
methylation mode of [fg-labs/bwa-mem3](https://github.com/fg-labs/bwa-mem3) 0.16, ported to
Rust. The options are spelled as in bwa-mem3, so a bwa-mem3 command line runs unchanged. On the
test set, the SAM output is byte-identical to bwa-mem3 0.16 (see [Validation](#validation)).

The port is a Rust reimplementation of bwa-mem3's `meth_*.cpp`, `bwamem.cpp` and
`bwamem_pair.cpp` meth paths, Copyright (C) 2026 Fulcrum Genomics LLC, MIT licensed. The notice
is kept in [`LICENSE`](../LICENSE).

Without `--meth`, nothing changes. Not one byte of the output differs from what it was before the
port: `scripts/oracle_diff.sh` still matches bwa-mem2 on every record, and so do the usual PE
md5s at `-t1` and `-t4`.

## Usage

```bash
bwa-mem4 index --meth genome.fa                         # once
bwa-mem4 mem --meth genome.fa R1.fq.gz R2.fq.gz > out.sam
```

| Library | Command |
|---|---|
| Bisulfite, EM-seq (unmethylated C reads as T) | `mem --meth` (= `--meth=emseq`, aliases `em-seq`, `bisulfite`) |
| TAPS (methylated C reads as T) | `mem --meth=taps` |
| Illumina 5-base | `mem --meth=5base` (alias `5-base`; TAPS chemistry) |

The chemistry binds only with `=`, as in bwa-mem3. In `--meth taps`, `taps` would be read as the
index.

The library must be directional: read 1 (or a single-end read) carries C→T and comes from the
original top (OT) or original bottom (OB) strand, and read 2 carries G→A. Non-directional
libraries (CTOT/CTOB) and PBAT are not aligned as such.

### Index

`index --meth genome.fa` writes the plain index of `genome.fa` and an index of the converted
genome:

| File | Content |
|---|---|
| `genome.fa.{amb,ann,pac,bwt.2bit.64}` | the plain index (RNAME, `@SQ`, `MD`, XM context) |
| `genome.fa.meth.fa` | for each contig, `>r<name>` (G→A) then `>f<name>` (C→T), upper-cased, 100-base lines |
| `genome.fa.meth.{amb,ann,pac,bwt.2bit.64}` | the index of `genome.fa.meth.fa` (seeding) |

After indexing, `genome.fa.meth.fa` and `genome.fa.bwt.2bit.64` are not read and can be
deleted. `mem` needs `genome.fa.{amb,ann,pac}` and `genome.fa.meth.{amb,ann,pac,bwt.2bit.64}`.
You may also pass the `.meth` prefix itself as the index.

These are the same files bwa-mem3 0.16 writes. They are byte-identical except for the 8-byte
footer that bwa-mem3 appends to `.bwt.2bit.64` and the format does not need. Each tool loads the
other's index and gives the same SAM. So an index built by `bwa-mem3 index --meth` (for example by
nf-core's `bwamem3/index`) works as is.

### Options

| Option | Default | Effect |
|---|---|---|
| `--meth[=CHEM]` | off; `emseq` when given bare | turns the mode on: `emseq`, `taps` or `5base` |
| `--meth-scoring MODE` | `collapsed` (EM-seq), `neutral` (TAPS, 5-base) | how a conversion scores: `collapsed` treats C/T (G/A) as a match in both directions, `genomic` treats it as a match only in the conversion's direction, `neutral` scores it 0 |
| `--meth-context all\|cpg` | `all` | where a conversion is free. Only `all` is supported (see [Differences](#differences-from-bwa-mem3-016)) |
| `--meth-tags SPEC` | `all` | which of `XR`, `XG`, `XM` to write: `all`, `none`, a list `XR,XG`, or exclusions `^XM` / `-XM`. A tag left out is not computed |
| `--meth-seed-prune[=MODE]` | `spec30` | drops short repetitive SMEMs before SA lookup: `spec30`, `baseline` or `off` |
| `--set-as-failed f\|r` | off | sets 0x200 on records of conversion strand `f` (`XG:Z:CT`) or `r` (`XG:Z:GA`) |
| `--chimera-qc` | off | bwameth's rule: when the longest `M` covers less than 44% of the read, sets 0x200, clears 0x2 and caps MAPQ at 1 |

`--meth` also changes these defaults, as bwameth does. A value you give yourself always wins.

| | Default under `--meth` |
|---|---|
| `-L` | `10a,10a` |
| `-U` | `100a` |
| `-T` | `40a` |
| `-M` | on |
| `-B` | `2a` under `collapsed` scoring |

Here `a` is the match score `-A`. The mismatch penalty must stay positive. `-x` presets are
refused with `--meth`, and any `--meth-*` option given without `--meth` is an error.

## How it aligns

1. **Seeding.** The read is converted (C→T for read 1 or single-end, G→A for read 2). Its SMEMs
   are then searched in the converted index, which holds both converted copies of every contig.
   With `spec30`, the SMEMs are pruned after the occurrence sort: thresholds T=30, L=25, U=22, and
   the first SMEM of maximal length is always kept. Each hit is mapped back to the original contig.
   A hit is dropped when it falls on the copy that does not match the read's chemistry, or when it
   runs off its copy.
2. **Chaining and extension** are bwa-mem's, against the original genome. Only the scoring
   matrix changes. Read 1 / single-end uses the OT matrix (a reference C read as T is free) and
   read 2 the OB matrix (G read as A). Mate rescue scores the mate with the mate's own matrix.
3. **NM** counts only the columns the matrix penalises, so a conversion is not an edit. **MD**
   stays literal: like Bismark's, it lists every converted base.
4. **Paired MAPQ** caps the score gap with the secondary scores of both ends, as bwa-mem3 does:
   `min(o - subo, o + 4a - (s1 + s2))`.

## Output

The header starts with `@HD VN:1.5 SO:unsorted GO:query`, unless `-H` gives an `@HD`. It
continues with the `@SQ` lines of the original contigs, the same names and lengths as
`genome.fa`. The `@PG` line is `ID:bwa-mem4`.

SEQ is the read as sequenced, not the converted one: C and T show as read, reverse-complemented
for a reverse-strand record. QUAL is untouched.

Optional tags come in bwa-mem3's order: `NM MD MC MQ AS XS RG SA pa XA HN XR XG XM`. After these
come the FASTQ comment's SAM tags.

| Tag | Written on | Meaning |
|---|---|---|
| `MQ:i` | records with a mate | the mate's MAPQ |
| `HN:i` | mapped records, not under `-a` | number of hits in the read's group |
| `XR:Z` | every record | read conversion: `CT` for read 1 / single-end, `GA` for read 2 |
| `XG:Z` | mapped records | genome strand: `CT` aligned as original top (OT), `GA` as original bottom (OB) |
| `XM:Z` | mapped records with SEQ | Bismark's call string, one letter per SEQ base, described below |

`XM:Z` uses Bismark's letters: `z`/`Z` for CpG, `x`/`X` for CHG, `h`/`H` for CHH, and `u`/`U`
for an unknown context (an N within the two following bases). Upper case means methylated.

- A letter is written only where the reference has a C (G on the bottom strand) and the read has
  C or T (G or A).
- Every other base gets `.`, including SNPs, insertions and soft clips.
- The context comes from the original reference. Bases outside the contig and ambiguous runs
  count as N.
- Which read base means methylated depends on the chemistry. Under EM-seq a kept C is
  methylated. Under TAPS and 5-base a C read as T is methylated, so the same read gets opposite
  case.

Under `--meth`, the SAM-shaped tokens of the FASTQ comment are appended as tags (`TAG:T:value`,
types `A i f Z H`; others are dropped). This happens without `-C`, as bwa-mem3's htslib writer
does. QC-fail (0x200) set on one record of a read or pair is spread to all its records, and 0x2
is cleared.

## Validation

`scripts/meth_oracle_diff.sh` builds a small reference, then compares bwa-mem4 against the
bwa-mem3 0.16 binary. The reference is chr20:44.0-44.4M and chr22:20.0-20.2M of GRCh38, with an
injected 30-N run and a 200-base soft-masked run, plus chrM.

It indexes the reference with both tools and checks that the index files are identical. It then
simulates 5,000 EM-seq and 5,000 TAPS pairs with `scripts/meth_sim.py`. The simulation includes:

- 0.5% sequencing errors and shared SNPs;
- 1-3 bp indels;
- chimeric read 1s;
- reads carrying N;
- random pairs;
- FASTQ comments.

Fifteen configurations are run, and each must give the same SAM, `@PG` excepted:

| Configuration | Options |
|---|---|
| Bismark | the exact options of Bismark#1209 |
| EM-seq, TAPS | PE and SE each |
| scoring | `genomic`, `neutral` |
| tags | `^XM`, `XR,XG` |
| seed prune | `off`, `baseline` |
| QC | `--set-as-failed r --chimera-qc` |
| other | `-t4`, `-A2`, `-a`, `-Y` |

```bash
FORK=/path/to/bwa-mem3 GENOME=/path/to/GRCh38.fa scripts/meth_oracle_diff.sh
```

Current result: `PASS: every configuration identical`.

The 200,000-pair sets of [Speed](#speed) are also compared, at `-K 10000000`. bwa-mem3 sizes its
batches by `-t`, and the insert-size model is per batch. Both sets are identical.

The script needs the bwa-mem3 binary and a GRCh38 FASTA, so it does not run in CI. CI runs the
unit tests:

- read conversion and the converted FASTA (`bwa-core`);
- the mapping back from the converted index and the chemistry filter (`bwa-chain`);
- XM calling over context, strand, chemistry, clips and indels (`bwa-mem`);
- comment re-encoding, `%g` formatting and the spreading of QC-fail.

## Speed

`--meth` runs the plain pipeline's batched stages: lockstep seeding and SA lookup, batched SIMD
extension, and batched mate rescue with K-mer pruning and the banded kernel. The meth matrices
reach those kernels in three ways:

- **Collapsed scoring** (the EM-seq default) frees C/T in both directions at the match score. It
  is exactly the uniform matrix on copies of both sequences with T rewritten as C (A as G for OB),
  so extension and rescue run the plain kernels on those copies.
- **Genomic and neutral scoring** (the TAPS and 5-base default) go through a 5×5 table. The
  extension and band kernels look up `5 × reference + read`: a two-register `TBL` on NEON, two
  `PSHUFB` on SSE4.1, AVX2 and AVX-512 (one `VPERMW` in the AVX-512 i16 extension kernel). The
  NEON rescue kernels do the same. The x86 rescue kernels give each base a 4-bit code such that
  `(reference code + read code) mod 16` indexes the matrix, so one `PSHUFB` reads it. Such codes
  exist for every `--meth` matrix; codes combined by XOR do not. With VBMI (Ice Lake and later,
  Zen 4 and later) the AVX-512 rescue kernel uses one `VPERMB` at `5 × reference + read`.
  Every kernel has an i16 form, which reads of more than 250 bp need.
- **Extension on the u8 kernel** takes a job whose query and score ceiling fit a byte, whatever
  its target length (up to the i16 kernel's 32,767): the row quantities live in 16-bit lanes. At
  250 bp most extension targets (the rest of the read plus the band) are longer than 255 bases,
  and took the 8-lane i16 kernel until 818ddf9 (M4 250 bp: EM-seq -13% cycles, TAPS -11%).
  bwa-mem3's 8-bit kernel goes to 1,088 bases the same way. Under the 5×5 table (TAPS) the u8
  batch is ordered by band class before it is cut into vectors, as the standard matrix's already
  was, so a vector's narrow bands do not run its widest lane's columns (M4 TAPS 250 bp: executed
  cells 1.62× the bands' → 1.37×, -3% instructions; 150 bp -1%).
- **Mate rescue on the u8 kernel** takes a job whenever its score ceiling plus the table's bias
  fits a byte, as bwa-mem3 0.16 does (`matesw_use_u8`). At the default scoring that includes the
  250 bp mate, which bwa sends to its i16 kernel. The query padding stays bwa's, so the output
  does not change.
- **Rescue pruning** under genomic / neutral counts the K-mer hits of the relation: reference C
  also matches read T (OT), reference G read A (OB). This is bwa-mem3's relation filter. A freed
  cell scores at most the match score, so the bound holds. The relation count runs on NEON and on
  SSE4.1 (`rescue_prune::count_hits`). The count does not shrink with the lanes and the full SW
  does, so x86 prunes only from a mean mate length of 200 bp per 32 lanes: 200 bp on AVX2, 400 bp
  on AVX-512 (TAPS `-t1`, ours / fork: EPYC 7763 150 bp 0.92 pruned against 0.83 unpruned, 250 bp
  0.80 against 0.82; EPYC 9V74 150 bp 0.91 against 0.81, and with the gate below still 0.82 against
  0.75 at c3c92a8). NEON prunes at every length.
  `BWA4_RESCUE_PRUNE=1` prunes anyway. bwa-mem3 does not prune `--meth` on x86. A window
  sharing more hits with its mate than the gate runs in full (its band would be the whole window).
  The plain gate is 1000 hits; relation hits are ~3x denser, so under a relation it is 3 per mate
  base (M4 TAPS 150 bp: -6% instructions, -5% cycles; 250 bp: -0.4% instructions).
- **EM-seq is not pruned**: its collapsed mismatch (2a) is below the bound's (K-1)a for any useful
  K. A better bound would not pay either: on 20,000 simulated chr21 pairs, 70% of the rescue jobs
  reach `minsc`, and in those windows 65% of the rows do (the three-letter alphabet is
  low-complexity). An exact, free filter would still have to run 57% of the cells.
- **The rescue's start pass** (the reverse DP that finds where the alignment begins) runs on the
  band kernel, around the end's diagonal, whenever that kernel can take the batch, pruned or not:
  EM-seq and x86 below the prune's length band it too (M4 EM-seq 150 bp: -6% cycles), as bwa-mem3
  does. A batch with a mate whose score could pass 255 (past ~250 bp) takes the forward kernel,
  since the band stops at the score as a byte. `BWA4_RESCUE_REVBAND=0` bands it only where the
  prune runs, for a same-host A/B.

A seed's starting score sums the read's matrix over the seed, 16 bases at a time (`TBL` on NEON,
two `PSHUFB` on x86); the scalar sum was 1.4% of an M4 EM-seq run (now -1.7% cycles at 150 bp).
A seed that scores 0 or less (a pair it matched only in converted space) extends on the scalar
kernel, as in bwa-mem3, unless its first cell cannot go positive, which is always the case at
match score 1. Then the DP stops at its first row having consumed nothing, so its result is
written directly. TAPS has about one such seed per read (M4: 250 bp -2.3% instructions, 150 bp
-1.3%).

200,000 simulated pairs on chr20 + chr22 + chrM, Apple M4 Max, `/usr/bin/time -l`:

Each cell is bwa-mem4 / bwa-mem3 0.16, the faster of two runs of each.

| Set | Wall (s) | User CPU (s) | Max RSS (MB) | Cycles (G) |
|---|---|---|---|---|
| EM-seq, `-t1` | 14.5 / 17.1 | 14.4 / 16.9 | 952 / 1061 | 56.3 / 65.9 |
| TAPS, `-t1` | 12.1 / 14.9 | 12.1 / 14.8 | 921 / 1037 | 47.3 / 57.7 |
| EM-seq, `-t12` | 1.39 / 1.89 | 15.6 / 18.5 | 1825 / 2126 | 60.1 / 68.9 |
| TAPS, `-t12` | 1.21 / 1.70 | 13.6 / 16.6 | 1771 / 2087 | 51.2 / 60.9 |

The machine was loaded (load average 6 to 9) during these runs. Cycles and instructions are the
steadier comparison.

100,000 pairs simulated from chr21 at 150 and 250 bp, same machine (commit c3c92a8, `-t1`, the
faster of three interleaved runs of each). Cycles (G), bwa-mem4 / bwa-mem3, then the ratio; load
moves wall times by up to 2x, cycles much less. The SAM output was identical. Max RSS is 0.92 to
0.97 of bwa-mem3's, except TAPS 150 bp, where both sit at 490-510 MB (run-to-run noise ±10 MB):

| Set | 150 bp | 250 bp |
|---|---|---|
| EM-seq | 17.2 / 26.3 (0.65) | 24.1 / 45.0 (0.54) |
| TAPS | 15.7 / 24.0 (0.66) | 20.5 / 38.0 (0.54) |

At `-t10` the wall times are, at 150 and 250 bp: EM-seq 0.59 / 0.90 s (0.66) and 1.52 / 2.93 s
(0.52), TAPS 0.54 / 0.84 s (0.64) and 0.74 / 1.29 s (0.57). The machine was loaded (load average
20 to 33); the cycle ratios are 0.55 to 0.68.

For scale, here is bwa-meth's route on the same pairs. bwa-mem2 aligns the converted reads (read 1
C>T, read 2 G>A) to the converted genome with bwa-meth's options, `-T 40 -B 2 -L 10 -M -U 100`.
bwa-meth's Python pass over its output is not counted. Homebrew's bwa-mem2 2.2.1 is not a
NEON-native build. Wall time, bwa-mem2 / bwa-mem4 `--meth`, the faster of two runs:

| Set | `-t1` | `-t10` |
|---|---|---|
| EM-seq 150 bp | 42.3 / 4.16 s (10.2x) | 5.51 / 0.52 s (10.6x) |
| EM-seq 250 bp | 78.5 / 6.04 s (13.0x) | 9.84 / 0.74 s (13.3x) |
| TAPS 150 bp | 42.0 / 4.03 s (10.4x) | 5.48 / 0.47 s (11.7x) |
| TAPS 250 bp | 88.9 / 5.20 s (17.1x) | 12.5 / 0.82 s (15.2x) |

On x86, `.github/workflows/perf-profile-x86.yml` with `read_set=meth` (150 bp) or `meth:250` times
both binaries on 200,000 pairs simulated from chr21, on the runner's CPU. Each cell is the wall-time
ratio bwa-mem4 / bwa-mem3 0.16, the faster of two runs of each, both binaries in the same job. A
range covers several jobs that drew the same CPU. Each row gives the newest commit that drew that
CPU; since 2b8f138 only the pruned path changed on x86 (AVX2 from 200 bp), so 2b8f138 and c3c92a8
rows time the same code elsewhere. The SAM output was byte-identical in every job.

| CPU (rescue kernel) | Commit | Reads | Jobs | EM-seq `-t1` | EM-seq `-t4` | TAPS `-t1` | TAPS `-t4` |
|---|---|---|---|---|---|---|---|
| EPYC 7763, Zen 3 (AVX2) | 2b8f138 | 150 bp | 1 | 0.747 | 0.746 | 0.762 | 0.769 |
| EPYC 7763, Zen 3 (AVX2) | 2b8f138, c3c92a8 | 250 bp | 2 | 0.683-0.696 | 0.672-0.680 | 0.697-0.703 | 0.675 |
| EPYC 9V74, Zen 4 (AVX-512, VBMI) | 2b8f138, c3c92a8 | 150 bp | 3 | 0.721-0.735 | 0.765-0.770 | 0.739-0.747 | 0.779-0.782 |
| EPYC 9V74, Zen 4 (AVX-512, VBMI) | 2b8f138 | 250 bp | 1 | 0.667 | 0.695 | 0.669 | 0.710 |
| Xeon 6975P-C (AVX-512, VBMI) | 3fba29d | 150 bp | 1 | 0.809 | 0.850 | 0.810 | 0.845 |
| Xeon Platinum 8573C (AVX-512, VBMI) | 46621ea | 250 bp | 1 | 0.855 | 0.867 | 0.824 | 0.857 |
| EPYC 9V45, Zen 5 (AVX-512, VBMI) | 0b328ab | 250 bp | 1 | 0.890 | 0.885 | 0.868 | 0.854 |

The user + sys CPU ratios are within 1.5% of the wall ratios. In the 2b8f138 and c3c92a8 jobs the
max RSS ratio is 0.89 to 0.98 at `-t1` and 0.60 to 0.77 at `-t4`. The A/B arms of those jobs moved
nothing worth keeping: `BWA4_LOCKSTEP_N` 12 or 16 stays within 1.5% of the probe's pick on the 7763,
the 9V74 and the 6975P-C, and `BWA4_RESCUE_PRUNE=1` at 150 bp on the 9V74 costs TAPS 7 points
(0.746 to 0.818). On the 9V45 the run-to-run noise is about 4%: over five jobs on the same EM-seq
code, its `-t1` ratio was 0.94 to 1.00. On that CPU, bwa-mem4's seeding takes
more perf samples than bwa-mem3's; extension makes up for it.

## Differences from bwa-mem3 0.16

None of these differences shows up on the test set.

- **`--meth-context cpg` is refused.** The CpG-only scoring model (`meth_model.cpp`) is not
  ported, because Bismark runs `all`.
- **The `BWAMEM3_METH_*` environment variables are ignored.** These are `SEED_PRUNE`,
  `SEED_FILTER`, `PRUNE_T/L/U/KB` and `BATCHED_RESCUE`. Use `--meth-seed-prune`. bwa-mem4 behaves
  like bwa-mem3 with the variables unset.
- **There is no second `@PG ID:bwa-mem3-meth` line.** The `@PG` names bwa-mem4 (see
  `DIVERGENCES.md`). The test strips `@PG`.
- **SEQ letters other than `ACGTN`** (IUPAC codes in the FASTQ) print as `N`. bwa-mem3 prints the
  original letter, upper-cased.
- **`pa:f`**, written only with ALT contigs, uses bwa-mem4's float formatting.
- **`-x` presets** cannot be combined with `--meth`.
- **bwa-mem3's `--bam` option** (uncompressed BAM output) is not available. Use `-o out.bam` in a
  build with the `hts-output` feature, or pipe the SAM into samtools.

## What Bismark needs

Bismark's bwa-mem3 engine ([FelixKrueger/Bismark#1209](https://github.com/FelixKrueger/Bismark/pull/1209),
`--illumina_5base --bwamem3`) relies on the following. This is what each point means for
bwa-mem4:

- **Index.** Bismark checks `.amb`, `.ann` and `.pac`, and that the contig names and lengths match
  the genome folder. `bwa-mem4 index --meth` writes the same files, and bwa-mem4 also reads an
  index built by bwa-mem3. Nothing to change.
- **Command line.** Bismark runs `mem --meth=5base --meth-scoring neutral --meth-context all
  --meth-seed-prune=spec30 --meth-tags none -K 100000000 -t N <prefix> R1 R2`, and reads SAM on
  stdout. bwa-mem4 accepts this argv as is, and its output is byte-identical to bwa-mem3's
  (`bismark` configuration above). Bismark also clears `BWAMEM3_METH_SEED_PRUNE`, which bwa-mem4
  ignores anyway.
- **Version gate.** Bismark runs `<binary> version` and reads stdout, rejecting anything below
  0.16.0 and warning on anything else. `bwa-mem4 version` prints `4.4.1` and exits 0. So the gate
  passes with a "not the validated release" warning. A bwa-mem4 engine would need its own
  minimum, the first bwa-mem4 release with `--meth`.
- **Binary name and `@PG`.** The engine is named `bwa-mem3`, and Bismark writes
  `@PG ID:bwa-mem3 PN:bwa-mem3 VN:<detected>` into its BAM. Running bwa-mem4 through it works if
  the path given to Bismark points at the bwa-mem4 binary. Recording the right provenance needs a
  small Bismark change: an engine alias such as `--bwamem4`, or the `PN` taken from the version
  probe.
- **Header and records.** Bismark checks the `@SQ` lines against the genome folder. It needs 0x2
  on both mates, and it reads RNAME, POS, FLAG and CIGAR. It computes its own XM, which is why it
  passes `--meth-tags none`. All of these are as in bwa-mem3.
- **Speed.** See [Speed](#speed): on Apple Silicon bwa-mem4 is faster than bwa-mem3 0.16 and
  uses less memory. On x86 see the per-CPU table there.

Bismark itself was not run here. The above comes from reading #1209 and the Bismark
documentation.
