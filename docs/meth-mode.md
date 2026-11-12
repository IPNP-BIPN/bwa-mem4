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
batches by `-t`, and the insert-size model is per batch. EM-seq is identical. TAPS differs on 15
pairs of 200,000, the same 15 before the speed work (commit 9f2a0df). In each, bwa-mem3 rescues
a mate from a read-1 region that bwa-mem4 does not list, on a low-complexity read
(`sim13347` reproduces it alone with `-I 298.36877747837184,49.874227483371115,533,64`).

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
- **Genomic and neutral scoring** (the TAPS and 5-base default) go through a 5×5 table. The NEON
  u8 extension, rescue and band kernels look up `5 × reference + read` with a two-register `TBL`.
- **Rescue pruning** under genomic / neutral counts the K-mer hits of the relation: reference C
  also matches read T (OT), reference G read A (OB). This is bwa-mem3's relation filter. A freed
  cell scores at most the match score, so the bound holds.

A seed that scores 0 or less under the read's matrix (a pair it matched only in converted space)
extends on the scalar kernel, as in bwa-mem3.

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

On x86, the genomic and neutral matrices (TAPS, 5-base) still extend and rescue with the scalar
kernels, and the x86 band kernels score the uniform matrix only. EM-seq (collapsed) uses the AVX2
and AVX-512 kernels. Neither case is measured on x86.

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
- **15 TAPS pairs in 200,000** differ (see [Validation](#validation)). The 5,000-pair test set
  does not hit them.

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
  uses less memory. On x86, TAPS / 5-base (neutral scoring) still runs scalar kernels.

Bismark itself was not run here. The above comes from reading #1209 and the Bismark
documentation.
