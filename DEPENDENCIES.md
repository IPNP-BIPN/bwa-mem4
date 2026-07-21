# Dependances

Versions exactes gelees dans `Cargo.lock` (pin de record). Justifications :

| Crate | Usage | Licence |
|---|---|---|
| `clap` | parsing CLI (`bwa-mem4 index`/`mem`) | MIT/Apache-2.0 |
| `thiserror` | type d'erreur de `bwa-core` | MIT/Apache-2.0 |
| `anyhow` | erreurs applicatives (CLI) | MIT/Apache-2.0 |
| `needletail` | lecture FASTQ/FASTA (hot path, analogue `kseq.h`) | MIT |
| `memmap2` | mmap de l'index (phase 2+) | MIT/Apache-2.0 |
| `rayon` | parallelisme (phase 8, differe) | MIT/Apache-2.0 |
| `serde`, `serde_json` | rapport de concordance JSON (`sam-diff`) | MIT/Apache-2.0 |
| `sha2` | shasum de determinisme dans le harnais | MIT/Apache-2.0 |
| `rust-htslib` | sortie BAM/CRAM (`mem -o out.bam` / `out.cram`) | MIT |

Dev-only (jamais dans le binaire livre) : `noodles-*` (validation SAM/BAM cote test), `rust-bio` /
`block-aligner` (bootstrap de seeding/SW).

## Provenance des contributions externes

- **Backend NEON (phase 9a)** : portage des optimisations Apple Silicon de **Nils Homer (@nh13)**,
  depuis sa PR `bwa-mem2/bwa-mem2#288` (fermee) et son fork `fg-labs/bwa-mem3` (licence MIT, comme
  bwa-mem2). Optimisations : `kswv` NEON natif (~7 %), blendv `vbslq` dans `bandedSWA` (~4 %),
  tuning Apple Silicon (P/E-core, cache L2, alignement 128 o, QoS, `Accelerate.framework`), build
  `arch=arm64` + PGO. Collaboration via fork/PR sur `IPNP-BIPN/bwa-mem4-rs` (acces lecture accorde a
  @nh13). Reimplementation en Rust natif (pas de copie de code C++) ; on s'inspire de l'algorithmique
  NEON, le SW scalaire restant la source de verite octet-identique.
  **Livre (phase 9a terminee)** : kernel `bandedSWA` NEON **int16x8** en Rust natif
  (`crates/bwa-neon/src/batched.rs`) reproduisant le layout SoA `[colonne*8 + lane]` et le blendv
  `vbslq` (`NEON_BLENDV` de son `neon_utils.h`), plus le portage de `mem_chain2aln_across_reads_V2`
  (collecte/tri-par-longueur/batch/scatter, `crates/bwa-mem/src/across.rs`). Octet-identique au
  scalaire/oracle (gate property + `oracle_diff` SE 5000/5000, PE 10000/10000), **~1,5x** mono-thread.
  Reliquat (perf pure) : `kswv` NEON natif, int8x16 (16 lanes), tuning P/E-core, PGO.
- **sse2neon** v1.8.0 (licence MIT) : uniquement cote *oracle* (rebuild instrumente bit-identique du
  binaire bwa-mem2 patche pour le diagnostic de parite), jamais dans notre binaire Rust.
