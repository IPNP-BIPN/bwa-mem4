# Dependances

Versions exactes gelees dans `Cargo.lock` (pin de record). Justifications :

| Crate | Usage | Licence |
|---|---|---|
| `clap` | parsing CLI (`bwa-mem3 index`/`mem`) | MIT/Apache-2.0 |
| `thiserror` | type d'erreur de `bwa-core` | MIT/Apache-2.0 |
| `anyhow` | erreurs applicatives (CLI) | MIT/Apache-2.0 |
| `needletail` | lecture FASTQ/FASTA (hot path, analogue `kseq.h`) | MIT |
| `memmap2` | mmap de l'index (phase 2+) | MIT/Apache-2.0 |
| `rayon` | parallelisme (phase 8, differe) | MIT/Apache-2.0 |
| `serde`, `serde_json` | rapport de concordance JSON (`sam-diff`) | MIT/Apache-2.0 |
| `sha2` | shasum de determinisme dans le harnais | MIT/Apache-2.0 |

Dev-only (jamais dans le binaire livre) : `noodles-*` (validation SAM/BAM cote test), `rust-bio` /
`block-aligner` (bootstrap de seeding/SW).

## Provenance des contributions externes

- **Backend NEON (phase 9a)** : portage des optimisations Apple Silicon de **Nils Homer (@nh13)**,
  depuis sa PR `bwa-mem2/bwa-mem2#288` (fermee) et son fork `fg-labs/bwa-mem3` (licence MIT, comme
  bwa-mem2). Optimisations : `kswv` NEON natif (~7 %), blendv `vbslq` dans `bandedSWA` (~4 %),
  tuning Apple Silicon (P/E-core, cache L2, alignement 128 o, QoS, `Accelerate.framework`), build
  `arch=arm64` + PGO. Collaboration via fork/PR sur `IPNP-BIPN/bwa-mem3-rs` (acces lecture accorde a
  @nh13). Reimplementation en Rust natif (pas de copie de code C++) ; on s'inspire de l'algorithmique
  NEON, le SW scalaire restant la source de verite octet-identique.
- **sse2neon** v1.8.0 (licence MIT) : uniquement cote *oracle* (rebuild instrumente bit-identique du
  binaire bwa-mem2 patche pour le diagnostic de parite), jamais dans notre binaire Rust.
