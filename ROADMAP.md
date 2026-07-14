# Roadmap

Une phase = une branche. Commits frequents ; merge vers `main` quand le gate de la phase passe.
Cible d'acceptation : index et SAM **octet-identiques** au binaire `bwa-mem2` 2.3 patche (oracle).

| Phase | Branche | But | Gate |
|---|---|---|---|
| 0 | `phase0-skeleton` | CLI `mem`/`index` (index stub), FASTQ -> SAM non-mappe, en-tete correct, harnais | `@SQ` octet-identique + JSON de concordance produit |
| 1 | `phase1-indexer` | `bwa-mem3 index` : `.pac/.ann/.amb/.bwt.2bit.64/.0123` | `cmp` octet vs `bwa-mem2 index` |
| 2 | `phase2-index-load` | chargeur + `get_occ`/`get_sa`/`backward_ext` + validation | assertions + cross-check occ/SMEM |
| 3 | `phase3-seeding` | SMEM + reseed + filtrage occ | seeds == oracle |
| 4 | `phase4-chaining` | `mem_chain` + `mem_chain_flt` | chaines == attendu |
| 5 | `phase5-extension` | SW bande scalaire (`SwBackend`) + `mem_chain2aln` | `AS` + coords == oracle |
| 6 | `phase6-se-sam` | primaire/MAPQ/CIGAR/tags | **SAM SE octet-identique** |
| 7 | `phase7-pe` | `mem_pestat`/`mem_matesw`/`mem_pair` | **SAM PE octet-identique** |
| 8 | `phase8-scale` | GRCh38 complet, rayon, resorption de la traine | ~100% concordance reads reels |
| 9a | `phase9a-neon` | backend NEON du SW derriere `SwBackend` (portage des optim. de @nh13, PR #288) | identique au scalaire + speedup mesure |
| 9b | `phase9b-gpu` | backend Metal du SW (entier -> bit-identique) | identique au scalaire + speedup |
| 10+ | | gate GIAB `hap.py`/`vcfeval` ; packaging | |

Statut : **phases 0-7 terminees, phase 8 quasi terminee**.
- **SE** : ligne entiere **5000/5000 byte-identique** (100%).
- **PE** : **9999/10000** enregistrements byte-identiques ; `mem_pestat` identique bit-a-bit,
  `mem_pair`/flags/TLEN/MC/MAPQ combinee OK. `XA:Z` (`mem_gen_alt`) et mate rescue
  (`mem_matesw`/`ksw_align2`) byte-identiques. **Traine resorbee** via oracle instrumente
  bit-identique (rebuild sse2neon dans `scratchpad`) : (1) ordre des chaines **par position** +
  portage exact de **`ks_introsort`** (tri instable) dans `mem_chain_flt` -> parite des regions
  sous-optimales (340 SE + 622 PE) ; (2) `mapQ_coef_fac` est un **int** dans bwa-mem2
  (`(int)log(50)=3`, pas `3.912`) -> parite MAPQ (23 SE + 6 PE). Reste **1 enregistrement PE**
  (`XS` cosmetique, MAPQ=60 intact) : composition de seeds d'une chaine sous-optimale sur locus
  repetitif -> extension SW differente. Voir `DIVERGENCES.md`.
- **Phase 8a (rayon)** : parallelisation SE+PE, sortie octet-identique quel que soit `-t` (a `-K`
  fixe), ~6.5x sur 8 coeurs. Sur `phase8-scale`.
- **Phase 8b (scaling)** : indexeur + aligneur valides **octet-identiques** jusqu'a **chr1 complet
  (248 Mbp)**. **chr20 complet (64 Mbp)** apres la resorption de la traine : les 5 fichiers d'index
  **octet-identiques**, PE **9994/10000** (99,94 %, contre 8886 avant le fix chaines+MAPQ). Les 6
  residus sont la meme famille (regions sous-optimales sur loci repetitifs) : 3 `XA`/`XS`
  cosmetiques, 1 MAPQ, 1 paire avec placement `POS`/`TLEN` different. **Bloqueur memoire resolu** :
  SA-IS **in-place**
  (`crate::sais`, sous-probleme empaquete dans le tableau SA, pas de copie i64 de l'entree ni de
  tableaux O(n) par niveau). Index **octet-identique** confirme chr20 (RSS 2,2 Go) et chr1 (RSS
  **8,1 Go** contre ~25 Go avant, ~16 o/base). `scripts/scale_test.sh` gere le gate par chromosome.
- **Phase 8b (genome entier)** : **GRCh38 complet (3,1 Gbp, 194 contigs) construit par notre
  indexeur et gate d'octet-identite PASSE** : les 5 fichiers (`.pac`/`.ann`/`.amb`/`.bwt.2bit.64`/
  `.0123`) **octet-identiques** a `bwa-mem2 index`. Pic RSS **~75 Go** (`/usr/bin/time -l`, sous les
  128 Go ; garde memoire jamais declenchee). **Bug trouve par le build genome-entier** : notre
  `.amb` fusionnait les N-runs a travers les frontieres de contigs (telomere N de chr1 + N de chr2
  comptes en un seul trou) ; `add1` de bwa-mem2 reset `lasts` par contig, corrige (`build.rs`,
  teste). chr1/chr20 seuls (contig unique) ne l'exposaient pas.
- **Phase 8c (Tier A, perf indexeur)** : parallelisation rayon du post-traitement FM (BWT, CP_OCC,
  echantillonnage SA, RC du `.0123`) + liberation des buffers d'entree avant le SA. Sortie
  **octet-identique** (verifie tiny + chr20 + genome complet). Genome entier : **741 s -> 518 s
  (~30 %, 12,3 -> 8,6 min)**, pic RSS ~inchange (le tableau SA i64 domine). **SA-IS reste
  mono-thread** (Tier B = SA-IS parallele, reporte).

## Phase 9a : backend NEON (collaboration @nh13, PR bwa-mem2#288)

Le SW scalaire est la source de verite ; NEON est un backend **entier** derriere le trait
`SwBackend`, donc l'octet-identite au scalaire/oracle est **preservee** et testable (property test
`scalaire == NEON` sur batches aleatoires + `oracle_diff.sh` toujours 100%). Optimisations a porter
depuis le fork de Nils Homer (`fg-labs/bwa-mem3`, ~2x plus rapide) et sa PR #288 :

- **`kswv` NEON natif** : Smith-Waterman vectoriel ecrit en NEON (vs traduction sse2neon), ~7 %.
- **`bandedSWA` blendv NEON** : `vbslq` a la place du blendv emule, ~4 %.
- **Tuning Apple Silicon** (hors chemin d'octet-identite, ne touche que perf/threads) : detection
  P-core/E-core via `sysctl`, taille du cache L2 pour dimensionner le batch, alignement lignes de
  cache 128 o (vs 64 o x86), hints QoS coeurs perf, link `Accelerate.framework`.
- **Build** : cible `arch=arm64`, PGO (`pgo-generate`/`pgo-use`), `simd_compat.h` (couche
  d'abstraction SSE/NEON), garde `memset_s` macOS.

**Gate** : sortie octet-identique au backend scalaire (property test + `oracle_diff.sh`) **et**
speedup mesure (`/usr/bin/time`). A faire sur une branche partagee avec @nh13 (acces lecture +
fork/PR accordes sur `IPNP-BIPN/bwa-mem3-rs`). Voir `DEPENDENCIES.md` pour la provenance.
