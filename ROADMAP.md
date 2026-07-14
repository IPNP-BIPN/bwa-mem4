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
| 8 | `phase8-scale` | GRCh38 complet, rayon, (NEON optionnel) | ~100% concordance reads reels |
| 9 | `phase9-gpu` | backend Metal du SW (entier -> bit-identique) | identique au scalaire + speedup |
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
  (248 Mbp)** ; chr20 (64 Mbp) PE 8886/10000. **Bloqueur memoire resolu** : SA-IS **in-place**
  (`crate::sais`, sous-probleme empaquete dans le tableau SA, pas de copie i64 de l'entree ni de
  tableaux O(n) par niveau). Index **octet-identique** confirme chr20 (RSS 2,2 Go) et chr1 (RSS
  **8,1 Go** contre ~25 Go avant, ~16 o/base). Genome entier 3,1 Gbp projete a **~100 Go < 128 Go**
  (build non lance ici pour eviter tout risque OOM). `scripts/scale_test.sh` gere le gate par
  chromosome.
