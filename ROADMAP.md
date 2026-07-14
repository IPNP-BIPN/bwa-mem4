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
| 9a | `phase9a-neon` | backend NEON du SW derriere `SwBackend` (portage des optim. de @nh13, PR #288) | **FAIT** : identique au scalaire + ~1,5x mesure |
| 9b | `phase9b-gpu` | backend Metal du SW (entier -> bit-identique) | identique au scalaire + speedup |
| 10+ | | gate GIAB `hap.py`/`vcfeval` ; packaging | |

Statut : **phases 0-7 terminees, phase 8 quasi terminee, phase 9a terminee**.
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

**Avancement** :
- **Fondations** (fait) : API `extend_batch` sur `SwBackend` (batch inter-sequences comme
  `bandedSWA`), gate de parite batch (`assert_backend_batch_matches_scalar`, 200 rounds), crate
  `bwa-neon` + `NeonBackend`.
- **Step 2b-i** (fait) : squelette DP **par lots** (`crates/bwa-neon/src/batched.rs`) : boucle
  ligne-cible + boucle query partagees sur la bande-union, chaque lane masquee sur sa propre bande
  et sa terminaison (ligne nulle / z-drop). Le **flot de controle divergent par lane** (le plus
  dur) est porte et **octet-identique** a `ksw_extend2` ; arithmetique des cellules encore scalaire.
- **Step 2b-ii** (fait) : kernel NEON **int16x8** (`crates/bwa-neon/src/batched.rs`, `#[cfg(aarch64)]`)
  suivant `bandedSWA` et `neon_utils.h` de @nh13 : layout SoA `[colonne*8 + lane]`, blendv `vbslq`
  pour le masque de bande par lane, 8 lanes int16 par registre. Recurrence en `vaddq_s16`/`vsubq_s16`
  **non saturants**, exacte car les valeurs `H`/`E`/`F` d'une extension locale sont bornees bien en
  deca de int16 (le kernel de @nh13 obtient la meme garantie via ops saturantes + binning
  `MAX_SEQ_LEN16`) ; garde `fits_i16` -> repli scalaire sinon. La boucle interne **sans branche**
  (masque de bande vectoriel + gather de score depuis une query paddee) fait passer le gain de 0,5x
  (squelette scalaire) a **1,37x** (longueurs melangees) et **2,16x** (uniformes 150 pb) sur
  `bench_batch`. Octet-identique : gate partagee + test property taille-read (qlen/tlen ~260).
- **Step 3** (fait) : portage de **`mem_chain2aln_across_reads_V2`** (`crates/bwa-mem/src/across.rs`,
  `align_reads_batched`) : collecte des extensions gauche/droite de **tous les reads** du batch, tri
  par longueur (packing des lanes SIMD), passage par le `SwBackend` NEON, scatter avec la logique
  d'acceptation exacte `MAX_BAND_TRY`. Chaque region ne dependant que de ses propres entrees, le
  batching est **preservant-resultat** (test d'equivalence batched == per-read sur 400 reads varies).
  Drivers SE/PE cables (chunk par pool rayon, invariant au nombre de threads). Verifie bout-en-bout :
  `oracle_diff` **SE 5000/5000** et **PE 10000/10000** `all_fields_match`, `-t1 == -t4`, **~1,5x**
  wall-clock mono-thread (SE 0,25 -> 0,17 s, PE 0,50 -> 0,33 s).

- **Step 2b-iii** (fait) : kernel **int8x16** (16 lanes) en plus de l'int16x8, **genere par une seule
  macro** (les deux largeurs ne peuvent pas diverger), avec binning par longueur facon
  `MAX_SEQ_LEN8`/`MAX_SEQ_LEN16` : int8 si borne < 120 (et qlen,tlen < 120), sinon int16 (< 30000),
  sinon scalaire ; fast-path homogene (pas de gather/scatter si tout le batch tombe dans un bin).
  Octet-identique (gate + test taille-read + `oracle_diff` SE 5000/5000, PE 10000/10000). Kernel
  `bench_batch` : **1,60x** (longueurs melangees, gain int8) et **2,16x** (uniformes 150 pb).

**Phase 9a terminee.** Gate rempli : octet-identique au scalaire/oracle **et** speedup mesure.

**Head-to-head M4 Max, `-t1`, 500k reads (region 2 Mbp), meme sortie octet-identique a l'oracle**
(mediane de 3) :

| Binaire | vs bwa-mem2 (mediane, ratio) |
|---|---|
| `bwa-mem2` 2.3 (sse2neon, oracle) | 1,00x |
| **bwa-mem3-rs (nous)** | **~2,2x** |
| `fg-labs/bwa-mem3` (fork @nh13, natif) | ~2,65x |

(Ratios ; les temps absolus derivent avec le thermique. Depart de la phase perf : nous 1,48x, ~15,9 s.)

**Optimisations perf (phase 9a-perf, toutes octet-identiques, verifiees `oracle_diff` SE 5000/5000 +
PE 10000/10000)** :
- **Score DNA en compare vectoriel** (plus de gather par cellule) : `score = N ? npen : (t==q ? a : mm)`
  via `vceqq`/`vbslq`, comme `bandedSWA`. Kernel 1,6->2,5x (melange), 2,2->3,4x (uniforme). Le plus gros gain.
- **`CpOcc` cache-line** : cp_count+one_hot interleaves en un enregistrement 64 o `#[repr(C, align(64))]`
  (comme le `CP_OCC` de bwa-mem2) -> 1 ligne de cache par lookup occ au lieu de 2. ~8 % (seeding).
- **mimalloc** (allocateur global), **`target-cpu=native`**, `backward_ext` (bloc unique), precompute
  des codes query/target du kernel.

Nous sommes passes de **1,48x a ~2,2x** vs bwa-mem2 ; l'ecart au fork de @nh13 est tombe de **1,74x a
~1,19x**. **Il reste devant** (nous ne l'avons pas battu). Le goulot alterne extension DP / seeding
FM-index selon le profil. Voir `DEPENDENCIES.md`.

## Phase 9c : recurrence bandedSWA (H) + evaluation du reliquat perf

Branche `phase9c-bandedswa`. Le « reliquat » du ROADMAP a ete instruit **avec mesure/profilage**, pas
par speculation. Resultats :

- **Recurrence bandedSWA (fait, commite, oracle-identique).** `ksw_extend2`, la reference scalaire
  batchee et le kernel NEON ouvrent desormais les gaps depuis **H = max(M, E, F)** (et non depuis M),
  soit la recurrence exacte de `MAIN_CODE16_CORE` de bwa-mem2/bandedSWA que le fork `fg-labs/bwa-mem3`
  documente comme byte-identique a bwa-mem2. Sur donnees reelles H == M a l'alignement, donc sortie
  inchangee : **oracle SE 5000/5000 + PE 10000/10000**, property test (NEON == scalaire) vert, tous les
  consommateurs de `ksw_extend2` (CIGAR, extension) passent. **Durcissement** : si un read reel touchait
  un cas H != M, l'ancien code M pouvait diverger de l'oracle ; le nouveau code H ne le peut pas.
  **Pas de speedup** (changement de fidelite/robustesse, pas de perf).

- **Kernel d'extension : pas de gain gate-safe (mesure).** La boucle interne est **latency-bound sur la
  chaine portee `f -> h -> f`** de la recurrence affine, pas ALU-bound. Preuves : (1) le **score-prepass**
  facon `SBT_PREPASS` (calcul du score hors de la boucle DP) **regresse a 0,90x** (le round-trip store/load
  `sbt` par cellule coute plus que recalculer le score en place, que l'OoO masque deja) ; (2) retrait de
  masques ALU redondants **perf-neutre**. Le fork partage la **meme** recurrence latency-bound
  (`MAIN_CODE16_CORE` a la meme chaine `f11 -> h11 -> f21`), donc son avance ne vient pas du kernel.

- **occ FM-index vectorise : NEGATIF (mesure).** Popcount NEON 4-bases (`vcntq_u8` + reduction pairwise)
  **0,89x** vs 4 `count_ones()` scalaires : LLVM compile deja `count_ones` en `cnt`/`addv` optimal, la
  chaine de reduction u8->u64 coute plus. Gain reserve au `GET_OCC` large d'AVX2, pas a la granularite
  u64 sur Apple Silicon.

- **kswv NEON mate-rescue (PE) : SANS OBJET (profil).** Le mate-rescue est saute sur les paires
  concordantes ; `ksw_align2`/`matesw` **n'apparait pas** au profil de la run PE. Zero gain mesurable.

- **minibwa (`nh13/minibwa`) n'est PAS une cible valide.** Il est rapide (2x bwa-mem2) parce qu'il
  utilise l'algo d'alignement de **minimap2 (`ksw2_extd2`)** + heuristiques (`skip mate rescue`,
  `reduced effort in repetitive regions`) : « comparable accuracy », **sortie differente** de bwa-mem2.
  Incompatible avec notre gate d'octet-identite.

**Conclusion.** Le kernel bandedSWA est proche de sa limite (latency-bound) et le fort byte-identique le
partage : son avance ~1,19x vit **ailleurs** que le kernel, dans le **prefetch de seeding** et le **skip
de travail prouve-inutile** (levier cite par minibwa), a resultat preserve.

## Phase 9d : prefetch de seeding (FM-index)

Branche `phase9d-seeding`. Port de `ENABLE_PREFETCH` de bwa-mem2/nh13 : dans la boucle de recherche
**backward** des SMEM, apres avoir garde un intervalle, `prfm pldl1keep` sur les deux blocs checkpoint
que le `backward_ext` de l'etape suivante lira (`cp_occ[k>>6]`, `cp_occ[(k+s)>>6]`), **une iteration
externe en avance**, pour masquer la latence DRAM des chargements data-dependent. Pur hint (resultat
inchange). `FmIndex::prefetch_occ` en `asm!("prfm ...")` (l'intrinseque `_prefetch` est instable).

- **Oracle-neutre** : SE 5000/5000, PE 10000/10000 ; alignements genome byte-identiques.
- **Gain (mesure, genome 10 Go, 100k reads genome-wide, `-t1`, min de 5)** : **~1,02x total, ~1,03x
  align-only** (prefetch < base a chaque rep). **Ne compte que quand `cp_occ` depasse le cache**
  (genome entier) ; nul sur la region 2 Mbp cache-residente.
- **Marche forward** : le walk reste dans une petite region deja en cache -> prefetch **net-negatif**,
  donc omis. Prochain palier possible : **recherche FM lockstep** (interleave de plusieurs reads,
  lookahead cross-slot T1) comme `getSMEMsOnePosOneThread_lockstep` du fork -> plus que le prefetch
  1-pas, mais gros restructure.

## Note DRAGEN (cadrage vitesse)

Objectif « battre DRAGEN sur la vitesse » : DRAGEN est un **accelerateur materiel FPGA/ASIC** (Illumina)
qui execute tout le pipeline sur du silicium dedie (~25-40 min pour un genome 30x). Un aligneur **CPU
logiciel ne peut pas battre du FPGA/ASIC dedie** a materiel comparable (1-2 ordres de grandeur). La seule
voie « classe DRAGEN » est **GPU** (cf. NVIDIA Parabricks qui *egale* DRAGEN via GPU) ou FPGA : c'est
exactement la **phase 9b (backend Metal GPU)**. De plus, DRAGEN n'utilise **pas** l'algo de bwa-mem2 (il
a son propre mappeur materiel), donc « octet-identique a bwa-mem2 » **et** « battre DRAGEN » sont deux
contraintes en tension. Cible CPU realiste : **le plus rapide des aligneurs octet-identiques a
bwa-mem2** (depasser le fork nh13). Cible « classe DRAGEN » : reactiver **9b (GPU)**.
