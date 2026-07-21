# Roadmap

Une phase = une branche. Commits frequents ; PR vers `dev`, `dev` promu sur `main` a la release.
Cible d'acceptation : index et SAM **octet-identiques** au binaire `bwa-mem2` 2.3 patche (oracle).

**Version courante : 4.0.0.** Les phases 0 a 10 sont terminees. Ce document reste le journal des
mesures : chaque phase garde son resultat, y compris les negatifs, pour ne pas re-instruire deux fois
la meme idee.

| Phase | Branche | But | Statut |
|---|---|---|---|
| 0 | `phase0-skeleton` | CLI `mem`/`index` (index stub), FASTQ -> SAM non-mappe, en-tete correct, harnais | fait |
| 1 | `phase1-indexer` | `bwa-mem4 index` : `.pac/.ann/.amb/.bwt.2bit.64/.0123` | fait, `cmp` octet |
| 2 | `phase2-index-load` | chargeur + `get_occ`/`get_sa`/`backward_ext` + validation | fait |
| 3 | `phase3-seeding` | SMEM + reseed + filtrage occ | fait |
| 4 | `phase4-chaining` | `mem_chain` + `mem_chain_flt` | fait |
| 5 | `phase5-extension` | SW bande scalaire (`SwBackend`) + `mem_chain2aln` | fait |
| 6 | `phase6-se-sam` | primaire/MAPQ/CIGAR/tags | fait, SAM SE octet-identique |
| 7 | `phase7-pe` | `mem_pestat`/`mem_matesw`/`mem_pair` | fait, SAM PE octet-identique |
| 8 | `phase8-scale` | GRCh38 complet, rayon, resorption de la traine | fait, WGS 30x reel |
| 9a | `phase9a-neon` | backend NEON du SW derriere `SwBackend` (optim. de @nh13, PR #288) | fait |
| 9b | `phase9b-gpu` | backend Metal du SW | **abandonnee**, backend retire (voir plus bas) |
| 9c-9e | | recurrence bandedSWA, prefetch de seeding, vague perf | fait |
| 10 | | ALT contigs, BAM/CRAM, CI, packaging, release 4.0.0 | fait |
| 11 | | gate GIAB `hap.py`/`vcfeval` (concordance variants) | a faire |

## Statut (4.0.0)

Parite mesuree sur un **WGS humain reel 32,9x** (GIAB HG002, 2x150), genome entier, pas un
sous-echantillon, les deux aligneurs lisant le meme index sur disque (`scripts/giab30x_pe.sh`) :

| | resultat |
|---|---|
| Index | **octet-identique**, les 5 fichiers |
| Single-end | **octet-identique** sur **353 517 767** enregistrements |
| Paired-end | **octet-identique** sur **707 312 349** enregistrements |
| ALT contigs | **octet-identique** sur l'analysis set GRCh38 reel, 261 contigs ALT |
| Vitesse | SE **2,62x**, PE **1,85x** vs bwa-mem2 a `-t8` (M4 Max) |
| Sorties | SAM, SAM compresse BGZF, BAM, CRAM |

Le ratio de vitesse **n'est pas une constante** : il decroit quand `-t` monte (3,28x a `-t1`,
2,45x a `-t16`), donc il se cite toujours avec son nombre de threads. Voir `docs/perf-levers.md`.

Trois gates : `scripts/check.sh` (fmt/clippy/tests), `scripts/opt_parity.sh` (58 combinaisons
d'options comparees `cmp` au binaire C), `scripts/alt_parity.sh` + `scripts/giab30x_pe.sh` (le WGS
complet). Les deux premiers tournent en CI a chaque push.

**Residus connus** : voir `DIVERGENCES.md`. Le principal n'est pas de nous : bwa-mem2 **ne s'accorde
pas avec lui-meme** entre x86_64 et arm64 sous scoring non defaut (`-A 2`), et c'est le build arm64
qui respecte la loi d'echelle imposee par l'algorithme. Notre parite est enoncee contre lui
(upstream `bwa-mem2#297`, ouvert depuis ce projet).

### Historique phase 7-8 (traine PE)

Traine resorbee via oracle instrumente bit-identique : (1) ordre des chaines **par position** +
portage exact de **`ks_introsort`** (tri instable) dans `mem_chain_flt` -> parite des regions
sous-optimales (340 SE + 622 PE) ; (2) `mapQ_coef_fac` est un **int** dans bwa-mem2
(`(int)log(50)=3`, pas `3.912`) -> parite MAPQ (23 SE + 6 PE). Puis, au passage a l'echelle du
genome, 13 causes racines supplementaires (passe de discard, semantique kbtree des positions
dupliquees, tris instables, gaps ouverts depuis M au lieu de H, padding du profil ksw, seuils
`mask_level` en f32 et non f64...), toutes documentees dans `DIVERGENCES.md`.
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
fork/PR accordes sur `IPNP-BIPN/bwa-mem4`). Voir `DEPENDENCIES.md` pour la provenance.

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
(mediane de 3). **Chiffres dates de la phase 9a** : ils precedent toute la vague perf decrite plus
bas, et la region 2 Mbp a un BWT cache-resident qui cache le seeding (voir l'avertissement en
phase 9e). Ils ne sont pas la mesure courante et le fork n'a pas ete re-mesure depuis.

| Binaire | vs bwa-mem2 (mediane, ratio) |
|---|---|
| `bwa-mem2` 2.3 (sse2neon, oracle) | 1,00x |
| **bwa-mem4 (nous)** | **~2,2x** |
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

## Phase 9e : la vague perf (tout octet-identique)

Tout ce qui suit est **octet-identique** (gate `oracle_diff` + `opt_parity`), sauf mention contraire.
Les gains sont isoles, donc ils ne s'additionnent pas naivement.

| Levier | Gain mesure |
|---|---|
| Mate rescue vectorise : kernel local-SW NEON inter-sequences (i16x8 puis **u8 16 lanes**), batching **inter-paires**, parallelise par chunks de paires | le plus gros levier PE (voir la mise en garde ci-dessous) |
| **Seeding lockstep** : round-2 de reseeding en lockstep, puis largeur N=8 -> **N=16** | SE +5,7 % puis +2,8 % |
| **Skip du SW bande** pour les seeds contenus sur la meme diagonale | SE +7,7 %, PE +5 % |
| **Recurrence f raccourcie** algebriquement dans le kernel SW (2 ops portees) | ~8 % sur le kernel |
| **PGO** (`scripts/pgo.sh`) | SE +6,1 %, PE +8,5 % a l'echelle du genome |
| `get_sa` resolu **a travers les reads** et non par read | +2-3 % a `-t8` |
| **mmap de l'index** + copie en bloc des tableaux BWT | **-6,2 Go de RSS**, chargement plus rapide |
| Fast path **ungapped** etendu aux diagonales portant des mismatches | inclus dans le skip ci-dessus |

**Mise en garde sur les chiffres PE historiques.** Le `PE 1,68x -> 2,42x` du commit de mate rescue a
ete mesure sur des reads **wgsim**. Sur GIAB reel le PE est plus bas (1,85-1,90x) parce que wgsim
produit des paires uniques qui **ne declenchent jamais le rescue** : la mesure tournait avec la
moitie du pipeline endormie. Les chiffres a citer sont ceux du WGS reel, en tete de ce document.

### Le profil reel, et pourquoi l'ancien etait faux

`work/region.fa` (2 Mbp) a un BWT cache-resident : le seeding y parait gratuit et l'extension y
parait dominante. L'ancien ROADMAP en a tire un « 85 % d'extension SW » qui a **cadre a tort** des
mois de travail. Sur le genome entier :

| profil SE (genome) | part |
|---|---|
| seeding + chainage | ~78 % |
| dont `get_sa` | ~19 % |
| kernel SW | ~4 % |

| profil PE (GIAB reel, `-t8`) | part |
|---|---|
| `matesw` (mate rescue) | **47,3 %** |
| `batched_extend` | 14,7 % |
| `mem_sort_dedup_patch` | 11,0 % |
| seeding | ~8 % |

**Le profil PE n'est pas le profil SE.** Trois sessions de travail sur le seeding touchent ~13 % du
PE. **Tout benchmark doit declarer ce qu'il desactive** : la region 2 Mbp cachait le seeding, `-K
100M` sur 500k reads cachait le pipeline lecteur/ecrivain (+8-9 %), wgsim cachait le mate rescue.

### Le mur du mate rescue : le nombre de cellules EST l'algorithme

Compte direct (`BWA4_MATESW_TIME=1`, GIAB reel, 500k paires, `-t1`) : **1 838 008 jobs, 381 032 465
824 cellules DP**, soit 148 pb x fenetre de 1401 pb par ancre. Le rescue fait ~17x le travail DP de
toute l'etape d'extension. Nous executons **deja ce meme compte 1,26x plus vite que bwa-mem2**
(12,87 s contre 16,19 s, mesure des deux cotes par `-S` / `BWA4_NO_RESCUE=1`). Aller plus vite
demande de faire **moins de cellules**, et ce compte est l'algorithme de bwa : le changer change la
sortie. Un prefiltre ungapped ne sauve rien non plus (148 bases x 1401 positions = exactement le
compte qu'il pretend eviter, la position du mate dans la fenetre etant inconnue).

### Le fait perf le plus structurant : le lockstep a mange le levier

Cinq techniques publiees du type « moins d'acces memoire » ont ete portees ou prototypees et
**mesurent ~0 ici**, parce que notre seeding lockstep W=16 + prefetch masque **deja** la latence
qu'elles visent. Avant de porter une technique de ce genre, la tester contre le lockstep en place,
pas contre un baseline naif.

**Leviers mesures morts (ne pas re-instruire) :**

- **LISA / index appris** : seeding BWA-MEME octet-identique construit et prouve identique sur le
  genome, puis mesure **beaucoup plus lent** que le FM. Abandonne.
- **Interleave ILP a 2 groupes du kernel SW** : parite. Register-bound (28 valeurs vives pour 32
  registres, 102 spills). Ne pas retenter sur NEON ni AVX2.
- **occ FM vectorise** (popcount NEON 4 bases) : **0,89x**. LLVM compile deja `count_ones` en
  `cnt`/`addv` optimal.
- **Score-prepass facon `SBT_PREPASS`** : **0,90x**. Le round-trip store/load coute plus que
  recalculer en place.
- **Table de prefixes facon STAR + recherche binaire** : morte a la mesure.
- **Cache 10-mer de minibwa** : 0 %.
- **Prefetch de ligne SA**, **prefetch sur la marche forward** : 0 % ou negatif.
- **Binning de regions BWT (Zhang)** : le genou est a 16,7M, hors de portee.
- **SME / SVE en streaming sur M4** : 64 lanes inexploitables pour ce kernel, NEON 16 lanes est le
  plafond.
- **DMP Apple / `PRFM` sur TLB miss / chicken bits HID** : portes fermees, un FM-index n'engage
  structurellement pas le prefetcher de pointeurs.
- **minibwa comme cible** : il utilise l'algo de minimap2 (`ksw2_extd2`) et des heuristiques, donc
  **sortie differente**. Incompatible avec le gate.

### Scaling en threads

| `-t` | bwa-mem2 | nous | ratio |
|---|---|---|---|
| 1 | 53,97 s | 16,44 s | **3,28x** |
| 4 | 15,63 s | 5,20 s | 3,00x |
| 8 | 9,28 s | 3,30 s | 2,81x |
| 12 | 7,16 s | **2,87 s** | 2,49x |
| 16 | 6,98 s | 2,84 s | 2,45x |

bwa-mem2 scale mieux que nous (7,73x contre 5,79x a 16 threads) : c'est le cout direct d'etre plus
rapide par thread, nous atteignons le plafond partage plus tot. **L'explication « `-t8` est
bandwidth-bound » a ete retractee** apres mesure : le M4 sert des gathers aleatoires dans 16 Go a
3,5 ns a 1 thread et 4,3 ns a 8 threads, il n'y a pas de contention de bande passante. Le cout de
`get_sa` (177 ns) est une chaine de ~7 defauts de cache **dependants**, donc de la latence serielle.
`-t12` est le genou (= le nombre de P-cores ; le pipeline prend 2 threads de plus, lecteur +
ecrivain).

## Phase 9b (GPU) : abandonnee, backend retire

Un backend Metal a existe et a ete **supprime** (`c20867d`). Raison : sur un genome entier le kernel
Smith-Waterman fait **~4 %** du temps, le seeding ~78 %. Amdahl plafonne tout offload SW a quelques
pourcents, et chaque backend ajoute une surface d'octet-identite a prouver contre le scalaire. Le
shader Metal avait justement livre un vrai bug (il ouvrait les gaps depuis `H` au lieu de `M`) parce
que cette preuve etait trop faible. Il reste dans l'historique git si le profil change.

## Note DRAGEN (cadrage vitesse)

Objectif « battre DRAGEN sur la vitesse » : DRAGEN est un **accelerateur materiel FPGA/ASIC** (Illumina)
qui execute tout le pipeline sur du silicium dedie (~25-40 min pour un genome 30x). Un aligneur **CPU
logiciel ne peut pas battre du FPGA/ASIC dedie** a materiel comparable (1-2 ordres de grandeur). La seule
voie « classe DRAGEN » est **GPU** (cf. NVIDIA Parabricks qui *egale* DRAGEN via GPU) ou FPGA : c'est
exactement la **phase 9b (backend Metal GPU)**. De plus, DRAGEN n'utilise **pas** l'algo de bwa-mem2 (il
a son propre mappeur materiel), donc « octet-identique a bwa-mem2 » **et** « battre DRAGEN » sont deux
contraintes en tension.

**Cadrage retenu.** La cible est **le plus rapide des aligneurs octet-identiques a bwa-mem2**, pas
DRAGEN. La voie GPU est fermee pour une raison independante de DRAGEN : le kernel SW ne fait que ~4 %
du temps genome, donc il n'y a rien a offloader (voir phase 9b). Une « classe DRAGEN » exigerait de
porter le **seeding** sur accelerateur, ce qui est un autre projet.

## Ce qui reste

1. **Gate GIAB `hap.py`/`vcfeval`** (phase 11) : montrer que la parite octet se traduit en
   concordance de variants sur le truth set, ce qui est le langage d'un utilisateur clinique.
2. **`mem_sort_dedup_patch`** : 11 % du profil PE, jamais regarde.
3. **SA-IS parallele** (Tier B de la phase 8c) : l'indexeur reste mono-thread sur le tableau de
   suffixes, qui domine son pic RSS et son temps.
4. **Re-mesurer le fork `fg-labs/bwa-mem3`** : la derniere comparaison date de la phase 9a, avant
   toute la vague perf, et sur un benchmark dont on sait maintenant qu'il cachait le seeding.
