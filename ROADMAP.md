# Roadmap

Une phase = une branche. Commits frequents ; PR vers `dev`, `dev` promu sur `main` a la release.
Cible d'acceptation : index et SAM **octet-identiques** au binaire `bwa-mem2` 2.3 patche (oracle).

**Version courante : 4.3.1.** Les phases 0 a 10 sont terminees. Ce document reste le journal des
mesures : chaque phase garde son resultat, y compris les negatifs, pour ne pas re-instruire deux fois
la meme idee.

La numerotation reste en 4.3.x et n'ira ni en 4.4 ni en 5.x : le binaire s'appelle `bwa-mem4`, la
version suit le nom, et les lots avancent sur le troisieme chiffre pour garder les releases
rapprochees.

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
| 11 | | gate GIAB `hap.py`/`vcfeval` (concordance variants) | a faire, jalon v4.3.3 |

## Releases

| Version | Date | Contenu |
|---|---|---|
| 4.0.0 / 4.0.1 | 2026-07-22 | ALT contigs, BAM/CRAM, packaging, CI |
| 4.1.0 / 4.1.1 | 2026-07-23 | correctifs de release |
| 4.2.0 | 2026-07-30 | vague perf (mate rescue, dedup, `.pac` vectorise, CaPS-SA), sonde par etage |
| 4.3.0 | 2026-07-31 | `-x` route vers rammap, sous-commande `version`, credits bwa-mem3 / @nh13 |
| 4.3.1 | 2026-08-04 | passe de documentation : couche mecanique Rust sur les 10 crates ; aucun changement de comportement |

## Jalons ouverts (GitHub Projects nº3)

| Jalon | Contenu | Etat |
|---|---|---|
| v4.3.2 | parite perf x86_64 (issues #20, #25, #27, #32, #33) | ouvert |
| v4.3.3 | phase 11, gate GIAB `hap.py`/`vcfeval` ; suivi upstream `bwa-mem2#297` | ouvert |
| v4.3.4 | SA-IS parallele (l'indexeur reste mono-thread sur le tableau de suffixes), structure de dedup incrementale | ouvert |

Les jalons avancent d'un cran sur le troisieme chiffre : une release courte et frequente plutot
qu'un saut de version mineure a chaque lot. Le jalon perf x86_64 demande une machine x86_64 et un
WGS complet, pas une mesure sur Apple Silicon.

## Statut (4.3.1)

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

Un quatrieme harnais, `scripts/docker_gates.sh`, rejoue toutes les portes **en conteneur**, sur les
deux plateformes, depuis une machine Apple Silicon : `build`, `check`, `parity`, `rss`, `bench`.

* `ARCH=amd64` (defaut) tourne sous Rosetta. `avx2` est expose donc ces noyaux s'executent vraiment,
  `avx512bw` ne l'est pas. Resultat au 2026-08-04 : `check` 89 tests verts, `parity` **62 sur 64**
  octet-identiques a l'oracle x86_64, les deux restantes etant le desaccord `-A 2` deja connu
  (205 enregistrements sur 8000, meme chiffre que depuis arm64, voir `DIVERGENCES.md` et l'upstream
  `bwa-mem2#297`).
* `ARCH=arm64` tourne **nativement**, donc ses temps sont reels. C'est le seul mode ou `bench`
  accepte de s'executer : sous emulation les rapports de vitesse ne se conservent pas, et
  `bench` refuse explicitement plutot que de produire un chiffre qui ressemblerait a une mesure.

`rss` compare le pic memoire de bwa-mem4 a celui du fork `fg-labs/bwa-mem3`, construit dans le
conteneur, dans les deux regimes de `-K`. Sur une reference de 200 kb (index retire du pic) le
rapport a `-t16` est de 0,84 a `-K` par defaut et 0,40 a `-K` fixe : la croissance par lot decrite
dans l'issue #25 ne se reproduit pas a cette echelle, seul le cout de base a `-t1` reste superieur
(1,53x, soit environ 100 MB constants). La mesure a l'echelle du genome reste a faire.

**Residus connus** : voir `DIVERGENCES.md`. Le principal n'est pas de nous : bwa-mem2 **ne s'accorde
pas avec lui-meme** entre x86_64 et arm64 sous scoring non defaut (`-A 2`), et c'est le build arm64
qui respecte la loi d'echelle imposee par l'algorithme. Notre parite est enoncee contre lui
(upstream `bwa-mem2#297`, ouvert depuis ce projet).

## hyalite 0.3.0 : nos deux corrections sont remontees en amont (2026-08-09)

Mise a jour de la section ci-dessous. Le pin de developpement passe de `0.2` a **`0.3.0`**, et
l'oracle independant reste vert sans modification.

Ce que 0.3.0 apporte pour nous : les **deux corrections de documentation** que ce projet avait
demandees en amont (`Psy-Fer/hyalite#2`) sont dedans, de l'aveu du mainteneur. Ce sont exactement les
deux choses que `third_party_oracle.rs` avait du decouvrir a la main :

* la matrice de substitution s'indexe `[requete][cible]` et non `[cible][requete]` comme chez bwa ;
* l'accord avec le `score2` de bwa **ne tient que si la longueur de requete est un multiple de la
  largeur SIMD**, parce que le maximum par ligne de bwa n'est pas le maximum par colonne du DP une
  fois le profil de requete bourre. C'est la meme observation que la section #47 fait de l'interieur.

Ce que 0.3.0 n'apporte pas : les **penalites de gap asymetriques**. `Scoring::new(alphabet_len,
matrix, gap_open, gap_ext)` reste le seul constructeur, donc `o_del != o_ins` reste inexprimable et
la portee de notre oracle est inchangee : il tourne en penalites symetriques, ce qui est le defaut de
bwa (`-O 6 -E 1`). La demande principale de l'issue reste ouverte en amont.

## hyalite comme noyau de production : non (2026-08-05)

hyalite 0.2 sert d'oracle independant (voir `crates/bwa-extend/tests/third_party_oracle.rs`). Question
posee : ses noyaux striped pourraient-ils remplacer les notres et nous faire gagner du temps ?

Mesure, formes de mate rescue reelles (requete 150 bp contre fenetre de 1672 bases, 2000 paires,
0,502 Gcellules, mono-thread, M4 Max ; conteneur Linux arm64 a 2 % pres) :

| noyau | debit |
|---|---|
| notre `ksw_align2` scalaire | 0,54-0,56 Gcell/s |
| hyalite `align_pairs` (striped SIMD) | **2,18-2,22 Gcell/s** |
| notre noyau NEON u8 batche (`BWA4_MATESW_TIME`, donnees reelles) | **10,59 Gcell/s/thread** |

hyalite est donc 4x notre repli scalaire et environ 5x plus lent que le noyau qui tourne reellement
en production. La raison est structurelle et non un defaut de leur code : ils font du striped
intra-sequence (Farrar), une paire a la fois, ce qui est le bon choix pour une bibliotheque a usage
general ; nous faisons du batch inter-sequence sur 16 voies u8, ce qui est le bon choix quand on a
des milliers de jobs independants et aucune envie de payer le profil raye par paire.

Correction d'une erreur de la premiere redaction de cette section : hyalite 0.2 **rend bien** les
coordonnees de debut, par traceback (`align`), et elles s'accordent avec la recuperation `KSW_XSTART`
de ksw sur les 499 paires (sur 600) dont le score franchit `minsc` et declenche donc cette passe.
C'est verifie dans l'oracle depuis. La barriere restante est `score2` : le max par ligne de bwa
inclut les colonnes de bourrage du profil de requete, celui de hyalite est le vrai max par colonne,
et le mate rescue consomme `score2` pour `csub`.

Le traceback est aussi une forme plus couteuse que ce que mesure le tableau ci-dessus, qui n'utilise
que le score, donc il n'ameliore pas le verdict de vitesse. Sa valeur pour ce projet est l'oracle,
pas la vitesse.

## Indexeur : le tableau de suffixes est parallelisable, mais pas par le portage Rust (2026-08-05)

Resume : `libsais64_omp` (portage rayon, pur Rust) perd ; **le C libsais gagne a tous les nombres de
threads, y compris a un seul**, et produit le meme tableau. Il est desormais le backend **par
defaut** (`libsais-c`), en version serie : la crate embarque la source C, donc cela ajoute un
compilateur C et aucun paquet systeme, et le binaire en exigeait deja un pour `rust-htslib`.
`--features libsais-c-omp` ajoute OpenMP (la ou sont les 4,3x, mais il faut libomp) et
`--features libsais` garde le portage pur Rust pour une compilation sans C du tout.

Mesure sur le texte 2L, tableaux **identiques dans tous les cas** (un tableau de suffixes est unique) :

| texte | pur Rust (defaut) | C 1 thread | C 4 | C 8 | C 16 |
|---|---|---|---|---|---|
| chr21, 93 M bases | 3,27 s | 1,95 s | 1,05 s | **0,85 s** | 1,22 s |
| chr1, 498 M bases | 20,15 s | 12,21 s | 5,85 s | **4,63 s** | 7,27 s |

Deux choses a retenir. Le C bat le portage Rust **meme a un seul thread** (1,7x), donc une partie du
gain n'est pas le parallelisme. Et le coude est a 8 threads : au-dela, les sections paralleles de
libsais perdent contre leur propre coordination, d'ou le plafond `min(rayon, 8)` et la variable
`BWA4_SA_THREADS` pour une machine dont le coude est ailleurs.

Bout en bout sur `bwa-mem4 index`, cinq fichiers octet-identiques a la fixture committee dans les
trois cas : conteneur Linux arm64 **3,347 s (pur Rust) -> 1,341 s (C + OpenMP)** ; hote macOS
**2,56 s -> 1,742 s** avec le nouveau defaut serie, qui est le gain que tout le monde recoit sans
installer quoi que ce soit.

### Pourquoi le portage pur Rust ne rattrape pas

Comparaison directe des trois sur le meme texte (chr21, 93 M bases), tableaux identiques partout,
`libsais-rs` en 0.2.2 :

| | serie | 4 threads | 8 threads | 16 threads |
|---|---|---|---|---|
| `libsais-rs` (pur Rust) | 3,02 s | 2,98 s | **2,81 s** | 3,34 s |
| `libsais` (C) | 1,92 s | 1,04 s | **0,81 s** | 1,00 s |

Le portage Rust **ne parallelise quasiment pas** : 7 % entre son mode serie et ses 8 threads, contre
2,4x pour le C. C'est ce qui explique le resultat de la veille, ou cabler `libsais64_omp` dans
l'indexeur etait neutre ou negatif : un etage qui gagne 7 % ne paie pas le surcout de mise en place
dans le contexte du builder. Le C, lui, gagne deja 1,6x **en serie** et 3,7x a 8 threads.

`libsais-rs` passe quand meme en 0.2.2 (0.2.0 -> 0.2.2 vaut ~8 % en serie), puisqu'il reste le
backend de repli sans compilateur C.

### Ce qui a ete essaye et rejete

* `libsais64_omp`, le portage rayon pur Rust, cable dans l'indexeur : **plus lent** que sa propre
  version serie, 3,23 s contre 2,56 s sur chr21 et 18,44 s contre 16,01 s sur chr1. Revert, et la
  mesure ci-dessus dit pourquoi.
* [`sufr`](https://github.com/TravisWheelerLab/sufr) : tri fusion parallele partitionne, avec
  `libsufr`. Abandonne apres **plus de 100 minutes de CPU sur chr21** la ou libsais met 3 s. Ce
  n'est pas le meme objet : il construit un index sur disque avec LCP et masques de graine, pour des
  requetes, pas un tableau de suffixes en memoire pour un builder d'index.

### Note historique

## Indexeur : le tableau de suffixes reste mono-thread, et ce n'est pas faute d'avoir essaye (2026-08-05)

Tout le reste de `bwa-mem4 index` est deja parallele (rayon) : le pack `.pac`, la moitie complement
inverse du `.0123`, la BWT, les blocs de checkpoints, le one-hot. Il ne restait que la construction
du tableau de suffixes, qui est l'issue #37 et le jalon v4.3.4.

`libsais-rs` expose `libsais64_omp(t, sa, fs, freq, threads)`, un point d'entree parallele backe par
rayon. Cable, verifie octet-identique (un tableau de suffixes est UNIQUE : une construction parallele
produit la meme permutation ou elle est fausse), et mesure sur deux tailles :

| reference | `libsais64` serial | `libsais64_omp`, 16 threads |
|---|---|---|
| chr21, 46 Mb | **2,56-2,60 s** | 3,23 s |
| chr1, 250 Mb | **16,01 s** | 18,44 s |

Plus lent aux deux echelles, d'environ 15 %. Le portage rayon de libsais paie sa mise en place sans
la rentabiliser a ces tailles. Change **revert**, negatif consigne : ne pas re-cabler `_omp` sans une
mesure a l'echelle du genome entier qui montre l'inversion.

Candidat restant pour #37 : [`sufr`](https://github.com/TravisWheelerLab/sufr), construction
parallele par partitionnement et tri fusion, avec sa bibliotheque `libsufr`. Aucune comparaison
publiee contre libsais, et il produit aussi un LCP dont nous n'avons pas besoin (donc de la memoire
en plus). A mesurer avant d'y toucher, pas a adopter sur la description.

## noodles contre htslib pour l'ecriture BAM : prototype mesure (2026-08-05)

Question posee : remplacer `rust-htslib` par `noodles` (pur Rust) pour la sortie binaire. Prototype
ecrit, deux transcodeurs texte SAM -> BAM sur **la meme entree** (chr21, 1 060 841 enregistrements,
439 MB de texte), donc la seule difference mesuree est l'ecrivain.

| threads | htslib (ce qu'on livre) | noodles |
|---|---|---|
| 1 | 5,24 s | **1,69 s** |
| 4 | 1,35 s | **0,93 s** |
| 8 | **0,68 s** | 0,93 s |

Correction, verifiee et pas supposee : **enregistrements identiques** (`samtools view` des deux BAM,
`cmp` octet a octet), en-tetes identiques, et les deux **redonnent exactement le SAM d'entree**.
Taille : 92 098 890 octets pour htslib, 91 640 005 pour noodles, soit 0,5 % de moins.

Lecture : noodles gagne largement a bas parallelisme (3,1x a un thread) et **plafonne a 0,93 s** quand
htslib continue de descendre. Le plafond dit ou est le vrai probleme : a 8 threads htslib a cache
toute sa compression et c'est le **parsing du texte** qui domine, des deux cotes. Autrement dit, notre
pipeline formate du texte SAM puis le **re-parse** pour ecrire le BAM.

Le vrai gain n'est donc pas le choix de la bibliotheque, c'est de construire les enregistrements BAM
**directement depuis les champs de l'aligneur**, sans passer par le texte. L'octet-identite ne
contraint que la sortie SAM ; le BAM est notre propre format de sortie et n'a pas d'oracle. C'est un
chantier separe, et il rendrait la comparaison ci-dessus caduque.

CRAM non evalue : c'est la ou htslib est l'implementation de reference et ou le risque se concentre.

## Le double passage texte du BAM : mesure faite, le chantier ne le vaut pas (2026-08-07)

La section ci-dessus concluait que le vrai gain serait de construire les enregistrements BAM
directement depuis les champs de l'aligneur. Cette conclusion etait tiree d'un prototype qui mesurait
le transcodeur **isole**, hors pipeline. Mesure dans le pipeline, elle ne tient pas.

Sonde `BWA4_WRITER_PROBE=1` : le fil ecrivain, seul etage serie du pipeline, compte ses secondes
passees a ecrire et ses secondes passees bloque sur le canal. Une paire d'`Instant::now()` par
**lot** (quelques dizaines par run), jamais par enregistrement, exactement pour ne pas repeter la
faute de la sonde `BWA4_ALIGN_SPLIT` decrite plus bas.

chr21, 1 M paires, `-t16`, 2 M enregistrements :

| sortie | ecrit | bloque |
|---|---|---|
| `out.sam` | 0,151 s | 14,254 s |
| `out.sam.gz` | 0,583 s | 18,178 s |
| `out.bam` | **1,417 s** | 13,879 s |

Lecture : sur un mur de 15,4 s l'ecrivain BAM travaille 1,4 s et attend 13,9 s, soit **9 %
d'occupation**. Supprimer entierement le parsing du texte rendrait au mieux ces 1,4 s de CPU sur ~207
s de CPU total, soit **0,7 %**, et **rien du tout** en temps mur, puisque l'etage n'est pas le goulot.
Le chantier (deux formateurs d'enregistrements a garder en phase, dont celui de `pe.rs`, sans oracle
pour le second) est hors de proportion avec ce que la mesure promet. **Abandonne.**

Ce que la mesure designe a la place : le surcout BAM n'est pas le parsing, c'est la **compression**.
`out.bam` coute ~19 s de CPU de plus que `out.sam` et l'ecrivain n'en explique que 1,4 : le reste est
du deflate sur les fils de fond de htslib.

### Le gain reel : libdeflate comme moteur deflate de htslib

`rust-htslib`, feature `libdeflate`. A/B entrelace, chr21, 1 M paires, `-t8`, `-o out.bam`, secondes
CPU :

| | zlib (avant) | libdeflate (apres) |
|---|---|---|
| repetitions | 176,39 / 181,42 / 178,62 / 181,11 / 183,70 | 171,07 / 172,15 / 174,41 / 174,06 / 181,53 |
| mediane CPU | 181,11 s | **174,06 s**, ~3,9 %, **5 victoires sur 5** |
| mediane mur | 22,24 s | 22,13 s, egalite |

Le mur ne bouge pas et c'est attendu : htslib compresse sur ses propres fils de fond, qui ont du mou a
`-t8`. Le gain est du CPU rendu au reste de la machine, pas du temps que ce run cesse d'attendre. A
`-t16` la mesure est plus bruitee (medianes 210,61 contre 204,38, 2 victoires sur 3), meme sens.

Cout en chaine d'approvisionnement : **nul**. `libdeflate-sys` est deja compile et lie dans ce binaire
pour le chemin `.sam.gz` (`bgzf` -> `libdeflater`), et `cargo tree -i libdeflate-sys` montre `hts-sys`
partageant la meme version 1.25.2 au lieu d'en ajouter une seconde copie.

Parite verifiee, pas supposee : `samtools view` du BAM et les enregistrements du SAM texte donnent le
meme md5 (`c833bb42...`), sur les deux moteurs de compression. Les octets compresses, eux, different
d'un moteur a l'autre : le BAM n'est pas l'oracle, le SAM texte l'est.

## Ou le fork est reellement moins cher, profil contre profil (2026-08-07)

Hypothese de depart, ecrite ici hier : le seeding pese ~30 % et l'ecart restant avec le fork y est
cache. **Faux, et mesure.** Profil par echantillonnage (`sample`, 17-18 s de fenetre) des deux
binaires sur **exactement la meme charge** : 1 M paires GIAB reelles contre GRCh38, `-t8`, sortie
jetee. Parts des echantillons de travail (les attentes de threads exclues des deux cotes).

| etage | bwa-mem4 | `fg-labs/bwa-mem3` |
|---|---|---|
| noyau de mate rescue | 39,2 % | 38,6 % |
| DP principal (extension) | 20,0 % | 18,5 % |
| **tri + dedup des regions** | **15,1 %** | **7,6 %** |
| **seeding (recherche arriere)** | **6,7 %** | **17,0 %** |
| resolution du suffix array | 3,9 % | 5,4 % |
| chainage | 5,1 % | 3,8 % |
| fenetre de reference | 1,1 % | 3,8 % |

Deux lectures, opposees a ce qui etait suppose :

1. **Notre seeding est deja ~2,5x moins cher que le sien** (6 634 echantillons contre 18 311). Le
   lockstep et le prefetch par lot font leur travail. L'hypothese "ecart d'implementation dans le
   seeding" est morte : il n'y a rien a rattraper, c'est nous qui sommes devant.
2. **Notre tri de regions coute deux fois le sien.** C'est la, et nulle part ailleurs, que le fork
   est structurellement moins cher.

Course complete sur cette charge, pour situer : nous 28,60 s de mur / 220,21 s de CPU, le fork
33,52 / 247,89. Nous gagnons de 15 % en mur, 11 % en CPU.

### Le tri : ce que l'octet-identite coute exactement, chiffre

Le fork trie avec **pdqsort**, nous avec le `ks_introsort` de klib, parce que la permutation est
**observable en sortie** : `mem_sort_dedup_patch` trie sur `re` seul puis tue, parmi des regions a
egalite, celle que le tri a mise en premier.

Ce n'est pas de la prudence heritee, c'est verifie : en remplacant les deux tris par
`sort_unstable_by` (le pdqsort de Rust), le **md5 du SAM change** sur 200k paires reelles.

Et le prix de cette contrainte est maintenant chiffre. Meme binaire, seuls les deux tris changent,
A/B entrelace, 1 M paires, `-t8`, secondes CPU : **219,56 s avec `ks_introsort` contre 213,38 s avec
pdqsort, 3 victoires sur 3, soit 2,8 %.** Voila la totalite de ce que l'octet-identite nous coute sur
cet etage. C'est le plafond de tout ce qui reste a gagner ici, et il est interdit.

### Ce qui etait recuperable dans ce plafond, et l'a ete

`ks_introsort_by_key` : meme algorithme, meme permutation (prouvee et testee), mais le tri porte sur
des paires `(cle, index)` et les `MemAlnReg` de 96 octets ne bougent qu'une fois, a la fin.
`BWA4_DEDUP_SHAPE` dit pourquoi ca vaut la peine : 970 814 appels pour 200k paires, longueur moyenne
105, dont 570k a 65 regions ou plus, tous venant du mate rescue qui retrie apres chaque insertion.

Deux versions, et l'ecart entre les deux est la lecon :

| version | echantillons de la fonction | A/B bout en bout |
|---|---|---|
| indirect, `Vec` alloues par appel | 14 812 -> 13 665 (-7,7 %) | **nul**, 2 victoires sur 5 |
| indirect, tampons `thread_local` reutilises | idem | **-1,9 %**, 4 victoires sur 5 |

~20 millions d'allocations par million de paires coutent exactement ce que l'indirection fait gagner.
Le protocole du gain final est celui a plus faible dispersion ici : `-t4`, 500k paires reelles contre
GRCh38, medianes 106,94 s contre 108,97 s. Sortie octet-identique (chr21 et GRCh38).

Note de methode, valable pour tout ce qui suit : a `-t8` sur cette machine la derive thermique
atteint 10 % au fil d'une serie, ce qui noie un effet de 2 %. Les series a `-t8` ci-dessus ne servent
qu'a comparer des choses tres differentes ; tout gain de l'ordre du pour-cent se mesure a `-t4`.

## Le mate rescue, le plus gros poste restant (2026-08-07)

Ce que la sonde `BWA4_MATESW_TIME` dit du poste, sur 500k paires reelles a `-t4` :

| | |
|---|---|
| CPU dans le noyau | **54,4 s sur les ~107 s du run**, soit ~51 % |
| travaux | 1 837 905, pour 381 Gcellules |
| forme moyenne | requete 148 bp contre fenetre 1401 bp = **207 417 cellules par travail** |
| taxe de divergence de voie | 1,09x, et trier par longueur en economiserait **0,0 %** |
| travaux dupliques dans un appel | 35 sur 1 837 905, soit 0,0 % |

Les deux dernieres lignes ferment deux pistes avant de les ouvrir : ni le tri par longueur ni la
deduplication n'ont quoi que ce soit a recuperer. Les cellules, elles, sont fixees par l'algorithme
(SW local plein sur le rectangle, comme bwa). Reste donc le cout par cellule.

### Ce qui a marche : quatre lignes de cible a la fois

Le corps par PAIRES etait deja la, et il paie toujours : `BWA4_RESCUE_ROWPAIR=0` fait tomber le debit
de 9,41 a 8,59 Gcell/s, **+9,5 %**, reproductible a 0,4 % pres.

Quatre lignes poussent l'idee d'un cran : un quadruplet charge la colonne de requete, `e[j]` et
`h_prev[j]` une fois et ecrit `e[j]` et `h_cur[j]` une fois, soit **cinq operations memoire pour
quatre cellules** au lieu de cinq pour deux. Les trois retenues E interieures et les trois diagonales
interieures ne touchent jamais la memoire.

Mesure au calme, trois repetitions par bras, dispersion 0,4 % : **16,00 / 16,06 / 16,01 s** de CPU
noyau contre **16,46 / 16,44 / 16,41**, soit **+2,6 %**. Rendements decroissants nets face aux +9,5 %
de la paire, la pression sur les registres etant l'explication la plus probable. NEON seulement :
AVX2 a 16 registres vectoriels contre 32, et quatre lignes demandent seize accumulateurs vivants
avant les constantes.

Bout en bout, meme protocole que partout ailleurs ici (`-t4`, 500k paires reelles contre GRCh38,
secondes CPU), sur un hote redevenu calme, **5 victoires sur 5** :

| | repetitions | mediane |
|---|---|---|
| avant | 105,29 / 105,57 / 106,05 / 106,24 / 106,27 | 106,05 s |
| apres | 104,51 / 104,83 / 105,18 / 105,12 / 105,02 | **105,02 s**, -0,97 % |

La projection depuis le noyau (+2,6 % sur 51 % du run) donnait -1,3 %. Le reel est **-0,97 %**, un
peu en dessous, ce qui est le sens attendu : le poste "mate rescue" du run inclut de la mise en lots
et de l'extraction que le quadruplet ne touche pas. C'est le chiffre a citer, pas la projection.

### Ce qui n'a pas marche : sauter la reparation du N

Le tableau de scores donne deja le -1 de bwa des que le XOR vaut 4-7, donc la seule cellule qu'il rate
est **N contre N**, qui lit XOR 0 et ressemble a un appariement. D'ou un `vbslq_u8` par ligne. Quand
aucune voie du groupe ne porte de N, ce melange est prouvablement inutile : condition invariante de
boucle, que LLVM devrait pouvoir sortir en dupliquant la boucle.

Il ne l'a pas fait. Debit tombe de 9,63 a **8,95 Gcell/s**, soit **-7 %** : le test dans la boucle
coute plus que les quatre operations qu'il economise. Annule. A ne pas retenter sous cette forme ; il
faudrait deux corps ecrits a la main, et le gain theorique (4 operations sur ~64 par colonne) ne le
justifie pas.

Effet de bord garde, lui : les tests d'equivalence au scalaire tirent maintenant le code 4 (N) en plus
des quatre bases reelles. Le N etait couvert bout en bout (le genome en est plein) mais pas en test
unitaire, ce qui laissait un noyau oubliant la reparation echouer sur un md5 de genome entier plutot
que dans `cargo test`.

### Ce qui a bien mieux marche : supprimer la reparation au lieu de la sauter (2026-08-08)

La bonne question n'etait pas "quand peut-on sauter le melange" mais "pourquoi la table se trompe".
Elle se trompe sur une seule cellule, N contre N, parce que `4 XOR 4 = 0` tombe sur le creneau
d'appariement. En **encodant le N de la cible comme 12** au moment de l'empaquetage, la collision
disparait et chaque cas obtient son creneau :

| index | cas |
|---|---|
| 0-3 | bases reelles : appariement / mesappariement |
| 4-7 | N de requete contre base reelle |
| **8** | **N contre N**, desormais son propre creneau |
| 12-15 | N de cible contre base reelle |
| 9 | ZPAD, ecrase par le melange zpad, indifferent |
| >= 16 | PAD, `vqtbl1q` rend 0 |

La table a 16 creneaux et six servaient. La reparation est supprimee des trois corps (quadruplet,
paire, ligne seule), le cout part dans une comparaison par base de cible a l'empaquetage : 1400 fois
par groupe au lieu de 1400 x 148 x 4.

**-7,2 % sur le noyau** (14,90 s de CPU contre 16,05, trois repetitions, dispersion 0,3 %), soit 10,34
Gcell/s contre 9,60. Pres de trois fois le quadruplet. NEON seulement : les noyaux u8 x86 n'ont jamais
adopte la table indexee par XOR, ils scorent par `(t == 4) | (q == 4)` et un melange, donc il n'y a
aucun creneau a donner au N contre N ni rien a reutiliser.

### Le DP principal : la meme forme biaisee, et un resultat decevant

Le DP bande gardait le score coupe en deux, partie positive par addition et negative par soustraction,
parce qu'une voie u8 ne porte pas de negatif : deux tampons, deux ecritures dans la pre-passe, deux
lectures dans le corps. La forme biaisee du rescue les ramene a un. Ecrit une fois dans
`define_sw_kernel!`, donc les huit instanciations (NEON, AVX2, SSE4.1, AVX-512, en u8 et i16) en
heritent. La contrainte u8 (`m + biais + a` ne doit pas plafonner a 255 avant que la soustraction
reprenne le biais) est reservee par `dispatch_bins` au moment de choisir le bin ; elle vaut 5 pour les
defauts de bwa et ne deplace aucun travail reel.

Estimation avant mesure : ~12 % du noyau DP, soit ~2,4 % bout en bout. **Reel : 0,4 %, 5 victoires sur
6.** Les operations retirees ne sont pas sur le chemin critique, c'est la chaine serie de F qui l'est,
donc les enlever ne rend presque rien. Garde parce que c'est strictement moins de travail pour une
sortie identique, et parce que la pre-passe existe justement a cause du x86 (moins de ports vectoriels
que de voies), ou l'effet devrait etre plus grand et n'est pas mesurable ici. Mais le chiffre a citer
est 0,4 %, pas l'estimation.

### La serie complete, un seul banc

`-t4`, 500k paires reelles contre GRCh38, secondes CPU, six repetitions entrelacees, dispersion 0,5 % :

| binaire | medianes | contre le precedent |
|---|---|---|
| quadruplet seul | 105,24 s | reference |
| + creneau N | **102,34 s** | **-2,76 %, 6 victoires sur 6** |
| + DP biaise | **101,99 s** | -0,34 %, 5 victoires sur 6 |

Sortie octet-identique a chaque etape, sur GRCh38 et sur chr21.

## Le plafond du noyau de rescue : 16 Gcell/s, pas 56 (2026-08-08)

Le noyau imprimait depuis toujours `ISA ceiling: 16 u8 lanes x ~3.5 GHz = ~56 Gcell/s if 1
cell/lane/cycle`, ce qui le faisait paraitre cinq fois pire qu'il n'est. **Le chiffre est faux comme
objectif**, et voici ce qu'une mesure sur la machine de developpement (M4 Max) donne a la place.

| | |
|---|---|
| pic d'operations NEON, sans dependance ni memoire | **16,63 G op/s**, ~3,8 par cycle |
| la sequence d'operations par cellule du noyau, registres seuls | **16,04 Gcell/s**, 93 % de ce pic |
| le noyau livre, donnees reelles | 10,39 Gcell/s, **64 % de ce plafond** |

La reconciliation tient en une phrase : **une operation vectorielle fait avancer 16 cellules**, parce
que la disposition est inter-sequence (seize voies = seize travaux differents). Une ligne coute ~15
operations et couvre 16 cellules, donc le bon ratio est **~0,98 operation par cellule**, pas une
cellule par voie et par cycle. Le "56" supposait implicitement une operation par cellule, ce qu'aucun
Smith-Waterman a gaps affines ne fait.

Le desassemblage du binaire livre confirme, boucle rapide du quadruplet a `0x10007ca08` :

| | |
|---|---|
| instructions | 71 pour 64 cellules, soit **1,11 par cellule** |
| dont vectorielles | 63, soit 0,98 par cellule |
| memoire | 6 chargements, 2 ecritures, dont 3 de deversement |

**L'arithmetique est donc finie.** Les 36 % restants ne sont pas dans le corps du DP mais autour :

- la taxe de divergence de voie, **1,09x**, deja mesuree comme irrecuperable par tri ;
- l'epilogue de ligne (`finish_row`), une boucle SCALAIRE sur seize voies par ligne, ~5 % ;
- les colonnes de queue paddees, 1,42 instruction par cellule contre 1,11, ~2,6 % ;
- l'empaquetage du groupe et `extract_group`.

C'est la carte pour la suite. Ce qu'il ne faut PAS retenter : raccourcir la recurrence par cellule.
Elle est a 0,98 operation par cellule contre un plafond machine de 0,97, et les trois candidats
evidents sont deja fermes (argmax paresseux mesure 18 % plus lent, saut de la reparation du N 7 % plus
lent, tri par longueur 0,0 %).

## Recherche multi-agents sur les noyaux SIMD (2026-08-08)

32 agents, 24 leviers examines, **8 survivants** apres passage devant un contradicteur dont le verdict
par defaut etait "refute". Tout est en issues GitHub, une par chantier, avec le detail technique
complet (intrinseques, numeros de ligne, argument d'octet-identite, sources uops.info et SWOG Arm).

**Index : issue nº51.** Plafond par classe de CPU et plan classe.

| issue | chantier | gain |
|---|---|---|
| **nº43** | table XOR sur les noyaux u8 AVX2 / SSE4.1 | **+18 a +50 %** |
| nº44 | idem AVX-512, plus reequilibrage de ports (a conditionner) | +11 a +40 % |
| nº45 | `USQADD` fusionne, et `qsub(h, oe)` partage entre E et F (NEON) | +7,5 % mesure, +6 % |
| nº46 | les deux passes scalaires, `finish_row` et `extract_group` | 2 a 7 % |
| nº47 | les 12 colonnes de padding qui tournent dans le corps de queue | +2,8 a +7 % |
| nº48 | `batched.rs`, 2,0 op/cellule contre 0,95, jamais regle par ISA | 1 a 12 % |
| nº50 | le lemme de portee d'alignement : borner la passe inverse, partager le DP | +2,5 %, +15 % |
| nº49 | **les 16 impasses**, avec leurs raisons | - |

### Le resultat qui explique le 1,25x de l'AVX-512

A 512 bits, chez Golden Cove (Sapphire/Emerald Rapids), **toutes** les operations entieres saturantes
et tous les `max` retombent sur **le seul port 0** (uops.info : `VPMAXUB`/`VPSUBUSB`/`VPADDUSB` ZMM,
1 uop, p0, debit 1,00), parce qu'a cette largeur l'unite vectorielle du port 1 est fusionnee dans p0.
Le noyau devient donc borne par p0 pendant que p5 chome a ~75 %. La largeur n'achete rien : c'est la
mesure du 8573C (10,39 contre 8,29 Gcell/s) expliquee.

### Ce que la recherche a confirme, et qui clot un sujet

**L'arithmetique par cellule est finie.** 0,98 operation par cellule contre un plafond machine de
0,97, a 93 % du debit d'emission crete. Tout ce qui reste est hors du chemin critique (les epilogues
scalaires), hors de l'ISA (la table XOR absente du x86), ou algorithmique (nº50). Quiconque propose de
raccourcir la recurrence doit d'abord lire l'issue nº49.

### Honnetete sur la couverture

7 agents sur 32 sont tombes sur la limite de session, dont la synthese finale et les contradicteurs
des domaines Apple, etat de l'art et algorithmique. Donc **nº50 n'a jamais ete relu par un lecteur
hostile**, et les mesures de nº45 sont celles de l'agent lui-meme. nº43, nº44, nº46, nº47 et nº48 ont,
eux, survecu a la verification.

Question restee ouverte, faute d'avoir fini l'etat de l'art : **quelqu'un a-t-il publie un chiffre
par coeur depassant 10,4 Gcell/s sur des voies de 128 bits, pour du Smith-Waterman local a gaps
affines avec second meilleur score ?**

## Etat de l'art : personne ne fait mieux a contrat de sortie egal (2026-08-08)

Question laissee ouverte par la recherche multi-agents (son agent "etat de l'art" n'a pas fini) :
**quelqu'un a-t-il publie un chiffre par coeur depassant nos 10,4 Gcell/s, pour du Smith-Waterman
local a gaps affines ?** Recherche refaite a la main.

La seule metrique qui permet de comparer des horloges et des largeurs de vecteur differentes est
**cellules par cycle et par 128 bits**. Comparer des GCUPS bruts entre un KNL a 1,4 GHz et un M4 a
4 GHz ne veut rien dire.

| implementation | annee | disposition | largeur | cell/cycle/128 bits | ce qu'elle calcule |
|---|---|---|---|---|---|
| **SWIPE** (Rognes) | 2011 | inter-sequence | SSE 128 | **3,41** | **le score maximum, rien d'autre** |
| **bwa-mem4**, plafond registres | 2026 | inter-sequence | NEON 128 | **4,00** | score, te, qe, score2, te2 |
| **bwa-mem4**, livre | 2026 | inter-sequence | NEON 128 | **2,60** | score, te, qe, score2, te2 |
| SWIMM 2.0 (Rucci), KNL | 2018 | inter-sequence | AVX-512 | ~1,34 | score maximum |
| SWIMM 2.0, bi-Skylake | 2018 | inter-sequence | AVX-512 | ~1,3 | score maximum |
| Parasail (Daily) | 2016 | **striped** | AVX2 256 | ~1,23 | score maximum |

Incertitudes signalees : le nombre de coeurs et l'horloge de Parasail (bi-E5-2670 "24 coeurs",
donc des v3 a 2,3 GHz) et le modele exact du serveur bi-Skylake de SWIMM 2.0 sont deduits, pas
confirmes. Les autres chiffres sont dans les papiers.

### Le seul qui nous depasse, et pourquoi ca ne compte pas tel quel

**SWIPE : 9,1 GCUPS mono-thread a 2,67 GHz, soit 3,41 cellules/cycle en 128 bits.** Le papier
l'ecrit lui-meme : "more than 3.3 cells per physical core in each clock cycle". Boucle interne de
**10 instructions** contre nos ~15,75 par ligne de 16 cellules.

Mais SWIPE **ne calcule que le score maximum**. Le papier est explicite : *"a later version that at
least computes the actual alignments (not just the alignment score) [...] is planned"*. Ni
coordonnees de fin, ni colonne d'argmax, ni second meilleur score.

Nos ~5,75 operations de plus par ligne sont exactement ce que ce contrat de sortie coute : le suivi
de l'argmax par cellule (3 operations), la publication du maximum de ligne pour `score2`, et la
retenue E qui passe par la memoire. Ce sont des sorties dont `mem_matesw` a besoin, pas du gras.

**A contrat de sortie egal, aucun chiffre publie ne nous depasse.** Et notre propre plafond registres
mesure, 4,00 cellules/cycle, est **au-dessus** des 3,41 de SWIPE tout en produisant cinq sorties au
lieu d'une.

### Corroboration inattendue, vieille de quinze ans

SWIPE calcule en **7 bits biaises de 128, avec des operations signees saturantes** (`paddsb`,
`psubsb`, `pmaxub`) et extrait le score de substitution par **`pshufb` sur un profil temporaire**.
C'est trait pour trait la representation "bias-128 signed-8" plus table XOR que la recherche a
proposee pour le portage x86 (issue nº43), trouvee independamment. Rognes avait raison en 2011 et nos
noyaux x86 ne l'ont jamais fait.

### Le comparatif le plus direct qui existe : le BSW de bwa-mem2 lui-meme

Le papier IPDPS 2019 de Vasimuddin, Misra, Li et Aluru donne, pour **le meme algorithme et le meme
contrat de sortie que nous**, sur Xeon Platinum 8180 en AVX-512 8 bits :

| | bwa-mem2 BSW | bwa-mem4 |
|---|---|---|
| IPC | 2,17 (limite par 2 ports SIMD, effet p0 documente plus haut) | 3,8 op/cycle mesure |
| part du temps en calcul de cellules | **43 %** | la sonde ne mesure que le noyau |
| cellules utiles / cellules calculees | **~50 %**, soit une taxe de divergence de **2x** | **1,04x** (`batched`), **1,09x** (`matesw`) |
| part du temps en cellules utiles | **21,5 %** | |

Leur pre-traitement AoS vers SoA pese 33 % du temps du noyau et leurs deux ajustements de bande 24 %.
Autrement dit, la reference dont nous devons reproduire la sortie octet pour octet passe **quatre
cinquiemes de son temps de BSW ailleurs que dans des cellules utiles**. Notre taxe de divergence est
deux fois plus faible que la leur d'un facteur ~2.

### Ce que ca ferme, et ce que ca n'ouvre pas

Ferme : la question de savoir si nous laissons un facteur connu sur la table. Non. Sur 128 bits, a
sorties egales, la litterature ne fait pas mieux, et l'ecart avec le seul chiffre superieur (SWIPE)
s'explique entierement par ce qu'il ne calcule pas.

N'ouvre rien de neuf : le champ CPU s'est arrete vers 2018, l'essentiel de l'activite recente est
GPU (CUDASW++ 4.0, 2024) et donc hors sujet pour une comparaison par coeur.

## #45 : `USQADD` et le `qsub(h, oe)` partage, +10,3 % de noyau, -2,7 % de CPU (2026-08-08)

Premiere issue du backlog SIMD ouverte par la recherche multi-agents, et la premiere ou une
projection annoncee tient a la mesure. Deux reecritures du corps de colonne du noyau de rescue u8
NEON (`crates/bwa-neon/src/matesw.rs`), independantes, chacune derriere son propre commutateur pour
un A/B a un seul levier :

* **A, `BWA4_RESCUE_USQADD`** : la paire `vqsubq_u8(vqaddq_u8(d, s), bias_v)` devient un seul
  `vsqaddq_u8(d, s)`, l'instruction `USQADD Vd.16B, Vn.16B` d'AArch64 (accumulation saturante NON
  signee d'un addend SIGNE). La table de scores XOR est reconstruite en deltas signes, le biais
  disparait entierement. Une operation de moins par cellule, et la chaine de dependance diagonale
  passe de 6 cycles a 3.
* **B, `BWA4_RESCUE_SHAREOE`** : quand `oe_del == oe_ins` (le `-O 6,6 -E 1,1` de bwa, c'est-a-dire le
  cas qui selectionne ce noyau en pratique), `vqsubq_u8(h, oe)` est calcule une fois et alimente E et
  F. Cinq operations la ou il y en avait six.

Les deux formes sont **monomorphisees** en parametres const generiques, pas testees dans la boucle :
le repli du N avait deja paye 7 % pour un `if` invariant que LLVM n'a pas sorti de la boucle.

### Ce que la mesure donne

A/B entrelace, meme binaire, `-t4`, 500 k paires GIAB contre l'index chr21, **medianes de 9 courses**,
secondes CPU. Sonde noyau `BWA4_MATESW_TIME` (18,33 G cellules nominales) et CPU du processus entier :

| levier | noyau (s CPU) | vs base | debit noyau | CPU total (s) | vs base | victoires |
|---|---|---|---|---|---|---|
| base (`USQADD=0 SHAREOE=0`) | 2,25 | | 8,15 Gcell/s | 7,39 | | |
| A seul | 2,16 | **+4,2 %** | 8,49 Gcell/s | 7,31 | -1,1 % | 9/9 |
| B seul | 2,15 | **+4,7 %** | 8,53 Gcell/s | 7,32 | -1,0 % | 9/9 |
| A + B | **2,04** | **+10,3 %** | **8,99 Gcell/s** | **7,19** | **-2,7 %** | 9/9 |

Les deux leviers s'additionnent presque exactement (4,2 + 4,7 = 8,9 attendu, 10,3 obtenu, le surplus
venant des registres liberes, voir ci-dessous). L'issue annoncait +7,5 % et +6 % sur le corps du
quadruplet seul, mesures en assembleur a la main ; la sonde couvre en plus les corps de queue,
`finish_row`, `extract_group` et l'empaquetage des groupes, donc un rendement moindre sur le noyau
entier etait attendu. L'ecart entre 2,7 % de CPU total et 10,3 % de noyau est simplement la part du
rescue dans cette course : 2,25 s sur 7,39 s.

### Le desassemblage, et les deversements qui disparaissent

Boucle rapide du quadruplet, binaire livre, comparee au releve de la section « plafond du noyau » :

| | avant | apres |
|---|---|---|
| instructions pour 64 cellules | 71 (1,11 / cellule) | **64 (1,00 / cellule)** |
| operations vectorielles arithmetiques | 63 | **53 (0,83 / cellule)** |
| chargements | 6, **dont 3 de deversement** | **3, aucun deversement** |
| ecritures | 2 | 2 |

Les trois deversements ont disparu : le jeu de constantes vivantes perd `bias_v` (levier A) et
`oe_ins_v` (levier B), ce qui ramene le corps sous la limite des 32 registres. C'est le « benefice
lateral » que l'issue annoncait sans le chiffrer, et il explique la super-additivite.

### L'octet-identite

Les deux leviers sont exacts, pas seulement egaux en score, et les preuves sont consignees aux points
d'usage dans le code. Pour B : la soustraction saturante non signee est monotone et son ecretage a 0
preserve l'ordre, donc `qsub(max(a, b), c) == max(qsub(a, c), qsub(b, c))` ; avec `h = max(mfe, f)` et
`oe_ins >= e_ins`, le terme surnumeraire est absorbe.

Pour A il reste **une** entree ou les deux formes different : une lecture de table hors plage, cas
qui ne survient que dans le corps rapide pour une voie deja au-dela de son `tlen` alors que d'autres
voies du meme groupe font tourner la boucle. `vqtbl1q` rend 0, donc la forme biaisee donne `d - bias`
et `USQADD` donne `d`. Ces cellules sont mortes par construction. Cela n'a **pas** ete laisse a
l'argument : `matesw_ragged_tlen_equals_scalar` construit deux groupes de seize fenetres de 100 a
3200 bases (la plus longue tourne 15x plus longtemps que la plus courte) avec un `endsc` fini pour
declencher aussi le gel, et compare **les quatre monomorphisations** au scalaire. Le
`debug_assert!` sur la non-saturation de la table est promu en `assert!`, la correctness en release
en dependant desormais.

Gates : `check.sh` vert (fmt, clippy `-D warnings`, tests, plus la passe x86_64 sous Rosetta),
`oracle_diff.sh` vert, et le corps SAM des 500 k paires PE **octet-identique a bwa-mem2 2.3** pour les
quatre combinaisons de commutateurs (md5 `7178e85d…`, seule la ligne `@PG` differant de l'oracle,
comme toujours).

## #47 : les colonnes de bourrage ont leur propre corps, et le gain depend de la longueur de lecture (2026-08-08)

`ksw_padded_qlen` arrondit chaque requete a un vecteur entier, donc les colonnes `[qlen, qmax)` sont
du pur bourrage de profil ksw. Elles ne sont **pas mortes** : une colonne ZPAD score 0 et propage donc
la diagonale, son H alimente `rowmax` et donc `score2`. Elles traversaient le corps de queue complet,
le plus cher des deux.

Option A de l'issue, livree : un **troisieme regime de colonnes**. Quand toutes les voies vivantes du
groupe se bourrent jusqu'au meme `qmax` (le cas ordinaire d'un lot de lectures d'une meme course), les
colonnes `[max qlen, qmax)` ont ZPAD sur **toutes** les voies vivantes, donc `zpad_mask` vaut
tout-a-un et le calcul de score se replie sur `diag = d`. Le corps garde la recurrence H/E/F et perd
tout le reste : pas de chargement de la colonne de requete, pas d'`EOR`, pas de `TBL`, pas d'addition
de substitution, aucun des quatre melanges de bourrage. Le commutateur `BWA4_RESCUE_ZPADCOL=0`
renvoie ces colonnes au corps de queue.

La borne est calculee comme `[max qlen, min qlen_bourree)`, ce qui est plus general que l'egalite des
`qlen` demandee par l'issue : deux lectures de 145 et 150 bases se bourrent toutes deux a 160 et
partagent donc les colonnes 150 a 160. Quand la condition tombe, `n_pad == qmax` et le noyau est
exactement celui d'avant.

### Le gain n'est pas un nombre, c'est une fonction de la fraction bourree

C'est le resultat principal de cette issue, et il n'etait pas dans l'enonce. A/B entrelace, meme
binaire, `-t4`, sonde `BWA4_MATESW_TIME`, medianes :

| jeu | longueur de lecture | colonnes bourrees | noyau base | noyau avec | gain noyau | victoires |
|---|---|---|---|---|---|---|
| GIAB `m1/m2` sur chr21, 500 k paires | 49 bp | 15 de 64, **23,4 %** | 2,04 s | **1,90 s** | **+7,4 %** | 9/9 (9 courses) |
| `r1_500k/r2_500k` sur `genome.fa` | 150 bp | 10 de 160, **6,25 %** | 1,30 s | **1,28 s** | **+1,6 %** | 4/5, 1 nul (5 courses) |

Sur le jeu 49 bp, le CPU du processus entier passe de **7,23 s a 7,07 s, -2,2 %**, 9 victoires sur 9.

L'issue annoncait +2,8 % pour 12 colonnes bourrees sur 160, soit 7,5 % de bourrage. Notre point a
6,25 % de bourrage donne +1,6 %, donc **legerement en dessous de la projection**, et le point a 23,4 %
donne +7,4 %. Les deux sont coherents avec une loi a peu pres proportionnelle a la fraction bourree.
La consequence pratique : **une lecture juste au-dessus d'un multiple de 16 paie beaucoup, une lecture
pile sur le multiple ne paie rien**, et 150 bp, la longueur Illumina standard, est presque le meilleur
cas possible. C'est le chiffre 1,6 % qui vaut pour un run de production typique.

### Ce qui n'a pas ete fait

* **Option B** (forme close supprimant les colonnes bourrees de la boucle, +7 % annonce) : non tentee.
  Sa preuve n'est jamais passee par le verificateur adversaire de la recherche, et l'option A la
  subsume deja pour la majeure partie du gain sur la forme de production.
* **Les noyaux x86** (AVX2, AVX-512, SSE4.1) gardent leurs deux regimes. Le portage se mesure sur une
  machine x86, pas ici.

### L'octet-identite

`diag = d` est exactement ce que le melange calcule quand `zpad_mask` vaut tout-a-un, donc le nouveau
corps est l'ancien avec un masque prouve constant replie. Le melange de bourrage est abandonne sur
l'argument que le corps rapide fait deja : une voie dont la cible est PAD ici est une voie au-dela de
son propre `tlen`, toutes les operations sont locales a la voie, et `finish_row` comme
`extract_group` s'arretent a `limit[l]`.

Le test `matesw_uniform_qlen_pad_columns_equal_scalar` ouvre le regime expres : seize voies de 147 bp
(bourrees a 160, donc 13 colonnes) avec des cibles de longueurs echelonnees de 290 a 3080. Une erreur
de la premiere redaction du test, corrigee et consignee dans le code parce qu'elle est instructive :
**les colonnes bourrees ne peuvent jamais deplacer `score`, `te` ni `qe`**. Une colonne bourree recopie
la diagonale, donc son H egale le H d'une ligne anterieure a une colonne reelle, deja verse dans
`gmax` quand cette ligne s'est terminee, et le `>` strict de `finish_row` le rejette. Elles peuvent en
revanche deplacer `score2`/`te2`, qui viennent de `rowmax` que toute colonne ecrit. C'est donc
`score2` qui porte la preuve, et le test verifie qu'il est bien peuple.

Gates : `check.sh` vert, `oracle_diff.sh` vert, et le corps SAM PE **octet-identique a bwa-mem2 2.3**
avec le commutateur dans les deux positions, sur les **deux** jeux (500 k paires 49 bp sur chr21,
md5 `7178e85d…` ; 500 k paires 150 bp sur `genome.fa`, md5 `3a51acef…`, celui-la compare directement
a la sortie de l'oracle).

## #46 : les deux passes scalaires ne coutent rien, et pourquoi le microbenchmark disait le contraire (2026-08-08)

Resultat **negatif**, les deux leviers ecrits, mesures, et retires. L'issue les presentait comme la
plus grosse part identifiee des 36 % d'ecart au plafond. Sur cette machine et sur les deux jeux de
lectures presents sur le disque, ils valent **0,0 %**.

### A, la porte vectorielle sur `finish_row`

L'epilogue de ligne est du code scalaire seize voies execute une fois par ligne cible. La porte
proposee tient en une comparaison : `hit = imax > gmax_v` avec `gmax_v` a 255 dans les voies mortes,
ce qui est exactement la condition sous laquelle le corps scalaire modifie quelque chose. Ecrite avec
le miroir u8 de `gmax`, le maintien des voies mortes dans le balayage de sortie anticipee (deja
present, une fois par groupe de lignes) et le commutateur `BWA4_RESCUE_ROWGATE`.

| jeu | noyau porte off | porte on | ecart |
|---|---|---|---|
| chr21, 49 bp | 1,90 s | 1,90 s | **0,0 %**, 9 courses |
| genome, 150 bp | 1,28 s | 1,27 s | -0,8 %, 5 victoires 2 nuls sur 7 |

La raison est mesurable et a ete mesuree : **la porte s'ouvre 20,3 % du temps sur le jeu 49 bp et
61,8 % sur le jeu 150 bp** (compteur temporaire sur 23,9 M et 6,9 M de lignes). L'issue supposait un
`gmax` qui se stabilise en quelques dizaines de lignes ; c'est vrai par voie, mais la porte est
l'union de seize voies, et une seule voie qui progresse suffit a l'ouvrir. Sauter 80 % des epilogues
ne rapporte rien, donc l'epilogue lui-meme est deja quasi gratuit : ses deux branches sont
parfaitement predites, contrairement au microbenchmark `bench3.c` qui les mesurait a 21,93 cycles.

### B, le pre-filtre vectoriel de `extract_group`, et le piege de mesure

Meme conclusion, avec une explication plus interessante. La version livree parcourt `rowmax` une fois
**par voie**, a pas de 16 octets. La version ecrite ici le parcourt une fois par **ligne**, une charge
de 16 octets et un `UMAXV` decidant les seize suivis a la fois, et ne descend en scalaire que sur les
lignes qui passent `minsc`. Sur la forme reelle, `minsc = 19` et 492 lignes par voie, il y a de
l'ordre de zero a trois lignes qualifiantes.

Les trois mesures, dans l'ordre ou elles ont ete faites :

| mesure | ancien | nouveau | rapport |
|---|---|---|---|
| microbenchmark isole, `tmax = 600` | 5,94 us/groupe | 0,86 us/groupe | **6,9x** |
| idem, `tmax = 1401` (la forme de la recherche) | 13,06 | 1,79 | **7,3x** |
| chronometre **autour de l'appel**, en situation, 48 098 groupes | 0,227 s CPU | 0,055 s CPU | **4,1x** |
| **noyau complet, binaire propre, A/B entrelace** | **1,77 s** | **1,78 s** | **0,0 %** |

Les trois premieres lignes disent 4x a 7x, la quatrieme dit rien. Ce n'est pas une contradiction,
c'est un artefact de mesure, et il a ete isole : **poser `#[inline(never)]` sur `extract_group` dans
le binaire propre, sans aucun chronometre, reproduit exactement l'ecart** (1,89 s contre 1,72 s). Le
chronometre `Instant::now()` autour de l'appel est une barriere d'optimisation ; il empeche LLVM
d'incorporer `extract_group` dans la boucle de groupes du noyau, ou le balayage sous-`minsc` devient
aussi bon marche que le pre-filtre vectoriel. Le 7,6x « pas croise contre sequentiel » de la recherche
est reel sur une fonction isolee et **ne survit pas a l'incorporation**.

C'est la lecon a retenir de cette issue : ici, chronometrer une sous-partie d'un noyau chaud la rend
plus chere. La seule mesure qui compte reste l'A/B entrelace du binaire livre.

### Ce qui reste ouvert

Le point **C** de l'issue est intact et n'a rien a voir avec ce qui precede : les noyaux u8 x86
(AVX2, SSE4.1, AVX-512) gardent un `rowmax` en **i32** et font 32 ou 64 ecritures scalaires gardees
par ligne, la ou NEON publie la ligne en un `vst1q_u8`. A la forme moyenne d'AVX-512 le tampon fait
359 KB par groupe, hors L1. Ce portage se mesure sur une machine x86, pas ici, et l'issue reste
ouverte pour lui.

## #48 : les deux leviers faisables ici etaient deja appliques par LLVM (2026-08-08)

Deuxieme resultat **negatif**, et celui-la se demontre plutot qu'il ne se mesure. Sur les quatre
leviers de l'issue, deux sont faisables sur cette machine (A, supprimer le `max(., 0)` redondant des
chemins d'ouverture de gap ; D, reutiliser `j_v` au lieu de le rediffuser). C sont des compares x86,
B est explicitement « a faire en dernier ».

A et D ont ete ecrits, avec le parametre const `TUNE` sur le noyau genere et la fente de macro
`clamp0` liee a une identite a deux arguments pour les huit instanciations u8. Puis :

* **A/B du processus entier**, `-t4`, 500 k paires 150 bp, 11 courses entrelacees : **32,65 s contre
  32,65 s**, exactement rien.
* **microbenchmark du noyau seul**, 4096 travaux d'extension, un bras apres l'autre : +1,2 %, +1,3 %,
  +4,1 %, +6,2 %. C'est ce qu'on aurait pu publier.
* **le meme microbenchmark, bras alternes, medianes de 15 rondes** : **0,999x et 0,995x, 7 et 6
  victoires sur 15**. Le +6 % etait un effet d'ordre, pas un levier. La discipline d'entrelacement
  n'est pas une precaution de style, elle est ce qui separe les deux chiffres.

La preuve arrive ensuite, et elle est sans appel. Dans le binaire livre il n'existe qu'**un seul
symbole** pour le noyau :

```
batched_extend_neon_u8Kb0_EB6_      (Kb0_ = TUNE:bool = false)
batched_extend_neon_i16Kb0_EB6_
```

La monomorphisation `TUNE = true` a ete **fusionnee avec la `false`** : les deux compilent vers un
code machine identique. LLVM applique deja les deux reecritures. C'est evident a posteriori :
`umax(x, 0)` sur un type **non signe** se replie en `x` sans meme regarder d'ou vient `x`, et une
diffusion identique du meme scalaire est eliminee par CSE. L'issue comptait 33 operations ALU par
colonne dans le source ; l'assembleur n'en a jamais eu 33.

Le tout est revenu en arriere : `batched.rs` est intact.

### Ce que ce resultat dit du reste de l'issue

Le corps de colonne de `batched.rs` n'est pas limite par l'ALU vectorielle comme le noyau de rescue
l'est. Deux consequences pour la suite :

* le levier **C** (masque de bande en espace signe sur x86, +8 a +12 % annonce) est a re-instruire
  avant d'etre code. Il compte lui aussi des uops de source ; `cge_epu8` compte pour 2 et `clt_epu8`
  pour 3 chez l'agent, mais rien ne prouve que le compilateur ne les a pas deja arranges. **Compter
  les instructions dans le desassemblage x86, pas dans le source**, avant d'ecrire une ligne ;
* le levier **B** (replier la pre-passe de substitution dans la boucle principale) reste le seul
  candidat serieux, parce que c'est le seul qui retire un **acces memoire** plutot que des operations
  ALU : une ecriture pleine largeur et une lecture pleine largeur par colonne, dont le compilateur ne
  peut pas se debarrasser tout seul puisqu'elles traversent deux boucles. Il reste aussi le plus
  risque, pour la raison enumeree dans l'issue.

## #50B : les fenetres de rescue ne se chevauchent pas, le 15 % n'existe pas (2026-08-08)

L'issue demandait explicitement de **mesurer avant d'implementer**. C'est fait, avec une sonde
permanente, `BWA4_RESCUE_CLUSTER`, et la reponse est nette : la distribution est fine, donc le levier
n'a pas de matiere.

La sonde prend, par appel de noyau, les coordonnees de reference de toutes les fenetres de rescue,
les trie, forme les grappes de chevauchement sur un meme contig, et facture la contre-proposition de
l'issue (`W + A * 271` lignes par grappe de `A` fenetres couvrant `W` bases) contre les lignes que le
noyau parcourt reellement.

| | 500 k paires 150 bp, `genome.fa` | 500 k paires 49 bp, chr21 |
|---|---|---|
| fenetres | 144 617 | 647 290 |
| grappes de chevauchement | 133 591 | 589 292 |
| **A moyen** | **1,08** | **1,10** |
| A maximal | 14 | 192 |
| fenetres seules dans leur grappe | **86,5 %** | **84,8 %** |
| fenetres dans une grappe de A >= 4 | 1,7 % | 2,8 % |

Le plafond du levier, calcule en n'appliquant le partage qu'aux grappes de `A >= 2` et en laissant
les solitaires sur le chemin actuel :

| | 150 bp | 49 bp |
|---|---|---|
| lignes dans des grappes A >= 2 | 13,5 % du total | 15,2 % |
| lignes du noyau apres partage | **1,0120x** | **0,9993x** |

Donc **+1,2 % de lignes de noyau au mieux, et -0,07 % sur l'autre jeu**, avant tout cout
d'implementation, sur un noyau qui represente lui-meme 10 a 25 % de la course. Le +15 % annonce
supposait `A = 50` dans une grappe de 3 kb ; cette forme existe (le A maximal est 192 sur chr21) mais
elle concerne moins de 1 % des fenetres. Applique a tout le monde, le partage serait **plus cher** que
le code actuel (0,72x), parce qu'une grappe solitaire paierait `W + 271` lignes la ou elle en paie
`W`.

Rien n'est implemente, et c'est le resultat attendu de l'issue. Seule la sonde est livree.

**Le levier #50A reste ouvert** et n'a rien a voir avec le chevauchement : la passe inverse
(`KSW_XSTART`) dimensionne son `rowmax`, son eparpillement scalaire et son `extract_group` sur
`te + 1` alors que son `endsc` se declenche toujours (la preuve de l'issue est correcte : le `gmax`
de la passe inverse vaut exactement `score`, donc elle gele toujours). C'est un travail de
dimensionnement de tampon, mesurable independamment, et il faudra le confronter au resultat de #46B :
`extract_group` ne coute rien une fois incorpore, donc sur les trois couts que #50A cite, le
troisieme est deja nul et il reste l'allocation et l'eparpillement.

## #53 etape 0 : la taille des lots, et elle repond deux choses differentes (2026-08-08)

L'issue interdisait de decider quoi que ce soit de l'architecture GPU avant ce chiffre. Il est
mesure. `BWA4_EXTEND_SHAPE` et `BWA4_MATESW_TIME` comptent desormais les **appels** de noyau et
publient un histogramme en puissances de deux de `jobs.len()`, cote extension comme cote rescue.

200 k paires, `-t8`, deux jeux :

### Extension (`batched_extend`, la couture `SwBackend`)

| | 49 bp, chr21 | **150 bp, `genome.fa`** |
|---|---|---|
| appels | 392 | 392 |
| **moyenne jobs/appel** | 1 302 | **5 381** |
| plus gros appel | 3 263 | 7 284 |
| ou sont les jobs | 80,3 % dans 1024-2047 | **99,7 % dans 4096-8191** |
| cellules nominales | 278 M | 24,0 G |

A la longueur de lecture de production, **les lots d'extension font plus de 4 000 jobs**, ce qui est
le seuil que l'issue posait. Donc : **un lancement de noyau par appel suffit**, la couture reste
quasi synchrone, et le travail est le petit des deux scenarios. C'est aussi le cote qui porte les
cellules : 24 G cellules nominales pour 200 k paires.

### Mate rescue (`batched_ksw_align2`)

| | 49 bp, chr21 | **150 bp, `genome.fa`** |
|---|---|---|
| appels | 800 | 800 |
| **moyenne jobs/appel** | 328 | **72** |
| plus gros appel | 2 031 | **275** |
| ou sont les jobs | 34 % dans 512-1023 | 36 % dans 32-63, rien au-dessus de 511 |
| cellules nominales | 7,3 G | 5,4 G |

**C'est l'inverse.** A 150 bp le rescue soumet des lots de quelques dizaines de jobs, deux ordres de
grandeur sous le seuil. Meme en agregeant les huit threads d'un `-t8`, on serait a environ 576 jobs,
toujours en dessous. Un backend GPU pour le rescue demande donc bien la **file d'agregation
inter-threads** que l'issue decrivait comme « nettement plus gros », et elle change le contrat de
`SwBackend`.

### Ce que ce chiffre decide

1. **L'extension d'abord, le rescue ensuite.** L'ordre inverse de l'intuition, qui mettait le rescue
   en premier parce qu'il est le plus gros poste CPU. C'est le lot, pas le poste, qui decide.
2. La couture asynchrone de l'etape 1 (`submit`/`collect`) reste utile pour recouvrir, mais elle
   n'est **pas** un prealable cote extension : a 5 381 jobs par appel un lancement synchrone est
   deja rentable.
3. Cote rescue, tant que la file inter-threads n'existe pas, il n'y a rien a porter. Note toutefois
   que le rescue a **93 900 cellules par job** contre 11 400 pour l'extension : peu de jobs, mais
   chacun est gros, donc le parallelisme intra-lot n'est pas si maigre qu'il en a l'air. C'est a
   re-instruire avec le modele de cout d'un noyau GPU, pas avec le seuil « nombre de jobs ».

## #54 : la barriere d'octet-identite GPU, ecrite avant le premier noyau (2026-08-08)

Le precedent de l'issue est le bon argument : GPU-BWA-MEM (ICS 2023) affirme rendre les memes
resultats que BWA-MEM et **ne publie aucune methode de validation**. Voici la methode, ecrite avant
qu'il y ait une ligne de Metal ou de CUDA, et deja verte sur les backends CPU.

### Deux barrieres generiques sur `SwBackend`

Ecrites contre le trait, pas contre NEON, donc un futur backend GPU les passe **sans modification**.
Elles ciblent les deux pieges que les balayages aleatoires existants ne peuvent pas atteindre.

* `assert_backend_tie_rule_matches_scalar` (**piege 2**). Le noyau de rescue garde le **premier**
  maximum (`vcgtq_u8`, `>` strict) et le noyau d'extension garde le **dernier** (`$cge`, non strict,
  `ksw.cpp:491`). Les deux sont opposes expres. Un portage qui les intervertit rend le bon `score` et
  des `te`/`qe` differents, uniquement sur des egalites, que la sequence aleatoire ne produit
  presque jamais. La barriere les fabrique : cibles periodiques (periodes 1, 2, 3, 4, 7) contre des
  requetes en phase, ou toutes les cellules d'une ligne sont a egalite. La periode 7 est la parce
  qu'elle n'a de facteur commun avec aucune largeur SIMD du projet, donc une egalite ne peut pas
  coincider avec une frontiere de voie par chance.
* `assert_backend_batch_order_invariant` (**piege 6**). Un noyau GPU voudra regrouper par longueur.
  C'est legal tant que le remplissage des voies mortes n'influence pas les voies vivantes. La
  barriere passe **les memes 200 jobs dans dix permutations tirees au hasard** et exige que le
  resultat de chaque job soit identique dans les dix, et egal au `extend` job par job.

Une troisieme, cote rescue, vit dans `matesw.rs` : `matesw_tie_rule_equals_scalar`, meme generateur
periodique, compare a `ksw_align2`.

Les trois sont vertes sur le backend scalaire et sur NEON. Sur le scalaire elles ne peuvent pas
echouer utilement, et c'est le but : cela prouve que les **generateurs** sont bien formes (des
entrees periodiques qui produisent vraiment des egalites, des permutations qui permutent vraiment)
avant qu'un backend GPU soit juge par elles.

### Le gate d'integration, `scripts/gpu_parity.sh`

Trois etapes, dans cet ordre :

1. **determinisme** : trois courses du chemin CPU doivent rendre **trois md5 egaux**. C'est ce qui
   attrape une erreur memoire non corrigee ou un pilote qui reordonne, et rien de ce qui suit n'a de
   sens si la reference n'est pas reproductible. **Cette etape tourne des aujourd'hui et passe**
   (md5 `791c21c2…` sur 200 k paires 150 bp, `-t8`) ;
2. **parite** `BWA4_GPU=off` contre `BWA4_GPU=<backend>`, meme binaire ;
3. **invariance au partage** (piege 7) : `BWA4_GPU_SPLIT` balaye 0,0 / 0,25 / 0,5 / 0,75 / 1,0 et les
   cinq md5 doivent etre egaux. Si cette etape passe, le co-ordonnancement dynamique est sur par
   construction et n'a plus a etre re-prouve.

Les etapes 2 et 3 **sautent bruyamment** tant qu'aucun backend GPU n'est compile dans le binaire :
elles annoncent le saut plutot que de passer a vide.

### Ce qui reste des sept pieges

Les pieges 1 (aucun flottant, a verifier en relisant le PTX ou l'IR Metal), 3 (le seuil de
saturation a 255 et le repli i16) et 5 (`max_off` et le compte de jobs requeues par round) demandent
soit du code GPU, soit une sonde de plus, et restent ouverts dans l'issue. Le piege 4 (le codage
`N_TARGET = 12`) est deja couvert par les generateurs, qui tirent `% 5` et non `% 4`, et la nouvelle
barriere d'ordre respecte la meme regle.

## #55 etape 0 : bloquee, il n'y a pas de compilateur Metal sur cette machine (2026-08-08)

L'issue interdit d'ecrire une ligne de Metal avant d'avoir repondu a une question : **MSL expose-t-il
l'arithmetique entiere saturee ?** Le plafond annonce en depend directement, un `uchar4` a 4 cellules
par thread donnant ~4 operations par cellule et ~2 Tcell/s, contre une emulation `min`/`max` qui en
coute deux de plus par saturation.

La question n'a pas pu etre tranchee ici, et c'est le resultat a consigner :

* **le compilateur Metal n'est pas installe.** `xcrun -sdk macosx metal --version` rend
  `unable to find utility "metal", not a developer tool or in PATH`, et il n'y a pas de
  `/Applications/Xcode.app` : cette machine n'a que les Command Line Tools. Le microbenchmark MSL
  autonome que l'issue demande **ne peut pas etre compile ici**, encore moins mesure. Installer Xcode
  est donc le prealable materiel de toute la piste #55, au meme titre qu'une machine x86 l'est pour
  #43 et #44 ;
* **la specification n'a pas non plus tranche la question par la documentation.** Le miroir public de
  la Metal Shading Language Specification atteignable d'ici est une traduction partielle : sa table
  des fonctions entieres liste bien `addsat(T x, T y)` mais s'arrete avant `subsat`, et le PDF
  officiel d'Apple depasse la taille que l'outil de recuperation accepte. **Donc : `addsat` semble
  exister, `subsat` n'est pas confirme, et rien n'est confirme sur les types vectoriels comme
  `uchar4`.** Cela s'affirme depuis un compilateur, pas depuis un moteur de recherche.

### Ce qui change dans le calcul si `subsat` manque

L'emulation existe et elle est exacte, ce n'est pas un obstacle de correction :

```
subsat_u(x, y) = x - min(x, y)          // non signe, exact
addsat_u(x, y) = x + min(y, MAX - x)    // non signe, exact
```

Mais elle coute **une operation de plus par soustraction saturee**, et le corps de colonne du noyau
de rescue en compte quatre par ligne sur un total de treize operations vectorielles (releve du
desassemblage NEON, section #45). Une emulation ferait donc passer ce corps de 13 a 17 operations,
soit **+31 %**, et le plafond de 2 Tcell/s annonce par l'issue tomberait dans la meme proportion. Ce
n'est pas fatal (le besoin est de 33 a 116 Gcell/s), mais **le chiffre de l'issue est a corriger de
+31 % dans le pire cas**, et il ne peut pas etre publie tel quel avant que le microbenchmark
tranche.

### Le protocole exact a executer quand Xcode sera la

1. compiler un `.metal` autonome qui utilise `addsat` et `subsat` sur `uchar4`, `ushort4` et `int` ;
2. si la compilation echoue sur `subsat`, refaire la variante emulee et **compter les instructions
   dans l'AIR / l'assembleur GPU**, pas dans le source : la lecon de #48 est que le compilateur
   applique deja des reecritures que le comptage au niveau du source attribue au developpeur ;
3. mesurer les trois variantes (`uchar4` sature, `ushort4` emule, `int` une cellule par voie) et ne
   choisir l'architecture qu'ensuite.

## #48B : la pre-passe de substitution repliee, +4 a +5 % de noyau, -0,9 % de CPU (2026-08-08)

Suite de la section #48 : apres deux leviers nuls parce que LLVM les appliquait deja, le troisieme
est le seul qui retire un **acces memoire** plutot que des operations ALU, et c'est celui qui paie.

La pre-passe calculait le score de substitution de chaque colonne de la bande dans une boucle
separee, l'ecrivait dans `sbt_buf`, et la boucle principale le rechargeait immediatement apres. Une
ecriture pleine largeur et une lecture pleine largeur par colonne, plus une seconde traversee de la
bande, pour transporter une valeur entre deux boucles de la meme iteration. Repliee dans la boucle de
colonne, la meme expression sur les memes entrees, donc l'octet-identite est immediate.

### La difference avec #48A et #48D se voit dans les symboles

C'etait le test decisif de la section precedente, il est repasse ici :

```
batched_extend_neon_u8Kb0_EB6_     batched_extend_neon_i16Kb0_EB6_
batched_extend_neon_u8Kb1_EB6_     batched_extend_neon_i16Kb1_EB6_
```

**Les deux monomorphisations existent**, la ou celles de #48A/D avaient fusionne. Le compilateur ne
sait pas faire cette transformation, et c'etait la prediction.

Boucle de colonne u8, desassemblage (identification par forme) :

| | forme separee | forme repliee |
|---|---|---|
| instructions de la boucle principale | 41 | 40 |
| chargements | 5 | **3** |
| ecritures | 3 | **2** |
| plus une boucle de pre-passe sur les memes colonnes | oui | **non** |

### Mesure

| mesure | separee | repliee | gain | victoires |
|---|---|---|---|---|
| microbenchmark du noyau, bras alternes, medianes de 15, deux executions | 9,87 / 9,79 ms | 9,47 / 9,28 ms | **+4,2 % et +5,5 %** | **15/15 et 15/15** |
| CPU du processus, 500 k paires **150 bp**, 11 rondes entrelacees | 33,04 s | **32,74 s** | **-0,91 %** | **11/11** |
| CPU du processus, 500 k paires **49 bp** | 7,09 s | 7,08 s | 0,0 % | nul |

Le contraste entre les deux jeux n'est pas du bruit et se lit dans la sonde #53 : l'extension
represente **24,0 G cellules nominales** sur le jeu 150 bp contre **278 M** sur le jeu 49 bp. Il n'y a
tout simplement presque pas d'extension a accelerer quand les lectures font 49 bases. **-0,91 % est
le chiffre pour un run de production.**

Note de methode, la meme que pour #48A : le microbenchmark a bras alternes est ce qui rend ce +4,2 %
croyable. Le meme harnais, un bras apres l'autre, avait attribue +6 % a un levier qui valait zero.

Le levier **C** (masque de bande en espace signe) reste ouvert et x86 uniquement.

## #50A : la passe inverse bornee par le lemme, 55 a 74 % de lignes en moins pour 0,8 % (2026-08-08)

La cible de la passe inverse est `target[..=te]` retournee, soit `te + 1` lignes, mais l'alignement
qu'elle doit retrouver ne peut pas en couvrir plus que le lemme d'envergure n'autorise :

```
score <= max_sc * Q - o_del - (T - Q) * e_del
  =>  T <= qlen + (max_sc * qlen - score - o_del) / e_del
```

avec `qlen = qe + 1` et `score` celui de la passe avant, pas `minsc` : c'est ce qui rend la borne
serree, un score plus haut imposant une envergure plus courte. La division entiere tronque, et c'est
le bon sens : `T` est un entier majore par une quantite reelle. Livre derriere
`BWA4_RESCUE_REVBOUND`.

Ce que la borne enleve, mesure :

| jeu | lignes de cible inverse, sans borne | avec borne | enleve |
|---|---|---|---|
| 500 k paires 150 bp, `genome.fa` | 35 790 921 | 16 162 618 | **54,8 %** |
| 500 k paires 49 bp, chr21 | 23 730 481 | 6 169 194 | **74,0 %** |

Ce que cela rapporte, A/B entrelace sur la sonde `BWA4_MATESW_TIME`, 15 rondes :

| jeu | sans borne | avec borne | ecart | victoires |
|---|---|---|---|---|
| 150 bp | 1,28 s | **1,27 s** | **-0,8 %** | 7 victoires, 1 defaite, 7 nuls |
| 49 bp | 1,90 s | 1,90 s | 0,0 % | nul |

**Enlever plus de la moitie des lignes de la cible inverse vaut 0,8 %.** C'est la meme lecon que
#46B, et il faut la consigner comme telle : le travail par ligne autour du DP (le remplissage de
l'arene, l'eparpillement SoA, l'allocation de `rowmax`) est bien moins cher que la recherche ne le
supposait. Le +2,5 % annonce par l'issue supposait trois couts, dont l'un (`extract_group`) avait
deja ete mesure a zero en #46B ; les deux autres valent ensemble 0,8 %.

Garde malgre tout : la borne est prouvee, octet-identique sur les deux jeux, et l'ecart est
consistant en signe (7 contre 1) meme s'il est a la resolution de la sonde. Elle divise aussi par
deux a quatre le trafic memoire de cette passe, ce qui ne se voit pas a `-t4` sur une machine a
546 GB/s mais n'est pas rien ailleurs.

Ce qui n'est PAS raccourci, et l'issue le disait deja : le DP lui-meme. `endsc = score` se declenche
toujours (le `gmax` de la passe inverse vaut exactement `score`), donc la boucle de lignes s'arretait
deja au gel. C'est du dimensionnement de tampon, pas des cellules en moins.

## #54 pieges 3 et 5 : la frontiere de saturation, et un `max_off` qui ne decide jamais rien (2026-08-08)

Les deux derniers pieges d'octet-identite testables sans GPU.

### Piege 3, la frontiere u8 / i16

`matesw_saturation_boundary_equals_scalar` construit des travaux qui se posent **exactement** sur le
seuil au lieu de le longer : pour chaque bonus de correspondance `a` de {1, 2, 3, 5, 7}, les
longueurs de requete sont choisies pour que le plafond `qlen * a` vaille 248, 249, 250, 251, 254, 255
et 256, encadrant a la fois la limite `U8_SCORE_LIMIT` de 250 et le debordement propre a la voie u8 a
255. Les `a` sont pris de part et d'autre de la divisibilite de 250 (2 et 5 la divisent, 3 et 7 non),
de sorte que le seuil est atteint pile aussi bien qu'encadre. Un second balayage traverse l'autre
precondition u8, `qlen < 250`, qui existe parce que la colonne d'argmax partage la voie avec le
score.

`score2` est compare autant que `score`, avec un `minsc` assez bas pour que la liste des rivaux soit
reellement peuplee : un bug de saturation qui epargnerait le meilleur alignement mais ecreterait un
second changerait le MAPQ et rien d'autre, ce qui est exactement le genre de divergence qui passe
une revue et tombe sur un md5.

### Piege 5, `max_off` et la boucle de rounds : il ne se declenche jamais

C'est le resultat le plus utile de la paire, et il n'etait pas attendu. `BWA4_ALIGN_SPLIT` compte
desormais les travaux entres dans chaque round de doublement de bande et ceux que le test
d'acceptation renvoie au suivant :

| jeu | bande | round 0 | requeues |
|---|---|---|---|
| 200 k paires 150 bp, `genome.fa` | `-w 100` (defaut) | 2 109 519 travaux | **0** |
| 200 k paires 49 bp, chr21 | `-w 100` (defaut) | 510 344 travaux | **0** |
| 200 k paires 150 bp | **`-w 5`** | 2 113 395 travaux | **29 710, soit 1,406 %** |

**Aux reglages par defaut la boucle de doublement de bande ne tourne jamais.** Le test d'acceptation
passe des le round 0 pour la totalite des travaux, donc `max_off` ne decide de rien et un GPU qui le
calculerait faux serait **invisible** sur ces donnees. Ce n'est pas une raison de relacher le piege,
c'est une raison de changer le gate : `scripts/gpu_parity.sh` execute maintenant **chacune de ses
trois etapes deux fois**, a la bande par defaut puis a `-w 5`, et c'est le second bras qui met
reellement le piege 5 sous test. Les deux references de determinisme sont vertes
(`791c21c2…` et `78d44e0f…`).

Cela dit aussi quelque chose du code CPU : le round 1 existe, il est correct, et il n'est
pratiquement jamais emprunte en production.

## #48C : le masque de bande en espace signe, et ce que le desassemblage x86 dit vraiment (2026-08-09)

Apres #48A et #48D, la regle etait posee : **compter les instructions dans le desassemblage x86, pas
dans le source**, avant d'ecrire une ligne. C'est ce qui a ete fait, et le comptage a la fois
confirme et corrige l'issue.

Binaire x86_64 construit ici (`cargo build --target x86_64-apple-darwin`), boucle de colonne du
noyau d'extension u8, avant tout changement :

| noyau | instructions de la boucle | `vpmaxub` | `vpcmpeqb` | `vpcmpgtb` |
|---|---|---|---|---|
| AVX2 u8 | 54 | 8 | 6 | 0 |
| SSE4.1 u8 | 54 | 8 | 6 | 0 |
| **AVX-512 u8** | **38** | 4 | **0** | 0 |

L'issue avait raison pour AVX2 et SSE4.1 : faute de comparaison entiere non signee, le test de bande
est emule en deux `vpmaxub` + deux `vpcmpeqb` + un `vpandn` + un `vpand`, six instructions sur 54.
**Elle avait tort pour AVX-512**, qui possede `vpcmpltub` / `vpcmpnltub` et dont LLVM les emet deja
depuis le meme code source : d'ou ses 38 instructions et zero `vpcmpeqb`. Un rewrite signe n'y aurait
rien apporte. NEON est dans le meme cas, avec ses `vcgeq_u8` / `vcltq_u8` natifs.

Livre : une fente de macro `band` (l'expression fusionnee `(j >= beg) & (j < end)`, une par ISA) et
une constante `band_bias` XOR-ee dans `j`, `beg` et `end`. `0x80` pour les deux noyaux u8 x86,
**0 partout ailleurs**, ce qui laisse NEON et AVX-512 exactement comme ils etaient. Le biais sur
`beg`/`end` est applique dans le prologue de ligne, ou ces deux tableaux scalaires sont remplis de
toute façon, donc il est gratuit. La fente `clt` disparait du macro : le test de bande etait son seul
usage.

Resultat, meme mesure :

| noyau | avant | apres | `vpmaxub` | `vpcmpeqb` | `vpcmpgtb` |
|---|---|---|---|---|---|
| AVX2 u8 | 54 | **51** | 8 -> **6** | 6 -> **4** | 0 -> 2 |
| SSE4.1 u8 | 54 | **51** | 8 -> **6** | 6 -> **4** | 0 -> 2 |
| AVX-512 u8 | 38 | 38 | 4 | 0 | 0 |

Trois instructions de moins par colonne, dont **deux `vpmaxub`**, qui sont precisement la classe p01
que l'issue designait comme liante sur Skylake et Zen 2.

### Ce qui n'est PAS mesure, et il faut le dire

**Il n'y a pas de chiffre de temps.** Cette machine est arm64 ; le binaire x86 ne tourne ici que sous
Rosetta, ou les rapports de vitesse ne se conservent pas. Le comptage ci-dessus est un comptage
d'**assembleur emis**, ce qui est une bien meilleure preuve qu'un comptage de source (c'est
exactement ce qui manquait a #48A et #48D), mais **ce n'est pas une mesure**. Le A/B doit venir d'un
runner x86, via `.github/workflows/bench-x86.yml`.

### L'octet-identite, elle, est verifiee ici

Le XOR par 0x80 est la bijection croissante entre u8 et i8 (`(a ^ 0x80) >s (b ^ 0x80)` ssi
`a >u b`), donc les compares signes sont exacts. Oublier de biaiser un des trois operandes ne donne
pas une derive subtile mais du grand n'importe quoi, que le premier cas des gates attrape. Verifie :

* `check.sh` vert, y compris la passe x86_64 sous Rosetta ou **les noyaux AVX2 s'executent vraiment** ;
* et le gate qui compte : le binaire **x86_64 sous Rosetta** et le binaire **arm64** rendent le
  **meme md5** (`791c21c2…`) sur 200 k paires reelles, corps SAM complet.

## #46C : le `rowmax` i32 des noyaux u8 x86 passe en u8 (2026-08-09)

Dernier point ouvert de #46, et le seul des trois qui n'avait pas ete infirme par la mesure : les
noyaux u8 x86 (AVX2, SSE4.1, AVX-512) portaient encore un `rowmax` en **i32** et publiaient la ligne
avec 32 ou 64 ecritures scalaires gardees, la ou NEON stocke la ligne entiere en un `vst1q_u8` depuis
longtemps. Aligne sur NEON.

Ce que cela change en taille de tampon, a la forme moyenne (`tmax` = 1401 lignes) :

| noyau | voies | `rowmax` i32 | `rowmax` u8 | ecritures par ligne |
|---|---|---|---|---|
| AVX-512 u8 | 64 | **350 KB par groupe** | **87,6 KB** | 64 -> **1** |
| AVX2 u8 | 32 | 175 KB | 43,8 KB | 32 -> **1** |
| SSE4.1 u8 | 16 | 87,6 KB | 21,9 KB | 16 -> **1** |

Le cas AVX-512 est celui que l'issue designait : 350 KB par groupe sort de L1 et va en L2, et le
parcours par voie de `extract_group` touchait alors une ligne de cache neuve a chaque chargement, en
re-lisant chaque ligne une fois par voie. En u8 le pas est exactement une ligne de 64 octets par
ligne de DP de 64 voies.

L'ecriture est inconditionnelle, sur l'argument que NEON tient deja : les maxima de ligne d'une voie
ne sont lus que pour les lignes `0..=limit[l]`, et `limit[l]` vaut `tlen[l] - 1` ou la ligne de gel,
donc tout ce que cette ecriture depose au-dela de l'ancienne garde n'est lu par personne.

### Mesure : aucune, et pour la meme raison que #48C

Machine arm64. Le temps x86 doit venir d'un runner x86. Ce qui est verifiable ici l'a ete :

* `check.sh` vert, dont la passe x86_64 sous Rosetta ou le test `avx2_matesw_u8_matches_scalar`
  appelle le noyau AVX2 **directement**, donc le portage AVX2 est reellement execute et compare au
  scalaire ;
* le binaire **x86_64 sous Rosetta** et le binaire **arm64** rendent le meme md5 sur les **deux** jeux
  reels (200 k paires 150 bp, `791c21c2…` ; 500 k paires 49 bp, `7178e85d…`). Sous Rosetta la
  detection `avx2` est fausse, donc le noyau de rescue reellement emprunte de bout en bout est le
  **SSE4.1**, qui est l'un des trois modifies ;
* **le noyau AVX-512 u8 n'a aucune couverture ici** : son test s'arrete faute de `avx512bw`. Le
  changement y est mecaniquement identique aux deux autres, mais c'est dit plutot que tu.

#46 est donc close : A et B mesures a zero et retires, C livre.

## Le cumul du lot SIMD, mesure bout a bout (2026-08-09)

Les sections precedentes donnent chacune leur levier. Voici le total, mesure et non additionne :
**A/B entrelace de deux binaires**, `7428cd0` (avant #45) contre `615653c`, `-t4`, medianes de 9
rondes, secondes CPU. Deux binaires plutot qu'un commutateur parce que la question porte sur le lot
entier ; l'entrelacement annule quand meme la derive thermique, qui est la raison d'etre de la regle.

| jeu | | avant | apres | rapport | gain | victoires |
|---|---|---|---|---|---|---|
| 500 k paires **150 bp**, `genome.fa` | CPU du processus | 33,06 s | **32,53 s** | **1,016x** | **-1,6 %** | **9/9** |
| | noyau de rescue | 1,50 s | **1,27 s** | **1,181x** | **-15,3 %** | **9/9** |
| 500 k paires **49 bp**, chr21 | CPU du processus | 7,64 s | **7,07 s** | **1,081x** | **-7,5 %** | **9/9** |
| | noyau de rescue | 2,44 s | **1,91 s** | **1,277x** | **-21,7 %** | **9/9** |

Les deux binaires rendent le meme corps SAM, `3a51acef…`, verifie avant de chronometrer.

**Pourquoi les deux jeux different d'un facteur cinq.** Ce n'est pas du bruit, c'est la composition
du travail. Le lot a surtout accelere le **noyau de rescue** (#45, #47, #50A), qui pese beaucoup plus
lourd a 49 bases qu'a 150 : 2,44 s sur 7,64 s de CPU contre 1,50 s sur 33,06 s. Et #47 depend en plus
de la fraction de colonnes bourrees, 23,4 % a 49 bp contre 6,25 % a 150 bp. Le seul levier qui vise
l'extension, #48B, vaut -0,9 % a 150 bp et rien a 49 bp, exactement l'inverse.

**Le chiffre a citer pour une course de production Illumina 2x150 est donc 1,016x sur le processus
entier, avec 1,18x sur le noyau de rescue.** Les deux modifications x86 (#48C, #46C) ne sont dans
aucune de ces colonnes : ce binaire est arm64.

## #43 : la table de scores XOR portee sur AVX2 et SSE4.1, et le +18 a +50 % est a revoir (2026-08-09)

Le plus gros levier annonce de toute la recherche. Porte : les noyaux u8 AVX2 et SSE4.1 du rescue
scorent desormais par `vpshufb` sur la meme table 16 entrees indexee par `target XOR query` que NEON,
avec le meme reencodage de la cible N en `N_TARGET = 12`. `N_TARGET` n'est plus `cfg`-limite a
aarch64.

Les deux choses que l'issue disait devoir arriver dans le meme commit y sont :

* le test de cellule morte `cge_epu8(t_v, m_v)` (t >= 5) aurait tue **toutes** les vraies bases N de
  la cible une fois celles-ci reencodees en 12. Remplace par le test de bit de poids fort de NEON,
  `cmpgt_epi8(zero, t_v)`, `PAD` etant le seul octet de l'alphabet dont le bit 7 est mis. La meme
  substitution ramene le test cote requete `cgt_epu8(q_v, zpad_v)` de trois instructions a une ;
* la garde de saturation `U8_SCORE_LIMIT + mispen + mtch <= 256` est repliquee, en `assert!` dur.

L'ecart `vpshufb` / `vqtbl1q_u8` est discharge par enumeration et le raisonnement est dans le code :
`vpshufb` lit `tbl[idx & 15]` la ou `vqtbl1q_u8` rend 0 pour tout index >= 16, donc les deux ne
different que sur 16..127, et cet intervalle est **inatteignable** (seul `PAD` a le bit 7 ; hors
`PAD` les deux operandes valent au plus 12 donc `t ^ q <= 15`).

### Ce que le desassemblage montre, et pourquoi il corrige la projection

Corps de colonne u8 inline dans `fwd_local_sw_batch`, boucle deroulee sur deux colonnes de la paire
de lignes (donc quatre cellules-lignes par iteration), AVX2 et SSE4.1 donnant les memes comptes :

| | avant | apres |
|---|---|---|
| instructions | 246 | **248** |
| **`VPBLENDVB`** | **14** | **6** |
| **`VPCMPEQB`** | **17** | **5** |
| `VPSHUFB` | 0 | 4 |
| `VPOR` | 5 | 1 |
| `VPSUBUSB` | 24 | 20 |
| `VPMAXUB` | 22 | 20 |
| `VPXOR` | 13 | 18 |

**Le nombre d'instructions ne bouge pas** (+2). Ce qui bouge est le melange, et c'est bien la ou
l'issue visait : moins 8 `VPBLENDVB` et moins 12 `VPCMPEQB` contre quatre `VPSHUFB` de plus. En
comptant les uops publies : `VPBLENDVB` ymm vaut 2 uops de Haswell a Tiger Lake, 3 sur Emerald
Rapids, environ 8 microcodes sur Gracemont, tout le reste ici valant 1.

| classe | avant | apres | ecart |
|---|---|---|---|
| uops totaux, Haswell a Tiger Lake | 260 | 254 | -2,3 % |
| uops totaux, Emerald Rapids | 274 | 260 | -5,1 % |
| uops totaux, Gracemont | 344 | 290 | **-15,7 %** |
| **uops p5, Haswell/Broadwell** (`VPBLENDVB` 2x p5, `VPSHUFB` 1x p5) | **28** | **16** | **-43 %** |
| uops p01 (`cmpeq`, `max`, `adds`, `subs`) | 67 | 49 | -27 % |

**Donc : le +18 a +50 % de l'issue n'est pas soutenu par le nombre d'instructions ni par le total
d'uops sur un coeur grand public.** Il l'est par la pression de port, si p5 est bien ce qui lie sur
Haswell/Broadwell, et par le cas Gracemont. Ces lignes restent des **projections a partir de tables
publiees**, pas des mesures : le meme genre de raisonnement s'est deja trompe deux fois dans ce
dossier (#48A et #48D). Le chiffre reel doit venir d'un runner x86.

### Ce qui est verifie ici

* `check.sh` vert, dont la passe x86_64 sous Rosetta ou `avx2_matesw_u8_matches_scalar` et son jumeau
  SSE4.1 appellent les noyaux **directement** ;
* generateurs renforces comme l'issue le demandait : **chaque** travail porte desormais un N dans la
  requete **et** dans la cible, plus une cellule both-N sur la diagonale d'une copie plantee, qui est
  exactement l'index `4 ^ 12 = 8` que la forme a blends n'avait pas ;
* `oracle_diff.sh` vert ;
* binaire **x86_64 sous Rosetta** et binaire **arm64** : meme md5 sur les deux jeux reels
  (`791c21c2…` et `7178e85d…`). Sous Rosetta `avx2` n'est pas detecte, donc le noyau reellement
  emprunte de bout en bout est le **SSE4.1**, l'un des deux portes.

## #44 : la meme table sur AVX-512, plus l'echange de port gratuit (2026-08-09)

Partie A portee, partie B **a moitie**, et la moitie retenue est celle qui ne coute rien.

**Partie A.** `fwd_local_sw_avx512_u8` score desormais par `_mm512_shuffle_epi8` sur la table 16
entrees diffusee aux quatre voies de 128 bits par `_mm512_broadcast_i32x4` (`vpshufb` est in-lane),
avec le meme `N_TARGET = 12`. Disparaissent : `mispen_v`, `four_v`, `m_v`, `mtch_v`, les
`_mm512_cmpeq_epi8_mask` par ligne, le `korq` et les deux blends de score. Comme sur AVX2, le test de
cellule morte `cmpge_epu8(t_v, m_v)` aurait tue toutes les vraies N de la cible : il devient
`_mm512_movepi8_mask`, qui est aussi ce qui remplace `cmpgt_epu8(q_v, zpad_v)` cote requete. La garde
de saturation est un `assert!` dur. `vpermb` n'a pas ete retenu, pour la raison de l'issue : une table
de 16 entrees ne demande pas une permutation sur 64.

**Partie B, seulement l'echange gratuit.** `imax = max_epu8(imax, h)` devient
`mask_blend(gt, imax, h)` en **reutilisant** le masque `gt` que la mise a jour de `col` calcule une
ligne plus haut. Aucune instruction nouvelle, et la max quitte p0, le port qui lie a 512 bits, pour
p05 ou `VPBLENDMB` tourne a 2 par cycle. Octet-identique par definition de la max, et le `>` strict
reutilise est exactement la regle d'egalite que le noyau applique deja.

**Les echanges payants (points 2 et 3 de la partie B) ne sont PAS faits**, et c'est delibere.
L'issue demande elle-meme de les conditionner, parce qu'ils **degradent sur Zen 5** ou les quatre
pipes FP prennent `vpmaxub` a 0,25 de debit tandis que `vpcmpub` coute 6 de latence. Ecrire une
porte dont le calibrage ne peut etre ni execute ni mesure sur cette machine reviendrait a livrer une
heuristique non verifiee dans le noyau le plus chaud du programme. Le point 3 de l'issue interdit de
toute maniere de toucher `mfe` et `h`, qui sont sur la chaine inter-lignes.

### Le trou de couverture, qu'il faut dire

**Le noyau AVX-512 u8 n'est execute nulle part ici.** `avx512_matesw_u8_matches_scalar` s'arrete
faute de `avx512bw`, Rosetta ne l'expose pas, et le binaire x86 n'emprunte donc jamais ce chemin.
Ce qui est verifiable a ete fait :

* le generateur partage `rescue_jobs` est renforce comme l'issue le demande : **chaque** travail porte
  desormais un N dans la requete **et** dans la cible, plus une cellule both-N, l'index `4 ^ 12 = 8`
  que la forme a blends n'avait pas. Ce generateur alimente aussi le test AVX-512, donc le prochain
  tirage Intel de la CI le verifiera reellement ;
* `check.sh` vert, `oracle_diff.sh` vert, et md5 identique entre binaire arm64 et binaire x86 sous
  Rosetta sur les jeux reels, ce qui prouve que le portage n'a rien casse **ailleurs** ;
* aucun chiffre de temps, ici non plus.

Autrement dit : ce commit est **ecrit et relu, pas execute**. C'est le seul du lot dans ce cas, et il
ne doit pas etre traite comme les autres tant qu'un tirage CI avec `avx512bw` n'a pas tourne.

## Les chiffres x86, enfin mesures : +17 % sur le noyau de rescue (2026-08-09)

Tout le travail x86 (#43, #44, #46C, #48C) avait ete ecrit et verrouille sur une machine arm64, donc
**sans aucune mesure**. Le harnais CI est desormais la et la dette est payee, sauf pour AVX-512.

`bench-x86.yml` prend trois entrees de plus : `base_ref` (le job construit et sonde le ref courant
**et** celui-la sur le **meme runner**, coup sur coup), `require_avx512` (echec immediat si le tirage
n'est pas Intel) et une nouvelle sonde `x86_extend_ab` pour les noyaux d'**extension**, qui n'en
avaient aucune. Deux branches de mesure existent pour cela : `bench-base-x86` (pre-#43) et
`bench-base-48b` (post-#48B), portant la meme sonde retro-portee.

Runner **AMD EPYC 7763 (Zen 3)**, `avx2` sans `avx512bw`.

### Noyau de rescue, `rescue_kernel_ab`, AVX2 u8

| | debit |
|---|---|
| base pre-#43 | 9,545 / 9,550 / 9,556 Gcell/s |
| HEAD | 11,129 / 11,170 / 11,241 Gcell/s |
| **ecart** | **+16,6 %, +17,0 %, +17,6 %** sur trois courses |

**+17 % sur le noyau de rescue AVX2**, c'est-a-dire l'effet de **#43 (table XOR) plus #46C
(`rowmax` en u8)**. L'issue #43 projetait +31 % pour Zen 2/3 ; le reel est un peu plus de la moitie,
et il est solide (trois courses, dispersion 1 point).

Note au passage : mon propre comptage d'uops depuis le desassemblage annoncait -2,3 % d'uops totaux
sur un coeur grand public et -43 % sur p5. Le vrai chiffre est +17 % de debit. **Le comptage
d'instructions, meme d'assembleur emis, ne predit pas le temps.** C'est la troisieme fois dans ce
dossier.

### Noyaux d'extension, `x86_extend_ab`

Trois points, ce qui permet de separer les deux leviers :

| | SSE4.1 u8 | AVX2 u8 |
|---|---|---|
| base pre-#43 | 1,582 Gcell/s | 2,215 |
| apres **#48B** (pre-passe repliee) | **1,675** | **2,195** |
| HEAD, apres **#48C** (masque signe) | 1,670 | 2,201 |

* **#48B** : **+5,9 % sur SSE4.1**, **-0,9 % sur AVX2**. Le levier qui retire un aller-retour
  memoire paie sur le noyau etroit et coute un point sur le large. Il reste garde : il gagne sur deux
  ISA sur trois (NEON +4 a +5 % de noyau, SSE4.1 +5,9 %) et perd 0,9 % sur la troisieme.
* **#48C** : **-0,3 % et +0,3 %**, donc **plat**. Les +8 a +12 % projetes ne se materialisent pas sur
  Zen 3. L'issue les argumentait sur la pression p5 de Skylake/Haswell, que ce runner n'est pas ; le
  chiffre reste donc a prendre sur un coeur Intel. En attendant, le changement est conserve parce
  qu'il est neutre en vitesse ici et strictement moins d'instructions.

### #44 reste non verifie, et le tirage est une loterie

`require_avx512` fonctionne exactement comme prevu : il fait echouer le job en une trentaine de
secondes quand le tirage n'a pas `avx512bw`. **Sept tirages consecutifs ont donne le meme AMD EPYC
7763.** Le noyau AVX-512 u8 n'a donc toujours ete execute nulle part, et #44 reste ouverte pour cette
seule raison. Le harnais est pret : un tirage Intel suffira a faire tourner
`avx512_matesw_u8_matches_scalar` et a imprimer la ligne `avx512_u8` des deux sondes.

## Le processus entier sur x86 : +1,33 %, et deux choses vues au passage (2026-08-09)

Les sondes de noyau repondaient « les noyaux ont-ils accelere ». Elles ne repondent pas « le binaire
a-t-il accelere », qui est ce que les issues x86 devaient vraiment. Le job bout-en-bout de
`bench-x86.yml` construit desormais le ref de base **sur le meme runner**, verifie que les deux
binaires s'accordent sur le corps SAM, puis les entrelace sur le meme index et les memes lectures,
avec **bwa-mem2 chronometre a cote comme canari de derive**.

Runner AMD EPYC 7763 (Zen 3), chr21, 500 k paires simulees, 8 threads :

| | |
|---|---|
| md5 du corps SAM, base et HEAD | **identiques**, `c32c321f…` |
| base | 53,58 / 53,90 / 54,02 s |
| HEAD | 52,97 / 52,87 / 53,15 s |
| canari bwa-mem2 | 79,45 / 79,43 / 79,54 s (dispersion 0,14 %, la machine n'a pas derive) |
| **resultat** | **1,0134x, +1,33 %, 3 victoires sur 3** |

**Donc non, on ne perd pas sur x86 : +1,33 % sur le processus entier.** Le chiffre colle a celui
d'arm64 sur la meme longueur de lecture (+1,6 % a 150 bp), ce qui est rassurant : les deux
plateformes voient le meme lot de leviers produire le meme ordre de grandeur.

Le -0,9 % du noyau d'extension AVX2 (#48B) existe toujours, il est simplement noye : le rescue prend
+17 % sur la meme machine et pese davantage.

### Deux choses vues au passage, qui ne sont pas de ce lot

1. **Contre le fork, sur x86, nous perdons : 0,71x** (bwa-mem3 37,69 s contre nos 53,30 s ; bwa-mem2
   79,62 s, donc 1,49x pour nous contre l'oracle). C'est le sujet des issues #20, #27 et du jalon
   v4.3.2, pas de la campagne SIMD, et le chiffre est ici confirme sur un runner neutre. A 150 bp sur
   Apple Silicon le rapport contre le fork etait de 0,98x ; sur Zen 3 il est de 0,71x. **L'ecart x86
   est reel et il est gros.**
2. **La sonde de tier confirme la lecture du runner** : `scalar` 486,93 s, `avx2` 53,11 s,
   `avx512` 484,58 s. Le tier `avx512` force retombe en scalaire faute de `avx512bw`, exactement le
   comportement documente, et cela redit que rien d'AVX-512 n'a tourne.

## AVX-512 : ce n'est pas une loterie, la fonctionnalite n'est pas exposee (2026-08-09)

#44 attendait « un tirage Intel ». Ce n'en est pas un, et la conclusion est plus utile que ce qu'on
cherchait.

**23 dispatches, 3 images de runner** (`ubuntu-22.04`, `ubuntu-24.04`, `ubuntu-latest`), **2 modeles
de CPU** : AMD EPYC 7763 (Zen 3) et AMD EPYC 9V74 (Zen 4). **Zero avec `avx512bw`.**

Le second modele est celui qui tranche : **Zen 4 possede AVX-512 dans le silicium**. Si un 9V74 ne
montre pas le drapeau, ce n'est pas le tirage qui est malchanceux, ce sont les machines virtuelles
Azure derriere les runners heberges de GitHub qui **ne l'exposent pas a l'invite**. Continuer a
relancer des tirages ne peut donc rien donner.

**#44 n'est pas verifiable sur un runner heberge par GitHub.** Il faut un SKU de runner plus gros
(payant), un runner auto-heberge, ou une machine Intel. C'est ecrit dans l'issue.

### Et un bug de sonde, qui mentait depuis toujours

En cherchant cela, la ligne de journal des fonctionnalites SIMD s'est revelee fausse :

```
grep -o -m1 -E 'avx2|avx512bw|avx512f' /proc/cpuinfo
```

Avec `-o`, le `-m1` de GNU grep compte les **correspondances**, pas les lignes correspondantes. La
ligne `flags` contient les trois, mais la sonde n'en imprimait qu'une : « `simd: avx2` », **meme sur
une machine qui aurait eu AVX-512**. Elle a menti dans toutes les executions de ce workflow. Corrigee
(sans `-m1`, plus `avx512vl`). Ici cela ne changeait rien, la porte `require_avx512` interrogeant
`/proc/cpuinfo` directement, mais sur une future machine Intel la ligne aurait cache le fait meme
qu'on cherchait.

## #53 etapes 1 et 2 : la couture asynchrone et le crate `bwa-gpu` (2026-08-09)

L'etape 0 avait mesure les lots. Voici le contrat et le crate, tous deux sans une ligne de code GPU
et sans une dependance systeme.

### Etape 1, le contrat non bloquant

`SwBackendAsync` dans `bwa-extend` : `submit` rend la main tout de suite, `collect` bloque jusqu'au
resultat, `queue_depth` dit combien de lots il vaut la peine de garder en vol. `BatchTicket` n'est ni
`Clone` ni `Copy`, donc un lot ne peut pas etre recolte deux fois.

**Les backends CPU ne changent pas d'une ligne.** L'adaptateur `SyncAsAsync<B>` donne le contrat
asynchrone a n'importe quel `SwBackend` bloquant : `submit` fait le travail et gare le resultat,
`collect` le reprend. `queue_depth` y vaut **1**, ce qui est la reponse honnete pour un CPU et fait
qu'un appelant qui pipeline jusqu'a `queue_depth()` se comporte exactement comme aujourd'hui.

L'invariant qui compte, et qui est teste : `collect(submit(..))` **egale** `extend_batch(..)`. Les
barrieres d'acceptation existantes restent donc valides telles quelles.

### Etape 2, le crate

`crates/bwa-gpu`, trois choses communes a tout GPU dont aucune n'est du GPU :

* **`JobArena`** : met a plat les `&[u8]` empruntes des `ExtendJob` en **une** allocation avec table
  d'offsets, et **reutilise** cette allocation d'un lot a l'autre. C'est le point « persistance » de
  l'issue : allouer et liberer par lot couterait plus que le lancement que cela doit amortir. Sur
  memoire unifiee, ce tampon plat peut **etre** le `MTLBuffer`, ecrit sur place, zero copie.
* **`select()`** : lit `BWA4_GPU=metal|cuda|off` et rend le backend plus la **raison** du repli
  (`NotRequested`, `NotCompiledIn`, `NoDevice`). Une valeur inconnue n'est pas une erreur, expres :
  cette variable vivra dans des scripts qui survivront a ce binaire, et une faute de frappe qui
  produit silencieusement le bon resultat sur CPU vaut mieux qu'une course qui meurt a la sixieme
  heure d'un WGS.
* les features `metal` et `cuda`, **declarees et vides**, pour que `--features metal` sur une machine
  sans GPU compile et emprunte le repli, ce qui est testable et teste des maintenant.

### Les trois criteres d'acceptation

| critere | resultat |
|---|---|
| `cargo build` sans feature GPU : rien ne change, aucune dependance nouvelle | **vert**, `cargo tree` ne montre que `bwa-mem4-extend` |
| `--features metal` sans GPU utilisable : repli CPU, meme md5, aucune erreur bloquante | **vert**, 5 tests dont le balayage des cinq valeurs de `BWA4_GPU` |
| histogramme de taille de lot publie dans le ROADMAP | **fait** (section #53 etape 0) |

### La limite, dite plutot que tue

**L'aligneur ne consomme pas encore ce crate.** Brancher `select()` dans `across.rs` aujourd'hui
ajouterait une indirection au chemin chaud pour zero benefice, puisqu'il n'existe aucun backend a
selectionner. Le `BWA4_GPU=metal` d'un binaire actuel est donc simplement ignore, et le md5 identique
que l'on obtient ne prouve rien de plus que cela ; c'est le test du crate qui exerce reellement le
repli. Le branchement se fera dans le meme commit que le premier backend, ou il aura un sens.

## #55 etape 0 : debloquee sans Xcode, et le GPU tient 330 Gcell/s (2026-08-09)

**Correction de la section « #55 etape 0 : bloquee » ci-dessus.** Elle concluait qu'il fallait Xcode
et que le chiffre de l'issue pouvait devoir etre corrige de +31 % si `subsat` manquait. Les deux
points tombent.

**Xcode n'est pas necessaire.** L'issue supposait le compilateur `metal` hors ligne, qui vient avec
Xcode. Mais MSL se compile **a l'execution** via `newLibraryWithSource:`, et `Metal.framework` fait
partie de macOS. Il ne faut donc que `clang`, que les Command Line Tools fournissent. C'est
`scripts/msl_probe.sh`, deux sondes de quelques dizaines de lignes en Objective-C.

### Ce que MSL expose reellement

| forme | resultat |
|---|---|
| `addsat(uchar4)`, `subsat(uchar4)` | **existent** |
| `addsat(uchar)`, `addsat(ushort4)`, `subsat(ushort4)` | existent |
| `add_sat` / `sub_sat` (l'orthographe OpenCL de l'issue) | **n'existent pas** : « use of undeclared identifier 'add_sat'; did you mean 'addsat'? » |

Donc **l'arithmetique entiere saturee est la**, sous le nom `addsat`/`subsat`. Le repli par `min`/`max`
n'a pas lieu d'etre, et la correction de +31 % que la section precedente reservait **n'est pas due**.

La semantique est verifiee sur l'appareil, pas seulement a la compilation :

```
addsat(250,3,0,255 + 10,0,0,1) = 255,3,0,255
subsat(3,255,0,10 - 4,1,7,10)  = 0,254,0,0
```

Ecretage a 255 et a 0, par voie, exactement ce dont la recurrence u8 a besoin.

### Le plafond, mesure au lieu d'etre projete

Seconde sonde : une colonne de la recurrence de rescue en `uchar4` (4 cellules par voie), 13
operations vectorielles, **registres seuls, aucun acces memoire dans la boucle**, meme discipline que
la mesure du plafond NEON. 1 M de threads, 4096 colonnes chacun, meilleur de 5, quatre executions :

| | |
|---|---|
| **plafond mesure** | **320,7 / 327,0 / 330,3 / 332,9 Gcell/s** |
| besoin, selon le cadrage de #52 | 33 a 116 Gcell/s |
| **marge** | **2,8x a 10x** |
| plafond CPU entier (12 P-cores x ~10,4) | ~125 Gcell/s |

**Le GPU integre vaut donc environ 2,6x le CPU entier sur cet etage**, ce que l'issue affirmait et
qui est maintenant mesure sur la machine.

Un ecart avec l'issue subsiste et vaut d'etre note : elle projetait ~570 Gcell/s theoriques, ~230 a
40 % d'efficacite, et **~2 Tcell/s** dans le cas ou `uchar4` saturee existerait. Elle existe, et le
reel est **330**, donc au-dessus de l'estimation a 40 % mais **six fois sous le 2 Tcell/s**. Ce
dernier supposait ~4 operations par cellule ; notre sequence en fait 13 par `uchar4`, soit 3,25 par
cellule, donc ce n'est pas l'arithmetique qui manque : c'est que le pic de 8,6 T op/s n'est pas
atteignable pour des operations sur octets empaquetes avec une chaine de dependance serielle.

**Conclusion pour la suite : la voie Metal est ouverte, son plafond est confortable, et l'etape 0 est
close sans qu'aucun logiciel n'ait ete installe.**

## #55 : le backend Metal existe, il est octet-identique, et il perd contre un coeur (2026-08-09)

Le premier noyau GPU du projet. `crates/bwa-metal`, feature `metal` **desactivee par defaut**, MSL
compile a l'execution donc **aucune chaine Metal au build**, ni pour ce crate ni pour le binaire.

### Ce qu'il fait

`ksw_local_fwd`, la passe avant du mate rescue, **un travail par thread**. Ce choix n'est pas
paresseux : il supprime toute communication inter-thread, donc il n'existe aucun `simd_shuffle`,
aucune reduction de groupe, rien par quoi les donnees d'un travail pourraient atteindre le resultat
d'un autre. C'est ce qui rend l'argument d'octet-identite court.

Les sequences arrivent dans un `MTLBuffer` **partage** que le CPU remplit sur place : sur memoire
unifiee, **rien n'est copie** pour atteindre le GPU. `score2`/`te2` ne sont pas calcules sur le GPU :
le noyau rend les maxima de ligne et le CPU les passe au `SuboptimalTracker` que le scalaire et NEON
utilisent deja, pour que la regle de fusion et la fenetre d'exclusion existent une fois et non trois.

### Les barrieres, appliquees au GPU

| gate | resultat |
|---|---|
| contre `ksw_local_fwd`, 2000 travaux de forme reelle, N des deux cotes, cellule both-N, moitie du lot avec `endsc` fini | **octet-identique** |
| #54 piege 2, regle d'egalite d'argmax sur cibles periodiques (periodes 1, 2, 3, 4, 7) | **vert** |
| #54 piege 6, dix permutations du meme lot | **vert** |

### Le chiffre, et il n'est pas bon

| | debit |
|---|---|
| plafond mesure du GPU, registres seuls (`msl_probe.sh`) | ~330 Gcell/s |
| **noyau livre, formes reelles 150 bp x 620 bp, 8192 travaux** | **10,95 Gcell/s** |
| un seul thread CPU (`BWA4_MATESW_TIME`) | ~10,6 Gcell/s |

**Le GPU fait donc jeu egal avec UN coeur CPU**, et realise 3,3 % de son propre plafond. La cause est
la memoire : les rails H/E vivent en memoire de peripherique, 3 lectures et 2 ecritures par cellule,
la ou la sonde de plafond garde tout en registres. Deux corrections ont deja ete faites, chacune
mesuree sur la meme sonde :

| version | debit |
|---|---|
| rails 32 bits, disposition **par travail** | 2,30 Gcell/s |
| rails 32 bits, disposition **par colonne** | **7,66** (3,3x) |
| rails **u8**, disposition par colonne | **10,95** (+43 %) |

La disposition par travail (`rail[gid * qmax + c]`) mettait les 32 threads d'un groupe SIMD sur 32
lignes de cache differentes ; par colonne (`rail[c * n_jobs + gid]`) elle les met sur 128 octets
consecutifs. Les rails u8 divisent ensuite le trafic par quatre. Les travaux dont le plafond de score
depasse `U8_SCORE_LIMIT` prennent le noyau 32 bits, exactement le meme binning que
`fwd_local_sw_batch` fait sur le CPU.

Ce qui reste des 30x manquants est le **`uchar4`** : quatre travaux par voie, c'est ce que la sonde de
plafond mesure reellement, et c'est une decomposition differente (quatre travaux par thread, avec
leurs geometries inegales a masquer) plutot qu'un changement de type. C'est la prochaine etape.

### Pourquoi il n'est pas branche dans l'aligneur

Parce que l'etape 0 de #53 a compte **72 travaux par appel de rescue** a 150 bp. Un lancement de
noyau ne s'amortit pas sur 72 travaux, et le noyau perd deja contre un coeur a 8192. Le brancher
aujourd'hui serait une regression sur les deux tableaux. Ce qui ouvre la voie est la file
inter-threads de #53, ou la couture d'extension et ses **5 381 travaux par appel**.

Verifie : `check.sh` vert, `cargo tree` sans la feature ne montre aucune dependance nouvelle, avec la
feature il montre `metal 0.33`, et le md5 du binaire est inchange.

## Pourquoi le GPU semblait lent : ce n'etait pas le noyau, c'etait le lot (2026-08-09)

Le noyau Metal mesurait 14,08 Gcell/s contre 10,78 pour **un** thread CPU, ce qui se lisait comme
« le GPU vaut a peine un coeur ». C'etait une conclusion tiree a une seule taille de lot. Balayee,
sonde decoupee, temps GPU seul :

| travaux soumis | debit du noyau |
|---|---|
| 2 048 | 3,63 Gcell/s |
| 8 192 | 14,06 |
| **32 768** | **46,2 / 46,8 / 46,8** |
| 65 536 | 44,5 / 45,0 |
| 131 072 | 45,5 |

**C'est de l'occupancy, pas de la lenteur.** Un travail par thread veut dire que la taille du lot
**est** le nombre de threads : a 8 192 les 40 coeurs GPU sont a jeun, et le debit monte quasi
lineairement (12,7x pour 16x de threads entre 2 048 et 32 768) jusqu'a saturer vers **46 Gcell/s**.

A ce regime le noyau vaut **4,3 coeurs CPU**, soit environ **37 % du CPU entier**, et il atteint
14 % de son propre plafond registres-seuls (le reste etant les rails en memoire et une cellule par
voie au lieu de quatre).

### Ce que cela dit du branchement

Le probleme n'a jamais ete la vitesse du noyau, il est la **taille des lots** : la production soumet
**180 travaux par appel** en moyenne (72 a 150 bp `-t8`), et il en faut **~32 000** pour remplir la
machine. C'est un facteur **180**.

L'etape 0 de #53 avait pose l'alternative dans les bons termes (« si les lots font quelques
centaines, il faut une file d'agregation inter-threads ») ; ce balayage donne enfin la taille que
cette file doit atteindre. Agreger les huit threads d'un `-t8` donne ~1 400 travaux, encore **20x
trop peu** : il faudrait accumuler sur plusieurs chunks, donc retarder le rescue de plusieurs lots de
lectures, ce qui touche l'ordonnancement du pipeline et pas seulement la couture.

Et le calcul de rentabilite reste celui de la section precedente : sur cette course, le mate rescue
pese **1,25 s sur 27,5 s de CPU, soit 4,5 %**. Meme un GPU parfait plafonne la.

## L'extension sur GPU, branchee et mesuree : -22 % de CPU, et une bascule a -t8 (2026-08-09)

Le noyau de la section suivante est desormais **branche dans l'aligneur**. `BWA4_GPU=metal` avec la
feature `metal` route toute l'extension de graines sur le GPU ; le mate rescue reste sur le CPU,
comme la mesure des tailles de lot le commandait.

Le branchement tient en une fonction : `align_chunk` choisit le backend, et comme `MetalExtend`
implemente `SwBackend`, `align_reads_batched` ne sait rien de tout cela. Une poignee de device par
worker rayon (`thread_local`), donc une compilation de noyau par thread au demarrage et aucun partage
entre threads. Repli CPU silencieux si la machine n'a pas de Metal utilisable.

**Octet-identique de bout en bout**, sur les deux jeux reels (`791c21c2…` et `7178e85d…`), avec toute
la programmation dynamique d'extension executee sur le GPU.

### La mesure, et elle corrige ma propre prediction

| | mur, CPU seul | mur, GPU | **CPU seul** | **CPU avec GPU** |
|---|---|---|---|---|
| `-t4` | 8,23 s | **7,82 s (-5,0 %)** | 32,26 s | **25,21 s (-21,9 %)** |
| `-t8` | 4,51 s | 4,63 s (+2,7 %) | 34,54 s | 26,59 s (-23,0 %) |
| `-t16` | **3,15 s** | 4,69 s (**+49 %**) | 43,45 s | 28,72 s (-33,9 %) |

Le GPU absorbe **22 a 34 % du travail CPU a tous les regimes**. Mais le mur ne gagne qu'a `-t4`, et
la bascule est entre `-t4` et `-t8`, **pas vers `-t12` comme je l'avais annonce**. L'erreur est
identifiee : j'avais raisonne avec le debit **a saturation** (22,4 Gcell/s, atteint a 32 000 travaux)
alors que la production soumet **10 800 travaux par appel**, ou le noyau ne fait que **9,08**. La
demande d'extension a `-t16` est de 19,1 Gcell/s : le GPU devient le goulot, et le mur de la variante
GPU plafonne vers 4,7 s quel que soit `-t`.

### Ce que cela dit de la suite

L'agregation inter-threads n'etait pas une optimisation de confort, c'est **la** condition pour que
le GPU serve au-dela de `-t4` : agreger les quatre travailleurs d'un chunk (4 x 10 800 = 43 000)
porte le noyau de 9,08 a 22,4 Gcell/s, ce qui couvre la demande de `-t16`. Et la barriere qui
l'autorise existe deja et est verte (#54 piege 6, dix permutations).

### Quand c'est utile aujourd'hui

Tel quel, a `-t4` : **-5 % de mur et -22 % de CPU**. Le second chiffre est le vrai : sur une machine
partagee, rendre un cinquieme du CPU a d'autres travaux pour un mur equivalent est un gain qui ne se
lit pas dans un `time`. A `-t16` sur cette machine, il ne faut pas l'activer.

## Le noyau d'extension Metal : ecrit, octet-identique, 6,4x un coeur (2026-08-09)

La suite immediate de la section ci-dessous : l'etage designe par la mesure est porte.
`crates/bwa-metal/src/extend.metal` implemente `ksw_extend2`, un travail par thread, avec la bande,
le z-drop, `max_off`, `gscore`, le resserrement de bande par ligne et son epilogue.

### Ce qui a rendu la validation gratuite

Le backend implemente **`SwBackend`**, donc **toute la batterie d'acceptation du projet s'applique
sans une ligne de nouveau harnais**, et elle est passee du premier coup :

| gate | ce qu'elle couvre |
|---|---|
| `assert_backend_matches_scalar` | le balayage complet : scorings, largeurs de bande, z-drop, formes |
| `assert_backend_batch_matches_scalar` | les lots, avec les rounds `w` / `2w` |
| `assert_backend_tie_rule_matches_scalar` | #54 piege 2, entrees ou toutes les cellules d'une ligne sont a egalite |
| `assert_backend_batch_order_invariant` | #54 piege 6, dix permutations |

La derniere est plus qu'un test ici : **c'est la preuve que regrouper les travaux de plusieurs
threads CPU dans un seul lancement GPU ne peut pas changer une reponse.** Elle avait ete ecrite pour
un futur backend GPU sans savoir qu'elle certifierait exactement ce regroupement.

Deux choses sont deliberement restees sur l'hote : le **serrage de bande** (`clamp_band`), parce que
le contrat `SwBackend` l'exige et parce qu'il est en `f64` alors qu'aucun noyau DP de ce projet ne
contient de flottant ; et le **profil de requete**, le noyau indexant directement la matrice `m * m`,
ce qui evite un tampon et garde le noyau general au lieu de supposer la forme ADN uniforme que les
noyaux SIMD exigent.

### Les chiffres

Formes d'extension reelles (requetes 40-110 bases, cibles legerement plus longues), meilleur de 5 :

| travaux | CPU NEON, 1 thread | **Metal** | rapport |
|---|---|---|---|
| **10 800** (la taille de production a `-t4`) | 3,61 Gcell/s | **9,08** | **2,5x** |
| **32 768** (saturation) | 3,52 | **22,4** | **6,4x** |
| 65 536 | 3,49 | 22,1 | 6,3x |

Meme courbe d'occupancy que le noyau de rescue, meme genou : le lot **est** le nombre de threads.
L'empaquetage coute 0,3 ms sur 8, soit 4 % — la memoire unifiee tient toujours sa promesse.

**A la taille de lot que la production soumet deja, sans aucune file d'agregation, le GPU fait 2,5x
un coeur.** En agregeant les quatre threads d'un chunk a `-t4` (4 x 10 800 = 43 000), il fait 6,4x.

### Ce que cela vaudrait, et ce qui manque

L'extension est 30 % de la course. Le CPU y met 8,0 s sur 27,5 s a `-t4`, soit 2,0 s par thread. Un
GPU a 6,4x un coeur absorberait le travail des quatre threads en ~1,25 s **pendant** que le CPU fait
autre chose, mais l'aligneur n'a aujourd'hui rien a faire d'autre a ce moment-la : il faudrait
recouvrir avec le seeding du chunk suivant, ce qui est la couture asynchrone de #53 etape 1, deja
ecrite mais sans consommateur.

Ce qui manque est donc l'**integration**, pas le noyau : agreger les travaux des threads d'un chunk,
lancer, redistribuer. Le noyau existe, il est prouve, et la barriere qui autorise le regroupement
aussi.

## Le chemin GPU est l'extension, pas le rescue, et les chiffres le disent (2026-08-09)

Suite directe du balayage d'occupancy. La question posee etait « comment atteindre 32 000 travaux par
lancement ». La reponse n'est pas une file d'agregation pour le rescue : c'est de changer d'etage.

### Les deux coutures, mesurees en production, meme course

| | mate rescue | **extension** |
|---|---|---|
| part de la course CPU | 4,5 % | **30 %** |
| travaux par appel, `-t4` | 181 | **10 779** (99,7 % dans 8k-16k, plus gros 13 499) |
| travaux par appel, `-t8` | 72 | 5 401 |
| facteur manquant pour saturer (~32 768) | **180x** | **3x** |
| plafond de gain si le GPU etait gratuit | 4,5 % | **30 %** |

Le rescue demande d'accumuler sur plusieurs chunks de lectures, donc de retarder un etage dont les
insertions dans `ma` sont **observables** dans la sortie (la permutation de `ks_introsort`, voir #38).
L'extension demande d'agreger les threads **d'un meme chunk** : 4 x 10 779 = 43 000, au-dessus de la
saturation, sans decaler quoi que ce soit dans le temps.

### Et l'agregation inter-threads y est deja prouvee sure

Deux proprietes que ce depot a etablies pour d'autres raisons se rejoignent ici :

1. le resultat d'un travail d'extension ne depend que de son propre `(query, target, h0, w)`, ce qui
   est l'invariant central de `across.rs` et ce qui autorise deja le tri par longueur ;
2. **la barriere #54 piege 6** (`assert_backend_batch_order_invariant`) l'exige et le verifie : les
   memes travaux passes dans dix permutations aleatoires doivent rendre le meme resultat, travail par
   travail. Elle a ete ecrite pour un futur backend GPU ; elle certifie exactement le regroupement
   inter-threads dont ce chemin a besoin.

3. et la boucle de rounds ne complique rien : #54 piege 5 a mesure **0 requeue sur 5 281 833 travaux**
   aux reglages par defaut. En pratique l'extension est un round unique.

### Ce qui manque, et sa taille

Le noyau. Le rescue portait une recurrence de 13 operations ; l'extension porte la bande, le z-drop,
`max_off`, `gscore`, le resserrement de bande par ligne et son epilogue scalaire. C'est le noyau de
700 lignes de macro, pas la recurrence. C'est le vrai chantier, et il est maintenant **designe par la
mesure** plutot que par l'intuition qui visait le rescue parce qu'il etait le plus gros poste CPU sur
un profil GIAB 30x a `-t8` que cette course ne reproduit pas.

## Le `uchar4` sur Metal : tente, non concluant, retire (2026-08-09)

Suite annoncee de la section precedente : quatre travaux par thread dans un `uchar4`, la
decomposition des noyaux CPU un cran plus bas. Ecrit en entier (noyau `rescue_fwd_u8x4`, empaquetage
des quadruplets avec PAD/ZPAD cote hote, troisieme pipeline, fusion des resultats), **octet-identique
aux trois gates**, puis **retire**. Voici pourquoi, parce que le chemin compte plus que le verdict.

### Ce qui a ete appris en route, et qui reste vrai

**Un piege de disposition memoire qu'aucun compilateur ne signale.** MSL aligne un type vectoriel sur
sa propre taille : un `int4` sur 16 octets, un `ushort4` sur 8. Rust aligne `[i32; 4]` sur 4. La
premiere version de la structure `Quad` melangeait les deux conventions, les champs tombaient a des
decalages differents des deux cotes, et **chaque quadruplet revenait a zero** sans le moindre
avertissement. Corrige en n'utilisant que des **tableaux scalaires** des deux cotes, dont
l'alignement naturel coincide. C'est la deuxieme fois de la journee qu'une disposition `repr(C)`
positionnelle mord (la premiere etait le renommage `rail` -> `kind`), et les deux fois **c'est le
gate d'octet-identite qui a attrape la chose, pas un test de layout**.

### Pourquoi c'est retire

| | 8192 travaux | 65536 travaux |
|---|---|---|
| quadruplets actifs | 4,58 Gcell/s | 23,25 |
| quadruplets desactives (meme arbre) | 4,57 | 23,20 |
| **version livree, sans tout cela** | **12,0** | |

Les deux bras donnent le meme chiffre a 0,2 % pres, ce qui veut dire que **la comparaison ne mesure
pas le levier** : le travail d'hote que la branche ajoute domine les deux. Et il coute cher, puisque
le meme noyau u8 fait 12,0 Gcell/s dans la version livree contre 4,6 avec cette branche presente,
meme quand elle est inactive. Une premiere piste, un `Box<dyn Fn>` choisi par travail et appele par
ligne (5 millions d'appels virtuels dans la region chronometree), a ete corrigee et n'a rien change.

**Un levier qu'on ne sait pas mesurer proprement ne se garde pas.** Le code est retire plutot que
laisse en place « au cas ou » : il ajoutait un troisieme noyau, une quatrieme structure partagee, et
un surcout non explique, pour un gain non demontre.

### Ce qui reste vrai pour la suite

Le plafond du `uchar4` reste mesure a ~330 Gcell/s par `scripts/msl_probe.sh`, et le noyau livre est
a 12. L'ecart n'est donc pas referme, mais la prochaine tentative doit commencer par **isoler le
temps GPU du temps hote** dans la sonde, ce que la sonde actuelle ne fait pas : elle chronometre
`forward_batch` en entier, empaquetage et calcul de `score2` compris. C'est exactement le genre de
mesure qui a induit en erreur ici, et c'est reparable avant d'ecrire une ligne de noyau.

## Le regime de colonnes pleinement en bande, trouve par la mesure (2026-08-09)

Le backlog d'issues etant epuise pour cette machine, ce levier vient d'un profil, pas d'une liste.

### Le profil qui l'a designe

Il a d'abord fallu **reparer la sonde** : les compteurs par etage sont derriere la feature Cargo
`align-split`, desactivee par defaut, mais `TOTAL_NS` ne l'est pas, donc `BWA4_ALIGN_SPLIT=1` sur un
binaire ordinaire imprimait `total=30,5s seeding=0,000s side=0,000s [...] rest=30,5s`. Une ligne de
zeros qui se lit exactement comme une mesure disant « le seeding est gratuit ». Elle dit maintenant
que les compteurs sont compiles hors du binaire, et la CLI expose la feature.

Avec la feature, 500 k paires 150 bp, `-t4`, secondes CPU cumulees :

| etage | s | part |
|---|---|---|
| seeding | 9,19 | 33 % |
| **extension** | **8,31** | **30 %** (dont noyaux 8,01) |
| chain_flt | 0,77 | 3 % |
| reste | 9,21 | 34 % |

Dans le reste, `BWA4_CHAIN_TIME` attribue **42,6 M de recherches SA a 103 ns** = 4,39 s, 16 % de la
course. **Deja au genou** : `get_sa_batch` tourne une fenetre logicielle de 128, dont le balayage est
consigne dans la source (152 ns a `-t16` sur GRCh38, plat au-dela de 128), et 103 ns a `-t4` fait
mieux. Moins de recherches demanderait un changement algorithmique, pas une acceleration.

### Le comptage qui a designe le levier

Boucle de colonne du noyau d'extension NEON u8, **desassemblage du binaire livre** :

| | extension | rescue |
|---|---|---|
| instructions par colonne de 16 cellules | 35 | |
| **operations vectorielles par cellule** | **1,63** | **0,83** |

L'ecart de 2x que #48 annonçait existe donc bien, meme si aucun de ses leviers ne l'expliquait. Neuf
de ces 26 operations vectorielles sont le **masque de bande** et les melanges qu'il pilote : deux
compares et deux `and` pour le construire, puis cinq `bsl`/`and` qui l'appliquent aux deux ecritures
gardees, a `h1`, a `f` et a l'argmax.

### Le levier

Dans les colonnes ou **toutes** les voies vivantes sont en bande, ce masque vaut tout-a-un et les
neuf operations se replient. Trois regimes de colonnes, exactement la forme de #47 :
`gbeg..fast_lo` masque, `fast_lo..fast_hi` **non masque**, `fast_hi..gend` masque, ou
`[fast_lo, fast_hi)` = `[max(beg), min(end))` sur les voies actives. Les bornes sont calculees dans
le prologue de ligne qui remplit deja `beg_lane`/`end_lane`, donc elles coutent un `max` et un `min`.

| | |
|---|---|
| colonnes prenant le regime rapide, **premiere forme** | 12,8 % |
| colonnes prenant le regime rapide, **forme livree** | **43,1 %** |
| **noyaux d'extension**, premiere forme (`EXTEND_NS`, 7 rondes) | 8,399 -> 8,279 s, -1,43 %, 7/7 |
| **noyaux d'extension, forme livree** | **8,182 -> 7,767 s, -5,07 %, 7 victoires sur 7** |
| **CPU du processus entier, forme livree** | **32,26 -> 31,83 s, -1,33 %, 9 victoires sur 9** |

Les deux dernieres lignes sont coherentes : l'extension pese 30 % de la course, donc -5,07 % du
noyau fait bien **-1,5 % du processus**, et le A/B de CPU total en mesure -1,33 % avec 9 victoires
sur 9. La premiere forme, elle, valait -0,44 % du processus, c'est-a-dire sous la resolution de ce
meme A/B, ou elle n'avait obtenu que 5 victoires sur 9 : **il a fallu mesurer au niveau du noyau pour
la voir du tout.**

Le gain reste sous le comptage d'operations (4,5 % attendus pour 1,43 % obtenus dans la premiere
forme, meme facteur d'un tiers ensuite) : les operations repliees sont des melanges bon marche sur
des pipes vectorielles qui ont du mou.

### La preuve qui a triple sa portee

La premiere forme exigeait que **toutes** les voies soient actives, et seules 19,7 % des lignes le
sont, d'ou 12,8 % de colonnes seulement. Le commentaire d'origine du noyau affirmait que le masque
`active_v` empeche la reecriture de l'etat d'une voie terminee, ce qui interdisait d'ignorer les
voies inactives. C'etait une question de correctness, donc elle a ete tranchee dans le code plutot
que supposee. Les trois points, verifies :

1. **Une voie ne se reactive jamais.** `active` est un local reconstruit a chaque ligne par
   `!done[l] && i < tlen[l]`, et `done[l]` comme `i >= tlen[l]` sont monotones en `i`.
2. **Rien ne relit ses rails.** Le seul lecteur de `eh_h`/`eh_e` hors de la boucle de colonnes est le
   resserrement de bande en tete de ligne (`first_live`/`last_live`), et il est a l'interieur de
   `if !active[l] { continue }`. L'epilogue de ligne aussi.
3. **Rien ne survit dans les registres.** `h1_v` est recharge depuis le tableau scalaire `h1[]` au
   debut de chaque ligne et `f_v` est remis a zero, et l'epilogue ne recrit `h1[l]` que pour les
   voies actives.

Donc une voie inactive peut voir ses rails, ses voies de registre et son `rowmax`/`mj` ecrits avec
des valeurs calculees : personne ne les lit. La condition tombe a « les voies **actives** partagent
une plage », et les colonnes rapides passent de **12,8 % a 43,1 %**, pour **-5,07 %** de noyau au
lieu de -1,43 %.

## La voie GPU : le plafond est 2,45x, et le GPU integre suffit deja (2026-08-08)

Suite directe de la section precedente : puisque l'activite recente est GPU, la question devient
« que vaut un GPU ici, et qu'est-ce que l'octet-identite en interdit ? ». Instruite le meme jour,
consignee dans les issues #52 a #57.

### Le plafond, calcule sur notre propre profil

Le profil PE mesure le 2026-08-07 (section « Ou le fork est reellement moins cher ») donne
**39,2 % de mate rescue + 20,0 % de DP principal = 59,2 % de Smith-Waterman entier**. Sur la course
de reference (28,60 s de mur, 220,21 s de CPU, 1 M paires) cela fait **130,4 s CPU de DP** contre
89,8 s pour tout le reste. Si le DP devient gratuit : **2,45x en CPU, mur de 28,60 s a ~11,7 s**.

C'est le plafond absolu de la voie GPU. Il ne bouge qu'en attaquant aussi le tri (15,1 %, #38) et le
seeding (6,7 %).

### Le resultat qui n'etait pas attendu : il ne faut presque pas de GPU

Pour absorber ces 130,4 s CPU pendant les ~11,7 s de mur restantes, il faut **33 a 116 Gcell/s**
selon la convention de comptage (10,4 Gcell/s/thread donne ~1,3 T cellules ; #50 chiffre l'etage
rescue seul a 381 G).

| materiel | debit | ce qu'il calcule |
|---|---|---|
| notre CPU, M4 Max, 12 P-cores | ~125 Gcell/s | score, te, qe, score2, te2 |
| GPU integre du M4 Max (40 coeurs, 5120 ALU, ~1,68 GHz) | ~8,6 T op/s, donc **~570 Gcell/s theoriques**, ~230 en tablant sur 40 % | a ecrire |
| CUDASW++ 4.0 sur A100 (half2) | 1,94 TCUPS | **score seul** |
| CUDASW++ 4.0 sur H100 (s16x2 + DPX) | **5,71 TCUPS** | score seul |

Notre contrat coute ~1,6x le score seul (15,75 operations par ligne de 16 cellules contre les 10 de
SWIPE). Meme en divisant encore par deux : H100 ~1,8 TCUPS, soit **15x le besoin**. Et le GPU
integre du Mac est deja **2 a 7x le besoin**.

**Il n'y a donc pas besoin d'un accelerateur de datacentre pour prendre les 2,45x.** Sur une machine
a GPU discret, l'interet n'est plus la vitesse du DP mais le fait qu'il rende 59,2 % du CPU au
seeding et au tri.

### Ce que l'octet-identite autorise, et c'est plus large qu'attendu

Chaque job de DP est une **fonction pure** de ses arguments. Donc n'importe quelle repartition des
jobs entre CPU et GPU, y compris dynamique et dependante de la charge, **produit exactement la meme
sortie**. Le co-ordonnancement opportuniste est gratuit en risque, ce qui est la difference entre ce
projet et un ordonnancement classique.

Interdits, en revanche : tout flottant (CUDASW++ 4.0 tire ses 5,01 TCUPS sur L40S en `half2` ; notre
recurrence est en entiers satures), le tri des regions sur GPU (permutation observable), et toute
inversion de l'egalite d'argmax (`>` strict, le premier maximum gagne).

### La couture existe deja

`SwBackend::extend_batch` est deja par lot et cite deja `"metal"` en exemple de `name()` ; `across.rs`
batche deja a travers les reads avec des rounds `w` puis `2w` et une requeue ; `batched_ksw_align2`
fait un appel par round pour tout le lot de paires. Il n'y a pas de refonte du pipeline a faire, il y
a une couture a rendre asynchrone.

### Ce que la concurrence fait, et pourquoi son levier principal ne nous est pas offert

GPU-BWA-MEM (ICS 2023) tire **3,2 a 3,8x sur bwa-mem2** en portant tout le pipeline, dont **3,7x sur
le seeding**. Notre seeding pese 6,7 % contre les 17,0 % du fork : ce levier vaut au mieux 6,7 % chez
nous, leur gain venant en grande partie d'une base que nous avons deja battue. Leur noyau SW, un warp
par read en front d'onde, ne rend que **2x** ; notre disposition inter-sequences donne un thread par
alignement, sans communication ni divergence interne, ce qui est structurellement meilleur.

Aucun de ces outils ne tient l'octet-identite comme critere eliminatoire : Parabricks la tient sous
condition de `-K` egal, GPU-BWA-MEM l'affirme sans methode de validation publiee. **Notre argument
reste « plus vite ET prouve identique ».**

## AVX-512 : correct, jamais chronometre, et pourquoi (2026-08-07)

Etat separe en deux, parce que les deux reponses sont differentes.

**Correction : validee.** Les quatre tests d'octet-identite `avx512_verify` (extension u8/i16, rescue
u8/i16) tournent pour de vrai des que le runner a `avx512bw`. Tirage CI du 2026-08-04 : `simd: avx2
avx512bw avx512f`, et les quatre tests apparaissent en `ok` dans le journal. Ce n'est donc pas du code
jamais execute.

**Vitesse : mesuree, enfin.** Le pool `ubuntu-22.04` de GitHub est heterogene et le tirage n'est pas
choisissable, ce qui est la seule raison pour laquelle ce chiffre a manque si longtemps. Cinq tirages,
un seul Intel :

| date | modele | `simd:` | scalaire | AVX2 u8 | AVX-512 u8 |
|---|---|---|---|---|---|
| 2026-08-03 (hebdo) | AMD EPYC 7763 | `avx2` | 0,130 | 10,09 | *skipped* |
| 2026-08-07 #1 | AMD EPYC 7763 | `avx2` | 0,130 | 10,09 | *skipped* |
| **2026-08-07 #2** | **Intel Xeon Platinum 8573C** | `avx2 avx512bw avx512f` | 0,131 | 8,29 | **10,39** |
| 2026-08-07 #3 | AMD EPYC 7763 | `avx2` | 0,128 | 9,78 | *skipped* |
| 2026-08-07 #4 | AMD EPYC 7763 | `avx2` | 0,131 | 9,85 | *skipped* |

En Gcell/s. Sur l'hote Intel : **AVX-512 vaut 1,25x l'AVX2** (79,2x le scalaire contre 63,2x). Une
seule observation, mais les trois arms sont chronometres **dans le meme processus sur le meme hote**,
donc le rapport 1,25x ne depend ni du tirage ni du voisinage.

Deux lectures, et la seconde compte plus que la premiere :

1. **Le noyau 512 bits sert.** 1,25x sur un etage qui pese ~39 % des echantillons de travail
   (profil arm64, voir la section precedente) vaudrait ~8 % du CPU total sur une telle machine. La
   calibration a l'execution le choisit deja toute seule : rien a cabler.
2. **Le gain vient de la largeur, pas de la machine.** Ce Xeon fait 8,29 Gcell/s en AVX2 la ou
   l'EPYC 7763 en fait 9,78 a 10,09. Autrement dit l'AVX-512 de l'Intel arrive tout juste au niveau
   que l'AMD atteint **en AVX2 seul**. Le 512 bits rattrape un hote plus lent par vecteur ; il ne
   place pas l'Intel devant.

Localement c'est hors d'atteinte et il ne faut pas y revenir : Rosetta n'expose pas `avx512bw`, et le
TCG de QEMU n'implemente pas AVX-512 (il s'arrete a AVX2). Il n'y a pas d'emulateur a essayer.

Ce qui a rendu la mesure possible : `bench-x86` accepte une entree `kernel_ab_only`, qui saute le job
bout-en-bout d'une heure et ne lance que le A/B de noyau d'une minute. Retirer au sort une machine
Intel coute une minute par essai au lieu d'une heure, et il en a fallu trois.

## Contre le fork, avec tout ce qui precede (2026-08-07)

4 M paires GIAB reelles contre GRCh38, `-t8`, A/B entrelace, sortie jetee.

| | mur | CPU |
|---|---|---|
| bwa-mem4 | **112,87 / 121,47 / 114,57 s** | **883,5 / 949,7 / 891,3 s** |
| `fg-labs/bwa-mem3` | 134,41 / 137,54 / 165,04 s | 1043,8 / 1070,1 / 1175,6 s |
| rapport | **0,84 / 0,88 / 0,69** | **0,85 / 0,89 / 0,76** |

Trois repetitions sur trois, dans les deux metriques. L'avance est de 12 a 16 % sur les deux premieres
repetitions ; la troisieme (0,69) est a lire avec prudence, le fork y prend 20 % de plus que dans les
deux precedentes alors que nos trois mesures a nous tiennent dans 8 %. C'est donc la mesure du fork
qui a derive, pas nous qui avons accelere. Retenir 0,84-0,88, pas 0,69.

Dans tous les cas l'ordre de grandeur de l'avance est tres au-dessus de la derive de la machine,
contrairement aux effets de 2 % mesures plus haut, qui eux exigent le protocole `-t4`.

## Petits gains verifies (2026-08-05)

Meme protocole que la correction ci-dessous, parce que c'est le seul qui a tenu : A/B entrelace,
`-t4` sur 500k paires contre GRCh38, secondes CPU, medianes.

| changement | avant | apres | verdict |
|---|---|---|---|
| `panic = "abort"` dans `[profile.release]` | 40,27 s | **39,69 s** | garde, 5 victoires sur 5, ~1,4 % |
| PGO entraine sur `testdata/tiny` seulement | 40,54 s | **38,42 s** | **garde, cable dans `release.yml`, ~4,7 %** |

Le PGO est le plus gros gain verifie de la journee, et la surprise est la taille du jeu
d'entrainement : `scripts/pgo.sh` avertit depuis longtemps qu'entrainer sur une petite reference
profile les mauvais chemins (sur simule contre 2 Mbp, le mate rescue pese ~10 % du mur ; sur GIAB
reel contre le genome entier, ~59 %). Mesure faite : entrainer sur la seule fixture `testdata/tiny`
committee, puis aligner 500k paires reelles contre GRCh38, donne 38,42 s contre 40,54 s de CPU,
quatre victoires sur quatre. Ce dont le PGO a besoin de cet entrainement, ce sont les branches prises
par les boucles de seeding et de DP, et leur forme ne depend pas de la taille de la reference.

Cable dans `.github/workflows/release.yml` : instrumentation, entrainement sur la fixture committee
(`index` puis `mem`, les deux moities du code FM), fusion avec le `llvm-profdata` de la toolchain
epinglee, reconstruction. Une compilation de plus par cible. Consequence a connaitre : un binaire de
release n'est plus reproductible bit a bit par un simple `cargo build --release`, il l'est en
rejouant cette recette.

`panic = "abort"` retire les tables de deroulement et les points d'atterrissage : le binaire n'a rien
a derouler, un panic ici est un bug et le processus ne detient rien que l'OS ne reprenne. Cargo
ignore le reglage pour les profils test et bench, donc `cargo test --release` compile toujours avec
deroulement et un `should_panic` continuerait de fonctionner.

Sortie inchangee (md5 GRCh38), `opt_parity.sh` 62/64, `check.sh` vert.

## Correction : la deuxieme vague (2026-08-05) n'a rien gagne, et pourquoi

Trois changements ont ete faits apres la campagne ci-dessous, sur la foi d'une sonde. Mesure faite
apres coup, A/B entrelace contre le binaire d'avant (`6fcdea9`), CPU a `-t4` sur 500k paires, ou la
dispersion est la plus faible : **egalite a 0 a 2 % pres**. Aucun gain demontrable.

Ce qui s'est passe merite d'etre ecrit, parce que c'est un piege de methode et pas de code. La sonde
`BWA4_ALIGN_SPLIT` attribuait **18,1 s de CPU** a la materialisation de la fenetre de reference. Elle
enveloppait cette materialisation dans une fermeture avec deux `Instant::now()`, **par chaine**, soit
des dizaines de millions de fois : l'essentiel des 18,1 s etait la sonde elle-meme. Passer au
depaquetage groupe puis au tampon reutilise a fait tomber le chiffre affiche a 8,0 s, ce qui semblait
un gain de 10 s ; le A/B bout en bout des deux variantes, tout le reste egal, donne **41,09 s contre
41,26 s de CPU**, c'est-a-dire rien.

Consequences tirees :

* Les deux sondes (`BWA4_ALIGN_SPLIT`, `BWA4_SEED_STATS`) sont desormais derriere des features cargo
  eteintes par defaut. Compilees en dur, les compteurs de seeding coutaient a eux seuls ~3 % de CPU,
  sur une boucle qui tourne 1,7 milliard de fois : une branche previsible et jamais prise n'y est
  pas gratuite.
* Une sonde qui enveloppe un appel dans une fermeture se paie deux fois : le temps qu'elle mesure et
  l'inlining qu'elle empeche. Elle doit envelopper du travail par lot, pas par chaine.
* Le depaquetage groupe et le tampon reutilise sont **gardes** : ils sont neutres en vitesse et
  suppriment une allocation par chaine, ce qui reste la bonne forme. Ils ne sont simplement pas le
  gain annonce.
* Le prefetch des candidats de la ronde arriere est **neutre** aussi (`BWA4_SEED_PREFETCH=0` contre
  `=8` : 40,7 contre 40,2, puis 41,6 contre 41,3). Garde, avec le bouton pour le rebalayer sur une
  machine ou la latence memoire domine davantage.

Le gain reel contre le fork reste celui de la campagne ci-dessous, mesure quand la machine etait
calme. La lecon : ne rien annoncer sur la foi d'une sonde sans un A/B bout en bout entrelace.

## Campagne perf : passer devant le fork a -t16 (2026-08-04, suite)

Point de depart : dans le regime du gist de @nh13 (GRCh38, `-t16`, `-K` par defaut, entree gzippee,
5M paires), le fork etait devant de 1,17x en mur. Point d'arrivee : **5 victoires sur 5, rapport
0,963 a 0,992, meilleur temps 38,66 s contre 39,72 s**.

Ce qui a paye, dans l'ordre ou ca a ete trouve :

1. **Fenetre de prefetch SA 32 -> 128** (voir la section suivante). 1,17x -> 1,14x.
2. **Chargement d'index par `pread` parallele et teardown supprime.** 1,14x -> 1,08x.
3. **Lecture demarree avant le chargement de l'index.** Le thread lecteur naissait dans le pipeline,
   donc apres 10 Go d'index en memoire ; inflater et parser le premier lot de 160M bases coute ~1 s
   et le chargement 1,8 s. Ils se recouvrent desormais. 6,43 s -> 5,85 s sur 500k paires a `-t16`.
4. **Recouvrement des lots.** Un lot traversait ses etages en serie et ils ne remplissent pas le pool
   egalement : align 98,9 % d'occupation, rescue 86 %, dedup 69 %, encode 29 %. La queue fine du lot
   N tourne maintenant contre l'align du lot N+1, le seul etage qui absorbe tous les coeurs. Deux
   lots en vol, l'ordre de sortie reste porte par l'ordre de retrait, pas par l'ordre d'achevement.
   Cout : ~2,2 GB de RSS en plus (13,5 -> 15,7 GB), et c'est le poste qui a rendu la victoire
   possible.
5. **PGO**, cycle reproductible dans le conteneur (`scripts/docker_gates.sh`, `llvm-tools` dans
   l'image) : -4 % de CPU. Contrairement a ce que le gist observait sur sa machine.

Negatifs enregistres : la fenetre lockstep du seeding (`BWA4_LOCKSTEP_N`) est deja optimale a 16 sur
un vrai index (32 : 18,8 s de seeding, 64 : 20,0 s, 128 : 20,2 s contre 17,0 s), et la
deduplication des lookups SA ne paie pas (section suivante).

Le CPU etait deja passe sous celui du fork avant le recouvrement (0,96), ce qui disait que le travail
restant n'etait pas du calcul mais du remplissage de pool. C'est ce que le point 4 est allé chercher.

## Campagne perf : la resolution du suffix array (2026-08-04)

Profil du regime du gist de @nh13 (GRCh38, `-t16`, `-K` par defaut, entree gzippee) : l'etage
`align` fait **77 % du mur**, occupation du pool 98,9 %, donc pas un probleme de barriere. Dedans,
`BWA4_CHAIN_TIME` designe la construction des chaines, et la construction des chaines c'est
**42 567 543 resolutions de suffix array pour 500k paires**, chacune une marche aleatoire dans un
index de 10 Go. Le goulot est la latence memoire, pas le DP.

**Ce qui a paye.** La fenetre de prefetch de `get_sa_batch` (nombre de marches LF en vol) etait a 32.
Balayage a `-t16` sur GRCh38, cout par lookup : W=8 369 ns, 16 272 ns, 32 211 ns, 64 185 ns,
96 162 ns, **128 152 ns**, 192 147 ns, 256 144 ns. Reglee a 128, le coude et non le minimum : au-dela
la courbe est plate et les tableaux de pile encombrent le L1. `BWA4_SA_WINDOW` permet de rebalayer
ailleurs. Neutre en sortie par construction, verifie par md5 d'un run GRCh38 complet a W=32 et W=128.

Deux autres, plus petites : chargement de l'index par `pread` depuis le pool (10 Go depuis un page
cache chaud etaient copies par un seul thread, 2,4 s contre 1,8 s pour le fork, maintenant 1,8 s,
RSS inchange), et suppression du teardown (`mem::forget` sur l'index plutot que rendre 10 Go region
par region apres le dernier octet ecrit).

Resultat dans le regime du gist, 5M paires, meilleur de 3, bras alternes : mur 41,35 s -> **39,58 s**
contre 36,52 s pour le fork, soit un rapport qui passe de 1,14x a **1,08x** ; CPU 573 s -> **564 s**
contre 550 s, soit 1,055x -> **1,025x**.

**Ce qui n'a pas paye, et qu'il ne faut pas re-instruire.** 20 a 24 % des lookups d'un lot demandent
une ligne qu'un autre lookup du meme lot demande deja (`BWA4_SA_DUP=1`). Resoudre chaque ligne
distincte une seule fois exige de trier les lookups d'abord, et le tri coute autant que les ~22 % de
marches evitees : meilleur de quatre passes alternees a `-t16` sur 500k paires, **63,42 s de CPU sans
la deduplication contre 64,49 s avec**, et le mur a egalite (6,48 s contre 6,50 s). C'est une
consequence du point precedent : a fenetre 128 les marches se recouvrent assez bien pour que retirer
un cinquieme d'entre elles rapporte moins qu'un passage en O(n log n) ne coute. A l'ancienne fenetre
de 32, l'arbitrage aurait pu s'inverser.

Le tri seul (spike CCGrid 2013, `BWA4_SA_SORT=1`) est lui aussi neutre desormais : 158 ns par lookup
dans les deux ordres.

## Campagne perf : le noyau de mate rescue, et le classement contre le fork

**Etat au 2026-07-29 (M4 Max, 1M paires GIAB reelles, PE `-t12 -K 10M`, binaire PGO, ordre alterne,
5 reps) : bwa-mem4 25.81s de moyenne contre 27.48s pour fg-labs/bwa-mem3, gagnant 5 fois sur 5,
et nous restons les seuls octet-identiques a bwa-mem2 2.3 (2 013 247 enregistrements).** Avant ce
tour la meme mesure donnait 0,981x, soit 2% de retard.

**Banc de Nils, son echelle (giab-4m, GRCh38, `-t16`, `-K` par defaut, 8 lots) : bwa-mem2 188,24s,
fork 98,13s, bwa-mem4 91,01s. 1,078x contre le fork, 3 fois sur 3, dispersion 0,5%, et 2,06x contre
bwa-mem2 la ou le fork fait 1,91x.** Octet-identique sur 8 052 432 enregistrements, le fork divergeant
sur 18. Son gist donnait le fork a 1,29x en sa faveur sur ce meme jeu : le renversement total vaut
~1,39x. Reproductible par `T=16 K=default scripts/fork_bench.sh pe 3`.

**Mise a jour finale (meme jour) : dans le regime EXACT du gist (`-t16`, `-K` par defaut, entree
gzippee, sortie /dev/null), 6 victoires sur 6, rapport apparie median 1,104**, moyennes 25,68s pour
le fork contre 23,38s. Le CPU confirme, 342,3s contre 313,0s. Le gist annoncait 1,27 a 1,50x en
faveur du fork sur ce meme regime.

Le dernier levier n'etait pas un kernel : a `-t16` notre pool rayon etait **plafonne a 12 workers**
par le cap P-core macOS pendant que le fork en utilisait 16. Le cap se justifiait sur le petit banc
simule ; sur donnees reelles il coute 9,2% de mur (24,25/26,31/26,39s plafonne contre
23,00/23,09/23,82s sans). Il est desormais opt-in (`BWA4_PCORE_CAP=1`). Neutre en sortie, verifie :
a `-K` fixe le md5 est inchange.

La sonde de barriere (`BWA4_BARRIER_TIME=1`, nouvelle) a permis de le trouver et a REFUTE au passage
l'hypothese du desequilibre de barriere : occupation 99,2% sur align, 97,5% sur rescue, 98,6% sur
sam_emit, queues a 0-1%.

Deux leviers, tous deux sur le noyau u8 de mate rescue, qui tournait a 7,5 Gcell/s alors que 16
voies u8 a ~4 GHz en promettent bien davantage. La boucle de colonnes porte
maintenant une plage rapide sans padding (`n_fast` = la plus courte requete vive du groupe), ce qui
retire six operations vectorielles par cellule sur 92,5% de la matrice. Mesure a deux instruments :
-14,2% de CPU noyau (3/3) et -9,7% de mur en build simple, -5,0% de mur en build PGO (4/4). Porte
sur les six noyaux (NEON u8/i16, AVX2 u8/i16, AVX-512BW u8/i16).

Le second est le **blocage deux lignes de cible** : les lignes `i` et `i+1` partagent le chargement
de colonne de requete, `h_prev[j]`, `e[j]` et le rangement `h_cur[j]`, parce que la diagonale de la
ligne `i+1` est le H de la colonne precedente de la ligne `i` et que son report E est celui que la
ligne `i` vient de produire. Cinq acces memoire pour deux cellules au lieu de dix, et deux chaines H
independantes a entrelacer. **-8,9% de CPU noyau (3/3, dispersion intra-bras de 0,02s)**, -6,3% sur
l'etage rescue, -2,2% de mur (sous le plancher, garde sur la force de la mesure noyau). Porte sur les
trois noyaux u8 (NEON, AVX2, AVX-512BW) ; les noyaux i16 gardent la boucle une ligne, c'est le chemin
froid.

**La verification x86 n'est plus une simple compilation** : Rosetta 2 execute AVX2 sur cette machine,
donc `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo test --workspace --target x86_64-apple-darwin`
execute reellement les noyaux AVX2 contre la reference scalaire. Ajoute a `scripts/check.sh`.
AVX-512 reste non couvert (pas de `avx512bw` sous Rosetta).

Pistes fermees par la mesure dans le meme tour, toutes documentees dans `docs/perf-levers.md` :
tri des jobs de rescue par longueur (les fenetres font toutes `2 * max_dist`, gain exactement nul),
deduplication des jobs (19 sur 739 868), argmax de colonne recupere a la demande (+18% de CPU
noyau), tri introsort indirect par tags d'index (-3,7% de mur), backend gzip (plafond 1,2%),
arene plate pour la passe inverse (nul).

## Phase 0 de la campagne perf : mesure de reference contre le fork

Trois bras entrelaces dans la meme passe (`scripts/fork_bench.sh`), index genome, binaire PGO,
4M reads/paires GIAB HG002, `-t8`, `-K` 10M (60 batches SE / 119 PE), mediane de 3, chaque binaire
prechauffe. Le tag `HN:i` que le fork ajoute a chaque enregistrement est retire sur les trois bras.

| | bwa-mem2 2.3 | `fg-labs/bwa-mem3` | bwa-mem4 | nous vs fork |
|---|---|---|---|---|
| SE temps | 78,25 s (1,00x) | 37,75 s (2,07x) | **29,98 s (2,61x)** | **1,26x, on gagne** |
| PE temps | 288,56 s (1,00x) | **136,45 s (2,11x)** | 155,00 s (1,86x) | **0,88x, on perd** |
| pic RSS | 16 797 Mo | **11 034 Mo** | 19 218 Mo | **0,57x, on perd** |

**Verdict : on bat le fork en SE (1,26x), on le perd en PE (il est 1,14x plus rapide), et on perd
nettement en RAM des deux cotes (1,75x son pic).** Le « 1,29x derriere » cite jusqu'ici etait un
artefact de regime : `-t1`, align-only, region 2 Mbp au BWT cache-resident. Log :
`work/forkbench/20260721_222522` (SE) et `.../20260721_223530` (PE).

Trois faits pour la suite de la campagne :
- **RAM (le levier le plus clair).** Le fork annonce `SA compression enabled with xfactor: 8` : il
  compresse le tableau de suffixes echantillonne, nous non. Cela explique vraisemblablement l'essentiel
  des 5,8 Go qu'il gagne sur bwa-mem2. C'est une technique nommee, pas une micro-optim a deviner. NB
  nos pages `.0123` sont file-backed (mmap), les siennes probablement anonymes : le pic RSS reste la
  bonne mesure de « ca tient sur la machine » mais les deux chiffres ne reagissent pas pareil sous
  pression.
- **PE (l'ecart de vitesse a fermer).** Notre sauvetage de mate vectorise est notre plus gros gain
  vs bwa-mem2 a `-t1` (1,51x -> 2,4x) mais le fork nous repasse devant a `-t8`. Piste : le
  `batch_mate_rescue` et son parallelisme (goulot d'Amdahl deja identifie). PE tres stable
  (bwa-mem2 +/-0,2 %), donc le 0,88x n'est pas du bruit.
- **Concordance du fork (a verifier avant d'affirmer).** Sur 4M reads le fork emet moins
  d'enregistrements que bwa-mem2 : **17 de moins en SE** (4025128 vs 4025145) et **18 de moins en PE**
  (8052420 vs 8052438), au-dela du tag `HN`. Nous sommes octet-identiques sur la totalite. A reduire a
  un cas reproductible avant toute affirmation publique : ce peut etre notre facon de l'invoquer.

**Decision phase B** : elle a lieu, et sa cible est le PE, pas le seeding `-t1`. Le regime cible
(`-t8` end-to-end) nous donne l'avantage en SE mais un deficit reel en PE et en RAM ; l'ancien chiffre
`-t1` qui la motivait est retire.

### Phase A, levier RAM : pic 19,2 -> 11,2 Go (deux commits, octet-identiques)

**Le deficit RAM est ferme, on est passe de 0,57x a ~egalite avec le fork.** Deux changements qui se
composent, aucun ne suffit seul, chacun garde l'index octet-identique a bwa-mem2 (les fichiers ne
changent pas sur disque, on change ce qu'on charge) :

1. **Reference depuis `.pac` (2-bit) au lieu du mmap `.0123` (1 octet/base)** (`413e313`). Deballe une
   base a la volee (shift-and-mask ; la moitie reverse-complement est reconstruite, pas stockee). En
   theorie -6,2 Go, **en pratique 0 sur le pic** : le pic n'etait pas la reference.
2. **BWT lu par `read()` dans les buffers alignes, au lieu de mmap+copie** (`79c7a32`). Le mmap+copie
   tenait le BWT en double (pages mappees **plus** copie) le temps du memcpy, ~20 Go transitoires : le
   vrai plafond, mesure. `read()` laisse la source dans le page cache du noyau (hors RSS), donc le pic
   n'est plus que les buffers destination. C'est ce commit qui deplace le pic, et il ne le fait que
   parce que le `.pac` a deja retire la reference (sinon `.0123` resterait a ~16 Go).

Mesure (index genome, `/usr/bin/time -l`) : **pic 19218 -> 11206 Mo, -8 Go**. Contre le fork a 10952 Mo,
on passe de **0,57x a 0,977x** (il garde ~2,3 %, sa compression SA, qu'on ne peut pas porter sans
casser l'octet-identite de l'index). Parite tenue : 21 tests d'index (chaque tableau charge passe par
`get_sa`/`backward_ext`), oracle SE 5000, PE `cmp`-clean sur 201266 enregistrements genome reels.
Impact vitesse du `read()` : **nul** (SE 4M `-t8` 28,54 s, le chargement est amorti sur 60 batches).

### Phase B, le deficit PE = le kernel de sauvetage de mate

> ⚠️ **2026-07-29 : cette section est PERIMEE. Le deficit de 1,45x sur le rescue n'existe plus, et
> les deux etapes suivantes qu'elle recommande ont ete testees et sont mortes.**
>
> Re-mesure sur **vraies donnees GIAB HG002** (1M paires, GRCh38, PE, `-t12`, bras entrelaces,
> `scripts/fork_bench.sh`), binaire PGO des deux cotes :
> **bwa-mem2 54,91 s / fork 28,18 s / bwa-mem4 28,71 s, soit 0,981x contre le fork** (RSS 0,985x),
> nous octet-identiques a bwa-mem2 sur 2 013 247 enregistrements, pas le fork. C'est une egalite
> sous le plancher de bruit de 3 %. Il n'y a plus de deficit PE a combler.
>
> Deux pieges que cette section illustre, tous deux mesures dans `docs/perf-levers.md` :
> * **Le tableau ci-dessous est sur wgsim**, ou le mate rescue fait 10 % du wall contre **59,5 %**
>   sur du reel. Le meme harnais donne 1,41x EN NOTRE FAVEUR sur wgsim et 1,10x CONTRE nous sur du
>   reel : un basculement de 1,55x du seul fait du jeu de reads. Aucun chiffre de vitesse mesure sur
>   `work/r1_500k.fq` n'est recevable.
> * **Le PGO vaut 12,4 % sur du reel** (32,28 -> 28,71 s), pas les 8,5 % mesures sur wgsim. Un bras
>   `cargo build --release` n'est pas comparable au binaire qu'on livre.
>
> Les deux suites recommandees ci-dessous, testees depuis :
> * *« comparer `batched_ksw_align2` au kswv du fork »* : le fork aplatit tous ses jobs en un lot
>   unique, nous appelons le kernel une fois par round. **Mort** : `BWA4_RESCUE_ROUNDS=1` mesure
>   102 jobs par appel en moyenne, 93,5 % du travail dans des appels de 64 jobs ou plus, contre
>   16 voies NEON u8. Les vecteurs sont deja pleins, aplatir ne refile rien.
> * *« le rescue est LE levier »* : son cout est son nombre de cellules (3,68 M jobs pour 1M paires,
>   ~207 k cellules chacun, ~762 G cellules), fixe par l'algorithme de bwa. `max_rounds` sature le
>   plafond `-m` a 50 sur **tous** les chunks.


**Decompose (t8, 500k paires, genome, PGO, mediane-3 ; `-S` isole le rescue) :**

| | full | `-S` (sans rescue) | cout rescue |
|---|---|---|---|
| **mem4** | 18,97 s | **8,33 s** | **10,64 s** |
| **fork** | 18,38 s | 11,03 s | **7,35 s** |

**Tout ce qui n'est pas le rescue, on gagne 1,32x** (8,33 vs 11,03), comme en SE. **Le rescue, on perd
1,45x** (10,64 vs 7,35) : ce deficit de 3,3 s efface notre avance de 2,7 s et fait tout le retard PE.
Le rescue est 56 % de notre temps PE.

**Ca renverse la these « le sauvetage vectorise est le gros gain PE ».** C'est un gain vs le rescue
SCALAIRE de bwa-mem2, mais le fork le vectorise AUSSI (base bwa-mem2 + kswv de nh13) et 1,45x plus
vite. **Le rescue est LE levier PE, seul.** Point important : les negatifs « fast-paths du kernel
epuises » portaient sur le kernel d'EXTENSION (bandedSWA/ksw_extend2) ; le rescue est un kernel
DIFFERENT (local-SW inter-sequences, `crates/bwa-neon/src/matesw.rs`), jamais audite contre celui du
fork. Les deux sont NEON 16 voies. Prochaine etape : comparer notre `batched_ksw_align2` au kswv du
fork ligne a ligne. Gate d'octet-identite du rescue : `matesw_equals_scalar` + oracle PE.

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

## Scalabilite : ou passent les 24 % manquants a `-t12` (2026-08-09)

Question posee : obtenir une scalabilite proportionnelle. Reponse mesuree : impossible par
l'ordonnancement, le manque est du trafic memoire. Le detail, parce que la conclusion negative n'a
de valeur qu'avec les chiffres qui l'excluent.

**Mesure de reference**, 2 M paires (le plancher de demarrage y pese 0,5 % au lieu de 18 %) :

| | `-t1` | `-t4` | `-t8` | `-t12` | `-t16` |
|---|---|---|---|---|---|
| mur | 45,36 | 12,11 | 6,61 | 4,97 | 4,77 |
| acceleration | 1x | 3,75x (94 %) | 6,86x (86 %) | 9,13x (**76 %**) | 9,51x (59 %) |
| CPU-s | 45,6 | 47,8 | 50,2 | 55,1 | 65,1 |

**Trois causes eliminees par la mesure.**

1. *L'equilibrage.* `BWA4_BARRIER_TIME` donne pour la region `align` a `-t16` une occupancy de
   **97,7 %** et une queue de **0,3 %** (95,1 % / 3,9 % pour `sam_emit`). Il n'y a pas de travailleur
   qui attend : les regions paralleles sont deja pleines.
2. *Le demarrage.* Un run de 1000 paires coute **0,23 s de mur et 2,45 s de CPU** : c'est le
   chargement des 15,9 GB d'index (9,4 BWT + 5,8 `.0123` + 0,7 pac), lu en tampons alignes et non
   mmap. Sur 500 k paires cela fait 18 % du mur et explique pourquoi ce jeu-la exagerait le probleme ;
   sur 2 M paires c'est 0,5 %, et la degradation reste entiere.
3. *Le reglage du parallelisme memoire.* La largeur lockstep a ete re-balayee **a `-t12` et `-t16`** :
   N16 et N32 se tiennent dans le bruit (4,53 / 4,66 contre 4,61 / 4,64 a `-t16`), le genou mesure a
   `-t4` s'aplatit des que les threads se disputent le controleur. Faire dependre la largeur du
   nombre de threads ne rendrait rien.

**Ce qui reste, et c'est la cause.** A `-t12`, c'est-a-dire **avant que l'ordonnanceur ne place quoi
que ce soit sur les 4 cœurs E**, le meme travail coute deja **+21 % de CPU** (55,1 contre 45,6). A
`-t16` l'inflation monte a +43 %, l'ecart supplementaire etant l'heterogeneite P/E. Les 21 % sont de
la contention du systeme memoire : le seeding fait deux acces aleatoires de 64 octets par extension
de base sur un BWT de 9,4 GB, et douze cœurs qui font ça saturent la file de misses.

**Trois leviers essayes ensuite, tous negatifs, tous mesures a `-t16` ou apparies.**

- *Profondeur de prefetch du seeding* (`BWA4_SEED_PREFETCH`). A `-t16`, P8 bat P0 sur **5 paires sur
  5** : le prefetch reste rentable sous contention, il ne l'aggrave pas. Defaut inchange a 8.
- *Fenetre de `get_sa_batch`* (`BWA4_SA_WINDOW`). 128 gagne les deux series contre 64 et 32
  (4,87 / 4,74 contre 5,03 / 4,97). Defaut inchange, et 128 est deja le plafond compile.
- *Faconnage des allocations dans la construction de chaines*. `chains` demarre a capacite nulle et
  chaque chaine alloue ses seeds a capacite 1, donc realloue a 2, 4, 8. Reserver (32 chaines /
  4 seeds) coute **+0,35 %** de CPU, 5 defaites sur 8 ; la variante sobre (8 / 2) coute exactement
  autant, 5 defaites sur 6. mimalloc sert ces petites allocations mieux qu'une pre-reservation, qui
  ne fait que changer de classe de taille et toucher plus de memoire. Retire.

Au passage, verifie : la sortie reste **octet-identique a `-t24`, `-t32` et `-t48`**, et au-dela de
16 threads sur cette machine il n'y a que de la sur-souscription (`-t16` 4,38 s, `-t24` 4,61,
`-t32` 4,64), ce qui est attendu et non un defaut.

Avertissement de methode, paye deux fois dans cette section : sur le jeu de 2 M paires a `-t16`, la
derive thermique atteint **15 %** en dix runs consecutifs. Un balayage lu ligne par ligne y montre
une fausse monotonie decroissante en performance. Seul l'appariement A/B est lisible a ce regime.

**Consequence.** Rendre la scalabilite proportionnelle demande de **reduire le trafic**, pas de mieux
ordonnancer : une structure d'index plus compacte ou plus locale (le `bwa_index::lisa::LearnedSa`
present dans l'arbre en est une piste), pas un knob. Note pratique en attendant : **`-t12` donne
96 % du mur de `-t16` pour 15 % de CPU en moins**, et sur le petit jeu il donne meme le meme mur.

## Campagne perf CPU et classement contre minibwa (2026-08-09)

Trois leviers mesures, tous octet-identiques, tous gardes :

| levier | effet (CPU-s, `-t4`, medianes) | victoires |
|---|---|---|
| Compteurs `traffic` compiles hors du binaire (`backward_ext`, `get_sa`, `get_sa_batch`) | 13,34 → 13,21, **-0,97 %** | 6/7 |
| `prefetch_ahead()` sorti de la boucle de `LsSlot::step` | 13,11 → 13,03, **-0,61 %** | 11/15 |
| Largeur lockstep 16 → 32 sur aarch64 | 13,00 → 12,835, **-1,27 %** | 6/6 |

Les deux premiers sont le meme defaut : une lecture de `OnceLock` (charge acquire) laissee sur un
chemin appele ~1e9 fois par lot de 500 k paires. Le troisieme etait une dette ecrite dans le code :
le defaut de 16 avait ete regle sur `work/region.fa`, dont le BWT tient en cache, donc sur un index
sans latence DRAM a masquer, ou des voies supplementaires sont du pur surcout. Sur un vrai index le
genou est a 32. La valeur reste a 16 sur x86, faute d'instrument : Rosetta tourne sur ce meme
systeme memoire M4 et ne peut pas repondre a une question de parallelisme memoire.

**Profil apres coup** (`-t8`, donnees reelles, part du busy) : seeding `LsSlot::step` 22,3 % +
`get_sa_batch` 13,6 % + `mem_collect_smem_batched` 5,3 % = **41 %** ; extension 22,9 % ; chaining
7,7 % ; rescue 6,2 %. C'est la reponse a « quel est le `p` d'Amdahl du GPU » : porter l'extension
entiere plafonne a 23 %.

### Contre le fork fg-labs/bwa-mem3 v0.9.0 (sorti le 2026-08-07)

La release n'apporte rien de performance : `--compat=bwa-mem` (octet-identique a bwa 0.7.19 en plus
de bwa-mem2), la derivation du FLAG 0x2 depuis la region de meilleur score, `pa:f` unifie entre les
ecrivains SAM et BAM, et la detection des echecs d'ajout de tag BAM. Les trois correctifs ne touchent
que les runs ALT-aware.

Verifie avant de chronometrer : `--compat=bwa-mem2` sort **exactement notre md5** (`791c21c2...`),
donc les deux binaires produisent le meme fichier et la comparaison porte sur le meme travail. Son
mode natif coute la meme chose que son mode compat, a 0,3 % pres.

| murs, medianes de 3 | `-t1` | `-t4` | `-t16` |
|---|---|---|---|
| fork v0.9.0 (compat) | 13,19 | 3,55 | 1,38 |
| **bwa-mem4** | **12,11** | **3,34** | **1,31** |
| avance | **1,089x** | **1,063x** | **1,053x** |

En CPU-s nous gagnons a `-t1` (12,17 contre 13,17) et `-t4` (12,82 contre 13,66) et nous perdons a
`-t16` (17,35 contre 16,99 sur 5 tirages, **+2,1 %**).

**Ce n'est pas un defaut de scalabilite, contrairement a ce que cette section affirmait d'abord.**
Verifie en balayant la frontiere P/E de la machine (12 cœurs P, 4 cœurs E) :

| CPU-s | `-t8` | `-t12` | `-t14` | `-t16` |
|---|---|---|---|---|
| bwa-mem4 | 13,47 | 14,28 | 16,18 | 17,24 |
| fork v0.9.0 | 14,19 | 14,52 | 16,12 | 17,03 |

Les deux courbes sont plates jusqu'a `-t12` et cassent entre `-t12` et `-t14`, c'est-a-dire quand
l'ordonnanceur commence a placer des threads sur les cœurs E, qui consomment plus de secondes CPU
pour le meme travail. Le fork gonfle de +17 % sur ce segment, nous de +21 %. L'ecart apparent
(+27 % contre +43 %) etait un artefact de normalisation : notre `-t1` etant meilleur d'une seconde,
la meme inflation absolue devient un plus gros pourcentage. Note pratique au passage : `-t12` donne
deja le mur de `-t16` (1,30 contre 1,31), donc les quatre cœurs E n'apportent rien ici.

**Levier essaye et retire.** Le chunk tombe de 4096 lectures a `-t4` a 1024 a `-t16`, ce qui laissait
soupconner un cout fixe par chunk. Il existe : a `-t16`, un chunk de 16 384 vaut **-5,3 % de CPU**
(16,84 → 15,95). Mais il coute **+12,5 % de mur** (1,28 → 1,44), 30 chunks pour 16 travailleurs
desequilibrant la queue. Et il n'explique pas l'inflation : a taille de chunk **identique** (4096),
le CPU-s passe quand meme de 12,82 a `-t4` a 16,63 a `-t16`. Le knob `BWA4_CHUNK_READS` a donc ete
retire. Ce que les profils montrent au passage, c'est que les etages par lots (`get_sa_batch` -11 %,
`build_chains_from_resolved` -15 %) profitent de lots plus gros : le gain est dans le remplissage des
lots, pas dans les allocations.

A ne pas superposer au 0,981x du 2026-07-29 : celui-la etait mesure sur donnees GIAB reelles et
**avec PGO** des deux cotes, celui-ci sur les 500 k paires **sans PGO**.

### Contre bwa-mem2 2.3 et contre minibwa

Meme machine (M4 Max), memes 500 k paires, entrelace, medianes de 3, murs en secondes :

| | `-t1` | `-t4` | `-t16` |
|---|---|---|---|
| bwa-mem2 2.3 | 43,04 | 13,36 | 5,81 |
| **bwa-mem4** | 12,34 | 3,38 | **1,27** |
| minibwa 0.7-r421 | **10,05** | **3,17** | 1,52 |

En CPU-s : bwa-mem2 41,8 / 44,9 / 52,2 ; bwa-mem4 12,4 / 12,9 / 16,8 ; minibwa 10,0 / 10,3 / 11,9.

**Lecture.** minibwa est **plus efficace en CPU partout** (-20 a -29 %), nous **scalons mieux** :
9,6x de `-t1` a `-t16` contre 6,6x pour lui, ce qui nous donne le mur a `-t16` (1,27 contre 1,52).
Son CPU-s ne monte presque pas pendant que son mur plafonne, ce qui pointe une partie serielle chez
lui plutot qu'une saturation memoire.

La comparaison est legitime et non un artefact de travail non fait : 400 000 enregistrements de part
et d'autre, aucun non-mappe, aucun secondaire, **99,948 %** de positions primaires identiques
(99,963 % a ±5 bp), MAPQ different sur 1,74 %.

Deux reserves. (1) bwa-mem2 sur Apple Silicon est le portage SSE2NEON de noyaux ecrits pour
AVX2/AVX-512 : le facteur 3,5x que nous obtenons et le 4,3x de minibwa sont gonfles dans la meme
direction, et le classement x86 reste a mesurer. (2) minibwa a un **algorithme different** (seeds
BWT facon bwa-mem, chaining et SIMD facon minimap2) et n'est pas octet-identique a bwa-mem2 ; il est
libre de tout ce que notre critere d'acceptation interdit. Le titre vise reste « le plus rapide des
aligneurs octet-identiques a bwa-mem2 », et sur ce terrain minibwa n'est pas un concurrent.

## GPU : la taille de lot, et pourquoi elle ne suffit pas (2026-08-09)

L'extension tourne sur le GPU depuis la section precedente, octet-identique, mais elle ne gagnait le
mur qu'a `-t4`. Le diagnostic ecrit alors etait que le noyau recevait des lots trop petits : le
budget de chunk est dimensionne en RAM (`INFLIGHT_READS / worker_count`), donc `-t16` distribue
1 024 lectures par chunk, soit ~2 700 travaux d'extension par lancement, la ou le noyau ne sature
qu'a ~32 000. Ce diagnostic est confirme. La conclusion qu'on en tirait ne l'est pas.

**Le levier.** `BWA4_GPU_CHUNK` fixe la taille de chunk. **Absolue et non un multiplicateur** :
un balayage de multiplicateur place l'optimum a 4x a `-t4` et 16x a `-t16`, et ces deux points sont
la meme valeur, 16 384 lectures. La grandeur qui compte est la taille du lot, et elle ne depend pas
du nombre de travailleurs qui l'ont produit. Le defaut est 16 384 des que `BWA4_GPU=metal`, parce que
sans lui le GPU **perd** le mur a tout `-t` > 4.

**Mesure** (jeu h, 500 k paires, medianes de 3, entrelace, `/usr/bin/time -l`) :

| | CPU seul (mur / CPU-s) | meilleur GPU | mur | CPU-s | RSS |
|---|---|---|---|---|---|
| `-t4`  | 3,40 / 13,09 | 32768 : 3,03 / 10,29 | **-10,9 %** | -21,4 % | +0,4 GB |
| `-t8`  | 1,90 / 14,21 | 16384 : 1,90 / 11,34 | **0 %**     | -20,2 % | +0,3 GB |
| `-t16` | 1,38 / 19,06 | 32768 : 1,54 / 15,30 | +11,6 %     | -19,7 % | +1,8 GB |

Le levier est reel : `-t8` passe de +9,5 % a l'egalite, `-t16` de +52 % a +12 %.

**Ce qui est infirme.** La section precedente annoncait que l'agregation « couvre la demande de
`-t16` ». Elle ne la couvre pas. Le mur de la variante GPU plafonne autour de 1,54 s **quel que
soit** `-t` : au-dela de `-t8` c'est le GPU qui est le chemin critique, et lui donner des lots plus
gros deplace le plafond sans le lever. La projection etait faite a partir du debit a saturation
(22,4 Gcell/s) sans tenir compte du fait que saturer le noyau et saturer la machine sont deux
choses. Ce qui leverait le plafond est le paquetage `uchar4` (plafond mesure ~330 Gcell/s), pas la
taille de lot.

**Le chiffre a retenir** est la colonne CPU-s, stable a **-20 % a tous les `-t`** : le GPU prend un
cinquieme du travail CPU. Sur une machine dediee ou seul le mur compte, cela ne vaut que jusqu'a
`-t8`. Sur une machine partagee, c'est un cinquieme de coeurs rendus.

Parite verifiee sur les deux jeux reels, a `-t3` et `-t8`, pour chaque taille de chunk : la taille de
chunk n'est que de l'ordonnancement, et les regions par lecture n'en dependent pas.

**Decision (2026-08-09) : le GPU est abandonne sur `dev`.** Le meilleur point absolu de la machine
reste le CPU seul a `-t16`, 1,38 s ; le meilleur point GPU, tous `-t` confondus, est 1,54 s, parce
que son mur plafonne la. Le GPU ne payait que dans trois cas etroits (`-t` plafonne a 4, facturation
au coeur-heure, machine partagee), et aucun n'est le cas d'usage vise. Les crates `bwa-metal` et
`bwa-gpu`, les scripts `gpu_parity.sh` / `msl_probe.sh` et le branchement CLI sont retires de `dev`.

**Pourquoi Parabricks n'est pas un contre-exemple (2026-08-09).** Le resultat ci-dessus (-20 % de
CPU, bascule vers `-t8`) est ce qu'on attend d'un GPU **integre** avec **un seul etage porte**, et il
ne contredit pas les 20-25 min par genome 30x annoncees sur 8x A100 :

1. **Materiel.** Memoire unifiee et pas de tensor core utile en DP entier, contre du HBM a 2-3 TB/s.
   L'ecart de debit brut est d'un a deux ordres de grandeur, donc leur bascule n'existe pas.
2. **Amdahl.** Ici seule l'extension part sur le GPU ; seeding FM, chaining et sortie SAM restent
   CPU. Le profil dit que le seeding pese **41 %** du busy et l'extension 23 % : meme une extension
   gratuite ne rendrait que 23 %. Parabricks porte la chaine entiere (alignement, tri, duplicats,
   BQSR) en VRAM sans aller-retour disque, et une bonne part de leur gain vient de la fusion des
   etages, pas du noyau.
3. **Ce qu'ils vendent** est le delai de restitution, pas le debit par dollar. Une machine a 200 k$
   perd souvent en $/genome et gagne en heures, ce qui est le bon arbitrage en clinique.
4. **Occupancy.** Leur lot est de l'ordre du million de lectures. Le notre est dimensionne par
   l'ordonnanceur de l'hote, ce qui est precisement le plafond mesure ci-dessus.
5. **La contrainte qu'ils n'ont pas.** Parabricks est *equivalent* a BWA-MEM, pas *identique* : ils
   divergent sur les egalites de score. Nous n'avons pas cette liberte, ce qui interdit la
   reassociation, les heuristiques approchees et la demi-precision. Elle vaut cher en perf.

S'en approcher demanderait de porter aussi le chaining et le seeding et d'accepter du
non-determinisme, c'est-a-dire de renoncer au critere d'acceptation du projet. Ce n'est pas un
arbitrage de performance, c'est un changement de projet.

Un chiffre manque pour completer le dossier : le debit d'**un cœur NEON** sur le noyau d'extension.
On a 22,4 Gcell/s GPU a saturation et 9,08 en lot de production, mais pas le denominateur, donc le
ratio GPU/cœur n'est connu que pour le rescue, ou il vaut **1x** (10,95 Gcell/s, `0638990`).

Le travail n'est pas perdu : il est **entier sur la branche `gpu`** (`6553a3d`), noyaux MSL, gate de
parite et mesures compris. Ce qui reste sur `dev` est la couture `SwBackendAsync` de #53 et les
barrieres de #54, gardees parce qu'elles sont agnostiques du backend, testees sur CPU, et que ce sont
elles qui rendraient un retour en arriere possible. Ce qu'il faudrait pour que ce retour vaille le
coup est ecrit ci-dessus : le paquetage `uchar4`, pas une taille de lot.

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
2. **`mem_sort_dedup_patch`** : **regarde (2026-07-29)**, 13,3 % du busy sur un profil `sample` en
   donnees reelles. Ce n'est PAS l'etage dedup (0,8 % du wall) : c'est le mate rescue qui reinsere une
   region puis retrie tout le vecteur, a chaque orientation, sur jusqu'a 50 rounds. `BWA4_DEDUP_SHAPE`
   mesure **n moyen = 94,87** et **1,84 M d'appels avec n >= 65**, tous venant du rescue.
   Deux correctifs octet-identiques livres (`dirty` pour sauter les re-tris a vide, **-14,2 %
   d'appels** ; pile d'introsort en tableau fixe, ~10 M d'allocations en moins) : **~2 % de wall en
   design apparie 4/4, donc sous le plancher de 3 %**. Le reste est incompressible sans une structure
   incrementale reproduisant l'ordre exact des egalites de klib, la permutation etant observable dans
   la sortie. Detail complet et tables dans `docs/perf-levers.md`.
3. **SA-IS parallele** (Tier B de la phase 8c) : l'indexeur reste mono-thread sur le tableau de
   suffixes, qui domine son pic RSS et son temps.
4. ~~**Re-mesurer le fork `fg-labs/bwa-mem3`**~~ : **fait (2026-07-29)**, sur vraies donnees GIAB et
   en PGO : **0,981x, egalite**. Voir l'encart de la phase B et `docs/perf-levers.md`.
5. **Verifier le PGO sur Graviton** : nh13 mesure le PGO a **-0,4 %** sur Graviton4 la ou il vaut
   **+12,4 %** ici. Les deux ne peuvent pas etre une propriete du code : soit son build n'a pas
   applique son profil, soit le benefice est propre a l'Apple Silicon. Seule question ouverte du
   dossier, et elle se tranche de son cote.
