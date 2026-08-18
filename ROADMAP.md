# Roadmap

Une phase = une branche. Commits frequents ; PR vers `dev`, `dev` promu sur `main` a la release.
Cible d'acceptation : index et SAM **octet-identiques** au binaire `bwa-mem2` 2.3 patche (oracle).

**Version courante : 4.3.3.** Les phases 0 a 10 sont terminees. Ce document reste le journal des
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
| 11 | | gate GIAB `hap.py`/`vcfeval` (concordance variants) | a faire, jalon v4.3.4 |

## Releases

| Version | Date | Contenu |
|---|---|---|
| 4.0.0 / 4.0.1 | 2026-07-22 | ALT contigs, BAM/CRAM, packaging, CI |
| 4.1.0 / 4.1.1 | 2026-07-23 | correctifs de release |
| 4.2.0 | 2026-07-30 | vague perf (mate rescue, dedup, `.pac` vectorise, CaPS-SA), sonde par etage |
| 4.3.0 | 2026-07-31 | `-x` route vers rammap, sous-commande `version`, credits bwa-mem3 / @nh13 |
| 4.3.1 | 2026-08-14 | couche mecanique Rust sur les 10 crates (preparee le 2026-08-04, publiee ici) ; lecteur hors du chemin critique et entree FIFO reparee, lecture parallele des deux fichiers de mates, une allocation par read au lieu de trois, tri par longueur du lot de rescue, backend C libsais par defaut (+ OpenMP en option), PGO reparee sur macos-14 |
| 4.3.2 | 2026-08-14 | cible bibliotheque : les implementations de commandes sont appelables sans sous-processus, et `MemArgs` derive `Default` pour etre constructible par un embarqueur |
| 4.3.3 | 2026-08-15 | le crate est enfin dependable : xz de needletail, sortie BAM/CRAM et mimalloc passent en features, donc la cible bibliotheque se compile sans toolchain C, sans conflit de lien lzma et sans allocateur global impose (#62, #63, #64) |

## Jalons ouverts (GitHub Projects nº3)

| Jalon | Contenu | Etat |
|---|---|---|
| v4.3.3 | parite perf x86_64 (issues #20, #25, #27, #32, #33) | ouvert |
| v4.3.4 | phase 11, gate GIAB `hap.py`/`vcfeval` ; suivi upstream `bwa-mem2#297` | ouvert |
| v4.3.5 | SA-IS parallele (l'indexeur reste mono-thread sur le tableau de suffixes), structure de dedup incrementale | ouvert |

Les trois jalons ont recule d'un cran : 4.3.2 portait le nom du jalon perf x86_64 et a ete publiee
avant lui, pour la cible bibliotheque. **Les milestones GitHub Projects portent encore les anciens
numeros et restent a renommer a la main**, ce fichier ne les renomme pas.

Les jalons avancent d'un cran sur le troisieme chiffre : une release courte et frequente plutot
qu'un saut de version mineure a chaque lot. Le jalon perf x86_64 demande une machine x86_64 et un
WGS complet, pas une mesure sur Apple Silicon.

## Statut (4.3.3)

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
du tableau de suffixes, qui est l'issue #37 et le jalon v4.3.5.

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

### #44 reste non verifie, et ce n'est pas une loterie (2026-08-15)

`require_avx512` fonctionne exactement comme prevu : il fait echouer le job en une trentaine de
secondes quand le tirage n'a pas `avx512bw`. Sept tirages consecutifs, puis dix-sept le 2026-08-09,
ont donne le meme AMD EPYC 7763, ce qui se lisait alors comme de la malchance. Ce n'en est pas :
**aucun runner accessible a ce projet n'expose `avx512bw`**, et deux d'entre eux ne le feront jamais.

| label | CPU | `avx512bw` |
|---|---|---|
| `ubuntu-22.04` | AMD EPYC 7763 (Zen 3), 17 tirages | non |
| `ubuntu-latest` | AMD EPYC 7763 (Zen 3) | non |
| `ubuntu-24.04` | AMD EPYC 9V74 (Zen 4) | non |
| `macos-15-intel` | Intel Core i7-8700B | non |
| `macos-26-intel` | Intel Core i7-8700B | non |
| `macos-13` | label retire, job jamais planifie | — |

Deux lectures a en tirer. Les runners Intel de macOS sont des Mac mini 2018 en Coffee Lake, une
generation grand public qui n'a jamais porte AVX-512 : cette piste est fermee par construction, pas
par le sort, et il est inutile de la retenter. Et `ubuntu-24.04` tourne sur du Zen 4, qui a bien
AVX-512 en silicium : c'est l'hyperviseur qui le masque, donc meme le materiel capable ne l'expose
pas. Retirer davantage ne changera rien.

Le harnais reste pret et il n'a pas de defaut : `avx512_matesw_u8_matches_scalar`,
`avx512_u8_and_i16_match_scalar` et la ligne `avx512_u8` des deux sondes attendent une machine, pas
du code. Condition de fermeture de #44, desormais precise : un coeur Intel Ice Lake ou Sapphire
Rapids (AWS `c6i`/`c7i`, Azure Dv5) le temps d'un `cargo test --release -p bwa-mem4-neon`, soit une
dizaine de minutes, ou un larger runner GitHub dont il faudra d'abord verifier le CPU de la meme
maniere. Tant que cette machine n'est pas louee, le noyau AVX-512 u8 reste du code jamais execute et
l'issue doit le dire ainsi plutot que d'attendre un tirage qui n'arrivera pas.

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
   v4.3.3, pas de la campagne SIMD, et le chiffre est ici confirme sur un runner neutre. A 150 bp sur
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

## Entree gzip : ou est le mur, et a combien de cœurs il tombe (2026-08-11)

Rob Patro publie un resultat qui vise exactement une fraction serielle que **tous nos bancs ont
manquee** : ils utilisent du FASTQ non compresse, alors que tout utilisateur reel a du `.fq.gz`. Son
diagnostic : l'inflation gzip est serielle (un decodeur par flux), donc un mappeur rapide affame ses
threads. Chiffres annonces : le split producteur/consommateur vaut **5,75x** a 64 threads sur
150 M paires, la ou doubler le budget de threads ne vaut que 9 %. Son correctif est en deux morceaux,
`rapidgzip-core` (inflation parallele, algorithme marqueur/fenetre de Knespel & Brunst) et
[`thread-broker`](https://crates.io/crates/thread-broker), qui **resout** le split au lieu de le
chercher : `d* = N * busy_producteur / (busy_producteur + busy_consommateur)`, mesure en temps **CPU**
et non en temps mur, parce que le temps mur contient le blocage et que le blocage depend du split.

**Mesure chez nous**, memes lectures, compressees contre non compressees :

| | mur non compresse | mur gzip | delta |
|---|---|---|---|
| `-t1` | 12,30 s | 12,11 | **0 %** |
| `-t8` | 1,86 | 1,90 | +2 % |
| `-t16` | 1,345 | 1,415 | **+5 %** |

Le probleme existe donc, et il **grandit avec les cœurs**, ce qui est la forme d'Amdahl attendue.
Mais il ne mord pas encore, et l'arithmetique dit a quelle distance il mord :

- notre demande a `-t16` est de **49 Mo/s par flux** (68 Mo en 1,4 s) ;
- un seul flux d'inflation tient **~1900 Mo/s** avec le gzip systeme de cette machine, et de l'ordre
  de 300-500 Mo/s avec notre `flate2`/`zlib-rs` ;
- il faudrait donc environ **6x notre debit d'alignement actuel**, soit de l'ordre de **100 cœurs**,
  avant que l'inflation elle-meme devienne le goulot.

Les +5 % mesures ne sont donc **pas** de l'inflation : c'est le **thread lecteur unique**, qui inflate
ET parse le FASTQ, et qui devient etroit quand seize travailleurs le sollicitent.

**Pourquoi le meme probleme vaut 5,75x chez lui et 5 % chez nous** : piscem mappe **20 M paires/s**,
nous faisons **0,37 M paires/s** a `-t16`. Un consommateur cinquante fois plus rapide par unite
d'entree atteint la limite du decodeur cinquante fois plus tot. Le levier est reel, il est bien
concu, et notre regime n'y est pas encore.

### Correction, apres avoir applique la loi de `thread-broker` a nos chiffres

Ce qui precede concluait « `rapidgzip-core` puis `thread-broker` le jour ou on vise 64 cœurs ».
**C'est faux**, et la mesure qui le montre a pris dix minutes.

Notre propre chemin de decompression (`flate2` + `zlib-rs`, celui que le lecteur recoit) fait
**~1500 Mo/s sur un flux**, et non les 300-500 supposes plus haut. Donc, sur un lot de 500 k paires :

- `busy_p` = 2 flux x 72 Mo / 1500 Mo/s = **0,096 s de CPU**
- `busy_c` = **16,9 s de CPU** a `-t16`
- **`d* = N * busy_p / (busy_p + busy_c) = 16 * 0,096 / 16,996 = 0,09 thread`**

Sa propre loi de commande alloue donc **un dixieme de thread** au decodage ; il faudrait `-t256`
pour qu'elle en demande 1,4. Et sa `EngagementPolicy` par defaut exige **8 threads par flux
producteur**, soit 16 pour nos deux, precisement parce qu'en dessous *« an inline, work-conserving
producer wins »*. Les deux verdicts concordent et ils viennent de son outil.

**L'inflation pese 0,6 % de notre CPU total.** La paralleliser ne rendrait rien, meme a 100 cœurs.

**D'ou viennent alors les +5 % ?** De l'arithmetique du lecteur : il paie ~0,048 s d'inflation par
fichier contre ~0,007 s de copie en clair, soit **+0,08 s pour deux fichiers**, contre un delta de
mur mesure de **+0,07 s** a `-t16`. C'est donc bien l'inflation, mais elle se voit **1:1 dans le mur
parce qu'elle est sur le chemin serie d'un lecteur unique**, pendant que seize travailleurs
consomment plus vite qu'un lecteur ne produit.

**Le correctif n'est donc ni un pool de decodeurs ni un broker**, c'est un recouvrement plus profond
de la lecture avec l'alignement (double buffering du lecteur). Meme cause qu'Amdahl, autre remede.
Et le premier geste reste de reparer le banc, pas le code : mesurer sur `.fq.gz`, puisque c'est ce
que les utilisateurs alignent.

Le crate reste juste pour le probleme qu'il vise. Son 5,75x est reel dans un regime ou le
consommateur est cinquante fois plus rapide que le notre par unite d'entree ; le notre n'y est pas,
et sa politique d'engagement le dit avant nous.

### hyalite 0.4.0

L'oracle tiers de `bwa-extend` passe de 0.3 a 0.4. C'est une **dev-dependency**, donc rien du binaire
livre ne bouge. Amont apporte `per-position maxima + bwa-compatible score2` et les maxima SIMD de
`align_pair_position_max`, c'est-a-dire exactement les deux points ou notre test documentait des
divergences de convention. Test `third_party_oracle` vert avec la nouvelle version.

## La largeur lockstep est maintenant mesuree au demarrage (2026-08-09)

Deuxieme round de recherche, hors bioinfo. Le chiffre qui declenche ce changement vient de mesures
publiees de pointer-chasing sur les CPU serveurs actuels ([Lemire, 2026](https://lemire.me/blog/2026/07/25/memory-level-parallelism-amd-is-the-king/)) :

| CPU | latence | debit 1 cœur | **acces concurrents** |
|---|---|---|---|
| Graviton 5 (2026) | 96 ns | 12,0 GiB/s | **19** |
| Apple M4 | | | **28** |
| Xeon Granite Rapids (2025) | 133 ns | 13,3 GiB/s | **30** |
| **EPYC Zen 5 Turin (2025)** | 142 ns | 24,5 GiB/s | **58** |

Progression AMD : Naples 15 -> Milan 22 -> Turin 58. Intel : Broadwell 10 -> Ice Lake 20 ->
Granite Rapids 30. **Un facteur trois entre CPU contemporains.**

Or la largeur lockstep doit valoir ce nombre : c'est le nombre de defauts de cache qu'un cœur peut
garder en vol. Nos deux constantes (32 sur aarch64, 16 ailleurs) avaient chacune ete mesurees sur
**une seule** machine, et sur Zen 5 la valeur x86 est **3,6x trop petite**. Aucune constante ne peut
suivre ça.

**Elle est donc mesuree au demarrage**, par l'experience meme dont ces chiffres proviennent : des
chaines de pointeurs dependantes, `k` a la fois, et on regarde ou ajouter des voies cesse de payer.

**La sonde chasse dans `cp_occ`, pas dans un tampon neuf**, et c'est la seule version qui marche.
Une premiere sonde sur 64 Mio alloues rendait une courbe non monotone et choisissait 64 : ce tampon
tient dans le cache systeme de cette machine, elle mesurait donc du cache. L'index fait des
gigaoctets, il est deja resident, il porte le meme comportement TLB, et c'est exactement la memoire
que le seeding va marteler.

Resultat sur cette machine, trois lancements sur trois :

```
  8 -> 14,6 ns    24 ->  4,3 ns    64 ->  2,7 ns
 12 ->  5,9 ns    32 ->  2,7 ns    largeur choisie : 32
 16 ->  4,6 ns    48 ->  3,0 ns
```

La courbe suit `latence / k` jusqu'a ~28 voies puis plafonne, ce qui est le comportement attendu, et
elle retombe sur **32**, exactement la valeur trouvee ce matin par balayage bout-en-bout. Sur une
machine Zen 5 elle choisirait plus large sans qu'on ait a le savoir.

Cout : **+10 ms** (0,24 s contre 0,23 s de demarrage), CPU inchange. Octet-identique sur les deux
jeux, `check.sh` et `oracle_diff.sh` vertes. `BWA4_LOCKSTEP_N` force toujours une valeur, et
`BWA4_LOCKSTEP_NO_PROBE` revient a la constante compilee pour une machine trop bruyante pour etre
mesuree.

## Un modele qui predit la scalabilite a 5 points pres (2026-08-09)

Revue de litterature hors bioinfo (architecture, bases de donnees en memoire, capacity planning) plus
trois mesures sur la machine. Resultat : la scalabilite de cet aligneur est le **produit de deux
grandeurs materielles**, toutes deux mesurables en deux minutes sur n'importe quelle machine.

```
efficacite(N) = ALU(N) x DEBIT_ALEATOIRE(N)
```

| threads | debit aleatoire | part memoire | plafond ALU | **predit** | **mesure** | ecart |
|---|---|---|---|---|---|---|
| 4 | 23,3 GB/s | 97 % | 99 % | **96 %** | 94 % | 2 pts |
| 8 | 43,8 GB/s | 91 % | 100 % | **91 %** | 86 % | 5 pts |
| 12 | 59,9 GB/s | 83 % | 97,5 % | **81 %** | 76 % | 5 pts |
| 16 | 69,7 GB/s | 73 % | 82 % | **60 %** | **59 %** | 1 pt |

Le chiffre qui manquait a toute la campagne : **le debit ALEATOIRE de la machine, pas le sequentiel**.
A 12 threads, 284,6 GB/s en sequentiel contre **59,9 en aleatoire**, soit 4,75x moins, et surtout il
**ne scale pas** : 97 / 91 / 83 / 73 % a 4 / 8 / 12 / 16 threads. Notre courbe est la sienne.

Deux autres mesures locales completent le tableau. Le **TLB de cette machine** porte 56 Mio
(~3200 entrees de 16 Kio) contre un `cp_occ` de **9,4 Go**, soit 168x trop peu, pour un surcout mesure
cache-chaud de **+10,7 ns par acces**. Et notre trafic reel, via `BWA4_TRAFFIC` : **858,6 M lignes de
128 octets** pour 500 k paires, 8,9 GB/s sur un seul thread.

**Ce que le modele interdit** : esperer qu'un changement de code depasse ce produit. Equilibrage,
taille de lots, allocateur, affinite, ordonnancement : tous mesures a 0 % ici, et c'est le resultat
correct, pas un echec des tentatives. **Ce qu'il autorise**, dans l'ordre : agrandir la page (seul
terme logiciel qui bouge le debit aleatoire, d'ou les 1,7x de THP sur Linux et l'interet des pages de
1 Gio), puis reduire le nombre d'acces, c'est-a-dire l'algorithme, ce qui est exactement d'ou vient
l'avance de minibwa.

**Corollaire sur la largeur lockstep** : Cimple (PACT'18) etablit que le plafond de MLP d'un cœur est
son nombre de MSHR, et Lemire mesure **28 voies soutenues sur un M4**. Notre largeur de 32 est donc
juste au-dessus du plafond materiel, ce qui explique le +1,27 % en passant de 16 a 32 et le genou
plat a `-t12`.

Le tout, avec la methode de mesure sur une machine neuve et les sources, est dans
[`docs/scaling-model.md`](docs/scaling-model.md).

## Pages de 1 GB : atteignables, mais pas mesurables sur un runner (2026-08-09)

BWA-MEM-SCALE mesure **+10,9 points** grace au HugeTLB en pages de 1 GB. `hugepage.rs` etablit deja
que les pages de **2 MiB valent ~1,7x sur l'etage seeding** et que **mimalloc les obtient toute
seule**, donc la seule part non payee de ce levier est le pas de 2 MiB a 1 GB, que mimalloc ne fait
pas. Avant d'ecrire un chemin `mmap(MAP_HUGETLB | MAP_HUGE_1GB)` sur une machine incapable de
l'executer, une sonde CI a demande si c'etait seulement atteignable (job `hugepage-probe`).

**Reponse : oui.** Sur `ubuntu-22.04` heberge (Azure westus, noyau 6.8) :

```
pdpe1gb pse                          <- 1 GB existe dans le silicium
hugepages-1048576kB, hugepages-2048kB <- les deux tailles sont exposees
2048kB: asked 4, got 4
1048576kB: asked 4, got 4             <- 4 GB de pages de 1 GB reservees a l'execution
mounted                               <- hugetlbfs monte
THP: [always]                         <- ce que mimalloc exploite deja
```

Rien dans `/proc/cmdline` ne reserve quoi que ce soit au boot : la reservation marche a chaud.

**Et pourtant le levier n'est pas mesurable ici**, pour une raison que la sonde met au jour : le
runner standard a **16 GB de RAM**, l'index GRCh38 en fait **15,9**, et loger `cp_occ` (6,2 GB) dans
des pages de 1 GB en demande 7 de plus. Le job bout-en-bout tourne donc sur chr21, dont le `cp_occ`
tient dans une seule page de 1 GB et ne fait subir aucune pression de TLB : on mesurerait zero, et ce
zero ne dirait rien.

**Etat du levier** : mecanisme atteignable et verifie, effet non mesurable sans un hote d'au moins
32 GB. Il reste donc une note de deploiement documentee, pas du code livre, tant qu'une machine
capable de le juger n'existe pas dans ce projet. Ecrire l'allocateur maintenant reviendrait a livrer
un chemin dont personne ne peut dire s'il gagne, ce que ce fichier interdit partout ailleurs.

**Erreur de methode a noter** : ce workflow porte `concurrency: group: bench-x86-<ref>`, donc
dispatcher la sonde a **annule le job bout-en-bout d'une heure** lance quelques minutes plus tot. Un
dispatch de plus sur la meme ref tue celui d'avant.

## Le PGO ne vaut plus 12,4 %, il vaut 3 % (2026-08-09)

Le dossier portait le PGO a **+12,4 % sur du reel** et +8,5 % sur wgsim, et s'en servait pour
expliquer une partie de l'ecart contre le fork. Re-mesure aujourd'hui sur `dev`, `cargo pgo build`,
un run de profilage, `cargo pgo optimize build`, binaire **octet-identique** sur les deux jeux :

| jeu | entrainement | CPU sans PGO | avec PGO | gain | victoires |
|---|---|---|---|---|---|
| reel (ERR356372) | wgsim | 28,76 | 27,91 | **-3,0 %** | 5/5 |
| reel (ERR356372) | le meme reel | 28,75 | 28,02 | **-2,5 %** | 5/5 |
| wgsim (500 k paires) | wgsim | 12,88 | 12,80 | -0,6 % | 5/5 |

Deux choses a en tirer.

**Le regime d'entrainement ne compte quasiment pas** : entrainer sur le jeu exact qu'on mesure donne
-2,5 %, entrainer sur wgsim donne -3,0 %, l'inverse de ce qu'on attendrait. Le profil n'est donc pas
le facteur limitant.

**Le lever a fondu parce que le code a ete optimise a la main depuis.** Le PGO gagne surtout par
l'inlining et la disposition des branches ; or la campagne a monomorphise les noyaux par const
generics, force `#[inline(always)]` sur `backward_ext`, supprime les branches de statistiques des
chemins chauds et sorti les lectures de `OnceLock` des boucles. Ce sont exactement les decisions que
le PGO prenait a notre place. Le profil `release` fait deja `lto = "fat"`, `codegen-units = 1` et
`panic = "abort"`.

**Consequence pratique** : le binaire livre gagne 3 %, pas 12,4 %, et le classement de cette campagne
reste juste puisque les trois concurrents etaient eux aussi sans PGO. La question ouverte du dossier
(« pourquoi nh13 mesure -0,4 % sur Graviton la ou nous mesurons +12,4 % ») **perd l'essentiel de son
objet** : l'ecart a expliquer n'est plus de 12,8 points mais de 3,4, ce qui est l'ordre de grandeur
d'une difference de microarchitecture ordinaire.

## Aucun aligneur de cette classe ne scale mieux, mesure (2026-08-09)

Objection legitime : « il y a des aligneurs qui scalent ». Verifie sur cette machine, meme jeu de
2 M paires, memes conditions, plutot que discute :

| | `-t1` | `-t4` | `-t16` | efficacite a `-t16` |
|---|---|---|---|---|
| **bwa-mem4** | 46,79 | 12,37 (95 %) | 4,58 | **64 %** |
| fork v0.9.0 (compat) | 50,43 | 13,08 (96 %) | 4,67 | **67 %** |
| minibwa 0.7 | 38,90 | 10,33 (94 %) | 3,94 | **62 %** |

Les trois tombent entre 62 % et 67 %, soit 5 points d'ecart sur une machine dont la derive thermique
vaut 15 %. **minibwa, le plus rapide des trois en absolu, scale moins bien que nous.** Ce qu'il gagne,
il le gagne en faisant moins de travail (38,6 CPU-s contre nos 46,8 a `-t1`, via un autre algorithme
SMEM et des heuristiques qui changent la sortie), pas en scalant mieux.

Rapportes au plafond materiel mesure (82 % a `-t16`, sonde ALU), les trois rendent **78 % de ce que
la machine peut donner**, et c'est le meme 78 % a `-t12`. La taxe memoire est donc une constante
d'environ 22 % a haut compte de threads, **independante de l'implementation**. Le papier bwa-mem2 le
dit depuis 2019 avec ses propres chiffres : 3,5x mono-thread mais **2,4x mono-socket**.

Conclusion : il n'y a pas de scalabilite a recuperer contre la concurrence, parce que la concurrence
ne l'a pas non plus. Il y a du **travail absolu** a recuperer, et c'est la que minibwa nous devance.
Le seul chemin restant vers une meilleure efficacite est celui que la revue de litterature designe :
des pages de 1 GB sur x86 (+10,9 points mesures par BWA-MEM-SCALE), ce qui est un levier de
deploiement, pas d'algorithme, et qui n'existe pas sur macOS.

## Le plafond reel de la machine, mesure (2026-08-09)

Demande : approcher 100 % d'efficacite a 16 cœurs, et par extension a 32. Le plafond a donc ete
mesure au lieu d'etre suppose, avec une sonde purement ALU (aucun acces memoire, donc tout ce qu'elle
perd est de l'horloge ou de l'heterogeneite) :

| threads | 1 | 4 | 8 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|
| Giter/s | 2,7 | 10,7 | 21,7 | 31,6 | 33,7 | 35,3 |
| efficacite | 100 % | 99 % | **100 %** | **97,5 %** | 89 % | **82 %** |

Deux faits en decoulent. **La frequence n'est pas en cause** : jusqu'a 12 threads la machine rend
97,5 % en calcul pur, il n'y a pas de perte d'horloge tous-cœurs a recuperer. Et **100 % a 16 est
physiquement inatteignable ici** : les 4 cœurs E ne valent que 1,4x a eux quatre, donc le plafond est
82 % quel que soit le code. La cible honnete a `-t16` sur cette machine est 82 %, pas 100 %.

**L'origine de l'ecart restant est la memoire, par elimination.** Le meme aligneur sur un index de
6 Mo (tenant en cache) au lieu de 15,9 GB voit son inflation CPU a `-t12` tomber de **+21 % a +10 %**.
La moitie de la contention est donc directement imputable au trafic vers le BWT.

**Hypotheses testees et rejetees**, dans l'ordre :

| hypothese | verdict | mesure |
|---|---|---|
| desequilibre de charge | non | occupancy 97,7 %, queue 0,3 % |
| cout de demarrage | non | 0,23 s, 0,5 % du mur sur 2 M paires |
| lecture FASTQ serialisee par `-K` | non | `-K 400000000` et le defaut donnent le meme mur |
| frequence tous-cœurs | non | sonde ALU a 97,5 % a `-t12` |
| retrogradation vers les cœurs E | non | QoS `USER_INTERACTIVE` sur les travailleurs rayon : 1,32 contre 1,32 a `-t12`, 1,28 contre 1,28 a `-t16`, CPU identique. Retire |
| reglage du parallelisme memoire | non | N16 et N32 a egalite a `-t12` et `-t16` |
| pre-reservation des allocations | non, **negatif** | +0,35 % de CPU dans deux variantes |

**Ce qui reste est le trafic lui-meme**, et c'est la seule voie vers une meilleure efficacite a
n'importe quel nombre de cœurs, sur arm comme sur x86 : deux acces aleatoires de 64 octets par
extension de base sur un BWT de 9,4 GB. Sur une machine homogene a 32 cœurs le plafond ALU
remonterait vers 100 %, mais notre courbe se degraderait davantage, puisque 32 cœurs se disputeraient
la meme DRAM. Autrement dit le probleme ne se dilue pas avec plus de cœurs, il s'aggrave, et la
reponse est la meme dans les deux cas : toucher moins de memoire. C'est un changement de structure
d'index, pas un reglage, et il devra passer la porte d'identite octet comme le reste.

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

> **Corrige le 2026-08-11 : les chiffres de cette sous-section sont invalides.** Le jeu de lectures
> utilise etait simule depuis **chr20** alors que l'index est **chr21**, d'ou 65 a 69 % de non-mappe :
> on y mesurait surtout la vitesse de rejet. Sur un jeu correct, le classement s'inverse. Voir
> « Le piege de jeu de donnees, et le classement refait » plus bas.

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

## Les deux fichiers d'une paire sont lus en parallele (2026-08-11)

Suite directe de la conclusion thread-broker : l'inflation ne pese que 0,6 % du CPU, mais elle apparait
1:1 dans le mur parce qu'elle est sur le chemin **serie** du lecteur. Le correctif n'est ni un pool de
decodeurs ni un broker, c'est du recouvrement. Mesure d'abord, avec `BWA4_STAGE_TIME=1` :

Le probleme est plus grand que l'inflation seule. `-K` vaut par defaut 10 M bases **par thread**, donc
a `-t16` le lot fait 160 M bases et un run de 500 k paires (151 M bases) tient **dans un seul lot**.
Un seul lot veut dire zero recouvrement possible avec le calcul : la lecture entiere precede
l'alignement entier. C'est un terme d'Amdahl qui **grandit avec `-t`**, puisque `-K` grandit avec `-t`.

Or `PairedFastqReader` lisait ses deux fichiers, independants, sur un seul thread. Chaque fichier a
maintenant son `RecordStream` : un thread, un parseur, des paquets de 8192 records sur un canal borne a
2. Le consommateur entrelace et applique la meme regle `-K` sur la meme sequence, donc **la frontiere de
lot ne bouge pas** et rien n'est visible en sortie.

chr21, 500 k paires, `-t16`, `wait_read` (le temps ou le thread principal est bloque sur le lecteur),
3 runs par bras :

| entree | avant | apres | |
|---|---|---|---|
| gzip | 0,415 / 0,436 / 0,428 s | **0,217 / 0,203 / 0,209 s** | **-51 %** |
| plain | 0,160 / 0,151 / 0,150 s | **0,070 / 0,065 / 0,069 s** | **-55 %** |

Les deux distributions ne se recouvrent pas. En bout de chaine, A/B entrelace 7 rondes, gzip, `-t16` :
mediane 18,63 s contre 18,57 s, **B gagne 5/7** (5/6 en ecartant la premiere ronde, froide). C'est
0,22 s de chemin serie retire pour ~0,1 s de mur mesure, **au plancher de bruit de ce banc** : je le
consigne comme tel, la mesure fiable est `wait_read`. Le gain est maximal quand `-K` est grand et les
lots peu nombreux, c'est-a-dire exactement au `-K` par defaut a haut `-t`.

Portee : paired-end deux fichiers. Le mono-fichier (SE, `-p`) ne gagne rien ainsi, son lecteur a deja
un thread a lui ; il faudrait y separer inflation et parsing, ce qui n'est pas fait.

Gates : `check.sh` vert, `oracle_diff.sh` PASS (5000/5000 `all_fields_match`), md5 inchange sur deux
points de la grille (`-t8 -K 10M` gzip et `-t3 -K 7M` plain, 500 k paires). Trois tests ajoutes :
ordre et frontieres de lot a travers une frontiere de paquet, fichiers de longueurs differentes
toujours refuses, fichier manquant toujours signale (l'ouverture est passee sur le thread lecteur,
donc l'erreur remonte au premier `next_batch` et plus a `from_paths`).

## Le raisonnement salmon (piscem-rs), applique a nos chiffres (2026-08-11)

Salmon 1.11 introduit un budget de slots unique partage entre decodage gzip et mapping, pilote par
`thread-broker`, avec un seuil d'engagement **mesure par mode** :

| mode salmon | engage le decodeur parallele a partir de |
|---|---|
| selective alignment | `-p` >= 50 |
| sketch | `-p` >= 10 |

et un encadrement empirique sur 26 M paires : a `-p 64` le decodeur **serie** gagne de 12 %, a `-p 128`
le parallele gagne de 7 %. Leur propre conclusion : « serial decoding is simply the right choice for SA
at most real budgets ».

**Notre seuil, calcule avec leur loi.** `d* = N x busy_p/(busy_p + busy_c)` ; `d* >= 1` demande
`N >= 1 + busy_c/busy_p`. Avec `busy_p` = 0,096 s CPU (2 x 72 Mo a 1500 Mo/s, backend reel) et
`busy_c` = 16,9 s CPU a `-t16` : **N >= 177 threads**. Nous sommes 3,5x plus loin du seuil que leur mode
selective alignment, ce qui est coherent : une paire BWA-MEM coute plus de calcul par octet d'entree
qu'un fragment de selective alignment. A 16 cœurs la question ne se pose pas.

**Le budget partage teste quand meme.** Leur reformulation la plus interessante (`-p` nomme un budget
de slots, pas un compte de threads de mapping) vaut a tout `N`, et chez nous `-t16` signifie 16 workers
**plus** un lecteur **plus** un writer sur une machine a 16 cœurs. Teste, 5 rondes entrelacees, gzip :

| | mediane |
|---|---|
| `-t15` (un cœur laisse au producteur) | 18,68 s |
| `-t16` | **18,47 s** |

`-t16` gagne 4/5. Reserver un slot au producteur **coute** 1,1 %, parce que le producteur a un taux
d'occupation de 1,1 % (0,209 s de lecture sur 18,4 s de run) : le controleur de salmon lui accorderait
0,18 slot, et un slot entier est trop.

**Ce qui est transferable et a ete pris.** Leur decision est **par fichier d'entree**. Nous l'avons
appliquee dans l'autre sens, et c'est le bon sens pour nous : pas un decodeur parallele par fichier,
mais un decodeur **serie par fichier**, deux fichiers, deux threads (section precedente). Meme
observation de depart, remede oppose, parce que le rapport calcul/octet n'est pas le meme.

**Ce qui est un argument de plus pour s'abstenir.** Leurs notes signalent une course dans le pool
partage de rapidgzip (reveil perdu dans la liberation de permis) capable de **bloquer un run**, corrigee
en rapidgzip-core 0.3.1. Prendre ce risque pour un composant a 0,6 % du CPU serait un mauvais echange.

**Reserve sur leurs propres chiffres.** Le seuil annonce (50) et l'encadrement empirique (serie gagnant
encore a 64) ne sont pas coherents entre eux dans les notes de version ; le seuil vient sans doute de la
politique et l'encadrement d'un jeu precis. Cela ne change pas notre conclusion, qui est a 177.

## Le piege de jeu de donnees, et le classement refait (2026-08-11)

Le classement publie contre le fork (« nous menons 1,089x / 1,063x / 1,053x ») **est faux**, et la
cause n'est pas une erreur de mesure : c'est le jeu de donnees.

`work/r1_500k.fq` porte des noms de la forme `20:2000000-4000000_...` : ces lectures sont simulees
depuis **chr20**. L'index de banc est **chr21**. Sur cette combinaison, 65 a 69 % des enregistrements
sortent non mappes, et ce qui est chronometre est en grande partie la vitesse a **rejeter** des
lectures qui n'appartiennent pas a la reference. Ce n'est pas un banc d'alignement.

Le bon jeu existait a cote : `work/chr21arm/r{1,2}.fq`, 1 M paires simulees depuis chr21, dont
**0 % de non-mappe**. Verifie avant de chronometrer : bwa-mem2 2.3, le fork `--compat=bwa-mem2` et
bwa-mem4 sortent le **meme corps SAM** (`c833bb42...`), donc les trois comparent bien le meme fichier.

### La discipline, d'abord

Un premier passage a `-t16` sans refroidissement a donne, pour le **meme binaire**, des murs de 15,98
a 45,95 s selon la ronde, et bwa-mem2 plus lent en entree plain (87,02 s) qu'en gzip (57,67 s), ce qui
n'a aucun sens physique. Quatre binaires lourds enchaines sur 16 cœurs, plus 690 Mo de FASTQ plain
relus a chaque ronde : derive thermique et page cache. **Inexploitable, et jete.**

La regle du projet le disait deja (`-t4` pour du pourcentage, la derive atteint 10 % a `-t8`). Avec
`-t4`, 500 k paires, gzip seul et **10 s de refroidissement entre chaque run**, la dispersion tombe
sous 1 % et les six rondes sont interchangeables.

### `-t4`, 500 k paires chr21, gzip, 6 rondes, medianes

| | mur | CPU | cœurs occupes |
|---|---|---|---|
| bwa-mem2 2.3 | 58,99 s | 230,9 s | 3,91 |
| fork bwa-mem3 v0.9.0 | **18,05 s** | 71,7 s | 3,97 |
| bwa-mem4 | 18,24 s | 71,8 s | 3,93 |
| minibwa 0.7-r424 | **7,56 s** | **29,1 s** | 3,85 |

### `-t16`, meme jeu, meme discipline, 6 rondes

| | mur | CPU | cœurs occupes |
|---|---|---|---|
| fork v0.9.0 | **5,95 s** | 85,4 s | 14,36 |
| bwa-mem4 | 6,36 s | 87,0 s | 13,67 |

### Ce que ca dit

**Contre le fork : egalite en calcul, retard en remplissage.** Les ratios CPU par ronde a `-t4` sont
0,996 / 1,001 / 1,001 / 1,000 / 1,000 / 0,998, c'est-a-dire le meme travail au millieme pres. Le fork
prend **1,0 %** sur le mur a `-t4` et **4,3 %** a `-t16`, en gagnant 6/6 dans les deux cas. Traduit en
efficacite de `-t4` vers `-t16` : **fork 76 %, nous 72 %**. Notre retard est entierement dans
l'occupation des cœurs, et pas du tout dans le noyau.

**Contre bwa-mem2 : 3,22x en CPU et 3,23x en mur** a `-t4`. C'est l'ordre de grandeur deja publie.

**Contre minibwa : 2,47x en sa faveur en CPU.** Le ROADMAP portait 1,21x (38,6 contre 46,8 CPU-s).
L'ecart s'est creuse et deux choses ont change en meme temps : minibwa est passe de r411 a r424, et
le banc precedent etait le jeu chr20-sur-chr21. Sa sortie n'est pas identique a la notre (autre algo
SMEM, plus d'alignement sans gap, prefiltre q-mer sur le mate rescue), donc c'est un autre produit,
mais 2,47x est trop gros pour etre range sous « il fait moins de travail » sans le chiffrer.

**La regle qui en sort**, et qui manquait a ce depot : avant de chronometrer, verifier le **taux de
non-mappe**. Un banc dont les lectures ne viennent pas de la reference ne mesure pas l'alignement, et
il peut inverser un classement sans qu'aucune mesure individuelle soit fausse.

## Le chemin critique du lecteur, et ce que rapidgzip y change (2026-08-11)

Suite de la correction du classement : contre le fork, notre CPU est a egalite et tout l'ecart est
dans l'occupation des cœurs. En chiffres, avec `occupation = (t_serie + T_par) / (t_serie + T_par/N)`
a N=16 : lui 14,36 cœurs occupes donc **0,77 %** de serie, nous 13,67 donc **1,15 %**. Il faut
recuperer **0,4 point de serie**, pas optimiser un noyau.

### Ou est le serie, mesure

`BWA4_STAGE_TIME=1`, 1 M paires GIAB reelles contre GRCh38 :

| | `wait_read` | ecart au fork (mur) |
|---|---|---|
| `-t4` | **0,000 s** | 2,6 % |
| `-t16` | **0,682 s, 4,2 % du run** | 4,3 % |

A `-t4` le lecteur est integralement recouvert et le fork gagne quand meme : cet ecart-la est
ailleurs, et il prend aussi 1,5 % de CPU. A `-t16` le lecteur explique presque tout le surplus.

La cause n'est pas le debit mais la **granularite** : `-K` vaut `10M x -t`, donc a `-t16` le lot fait
160 M bases et 1 M paires (302 M bases) tiennent en **deux lots**. Le premier est expose
integralement, il n'a pas de predecesseur avec quoi se recouvrir. Le chargement d'index, la seule
autre chose derriere quoi il pourrait se cacher, dure 0,28 s en cache chaud et est deja recouvert.

| `-K` | lots | `wait_read` gzip | plain |
|---|---|---|---|
| 20 M | 15 | 0,097 s | 0,000 s |
| 40 M | 8 | 0,184 s | |
| 160 M (defaut `-t16`) | 2 | 0,682 s | 0,219 s |

**Correction d'une erreur de raisonnement au passage.** J'ai d'abord ecrit que les boucles serie de
`process` s'allongent proportionnellement a `-t` parce que `-K` grandit avec `-t`. Faux : elles sont
O(paires), donc si `-K` double il y a deux fois moins de lots et le produit est constant. Seul
`mem_pestat` bouge, en `n log(n/B)`, un facteur logarithmique. Ce qui grandit avec N est la **part de
mur** que le serie represente, ce qui est Amdahl ordinaire. Le raisonnement `-K` ne vaut que pour le
lecteur, dont le premier lot grossit bien avec `-K`. Le commentaire de `Stage::Encode` dans
`stage_time.rs` portait deja cette erreur, et `Encode` n'est plus serie depuis longtemps.

### Deux correctifs, mesures

**Un : l'inflation sur son propre thread.** `parse_fastx_file` construit `MultiGzDecoder` *dans* le
parseur, donc un thread alterne inflation et parsing et un fichier coute `inflate + parse` au lieu de
`max(inflate, parse)`. Un thread d'inflation par fichier, blocs de 4 Mio, canal borne a 3.

**Deux : rapidgzip-core 0.3.1**, qui decode UN flux sur plusieurs threads. Debit mesure isolement sur
`r1_1m.fq.gz`, octets de sortie par seconde :

| decodeur | debit |
|---|---|
| zlib-rs (l'actuel) | 823-876 Mo/s |
| rapidgzip 1 thread | 917-937 Mo/s |
| rapidgzip 4 | ~1240 Mo/s |
| rapidgzip 8 | ~2240 Mo/s |
| rapidgzip 12 | **2863-2999 Mo/s** |

`wait_read` a `-t16`, `-K` par defaut :

| version | `wait_read` |
|---|---|
| lecteur d'origine | 0,682 s |
| + inflation sur son thread | 0,531 s |
| + rapidgzip | **0,458 s** |
| plancher, entree plain | 0,219 s |

**-33 % au total, dont -11 % pour rapidgzip seul.** Beaucoup moins que le 3,4x isole, et la raison
est mesuree : un balayage `BWA4_GZIP_THREADS` de 1 a 16 est **plat** (0,377 / 0,374 / 0,383 / 0,210
sur fichier entier). L'inflation n'est plus le goulot une fois sur son thread ; ce qui reste
au-dessus du plain est la copie a travers son `Read` et la latence de demarrage, pas le debit.

Garde quand meme : 0,2 % de CPU, 23 crates transitifs tous en Rust pur dont les runtime (`zlib-rs`,
`crossbeam-deque`) sont deja dans le graphe, et ca supprime une classe de goulot plutot qu'un reglage
valable pour ce fichier-ci. `BWA4_GZIP_THREADS` permet de rebalayer sur une autre machine, et
`--no-default-features` retombe sur l'inflateur mono-thread, lui-meme toujours sur son thread.

**Ce que thread-broker ne peut pas voir.** Sa loi alloue d'apres le temps occupe cumule, donc elle
optimise le regime permanent et repond 0,09 thread. C'est juste pour sa question. Notre probleme est
une **latence de remplissage de pipeline** : un thread inutile 99 % du run et decisif pendant 0,4 s.
Les deux outils de Patro ne sont pas interchangeables, et c'est rapidgzip qui correspond ici.

### Un bug trouve en chemin, et repare

`open_reader` lisait deux octets de magie puis **rouvrait le fichier par son chemin**. Sur un tube,
une substitution de processus (`<(zcat r1.gz)`) ou un FIFO, ces deux octets ne reviennent jamais : le
parseur voyait un flux deja entame et rendait **zero enregistrement, sans erreur**. `bwa-mem4 mem ref
<(zcat r1.fq.gz)` ecrivait un en-tete SAM et aucun alignement. bwa-mem2 lit cette entree, donc
c'etait aussi un ecart de parite. Pre-existant, verifie sur le binaire d'avant ce lot.

Repare : une entree non-seekable garde ses deux octets (chainage `Cursor` + fichier) et n'est jamais
rouverte. Verifie, meme corps SAM `61af65ce...` pour fichier regulier, FIFO gzip, FIFO plain **et
bwa-mem2 sur FIFO**. Un test cree un vrai FIFO avec `mkfifo` et lit 2500 enregistrements a travers.

### Ce qui reste sur ce chemin

Le plancher du lecteur est maintenant le **parseur** : ~926 Mo/s par fichier, trois allocations par
enregistrement (`name`, `seq`, `qual`). C'est le prochain levier, pas le decodeur. Et l'ecart a `-t4`
(2,6 % de mur, 1,5 % de CPU, lecteur totalement recouvert) est un probleme distinct, non localise.

## Le lecteur n'est plus le goulot, et la sonde mentait (2026-08-11)

### `Record` : trois allocations par read, devenues une

`Record` etait `String` + `Vec<u8>` + `Option<Vec<u8>>` + `Option<String>` : trois allocations et
trois copies par read, 96 octets de structure. Il est maintenant **un seul tampon**
`name | seq | qual | comment` plus trois longueurs et un drapeau : **une allocation, 32 octets**, avec
accesseurs.

Le decoupage qui a designe ce levier, 1 M reads, 352 Mo, un fichier :

| | temps |
|---|---|
| needletail, zero copie | 0,048-0,056 s |
| seq_io, zero copie | 0,049-0,050 s |
| **+ notre `Record` possede** | **0,113-0,114 s** |
| + une arene | 0,067-0,068 s |

**Changer de bibliotheque ne rendait rien** : needletail et seq_io sont a egalite, et paraseq annonce
lui-meme « matches the performance of the zero-copy parsers », son gain portant sur les parseurs
*une-copie*, ce que notre `Record` etait. Le cout etait notre representation.

Lecteur isole, 1 M paires, via `PairedFastqReader::next_batch` :

| | avant | apres |
|---|---|---|
| plain, 1 lot | 0,360 s | **0,192 s** (-47 %) |
| plain, 2 lots | 0,302 s | 0,188 s |
| plain, 8 lots | 0,234 s | 0,139 s |
| gzip, 1 lot | 0,355 s | 0,270 s |

Plus que les 0,046 s que la sonde predisait : la structure divisee par trois allege aussi les **deux
deplacements** par enregistrement (canal de chunks, puis vecteur de lot) et la pression allocateur.

Bout en bout a `-t4`, 8 rondes appariees : **CPU median -0,75 s sur 194, 6/8 en faveur**. Petit, parce
qu'a `-t4` le lecteur est integralement recouvert (`wait_read` = 0,000) : ce qui reste visible est le
calcul economise, pas le mur. Une ronde a donne -25 % de CPU, ce qu'aucun changement de representation
ne peut produire ; **ecartee**, cause inconnue.

Parite verifiee sur les trois chemins que la nouvelle representation encode differemment : GIAB gzip
`968a2331...`, `-C` avec commentaires `a16e64eb...`, FASTA sans qualites `578281f3...`, les trois
identiques a bwa-mem2.

### Deux zeros, consignes

**Reserver le `Vec` de lot.** L'isole montrait 1 lot a 0,360 s contre 8 lots a 0,234 s, ce qui
ressemblait a de la croissance par doublement. Apres reservation : 0,368 s. Sur le binaire reel,
`wait_read` median 0,332 contre 0,342. **Retire.** La difference 1-lot/8-lots vient des defauts de
page sur de la memoire fraiche, que l'allocateur recycle quand les lots sont petits.

**Deplacer au lieu de cloner dans `PrepPair`.** Six clones par paire supprimes (~376 Mo par lot de
1 M), dans un etage parallele : mur median +0,03 s, CPU median +0,1 s sur 190. **Zero.** Garde
malgre tout, parce que c'est du code en moins qui ne peut pas etre plus lent, mais compte comme nul.

### rapidgzip redevient utile apres coup

Avant l'arene, le lecteur isole donnait gzip 0,355 s contre plain 0,360 : le gzip etait **gratuit**,
masque par le parsing, et le balayage `BWA4_GZIP_THREADS` etait plat. Apres l'arene, gzip 0,270 contre
plain 0,192, et le balayage repond :

| threads | lecteur isole, gzip |
|---|---|
| 1 | 0,469 s |
| 4 | 0,357 s |
| **8** | **0,280 s** |
| 16 | 0,294 s |
| plain (plancher) | 0,167 s |

**-40 %**, et notre defaut (`-t`/2, soit 8 a `-t16`) est exactement l'optimum. Lecon generale :
optimiser le consommateur **redonne du travail au producteur**, et un levier juge nul peut redevenir
utile une fois corrige ce qui le masquait. Un zero est date, pas definitif.

### La sonde `stage_time` cachait 95 % du run

`NS` etait un `thread_local`, sur la croyance que seul le thread principal enregistre. Faux :
`run_pipeline` execute chaque `process` sur un thread `scope.spawn`, donc **tous les etages entre
`encode` et `sam_emit` etaient credites a un thread qui mourait ensuite**. La table ne montrait que
`wait_read`, `wait_write` et un enorme « unaccounted » etiquete « index load, header, teardown ».
Passe en `AtomicU64` globaux.

Profil enfin visible, `-t16`, GIAB, ms par lot : rescue 7287, align 5058, encode 1808, sam_emit 467,
wait_read 138, dedup_prep 105, pestat 21, deinterleave 4. Reserve : `encode`, `dedup_prep` et
`sam_emit` appellent `barrier::worker` **par read**, donc leurs chiffres sont gonfles par
l'instrument ; `rescue` est instrumente par chunk et `align` pas du tout.

Deux consequences a noter dans la table : les etages **recouvrent** `wait_read` (le pipeline lance
`process` du lot N puis attend le lot N+1), et deux `process` sont en vol a la fois, donc la colonne
`%_run` peut depasser 100 % et le reste est signe.

### Ou est vraiment le temps, hors sonde

`sample` sur un run `-t4`, feuilles seulement, normalise au busy :

| symbole | % du busy |
|---|---|
| `fwd_local_sw_neon_u8` (SW du mate rescue) | **18,0 %** |
| `batched_extend_neon_u8` (extension) | **15,6 %** |
| `mem_sort_dedup_patch` | **11,7 %** |
| `LsSlot::step` | 5,4 % |
| `align_reads_batched` | 3,8 % |
| `get_sa_batch` | 3,6 % |
| `build_chains_from_resolved` | 3,1 % |
| `mem_chain_flt` | 2,5 % |
| `batch_mate_rescue` | 2,2 % |
| `gen_cigar2` | 2,1 % |

**Le mate rescue et ce qu'il declenche font ~29 % du busy** : son noyau SW, sa boucle, et l'essentiel
de `mem_sort_dedup_patch`, qui existe surtout parce que le rescue reinsere une region puis retrie tout
le vecteur. Le seeding, suspect principal de toute la campagne precedente, ne fait que **12,8 %** en
cumule. C'est la cible du prochain lot, et elle est enfin chiffree plutot que supposee.

## Le mate rescue, six angles, et le seul qui paie (2026-08-11)

Le profil hors sonde designait le mate rescue : **~29 % du busy**, son noyau SW a 18,0 %,
`mem_sort_dedup_patch` a 11,7 %. Six leviers testes, un seul rend.

### Ce que la sonde `BWA4_MATESW_TIME` dit du noyau

3,68 M jobs, 760 Gcellules, 61,8 s CPU sur ~194, soit **32 % du run**. Requete moyenne 148 pb pour
une fenetre cible de 1396 pb : 206 537 cellules par job.

| angle | resultat |
|---|---|
| debit du noyau | **13,4 Gcell/s sur les cellules executees**, plafond machine ~16, donc **84 %** |
| taxe de divergence de lanes | 1,09x |
| tri par longueur (lectures a 151 pb) | **-0,0 %** |
| jobs dupliques dans un appel | **0,0 %** |
| partage des fenetres qui se recouvrent (#50B) | **0,87x, donc plus cher** : 87,9 % des fenetres sont isolees, A moyen 1,08 |
| pre-filtre sur avant le DP | plafonne a **7 %** : 78,1 % des DP sont acceptes |

Cinq zeros, chacun avec son chiffre. Le noyau n'a rien a rendre.

### Le dedup incremental : prouve, octet-identique, et nul

Deux proprietes rendent le scan arriere theoriquement O(n log n) au lieu de O(n²). Le tableau est
trie par `re` **croissant** et le scan va vers l'arriere, donc la condition d'arret est **monotone** :
atteindre une position coute **un test**, pas un parcours. Et le rescue insere dans un tableau deja
dedupe, ou aucune paire ne passait le test de redondance, donc seules les paires impliquant une
nouvelle region peuvent tuer. Marqueur gratuit : `n_comp` vaut 0 a la construction et 1 apres chaque
passe.

**Un trou trouve par la mesure, pas par le raisonnement.** La premiere version changeait le md5
(`c03b89c5...` contre `968a2331...`). Cause : quand la fusion collineaire est active, `mem_patch_reg`
reecrit les coordonnees de `p` **en cours de scan** et les paires deja depassees ne sont jamais
retestees, donc une passe **avec** fusion peut laisser deux survivants atteignables redondants. Le
premier dedup du rescue suit `DedupPrep`, qui fusionne. Corrige en portant la precondition en
parametre explicite ; md5 redevenu identique.

Puis **A/B : zero.** Delta CPU moyen -0,01 s sur 194, 5 rondes, `-t4`. ~9000 operations par appel
ramenees a ~800 ne se voient pas. **Retire**, la preuve conservee ici.

### Le tri par longueur du batch de rescue : +3,5 %, et pourquoi il etait nul le matin

Le meme levier, mesure **-0,0 %** le matin, vaut **-3,5 %** l'apres-midi. Rien n'a change dans le
code : le jeu de donnees est passe par **fastp**.

| entree, meme nombre de lectures | taxe de lanes | gain d'un tri |
|---|---|---|
| non trimmee, toutes a 151 pb | 1,09x | **0,0 %** |
| trimmee par fastp, longueurs variables | **1,15x** | **4,5 %** |

Des lectures toutes de meme longueur n'ont rien a trier. **Les donnees reelles sont trimmees.** Un
banc sur du non-trimme avait donc rendu ce levier invisible, exactement comme le banc chr20-sur-chr21
avait inverse le classement contre le fork.

Implemente sur la passe avant du noyau (`batched_ksw_align2`), tri par `(longueur cible, longueur de
requete padee)`, dispersion des resultats vers l'ordre d'appel. Result-preserving par l'argument deja
ecrit dans l'en-tete du module : un job ne depend que de ses propres entrees, l'ordre ne decide que
du remplissage des lanes. `BWA4_MATESW_SORT=0` restaure l'ordre d'appel.

A/B, `-t4`, 2 M paires GIAB trimmees, genome entier, 5 rondes :

| ronde | mur | CPU |
|---|---|---|
| r1 | 96,15 -> 93,12 (-3,2 %) | 379,8 -> 370,0 (-2,6 %) |
| r2 | 96,84 -> 92,60 (-4,4 %) | 386,7 -> 369,8 (-4,4 %) |
| r3 | 96,57 -> 92,89 (-3,8 %) | 383,4 -> 368,8 (-3,8 %) |
| r4 | 95,98 -> 92,63 (-3,5 %) | 380,0 -> 368,5 (-3,0 %) |
| r5 | 96,12 -> 93,29 (-2,9 %) | 384,1 -> 369,8 (-3,7 %) |
| **mediane** | **-3,55 %** | **-3,50 %** |

**5/5.** Et le gain mesure est **plus du double du contrefactuel** : 4,5 % d'un noyau a 34 % du CPU
predisait 1,5 %. Le tri achete donc autre chose que l'occupation des lanes, vraisemblablement de la
localite memoire. Le contrefactuel `EXEC_SORTED` est un plancher, pas une estimation.

Parite : `a8d54127...` identique avec et sans tri sur 2 M paires trimmees, et `6b3bfc2c...` contre
bwa-mem2 sur une tranche de 200 k.

### Classement sur echantillon complet apres fastp

10 813 312 paires trimmees (fastp `--detect_adapter_for_pe`, 3,19 Gbases, Q20 98,7 %/96,4 %),
GRCh38 entier, `-t16`, **une seule ronde**, donc a confirmer :

| | mur | CPU | cœurs |
|---|---|---|---|
| bwa-mem2 2.3 | 601,65 s | 7292,0 s | 12,12 |
| fork v0.9.0 | 231,39 s | 2780,0 s | 12,01 |
| bwa-mem4 (avant le tri) | 242,02 s | 3027,8 s | 12,51 |
| minibwa 0.7-r424 | 159,88 s | 1371,6 s | 8,58 |
| minimap2 2.31 `-ax sr` | 176,39 s | 2272,8 s | 12,88 |

**2,49x contre bwa-mem2 en mur.** Deux corrections a des affirmations plus tot dans la journee :
minimap2 nous **repasse devant** sur ce banc (176 s contre 242), alors qu'on le battait sur 1 M paires
non trimmees a `-t4` ; et l'ecart de CPU au fork passe de 1,5 % a **8,2 %**, sans que ce soit de
l'occupation puisqu'on tient 12,51 cœurs contre ses 12,01. Le tri en reprend 3,5.

### Une anomalie non expliquee

Le debit du noyau de rescue tombe de **12,27 a 10,16 Gcell/s** entre non trimme et trimme, soit
-17 %, alors que la divergence de lanes n'en explique que 5,5 points. **Onze points manquent**, et
c'est la piste ouverte la plus prometteuse.

## Les quatre backends de tableau des suffixes, mesures (2026-08-11)

`bwa-mem4 index` sur le genome humain entier (GRCh38, 2L = 6,2 G symboles), cache de pages chaud,
mesures appariees. **L'index produit est octet-identique dans les quatre cas**, et identique a
l'index de reference du depot : un tableau des suffixes est unique, donc ce choix ne peut pas toucher
la sortie de `mem`, seulement le temps de construction.

| backend | mur | CPU | pic RSS | dependance systeme |
|---|---|---|---|---|
| `libsais-rs` (Rust pur) | 226,9 s | 453,8 s | 76,9 Go | aucune |
| **C libsais serial (defaut actuel)** | 150,6 s | 301,2 s | 92,6 Go | aucune |
| **C libsais + OpenMP, 8 threads** | **87,9 s** | **175,8 s** | 95,0 Go | `libomp` |
| CaPS-SA (rayon) | voir plus bas | | | aucune |

### Paralleliser divise le CPU total, et ce n'est pas une erreur de mesure

Le premier chiffre OpenMP paraissait faux : le mur ET le CPU divises par 1,70. Paralleliser ne reduit
pas le travail. Deux verifications :

1. **Cache chaud, deux rondes appariees** : C serial 153,3/150,6 s de mur pour 306,6/301,2 s de CPU ;
   OpenMP 8 threads 92,5/87,9 s pour 185,0/175,8 s. L'effet tient.
2. **Le meme binaire OpenMP force a 1 thread** : 153,8 s de mur, 307,6 s de CPU, c'est-a-dire
   **exactement le bras serial**. Donc pas de mauvais routage FFI : le gain vient bien du threading.

L'explication est celle de `scaling-model.md`. La construction SA-IS sur 6,2 G symboles est **liee a
la memoire aleatoire** ; partitionner divise le working set de chaque thread, ce qui ameliore assez le
cache et le TLB pour reduire le **nombre de cycles**, pas seulement le temps mur.

**Et le signe s'inverse selon la taille.** Sur chr21 (46 Mb), OpenMP fait *monter* le CPU :

| threads, chr21 | mur | CPU |
|---|---|---|
| C serial | 1,85 s | 3,24 s |
| 1 | 2,15 s | 3,21 s |
| 4 | 1,30 s | 3,98 s |
| **8** | **1,14 s** | 4,48 s |
| 16 | 1,11 s | 4,98 s |

Un banc sur chr21 aurait conclu « la parallelisation coute 38 % de CPU pour 1,6x de mur ». Sur le
genome elle en rend 42 %. Troisieme fois de la journee qu'un banc trop petit inverse une conclusion,
apres le jeu chr20-sur-chr21 et le tri du batch de rescue sur lectures non trimmees.

### CaPS-SA : inutilisable ici, pour une raison algorithmique

| variante | chr21, mur | chr21, CPU |
|---|---|---|
| memoire externe (`build_ext_mem`, le cablage en place) | 50,5 s | 553 s |
| en memoire (`build_in_memory`) | 242 s | 455 s |
| libsais C serial | **1,85 s** | **3,24 s** |

Index octet-identique dans les deux cas, donc correct, mais **27x a 130x plus lent**. J'ai d'abord cru
a un miscablage : on appelait la construction a **memoire externe**, qui deverse ses phases dans
`temp_dir()`, alors que tous les autres backends materialisent le `Vec` entier de toute facon.
Corrige en `build_in_memory`, c'est **cinq fois pire en mur**.

La cause est plus profonde. CaPS-SA est un SACA **par comparaison** et son `max_context` vaut
`usize::MAX`, ce que le crate documente comme *« required for full lexicographic correctness when the
caller's text doesn't guarantee comparisons terminate via sentinels within a known window »*. Notre
`bref` est un texte de **4 symboles sans sentinelle interne** : les LCP y sont enormes et chaque
comparaison degenere. Borner `max_context` produirait un tableau **faux**, pas seulement different.
libsais est un SA-IS lineaire, la longueur des repetitions ne le concerne pas.

**Consequence pour la question « rayon plutot qu'OpenMP ».** `openmp-sys` recommande rayon, et la
recommandation vise l'usage de pragmas OpenMP depuis du Rust, qui est impossible ; notre cas est
l'autre, activer OpenMP *dans une bibliotheque C vendoree*, ce pour quoi le crate existe.

J'ai d'abord conclu qu'**aucun SACA rayon n'etait utilisable sur ce texte**. **Faux, et corrige
ci-dessous** : je n'avais regarde que `caps-sa`, et j'avais mesure `libsais-rs` dans sa version
publiee, dont le chemin parallele ne scale pas.

### Ce qui est livre

Un passthrough `libsais-c-omp` dans `bwa-cli`, OFF par defaut. C'est la seule feature du manifeste
qui exige un **paquet systeme** (`brew install libomp`, ou libgomp) : tout le reste se construit
depuis la tarball crates.io, et un artefact de release qui gagnerait silencieusement une dependance
dylib serait pire qu'un index plus lent a construire.

    brew install libomp
    cargo build --release -p bwa-mem4 --features libsais-c-omp

### Un zero de plus

Le meme tri par longueur qui rend -3,5 % sur le batch de rescue mesure **+0,63 % de CPU, 2 victoires
sur 4**, sur les bins de l'extension. Contrairement au rescue, le contrefactuel `EXEC_SORTED` (0,4 %)
disait vrai ici : les jobs d'extension sont courts et homogenes, la taxe de lanes n'est que de
1,04-1,05x, et le gather/scatter impose a un lot autrement homogene coute plus qu'il ne rend. Retire.

## Le SACA rayon existe, et il change le defaut (2026-08-11)

Correction de la section precedente. `libsais-rs` **0.2 publie** n'est pas la mesure de ce que le Rust
peut faire ici : son chemin parallele ne scale pas, et son prefetch logiciel etait compile
**uniquement pour x86_64**, donc tout le port tournait **sans prefetch sur arm64** la ou le C emet
`prfm`. Mes 226,9 s sur le genome mesuraient cela, pas une limite du langage.

Teste sur la branche `perf/omp-scaling` de `BenjaminDEMAILLE/libsais-rs` (PR #2 en amont), appelee via
`libsais64_omp` a 8 threads. Genome humain entier, index **octet-identique a l'index de reference du
depot** :

| backend | mur | CPU | pic RSS | dependance systeme |
|---|---|---|---|---|
| `libsais-rs` 0.2 publie | 226,9 s | 453,8 s | 76,9 Go | aucune |
| C libsais serial (defaut actuel) | 150,6 s | 301,2 s | 92,6 Go | aucune |
| **`libsais-rs` PR, 8 threads** | **103,9 s** | **207,8 s** | **77,3 Go** | **aucune** |
| C libsais + OpenMP, 8 threads | 87,9 s | 175,8 s | 95,0 Go | `libomp` |

**2,18x plus rapide que la version publiee, 1,45x plus rapide que notre defaut**, et 1,18x derriere le
C+OpenMP. Sur chr21 l'ecart disparait: 1,15 s contre 1,14 s.

Le chiffre que la comparaison en temps masque est le RSS : **17,7 Go de moins que le C+OpenMP**
(77,3 contre 95,0). Sur une machine qui n'a pas 137 Go, c'est ce chiffre qui decide si l'index se
construit.

### Ce qui n'est PAS fait, et pourquoi

Rien n'est bascule. Deux raisons, la seconde mesuree :

1. La branche est une **dependance git**, pas une version publiee. Un defaut ne peut pas pointer sur
   une branche. Note : notre contrainte est `version = "0.2"`, donc 0.2.3 sera prise
   **automatiquement** des sa publication, et le correctif de prefetch ARM s'applique aussi au
   chemin serie (2,03 s -> 1,32 s sur chr21 d'apres la PR). Le gain arrive sans rien changer ; seul
   le passage a `libsais64_omp` demande un minimum explicite.
2. **Sur la 0.2 publiee, appeler l'entree `omp` est une REGRESSION** : chr21 passe de 2,80 s / 4,20 s
   (entree serie) a 3,61 s / 6,10 s a 8 threads. Le changement d'appel doit donc accompagner le bump
   0.3, jamais le preceder.

Action a la sortie de **0.2.3** (la PR est mergee en amont le 2026-08-11, `main` porte deja
0.2.3, mais crates.io s'arrete a 0.2.2) : bumper `libsais-rs`, remplacer `libsais64` par `libsais64_omp` avec le
meme knob `BWA4_SA_THREADS`, remettre `default = ["libsais"]`, et **retirer le passthrough
`libsais-c-omp`** livre plus haut, qui n'aura plus d'objet. Le gate est `scripts/index_diff.sh`.

## L'oracle tiers couvre enfin les penalites de gap asymetriques (2026-08-12)

`hyalite` 0.4.0 (2026-08-10) ajoute `Scoring::new_asymmetric`, ce qui leve une limitation que
`tests/third_party_oracle.rs` documentait depuis sa creation :

> *« hyalite carries a single gap-penalty pair, so it cannot express `o_del != o_ins`. The test
> therefore runs symmetric penalties, which is bwa's default (`-O 6 -E 1`). »*

Consequence : les bras **asymetriques** de `ksw_extend2` et du noyau de rescue n'avaient **aucun
controle par un tiers**, alors que ce noyau fait 32 % du CPU d'un run paired-end et que `-O 6,7 -E 1,2`
les atteint. Ils n'etaient valides que contre notre propre reference scalaire, c'est-a-dire contre du
code qui partagerait une eventuelle mauvaise lecture de `ksw.cpp`, ce qui est precisement le trou que
ce fichier existe pour fermer.

Le risque de cet elargissement etait la **convention de direction** : se tromper de sens entre
deletion et insertion ne casse que les schemas asymetriques, donc exactement ceux qu'on ajoute. La
0.4 l'ecrit elle-meme, il n'y a rien a deviner : *« This maps onto bwa's `-O del,ins -E del,ins`
(`ksw_extend2`): the `E` chain charges `(open_del, ext_del)`, the `F` chain `(open_ins, ext_ins)`. »*

Le balayage passe de deux schemas a quatre, dont deux asymetriques (`(1,4,6,1,7,2)` et
`(2,3,7,2,5,1)`), 300 paires chacun. **Le score concorde sur les quatre.**

### Ce que l'elargissement a trouve, et qui n'est pas un bug

Sur `A=2 B=3 O=7,5 E=2,1`, round 165 :

| | score | qb | qe | tb | te |
|---|---|---|---|---|---|
| nous | 205 | **3** | 111 | **5** | 112 |
| hyalite | 205 | **0** | 111 | **0** | 112 |

Meme score, memes extremites de **fin**, debuts differents. Deux alignements locaux de score egal,
celui de hyalite s'etendant plus a gauche. bwa recupere le debut par sa passe inverse `KSW_XSTART`,
qui s'arrete des que le score avant est atteint ; une traceback ordinaire peut continuer a travers des
cellules qui ne contribuent rien. Les penalites asymetriques creent ces egalites la ou le cas
symetrique n'en produisait pas.

Asserter l'egalite la-dessus testerait une **convention**, pas un calcul. Les spans et le quadruplet
`(score, te, score2, te2)` restent donc compares sur les schemas **symetriques** seulement, ou ils
etaient verifies 499/499 ; le **score** l'est sur les quatre, et c'est lui qui pilote `csub`, la MAPQ
et chaque acceptation dans `mem_matesw`.

### Verifie que le nouveau bras est vivant

Une premiere tentative de mutation ne prouvait rien : changer le schema des deux cotes a la fois ne
peut pas creer de desaccord. La bonne mutation perturbe **un seul** cote, `o_ins + 1` sur le notre :
le test echoue alors des le round 58. Et l'echec de span cite plus haut porte sur le quatrieme
schema, ce qui montre que les schemas asymetriques sont bien atteints.

## La RAM par batch (#25) : le modele, trois suspects innocentes, deux correctifs nuls (2026-08-15)

Protocole : 4M paires GIAB, GRCh38 entier, M4 Max 16 coeurs / 128 GB, `/usr/bin/time -l`. Le
plancher est l'index seul, 9,39 GB, mesure sur un run qui echoue juste apres le chargement.

| `-t` | `-K` | RSS | dont batch |
|---|---|---|---|
| 8 | 10M | 11,14 GB | 1,75 GB |
| 8 | 80M (defaut) | 14,88 GB | 5,49 GB |
| 8 | 160M | 19,15 GB | 9,76 GB |
| 16 | 10M | 11,37 GB | 1,98 GB |
| 16 | 80M | 15,14 GB | 5,75 GB |
| 16 | 160M (defaut) | 19,45 GB | 10,06 GB |
| 16 | 320M | 27,49 GB | 18,10 GB |

**Le modele.** Les deux droites, prises a `-t` fixe, donnent la meme pente : **52 a 53 octets
residents par base d'entree**, socle de 1,2 a 1,4 GB, et **+0,25 GB par doublement de threads**. La
RSS est donc gouvernee par `-K` SEUL ; `-t` n'y entre que parce qu'il multiplie le `-K` par defaut.
Ce qui verrouille la sortie evidente : ce defaut vaut `chunk_size * n_threads` pour reproduire
`aux.task_size` (`fastmap.cpp:964`), et le baisser deplacerait les frontieres de batch, donc le
modele d'insert, donc les octets. Il faut reduire les 53 octets, pas le batch.

**L'hypothese centrale de l'issue est fausse.** « La RAM tire le wall » ne tient pas : a `-t16`,
63,33 s a K=160M, 64,47 s a 80M, 65,52 s a 320M, 66,28 s a 10M. Diviser le batch par seize rend
8,1 GB et COUTE 4,7 % de wall. Les 10 % d'ecart a `-t16` sont un autre dossier, a ne pas chercher ici.

**Trois suspects innocentes, chacun par une mesure.**

* Les files : (1,1) 19,45 GB, (2,2) 19,77, (3,3) 20,35. Un batch de plus en vol coute ~0,3 GB sur
  10, pas la moitie. Le correctif `68a92bc` avait bien ferme cette part, et son commentaire qui dit
  « c'etait tout l'ecart » est trop genereux : l'ecart est passe de 1,5x a 1,27x le fork, il n'a pas
  disparu.
* Les intermediaires de seeding : deja plafonnes a `INFLIGHT_READS = 16384` reads quel que soit `-t`.
* mimalloc : `MIMALLOC_PURGE_DELAY=0` donne 19,42 GB, avec arenes non pre-engagees 19,40, contre
  19,45. Et l'allocateur systeme, teste en retirant le `#[global_allocator]`, donne **21,27 GB et
  67,35 s** contre 19,45 GB et 63,33 s. mimalloc ECONOMISE ici, il ne thesaurise pas.

**Ou vont les octets.** Sonde temporaire, un batch de 160M bases (540 541 paires) juste apres
l'alignement : records 344 MB, codes 153 MB plus 25 MB d'en-tetes, regions 14,1 M vivantes pour
**24,0 M d'emplacements** soit 1287 MB utiles dans 2196 MB alloues, a 96 octets la region (88 utiles,
8 de padding). Total comptabilise **2741 MB, soit 18 octets par base**, sur les 53 mesures. Apres
rescue les regions montent a 15,6 M vivantes dans 27,6 M d'emplacements. Le reste n'est donc pas
vivant entre les etages : il est transitoire A L'INTERIEUR d'un etage.

**Ce que la courbe dit.** Echantillonnee toutes les 2 s, la RSS monte a 17,4 GB en 8 s puis derive
lentement jusqu'a 19,34 et **ne redescend jamais**, sans dent de scie par batch, avec les deux
allocateurs. Sur macOS les pages rendues par `madvise(MADV_FREE)` restent comptees tant que le
systeme n'en a pas besoin : **la RSS de pointe mesure le plafond de tout ce qui a ete alloue, pas ce
qui est vivant.** C'est la cle du dossier, et elle explique les deux echecs qui suivent.

**Correctif 1, rendre la capacite : negatif.** `shrink_to_fit` sur les listes de regions dont le
slack depasse 16 emplacements, ce qui visait les 909 MB de headroom par batch. A/B entrelace, deux
repetitions : **+0,94 et +0,95 GB de RSS**, wall inchange. Le gain vise etait de 909 MB et la perte
est de 950 MB : la capacite liberee n'est pas rendue, et les copies neuves montent le plafond
d'autant. Reverte.

**Correctif 2, ne jamais allouer le slack : nul.** `reserve_exact` sur les quatre listes paralleles
(`regs`, `reg_chain`, `reg_meta`, `reg_preskip`), le compte de seeds etant connu avant la premiere
poussee, ce qui remplace l'echelle 1, 2, 4, 8 par une allocation unique. A/B entrelace, cinq paires :
RSS 19,10 / 19,13 / 19,26 / 19,26 / 18,92 contre 19,00 / 19,43 / 19,29 / 19,13 / 18,86, et wall
gagnant **3 fois sur 5**, medianes 67,8 contre 67,5 s. Un nul des deux cotes. SAM octet-identique
verifie sur les 3,4 GB de sortie. Reverte, comme le fast-path d'epilogue avant lui.

**Ce qui reste ouvert, et sous quelle forme.** La cible n'est pas la memoire vivante mais le VOLUME
TOTAL alloue pendant un batch, puisque c'est lui qui fixe le plafond sur cette plateforme. Deux
suites possibles : instrumenter l'allocation par etage (compteur d'octets alloues, pas des
snapshots, ceux-ci ayant deja montre qu'ils ratent le pic), ou refaire la mesure sous Linux, ou les
pages liberees reviennent, pour savoir si l'ecart de 1,27x contre le fork est un ecart de code ou un
artefact de comptage macOS. Tant que ce point n'est pas tranche, aucun chiffre de RAM de ce projet
sur macOS ne doit etre lu comme « memoire necessaire ».

## Linux tranche #25 : la RAM revient, l'ecart est reel, et l'allocateur n'y est pour rien (2026-08-15)

La section precedente montrait que la RSS de pointe sur macOS est un plafond d'allocation et non une
mesure de memoire. Restaient deux questions, et une machine Linux les tranche toutes les deux :
chr21, 2M paires simulees (`wgsim -S 11`), runner heberge 4 coeurs, index construit par
`bwa-mem2 index` et lu par les trois aligneurs.

**La memoire revient.** Chronologie echantillonnee a 0,5 s, `-K 100M` : 1,72 GB, montee a 4,54,
**chute a 1,87**, remontee a 4,20, **chute a 1,33**, remontee. Une dent de scie par batch, la
signature que macOS n'a jamais montree. Le plafond macOS etait bien de la comptabilite.

**Mais l'ecart contre le fork est reel, et plus grand que l'estimation macOS.** Meme machine, meme
entree, meme index, deux repetitions :

| | bwa-mem4 | `bwa-mem3` (fork) | ratio |
|---|---|---|---|
| `-K` par defaut | 2,00 / 2,00 GB | 1,18 / 1,16 GB | **1,71x** |
| `-K` epingle a 100M | 4,28 / 4,56 GB | 2,29 / 2,28 GB | **1,93x** |

A batch identique nous prenons donc pres du double. Ce n'etait pas un artefact de plateforme : c'est
le sujet d'origine de #25, cette fois avec un repere fiable.

**Et l'allocateur n'y est pour rien.** jemalloc (`tikv-jemallocator` 0.7) branche derriere une
feature temporaire, meme entree, batch epingle :

| bras | wall | peak RSS |
|---|---|---|
| mimalloc | 201,8 / 201,6 s | 4,58 / 4,28 GB |
| jemalloc | 207,7 / 207,5 s | 4,47 / 4,47 GB |
| jemalloc `dirty_decay_ms:0,muzzy_decay_ms:0` | 246,6 / 245,9 s | 4,12 / 4,13 GB |
| `bwa-mem3` | 150,4 / 150,5 s | 2,30 / 2,27 GB |

**mimalloc reste.** jemalloc coute 2,9 % de wall dans les deux repetitions pour une RSS equivalente,
et sa restitution forcee rend 8 % de RAM contre 22 % de wall. Les corps SAM des deux allocateurs
sont identiques (seul `@PG CL` differe, il porte `argv[0]`). Sur macOS le meme A/B donnait jemalloc
gagnant de 0,7 GB, ce qui etait une lecture du plafond et non de la memoire, et ses wall allaient de
67 a 373 s : cette mesure-la ne valait rien, celle-ci vaut.

**Troisieme resultat, non demande.** Le wall du meme tableau donne **0,75x contre le fork** (202 s
contre 150 s), l'ecart x86 de #20 reproduit sur runner heberge, sur lectures simulees et 4 coeurs.
Ce n'est pas le WGS que #32 demande, mais c'est le premier chiffre x86 de ce projet obtenu sans
machine dediee, et il est du meme ordre que le 0,71x mesure sur Zen 3.

Ce qui reste pour #25 est donc etroit et bien pose : nos structures par batch, pas l'allocateur, pas
la plateforme, pas les files. La cible chiffree est de rendre ~2x moins de memoire par base a batch
egal, et la mesure de reference est desormais ce job Linux plutot qu'un chiffre macOS.

## Ou vont les octets (#25) : l'allocation par etage, mesuree depuis l'allocateur (2026-08-18)

Les deux sections precedentes fermaient les fausses pistes (les files, mimalloc, la plateforme) et
laissaient une seule suite utile : « instrumenter l'allocation par etage, un compteur d'octets
alloues, pas des snapshots ». C'est fait. `BWA4_STAGE_ALLOC` enveloppe l'allocateur global
(`crates/bwa-cli/src/stage_alloc.rs`) et compte chaque requete au moment ou elle passe, en
l'imputant a l'etage du pipeline qui l'a demandee.

**Et il ne reste PAS dans le binaire livre, contrairement aux autres sondes.** Desarme il ne coute
qu'une lecture d'`AtomicBool` et un branchement non pris, mais par ALLOCATION et non par batch, et
ce pipeline alloue 167 M de fois par 500k paires GIAB. A/B entrelace, min de 6 sur hote calme :
**12,49 s contre 12,56 s de wall et 95,14 s contre 95,67 s de user, +0,5 %**. Le wrapper vit donc
derriere la feature `stage-alloc`, off par defaut, exactement comme `align-split` avant lui ; le
meme A/B refait apres la mise sous feature ne distingue plus les deux binaires (12,47 contre
12,47 s). Sans la feature, `BWA4_STAGE_ALLOC` affiche une ligne qui le dit et ne change rien.

Trois colonnes, et elles ne disent pas la meme chose. Le **volume** est la somme de tout ce qui a
ete demande, jamais decremente : c'est le chiffre que la RSS de pointe macOS mesure reellement. Le
**vivant** est alloue moins libere, exact parce que `dealloc` recoit le `Layout` d'origine, et son
maximum courant est le vrai pic. Les **classes de taille** disent si un chiffre par base vient de
beaucoup de petites requetes ou de quelques grosses. Un appel `mark_baseline()` juste apres le
chargement de l'index gele le vivant a cet instant, si bien que la colonne de pic parle du batch et
non des 10 GB de genome.

Deux modes, parce qu'une seule execution ne peut pas repondre aux deux questions. L'attribution par
etage passe par une etiquette globale au processus, la ou allouent surtout les workers rayon, donc
elle exige **un seul batch en vol** : `BWA4_STAGE_ALLOC=1` retire chaque batch avant de lire le
suivant. `BWA4_STAGE_ALLOC=overlap` garde les deux batchs en vol du pipeline livre, donc le pic est
le vrai, mais le decoupage par etage devient un melange de deux batchs. Le reader et le writer sont
etiquetes par THREAD et echappent aux deux problemes.

**Protocole.** GIAB HG002, 500k paires, index genome GRCh38, M4 Max, `-t8`, `-K` par defaut (80M),
soit 148M bases et 2 batchs. Sortie SAM `cmp`-identique a la course non instrumentee dans les deux
modes. Le wall d'une course instrumentee ne vaut rien (34,5 s contre 13,0 s) : quatre atomiques par
allocation, et la serialisation en plus.

| | RSS de pointe | vivant de pointe | dont index | batch vivant | batch RSS |
|---|---|---|---|---|---|
| sans sonde | 14,72 GB | | | | 4,25 GB |
| `overlap` (pipeline livre) | 14,47 GB | 13,31 GB | 10,47 GB | **2,84 GB** | 4,00 GB |
| serialise (1 batch) | 13,22 GB | 12,12 GB | 10,47 GB | **1,65 GB** | 2,75 GB |

**Premier resultat : un tiers de la RSS du batch n'est vivante a aucun instant.** 4,00 GB de
resident pour 2,84 GB de vivant au pic, soit **1,41x** dans la configuration livree, et 1,67x a un
seul batch. Ce n'est pas une erreur de mesure : c'est ce que la section macOS annoncait sans pouvoir
le chiffrer. Consequence directe pour #25 : diviser par deux nos structures vivantes ne diviserait
pas par deux la RSS du batch, il resterait le ~1,2 GB que l'allocateur detient sans que personne ne
le tienne.

**Deuxieme resultat : le rapport churn sur vivant est de 23.** 66,87 GB alloues au total pour 148M
bases, soit **452 octets par base d'entree**, contre **19 octets par base** vivants au pic. Le
tableau par etage, mode serialise :

| bucket | volume | part | appels | taille moyenne | o/base | pic vivant batch |
|---|---|---|---|---|---|---|
| encode | 0,17 GB | 0,3 % | 1,00 M | 172 o | 1,2 | 0,09 GB |
| align | **31,49 GB** | **47,1 %** | **110,84 M** | 284 o | 212,8 | 1,37 GB |
| deinterleave | 0,05 GB | 0,1 % | 2 | 24 MB | 0,3 | 1,28 GB |
| dedup_prep | 0,16 GB | 0,2 % | 0,05 M | 3392 o | 1,1 | 1,31 GB |
| pestat | 0,02 GB | 0,0 % | 61 | 400 kB | 0,2 | 1,27 GB |
| rescue | **19,10 GB** | **28,6 %** | 14,95 M | 1277 o | 129,0 | 1,61 GB |
| sam_emit | 5,32 GB | 8,0 % | 39,89 M | 133 o | 36,0 | **1,65 GB** |
| reader | 0,10 GB | 0,2 % | 53 | 1,9 MB | 0,7 | |
| writer | 0 | 0,0 % | 4 | | 0,0 | |
| unstaged (index, main) | 10,45 GB | 15,6 % | 1,00 M | 10 kB | 70,6 | |
| **TOTAL** | **66,87 GB** | 100 % | **167,73 M** | 399 o | 451,8 | 1,65 GB |

L'align alloue **0,75 fois par base d'entree** et libere presque tout dans l'etage meme ; le rescue
suit avec 15 M d'appels a 1277 octets de moyenne. Le pic vivant, lui, arrive dans `sam_emit`, ou les
regions du batch et son texte SAM coexistent.

**Troisieme resultat : la forme.** 117 M des 167 M d'appels demandent moins de 128 octets et ne
pesent que 6 % du volume, tandis que la seule classe 16 ko - 32 ko pese **7,76 GB, 11,6 % du
volume, en 336k appels**. Les 452 octets par base ne sont donc ni « beaucoup de petites » ni
« quelques grosses » : ce sont deux populations distinctes, et elles appellent deux correctifs
differents.

### Et dans `align`, aucun point chaud : les quatre sous-phases se partagent le churn

La ligne `align` pesait 47 % du volume sans dire laquelle de ses phases le depense. Le compteur a
donc ete descendu dans `bwa-core` (`alloc_probe`), seul crate que le binaire ET `bwa-mem` partagent,
et `align_reads_batched_inner` porte maintenant quatre etiquettes. Meme protocole, mode serialise :

| sous-bucket | volume | part | appels | taille moyenne | o/base |
|---|---|---|---|---|---|
| align:seed | 7,27 GB | 10,9 % | 26,07 M | 279 o | 49,1 |
| align:sa | 7,40 GB | 11,1 % | 22,72 M | 326 o | 50,0 |
| align:chain | 8,41 GB | 12,6 % | 30,57 M | 275 o | 56,8 |
| align:extend | 8,38 GB | 12,5 % | 31,35 M | 267 o | 56,6 |
| align (le reste) | 0,03 GB | 0,0 % | 0,14 M | 236 o | 0,2 |

**Il n'y a pas de tampon a corriger, il y a un style d'allocation.** Les quatre phases sont a moins
de 16 % l'une de l'autre en volume comme en nombre d'appels, avec des tailles moyennes entre 267 et
326 octets, soit **une centaine d'allocations par read pour le seul `align`**, et 40 de plus par read
dans `sam_emit` a 133 octets de moyenne. C'est la signature du `Vec` par read et par seed repete
partout, pas d'un gros tampon mal dimensionne. Un correctif ponctuel ne peut donc pas payer : ce qui
paierait est une arene par worker reutilisee d'un read a l'autre a l'interieur d'un chunk, ce qui
touche les quatre phases a la fois. C'est aussi ce qui explique que les deux tentatives precedentes,
toutes deux ponctuelles, aient mesure nul.

**Les etiquettes ne sont pas livrees non plus.** Elles sont derriere la feature `stage-alloc` de
`bwa-mem`, pour la raison qui a mis `align-split` derriere la sienne : desarmee une etiquette est une
lecture d'`AtomicBool` par chunk de reads, donc rien, mais elle vit dans une fonction
`inline(always)` qui est le coeur chaud de l'aligneur, et un garde dont la portee couvre la moitie de
la fonction est exactement ce qui perturbe l'allocation de registres. L'A/B de la version non gatee
lisait +1,2 % en min de 4 et un nul en medianes : impossible a montrer gratuite, donc pas livree.
Apres la mise sous feature, binaire par defaut contre binaire d'avant le chantier, min de 5 :
12,46 s contre 12,43 s de wall, medianes 12,56 contre 12,54. Nul, cette fois pour de bon.

**Ce que ca change pour #25.** La cible n'est plus « nos structures sont deux fois trop grosses »,
elle se scinde en deux, avec des chiffres :

1. **Le churn de `align` et de `rescue`**, 50 GB de volume transitoire par 148M bases pour ~19
   octets par base reellement vivants, reparti a peu pres egalement sur les quatre sous-phases de
   `align`. C'est lui qui fixe le plafond que la RSS macOS reporte, et c'est un probleme d'arene par
   worker, pas de taille de structure ni de tampon isole. Les deux correctifs deja tentes
   (`shrink_to_fit`, `reserve_exact`) visaient le vivant et un site a la fois, ce qui explique apres
   coup pourquoi l'un a coute 0,95 GB et l'autre rien.
2. **Le 1,4x de retention** entre vivant et resident, qui survivra a toute reduction des structures
   et qui doit etre mesure sous Linux avec la meme sonde avant qu'on lui attribue un correctif.

La suite immediate est donc de relancer exactement cette sonde sur le job Linux de reference, ou les
pages reviennent, pour savoir laquelle des deux moities de l'ecart de 1,71x contre le fork est
vivante et laquelle est de la retention.

## #25 tranche : le resident EST le vivant sous Linux, et le 2x est le double batch en vol (2026-08-18)

La sonde par etage, rejouee la ou les pages reviennent. Protocole ROADMAP Linux : chr21, `wgsim -S 11`,
2M paires, index `bwa-mem2`, runner heberge 4 coeurs, `-K` epingle a 100M pour que le fork et nous
voyions le meme batch. Workflow `alloc-probe.yml`, run 32143186293.

**Premiere ligne, celle qui ferme la piste macOS.**

| | RSS de pointe | vivant de pointe | dont index | batch RSS | batch vivant | resident/vivant |
|---|---|---|---|---|---|---|
| pipeline livre (`overlap`) | 4,553 GB | 4,461 GB | 0,181 GB | 4,372 GB | 4,280 GB | **1,02x** |
| un batch en vol (serialise) | 2,680 GB | 2,587 GB | 0,188 GB | 2,492 GB | 2,399 GB | **1,04x** |

**Sous Linux le resident est le vivant.** Le 1,41x macOS etait de la comptabilite, entierement. Il
n'y a rien a recuperer du cote de l'allocateur, et le 1,4x de retention annonce comme « la moitie du
dossier » deux sections plus haut n'existe pas sur la plateforme ou les utilisateurs tournent.

**Deuxieme ligne, celle qui ferme #25.** Meme job, memes reads, memes index :

| | wall | RSS de pointe |
|---|---|---|
| bwa-mem4 | 201,71 / 202,09 s | 4,78 / 4,78 GB |
| `bwa-mem3` (fork) | 150,74 / 151,55 s | 2,36 / 2,39 GB |
| bwa-mem4, **un batch en vol** | | **2,49 GB** |

Contre le fork : **2,02x en RSS avec le recouvrement, 1,05x sans**. Notre batch n'est donc pas deux
fois plus gros que le sien, et nos structures ne sont pas en cause : nous en tenons deux a la fois.
`run_pipeline` lance le batch N+1 avant de retirer le N pour que la queue peu occupee du N (dedup a
69 %, encode a 29 %) tourne contre l'`align` du N+1, qui est la seule etape capable de remplir le
pool. Le commentaire du site disait deja « le cout est un batch resident de plus » ; ce qui manquait
etait son prix, et le prix est exactement l'ecart de l'issue.

**`BWA4_NO_BATCH_OVERLAP=1`** rend cette moitie. Sortie inchangee : l'ordre de sortie vient du join
avant l'envoi, pas du nombre de batchs en vol. Verifie `cmp`-propre sur 500k paires GIAB en local et
sur chr21 en CI. Sur macOS, `-t8`, 500k paires GIAB : 14,72 GB contre 13,28 GB de RSS de pointe, soit
la part batch de 4,25 a 2,81 GB, pour 1,4 % de wall sur un jeu a deux batchs, ou le recouvrement n'a
presque rien a recouvrir.

**Et c'est le seul levier qui rende cette memoire sans toucher a la sortie.** Baisser `-K` de moitie
donnerait le meme resident, deux batchs de `-K/2` valant un de `-K`, mais deplacerait les frontieres
de batch, donc le modele d'insert par batch, donc les octets : c'est ce que la section « la RAM par
batch » avait deja verrouille. Le nombre de batchs en vol, lui, n'est observable nulle part dans le
SAM.

Deuxieme point de mesure, macOS `-t8` avec `-K` 10M (15 batchs) : 11,79 GB contre 11,61 GB, soit
0,19 GB seulement. Attendu, et il faut le dire pour que personne ne cite le levier hors de son
regime : ce qu'il rend est UN batch, donc son gain est proportionnel a `-K`. A `-K` par defaut il
rend 1,44 GB, a `-K` 10M presque rien. Le wall de ce meme jeu n'est pas exploitable, l'hote allant de
13,4 a 20,3 s d'une repetition a l'autre.

### Et le churn ne coute pas de wall non plus : l'arene par worker est morte des deux cotes

La section precedente concluait que les 452 octets alloues par base appelaient une arene par worker.
Le resultat Linux lui retire sa raison memoire : ce volume transitoire ne fait pas de resident, la
RSS du batch etant son vivant a 1,02x pres. Restait sa raison wall, et un profil `sample` la retire
aussi. macOS, `-t8`, 500k paires GIAB, 6 s de profil en regime : **651 echantillons self dans
l'allocateur sur 55 299**, dont 18 876 en attente (`psynch_cvwait`, `ulock_wait`, `semaphore_wait`).
Soit **1,2 % du profil et 1,8 % du temps occupe**, contre 13 163 pour le seul `batched_extend_neon_u8`.

Un remplacement PARFAIT de l'allocateur par une arene rendrait donc au mieux 1,8 %, sous le plancher
de 3 % que ce projet s'est donne, pour un refactor qui traverse quatre crates et met l'octet-identite
en jeu. Le dossier « churn » se ferme sur ce chiffre : mimalloc encaisse 167 M d'allocations par
500k paires pour 1,8 % du busy, ce qui est le vrai enseignement du tableau des volumes.

### Le prix du recouvrement, mesure : 1,1 % de wall pour 41 % de la RSS

A/B entrelace sur le binaire LIVRE, sans instrumentation, meme runner Linux 4 coeurs, chr21, 2M
paires, `-K` 100M, six batchs, le fork chronometre a cote comme temoin de derive (run 32149818162) :

| | wall (3 reps) | RSS de pointe |
|---|---|---|
| recouvrement (defaut) | 206,16 / 206,49 / 206,85 s | 4,786 / 4,797 / 4,783 GB |
| `BWA4_NO_BATCH_OVERLAP=1` | 208,51 / 208,10 / 209,21 s | **2,818 / 2,833 / 2,816 GB** |
| `bwa-mem3` (fork) | 156,81 / 158,08 / 157,60 s | 2,371 / 2,385 / 2,374 GB |

Les trois repetitions vont dans le meme sens sans se croiser : **+1,1 % de wall pour -41 % de RSS de
pointe**, la part batch passant de 4,60 a 2,63 GB. Contre le fork nous passons de **2,02x a 1,19x en
memoire**, a wall inchange (0,76x dans les deux bras). Sortie identique, md5 hors `@PG`. macOS `-t8`
sur 500k paires GIAB donnait le meme ordre : 1,5 % de wall pour 1,46 GB, soit 10 % de la RSS totale
la ou l'index pese 10 GB, et 34 % de la part batch.

**#25 est donc chiffree et fermable, et la seule question qui reste est un choix de defaut**, pas une
mesure : 1,1 % de wall est en dessous du plancher de 3 % que ce projet applique a ses propres
optimisations, et 2 GB de RSS sur un run WGS est souvent ce qui decide si le job passe dans la
machine. Ce qui plaide pour garder le defaut actuel est qu'il a ete mesure gagnant sur les etages a
faible occupation (dedup 69 %, encode 29 %), et ces 1,1 % sont reels. Ce qui plaide pour l'inverser
est qu'un utilisateur memoire-contraint ne peut pas deviner l'existence d'une variable d'environnement,
alors qu'un utilisateur presse lit un chiffre de wall. Le choix n'appartient pas a la mesure.

## Trois questions fermees le meme jour : le gate GIAB, le PGO hors Apple, et l'indexeur parallele (2026-08-18)

### Phase 11 : l'identite octet, dite en precision et rappel (#35)

`scripts/giab_happy.sh` aligne les memes reads avec les deux aligneurs, appelle les variants avec le
meme caller, et note les deux VCF contre le benchmark NIST v4.2.1. Le workflow `giab-gate` le fait
tourner sur des reads HG002 REELS, decoupes dans le BAM 300x de GIAB par HTTPS : 1 089 635 paires
sur une fenetre de 10 Mb du bras q de chr21, 17 806 variants de verite dans 9,75 Mb de regions de
confiance.

| bras | TP | FP | FN | precision | sensibilite | F |
|---|---|---|---|---|---|---|
| bwa-mem4 | 16889 | 172 | 171 | 0,9899 | 0,9900 | 0,9899 |
| bwa-mem2 | 16889 | 172 | 171 | 0,9899 | 0,9900 | 0,9899 |

Enregistrements BAM identiques, enregistrements VCF identiques, donc la meme ligne deux fois. Ce
n'est pas une decouverte, c'est le livrable : l'identite existe desormais dans les unites que lit un
utilisateur de variant calling. Deux pieges rencontres et documentes sur place : Ensembl nomme le
contig `21` la ou le benchmark le nomme `chr21`, et un caller ne produit alors rien du tout sans se
plaindre ; et des reads extraits d'un alignement existant sont les reads que CET aligneur a places
la, ce qui est propre pour comparer deux aligneurs sur la meme entree et n'est pas une mesure de
sensibilite.

### Le PGO est un levier Apple Silicon, et rien d'autre (#33)

nh13 mesurait -0,4 % sur Graviton4 la ou nous mesurons +12,4 % sur M4. Un troisieme point tranche,
sans Graviton : ARM heberge (Ampere Altra, `CPU implementer 0x41`), notre chaine, notre procedure
d'entrainement, `llvm-profdata` de la toolchain et non de la distribution, entrainement sur une
graine wgsim differente de celle mesuree.

| rep | plain | PGO |
|---|---|---|
| 1 | 69,43 s | 69,99 s |
| 2 | 69,90 s | 70,32 s |
| 3 | 69,69 s | 69,95 s |

**PGO est 0,6 % plus lent**, trois repetitions sans croisement. Deux coeurs ARM non-Apple disent
donc la meme chose contre +12,4 % sur M4 : le gain appartient a l'Apple Silicon. C'est coherent avec
ce que le PGO achete ici, de la disposition de branches sur le chemin branchu (driver, seeding,
SAM), dont la valeur depend entierement du predicteur du coeur. `scripts/pgo.sh` doit donc etre
presente comme un levier Apple Silicon, pas comme une propriete du binaire.

### L'indexeur n'est pas mono-thread, il est mono-thread PAR DEFAUT (#37)

Mesure a l'echelle genome, GRCh38 (3,15 GB de FASTA), M4 Max, index octet-identique dans les trois
cas (md5 du `.bwt.2bit.64`) :

| backend | wall | RSS de pointe |
|---|---|---|
| defaut (libsais C, serie) | 152,45 s | 91,6 GB |
| `--features libsais-c-omp`, 8 threads | **90,96 s** | 98,4 GB |
| `--features capsa` (chr21 seulement) | 24x plus lent | 2,2x moins de RAM |

**1,68x pour +7 % de RAM**, et le code existe deja. Ce qui empeche d'en faire le defaut n'est pas
l'algorithme mais la dependance : `libsais-c-omp` exige un runtime OpenMP a la compilation, et un
artefact de release qui gagnerait silencieusement une dylib serait pire qu'un index plus lent, ce que
`crates/bwa-cli/Cargo.toml` disait deja. Le vrai reste de #37 est donc une question de packaging, pas
de parallelisation.

## Le dossier x86 : ou nous perdons, mesure sur le meme hote (2026-08-18)

Toutes les mesures de ce projet contre `bwa-mem2` etaient arm64. Voici les deux architectures cote a
cote, chr21 wgsim, `-t4`, chaque ratio pris **a l'interieur d'un seul hote** pour qu'aucune
comparaison de machines n'y entre :

| hote | bwa-mem4 | `bwa-mem3` (fork) | `bwa-mem2` | nous / mem2 | fork / mem2 | nous / fork |
|---|---|---|---|---|---|---|
| M4 Max (1M paires) | 36,55 / 36,54 s | parite (0,981x, GIAB) | 129,84 / 116,86 s | **3,20x** | ~3,2x | **1,00x** |
| EPYC 7763, Zen 3 (2M paires) | 205,4 / 208,0 s | 150,9 / 151,1 s | 321,7 / 324,6 s | **1,57x** | **2,13x** | **0,74x** |

**Les deux aligneurs perdent en passant sur x86, et c'est attendu** : `bwa-mem2` y est chez lui,
avec ses noyaux AVX2/AVX-512 ecrits pour Intel, tandis que sur arm64 il tourne a travers une couche
d'emulation SSE. Un ratio contre `bwa-mem2` mesure donc en partie le handicap de `bwa-mem2`, ce qui
est une raison de plus de citer le fork comme reference.

**Mais nous perdons deux fois plus.** Le fork passe de ~3,2x a 2,13x contre `bwa-mem2` ; nous
passons de 3,20x a 1,57x. La difference entre ces deux chutes EST l'ecart de 1,36x que nous avons
contre lui sur x86, et c'est le seul chiffre de ce tableau qui decrit du code a nous.

Deux causes possibles, et elles se separent par la mesure :

1. **Nos noyaux AVX2 sont plus faibles, relativement, que nos noyaux NEON.** Le kernel d'extension
   pese 56 % du CPU d'`align` sur ARM (profil `align-split` sur ces memes donnees), donc un deficit
   de kernel s'y verrait.
2. **Notre code SCALAIRE x86 est compile en baseline.** `bwa-mem2` livre un binaire par jeu
   d'instructions (`bwa-mem2.avx2`, `bwa-mem2.avx512bw`) et choisit a l'exec ; nous livrons un
   binaire baseline et ne selectionnons au runtime que les noyaux vectoriels. Toutes nos boucles
   scalaires de seeding, de chainage et de resolution SA sont donc compilees pour un CPU de 2003.
   Sur ARM ce choix ne coute rien, la baseline aarch64 contenant deja NEON : controle mesure ici,
   `target-cpu=native` contre baseline donne 36,35 s contre 36,48 s, un nul.

### Le profil x86, lu enfin, et il ne dit pas ce que les issues supposaient

Runner heberge 4 coeurs, AMD EPYC 7763 (Zen 3, pas d'AVX-512), chr21, `-K` par defaut, un batch en
vol pour que les parts soient exactes. Deux jeux de donnees, et c'est le point le plus important de
toute la section.

| jeu | bwa-mem4 | `bwa-mem3` | `bwa-mem2` | nous / fork |
|---|---|---|---|---|
| wgsim, 2M paires | 216,1 / 217,5 / 216,8 s | 162,9 / 165,7 / 165,3 s | 355 / 359 / 355 s | **1,32x** |
| HG002 reel, 5,45M paires | 128,2 / 129,6 / 129,5 s | 118,3 / 120,8 / 126,2 s | 278 / 283 / 280 s | **1,05x** |

**L'ecart x86 depend du jeu de donnees, et il n'est pas de 1,3x partout.** Sur des reads simules
nous sommes a 1,32x du fork ; sur des reads HG002 reels, decoupes dans une fenetre de 10 Mb, a
**1,05x**. Aucun des deux chiffres n'est le chiffre WGS que #32 demande : les reads reels utilises
ici viennent d'un alignement existant restreint a une fenetre, donc leurs mates tombent presque
toujours a cote et le rescue ne travaille pas (10,7 % du wall contre 21,4 % sur wgsim). La lecture
honnete est que le deficit va de 5 % a 32 % selon ce qu'on aligne, et que le protocole doit etre
cite avec le chiffre.

**Par etage, wgsim, normalise par million de paires, contre le meme profil sur M4 :**

| etage | M4 (s/Mpaires) | Zen 3 (s/Mpaires) | rapport |
|---|---|---|---|
| align | 24,86 | 70,3 | 2,83x |
| rescue | 9,10 | 23,4 | 2,57x |
| `sam_emit` | 2,79 | 14,4 | **5,15x** |

Un coeur M4 est plus rapide qu'un coeur Zen 3 de cette generation, donc un rapport uniforme autour
de 2,8x ne dit rien. Ce qui parle est **l'ecart entre les etages** : `sam_emit`, qui est du
formatage de chaines et d'entiers sans une seule instruction vectorielle, est deux fois plus penalise
que le reste. `chain_flt` suit a 4,3x et le `rest` du profil `align-split` (marche SA et fusion de
chaines) a 3,5x, contre 2,57x pour les noyaux d'extension eux-memes.

**Les deux noyaux vectoriels, eux, tiennent :** le kernel de rescue fait 4,56 Gcell/s/thread en AVX2
contre 10,03 en NEON, et les noyaux d'extension sont a 2,57x du M4. C'est le rapport attendu entre
ces deux coeurs. Ce ne sont donc pas les noyaux qui expliquent que nous perdions plus que le fork.

**La fenetre SA n'est pas un levier sur x86** : 219,4 / 218,7 / 217,3 / 218,3 / 217,3 s pour
W = 16, 32, 64, 96, 128. Plate, la ou sur M4 passer de 128 a 32 coute 18 %. Zen 3 ne profite pas du
recouvrement de defauts de cache que le M4 exploite, ce qui est coherent avec les 163 ns par lookup
mesures ici contre 60 ns sur M4.

**Ce que cela designe** : nos chemins SCALAIRES sur x86, pas nos noyaux. Et c'est exactement la ou la
difference de packaging joue, `bwa-mem2` livrant un binaire par jeu d'instructions quand nous livrons
un binaire baseline dont seules les fonctions vectorielles sont selectionnees au runtime. Le bras
baseline / v2 / v3 mesure precisement cela.

### Le PGO de la release etait un chiffre Apple generalise, et il coutait 3,3 % sur x86 (2026-08-18)

`release.yml` construit chaque artefact en PGO, avec un commentaire qui annonce +4,4 % mesure sur
« 500k paires reelles contre GRCh38 ». Cette mesure etait sur Apple Silicon. Refaite sur trois
architectures, trois repetitions entrelacees chacune, aucune ne se croisant :

| hote | plain | PGO | verdict |
|---|---|---|---|
| M4 (Apple Silicon) | | | **+12,4 %** |
| Ampere Altra (ARM heberge) | 69,43 / 69,90 / 69,69 s | 69,99 / 70,32 / 69,95 s | **-0,6 %** |
| AMD EPYC 7763 (Zen 3), recette cargo-pgo | 103,95 / 103,59 / 104,01 s | 108,66 / 107,98 / 107,80 s | **-4,1 %** |
| AMD EPYC 7763, **recette exacte de `release.yml`** | 104,22 / 104,27 / 104,27 s | 107,36 / 108,15 / 108,24 s | **-3,3 %** |

La derniere ligne est celle qui decide : meme recette, meme entrainement sur `testdata/tiny`, memes
CFLAGS. Ce n'est pas un proxy, c'est le chemin livre. **Chaque artefact `linux-x86_64` publie
jusqu'ici est donc ~3 % plus lent qu'un build ordinaire**, sur la plateforme ou tourne l'essentiel du
WGS, et le -0,4 % de nh13 sur Graviton4 disait la meme chose depuis le debut sans qu'on le croie.

Le build PGO est maintenant conditionne par un champ de matrice : Apple Silicon profile, les trois
autres compilent en clair. Effet de bord agreable : ces trois artefacts redeviennent reproductibles
depuis un arbre propre par `cargo build --release`, ce qu'un artefact PGO n'est pas.

`macos-x86_64` est desactive sans mesure propre, deliberement : deux architectures non-Apple sur deux
perdent, et livrer des octets plus lents sur une supposition non mesuree est le mauvais defaut.

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
