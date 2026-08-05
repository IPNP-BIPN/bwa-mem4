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
