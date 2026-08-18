# Adding a GPU, without giving up byte-identity

The 2026-08-08/09 research into whether a GPU can accelerate this aligner while keeping the SAM
bytes identical to bwa-mem2, and what it would take to build. It lived as GitHub issues #52 to #57
until 2026-08-18; the issues are closed and the plan is here, for the same reason the kernel
research moved in `kernel-ceiling-and-dead-ends.md`: a plan nobody can find is not a plan, and an
issue tracker is where documents go to be forgotten. **Closing those issues was a filing decision
and not a rejection.** Nothing here has been refuted; none of it has been built either.

Read the first section for the number that decides whether any of this is worth doing (2.45x, and
an integrated GPU already reaches it), then the prerequisites, then the barrier that must exist
before the first kernel, then the backends.

One thing to know before reading: every DP job in this pipeline is a pure function of its inputs,
which is what makes a GPU backend safe to attempt at all. The risk is not correctness-by-design, it
is correctness-in-practice, and section #54 is the whole answer to that.

## #52 : meta: hybride CPU+GPU, le plafond est 2,45x et le GPU integre suffit deja

Cette issue est le chapeau des cinq suivantes. Elle repond a une seule question : **jusqu'ou peut-on aller en gardant la sortie SAM octet-identique a bwa-mem2, si on ajoute un GPU a la machine ?**

La reponse est un nombre, et il est deja mesure : **2,45x**.

### 1. Le plafond d'Amdahl, calcule sur notre propre profil

Profil par echantillonnage, 1 M paires GIAB reelles contre GRCh38, `-t8`, sortie jetee (ROADMAP, section « Ou le fork est reellement moins cher »). Parts des echantillons de travail :

| etage | part | va au GPU ? |
|---|---|---|
| noyau de mate rescue (`matesw`) | 39,2 % | **oui** |
| DP principal, extension (`batched`) | 20,0 % | **oui** |
| tri + dedup des regions | 15,1 % | **non**, permutation observable en sortie |
| seeding (recherche arriere FM) | 6,7 % | non, deja 2,5x moins cher que le fork |
| resolution du suffix array | 3,9 % | non |
| chainage | 5,1 % | non |
| fenetre de reference | 1,1 % | non |

**59,2 % du travail est du Smith-Waterman a gaps affines sur des entiers**, et c'est exactement ce qu'un GPU fait bien.

Course de reference sur cette charge : **28,60 s de mur, 220,21 s de CPU**. Donc :

- DP = 0,592 x 220,21 = **130,4 s CPU**
- le reste = **89,8 s CPU**
- si le DP devient gratuit : 220,21 s tombe a 89,8 s, soit **2,45x en CPU**, et le mur passe de 28,60 s a **~11,7 s**.

C'est le plafond. Aucun GPU au monde ne va au-dela sans qu'on touche aussi au tri et au seeding.

### 2. Combien de GPU faut-il pour atteindre ce plafond ? Bien moins qu'on ne croit

Pour que le DP disparaisse sous le reste, le GPU doit finir 130,4 s CPU de cellules pendant les ~11,7 s de mur que le CPU passe sur le reste.

Le compte de cellules depend de la convention : a 10,4 Gcell/s/thread (mesure `BWA4_MATESW_TIME`) cela fait ~1,3 T cellules, alors que #50 chiffre l'etage rescue seul a 381 G cellules sur sa charge. Selon la convention, il faut donc au GPU entre **33 et 116 Gcell/s soutenus**.

Ce qui existe :

| materiel | debit publie | ce qu'il calcule |
|---|---|---|
| notre CPU, M4 Max, 12 P-cores | ~125 Gcell/s | score, te, qe, score2, te2 |
| **M4 Max GPU integre, 40 coeurs, 5120 ALU a ~1,68 GHz** | **~8,6 T operations entieres/s, soit ~570 Gcell/s a 15 op/cellule en voies scalaires** | a ecrire |
| CUDASW++4.0 sur A100 (half2) | 1,94 TCUPS | **score seul, pas de traceback** |
| CUDASW++4.0 sur L40S (half2) | 5,01 TCUPS | score seul |
| CUDASW++4.0 sur H100 (s16x2 + DPX) | **5,71 TCUPS** | score seul |

Notre contrat de sortie coute ~1,6x le score seul (mesure : 15,75 operations par ligne de 16 cellules contre les 10 de SWIPE, voir ROADMAP « l'etat de l'art »). Meme en divisant encore par deux pour la divergence de longueurs et le cout de lancement :

- **H100 : ~1,8 TCUPS pour notre contrat, soit 15x le besoin maximal.**
- **A100 : ~0,6 TCUPS, soit 5x le besoin.**
- **GPU integre du M4 Max : ~200 a 300 Gcell/s en pratique, soit 2 a 9x le besoin.**

**Conclusion, et c'est le resultat qui compte : meme le GPU integre d'un MacBook suffit a faire disparaitre entierement le DP sous le travail restant du CPU.** Il n'y a pas besoin d'un accelerateur de datacentre pour prendre les 2,45x. Un H100 est surdimensionne d'un facteur 15 : sur une telle machine, l'interet du GPU n'est plus la vitesse du DP mais le fait qu'il libere les 12 coeurs pour le seeding et le tri.

### 3. Ce que l'octet-identite autorise et interdit sur GPU

**Autorise, et c'est plus large qu'attendu.** Chaque job de DP est une **fonction pure** de (query, target, matrice, penalites, w, zdrop, h0). Le resultat ne depend ni de l'ordre d'execution, ni du thread, ni de la machine. Donc :

- repartir les jobs entre CPU et GPU dans n'importe quelle proportion est **sur par construction**. Le partage peut meme etre dynamique et varier d'un lot a l'autre : la sortie ne bouge pas. C'est ce qui rend le co-ordonnancement de #57 gratuit en risque.
- reordonner les jobs (tri par longueur pour reduire la divergence) est legal, ce que le probe `BWA4_EXTEND_SHAPE` documente deja.

**Interdit.**

1. **Aucun flottant, nulle part.** CUDASW++4.0 tire ses 5,01 TCUPS sur L40S en `half2`. Notre recurrence est en entiers satures ; un `half` ne reproduit pas la saturation a 255 ni les egalites exactes. Les seuls types admissibles sont `int32`, les octets satures SIMD-dans-le-registre (`__vaddus4` / `__vsubus4` / `__vmaxu4`), et `s16x2` avec DPX.
2. **Le tri des regions ne va pas sur GPU.** La permutation de `ks_introsort` est observable : `mem_sort_dedup_patch` trie sur `re` seul puis tue, parmi des regions a egalite, celle que le tri a mise en premier. Remplacer par pdqsort change le md5 du SAM, c'est verifie. Un tri GPU (radix, bitonic) change la permutation par definition.
3. **Les egalites d'argmax.** Le noyau met a jour la colonne avec un `>` strict : le **premier** maximum gagne. Tout portage doit refaire ce test dans ce sens.
4. **La boucle de reprise de bande reste au CPU.** `across.rs` execute les rounds `w`, puis `2w`, avec un test d'acceptation sur `max_off` et une requeue des jobs non converges. Le GPU calcule, le CPU decide. Cette structure est deja exactement celle que veut un GPU : deux passes, la seconde sur les survivants.

### 4. La concurrence, pour situer la cible

| outil | annee | gain annonce | base | sortie identique ? |
|---|---|---|---|---|
| **bwa-mem4 aujourd'hui** | 2026 | 1,11x CPU, 1,15x mur | `fg-labs/bwa-mem3` | **oui, octet pour octet** |
| GASAL2 dans BWA-MEM | 2019 | local x20, **total 1,3x** | BWA-MEM CPU | non teste |
| GPU-BWA-MEM (ICS'23) | 2023 | **3,2 a 3,8x** | bwa-mem2 sur EPYC 7662, 64 coeurs | revendique « same results » |
| Parabricks fq2bam | 2025 | 5-8 min contre 14-40 min | bwa-mem2, 8 coeurs | identique **si meme `-K`** |
| minibwa (Heng Li) | 2026 | **2x bwa-mem2**, CPU seul | bwa-mem2 | non, « comparable accuracy » |

Deux lectures.

- GPU-BWA-MEM tire 3,8x en portant **tout** le pipeline, seeding compris (3,7x sur le seeding). **Ce levier ne nous est pas offert** : notre seeding pese 6,7 % contre les 17,0 % du fork, donc le porter vaut au mieux 6,7 %. Leur gain vient en grande partie d'une base que nous avons deja battue.
- Personne dans ce tableau ne tient l'octet-identite comme critere eliminatoire. Parabricks la tient sous condition, GPU-BWA-MEM l'affirme sans methode de validation publiee (« no detailed validation methodology is provided beyond this assertion »). **Notre argument n'est pas « plus vite », c'est « plus vite ET prouve identique ».**

Cible : **2,4x notre binaire actuel, soit ~2,7x bwa-mem2 sur la meme machine avec un seul GPU**, en gardant le md5 du SAM.

### 5. L'ordre des travaux

1. #53: la couture asynchrone dans `SwBackend`, et la mesure de la taille des lots. Prealable a tout.
2. #54: la barriere d'octet-identite pour un backend GPU. A ecrire **avant** le premier noyau, pas apres.
3. #55: backend Metal, Apple Silicon, memoire unifiee. C'est la machine de developpement, et le calcul du 2 dit qu'elle suffit.
4. #56: backend CUDA, octets satures puis DPX `s16x2`.
5. #57: co-ordonnancement CPU+GPU et restructuration du pipeline.

Apres quoi le profil restant est : tri + dedup 37 %, seeding 16 %, chainage 12 %, et le sujet redevient #38 et #50.

### Sources

- [GPU-BWA-MEM, ICS 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10425913/)
- [CUDASW++4.0, BMC Bioinformatics 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11531700/)
- [GASAL2, BMC Bioinformatics 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6815017/)
- [Parabricks fq2bam](https://docs.nvidia.com/clara/parabricks/latest/documentation/tooldocs/man_fq2bam.html)
- [minibwa, arXiv 2606.15357](https://arxiv.org/pdf/2606.15357)
- [Hopper DPX](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
## #53 : perf(gpu): la couture asynchrone dans SwBackend, et mesurer la taille des lots

Prealable a tout backend GPU. Aucune ligne de CUDA ni de Metal n'a de sens avant que ces deux points soient regles.

### Ce qui existe deja, et qui est mieux que prevu

La couture est **deja au bon endroit** :

- `bwa_extend::SwBackend` a deja une methode par lot, `extend_batch(&[ExtendJob], m, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop)`, dont la doc dit deja « le resultat a l'index `k` doit egaler `extend(jobs[k]...)` exactement » et dont les exemples de `name()` citent litteralement `"metal"` ;
- `crates/bwa-mem/src/across.rs` batche deja les extensions **a travers les reads** (pas par read), avec une boucle de rounds `w` puis `2w`, un tri par longueur decroissante, et une requeue des jobs non converges ;
- le mate rescue passe par `bwa_neon::batched_ksw_align2(&[KswJob], ...)`, un appel par round pour tout le lot de paires (`pe.rs:1090`) ;
- les barrieres d'acceptation existent : `assert_backend_matches_scalar`, `assert_backend_batch_matches_scalar`, `matesw_equals_scalar`.

Autrement dit : **il n'y a pas de refonte du pipeline a faire pour brancher un GPU.** Il y a une couture a rendre asynchrone.

### Etape 0 : mesurer la taille des lots (a faire en premier, c'est une demi-journee)

Un GPU ne rentabilise un lancement de noyau qu'a partir de quelques milliers de jobs. Personne ici ne sait combien de jobs passent par appel.

`BWA4_EXTEND_SHAPE` compte les jobs et les cellules mais **pas les appels**, donc pas la moyenne par appel, ni la distribution. Il faut :

1. ajouter un compteur `CALLS` a `extend_shape` et un histogramme grossier de `jobs.len()` (puissances de deux) ;
2. le meme pour `batched_ksw_align2` ;
3. faire tourner sur 200k paires GIAB reelles, `-t8`.

Le resultat decide de l'architecture :

- **si les lots font >= 4k jobs** : un lancement de noyau par appel suffit, la couture reste presque synchrone, et le travail est petit ;
- **si les lots font quelques centaines** : il faut agreger les lots de plusieurs threads CPU dans une file unique cote GPU, ce qui est un travail nettement plus gros et change le contrat de `SwBackend`.

Ne rien decider avant ce chiffre.

### Etape 1 : la couture asynchrone

Le contrat actuel est bloquant : `extend_batch` retourne les resultats. Un GPU veut recouvrir. Proposition, en gardant l'ancien contrat intact :

```rust
/// Un lot soumis, pas encore recolte.
pub struct BatchTicket(u64);

pub trait SwBackendAsync: SwBackend {
    /// Soumet le lot, rend la main immediatement.
    fn submit(&self, jobs: &[ExtendJob], /* ... memes parametres partages ... */) -> BatchTicket;
    /// Bloque jusqu'a ce que le lot soit pret.
    fn collect(&self, t: BatchTicket) -> Vec<ExtendResult>;
    /// Profondeur de file utile, pour que l'appelant sache combien de lots garder en vol.
    fn queue_depth(&self) -> usize { 1 }
}
```

`extend_batch` devient `collect(submit(..))` pour tout backend, ce qui garde les backends CPU inchanges et les tests existants valides tels quels.

Points a trancher dans l'implementation, pas dans le contrat :

- **memoire.** Les `ExtendJob` portent des `&[u8]` vers query et target. Un GPU veut un buffer plat unique, avec des offsets. La conversion doit se faire une fois par lot dans le backend, pas dans l'appelant.
- **Apple.** En memoire unifiee, ce buffer plat peut etre alloue en `MTLBuffer` partage et rempli directement par le CPU : **zero copie**. C'est un avantage structurel du M4 Max sur un GPU PCIe, et il faut que le contrat le permette (donc : le backend possede l'allocation, l'appelant ecrit dedans).
- **persistance.** Allouer et liberer les buffers a chaque lot tue le gain. Les buffers doivent etre reutilises, dimensionnes sur le plus gros lot vu.

### Etape 2 : le crate

Nouveau crate `bwa-gpu`, sans dependance a CUDA ni a Metal :

- le trait `SwBackendAsync`, le `BatchTicket`, la mise a plat des jobs, la logique de file ;
- la selection de backend a l'execution (`BWA4_GPU=metal|cuda|off`), avec **repli CPU silencieux** si le GPU est absent ou si l'initialisation echoue ;
- les backends concrets vivent dans `bwa-metal` et `bwa-cuda`, derriere des features Cargo desactivees par defaut, pour que `cargo build` reste sans dependance systeme.

### Critere d'acceptation

- `cargo build` sans feature GPU : rien ne change, aucune dependance nouvelle.
- Avec `--features metal` sur une machine sans GPU utilisable : repli CPU, meme md5 SAM, aucun message d'erreur bloquant.
- L'histogramme de taille de lot est publie dans le ROADMAP.

Chapeau : #52. Barriere d'identite : #54.
## #54 : test(gpu): la barriere d'octet-identite pour un backend GPU, a ecrire avant le premier noyau

A ecrire **avant** le premier noyau GPU, pas apres. Un noyau GPU qui donne 99,99 % des jobs justes est un noyau faux : sur 1 M paires, 0,01 % fait des centaines d'enregistrements SAM differents, et le md5 tombe.

Le precedent le dit : GPU-BWA-MEM (ICS 2023) affirme « our GPU code yields the same results as the original BWA-MEM code » et **ne publie aucune methode de validation**. C'est exactement ce que nous ne voulons pas faire.

### Ce qui existe et qu'il suffit d'etendre

- `bwa_extend::assert_backend_matches_scalar<B>` : compare `backend.extend` a `ksw_extend2` champ par champ, sur des jeux generes.
- `bwa_extend::assert_backend_batch_matches_scalar<B>` : idem pour `extend_batch`, avec les rounds `w` / `2w`.
- `matesw_equals_scalar` : la barriere du noyau de rescue.

Un backend GPU doit passer les trois **sans modification**, ce qui est deja la moitie du travail. Le reste est ce que ces barrieres ne couvrent pas.

### Les sept pieges specifiques au GPU

Chacun merite un test nomme.

#### 1. Aucun flottant

CUDASW++4.0 atteint 5,01 TCUPS sur L40S en `half2` et 1,94 sur A100 en `half2`. **Interdit ici.** Notre recurrence est en entiers satures ; un `half` a 11 bits de mantisse reproduit les scores usuels par accident et casse sur les egalites et la saturation. Types admissibles, et rien d'autre :

- `int32` par voie (simple, lent) ;
- octets non signes satures dans un registre 32 bits : `__vaddus4`, `__vsubus4`, `__vmaxu4` (CUDA, disponible depuis Kepler), 4 cellules par thread ;
- `s16x2` avec DPX sur Hopper et au-dela.

**Test :** un `static_assert` de type dans le noyau, plus une relecture du PTX / de l'IR Metal cherchant toute instruction flottante dans la boucle interne.

#### 2. L'egalite d'argmax : le premier maximum gagne

Le noyau NEON met a jour la colonne par `col = bsl(vcgtq(h, imax), j, col)`, c'est-a-dire un `>` **strict** : a egalite, la colonne deja enregistree est conservee, donc **le premier maximum rencontre gagne**. Un portage qui ecrit `>=` produit des `te`/`qe` differents sur les repetitions parfaites, ou les egalites sont la norme et pas l'exception.

**Test :** un generateur qui produit des cibles periodiques (`ACGTACGTACGT...`) contre des requetes en phase, ou tous les maxima sont a egalite. Comparer `te`, `qe`, `te2` a l'implementation scalaire.

#### 3. La saturation a 255 et le repli i16

Le noyau u8 travaille biaise et sature. Quand il sature, le pipeline **rejoue le job en i16**. Le GPU doit detecter la saturation exactement au meme seuil, sinon il rend un score ecrete la ou le CPU aurait rejoue.

**Test :** jobs construits pour finir a 254, 255 et 256 exactement, dans les deux sens (score et score2).

#### 4. Le codage de N

Le noyau u8 encode la cible N en **12** et non en 4 (`N_TARGET`), avec une table de substitution indexee par XOR ou les creneaux 8 et 12..16 valent `bias - 1`. Ce n'est pas un detail de perf : c'est le codage. Un portage qui reprend « 4 = N » et applique une reparation par `bsl` donne les memes scores mais il faut le prouver, et l'histoire de ce fichier dit que la reparation a ete supprimee parce qu'elle etait fausse dans un cas.

**Test :** les generateurs d'equivalence scalaire tirent deja `% 5` et non `% 4`, donc N sort. **Ne pas revenir a `% 4` dans les tests GPU.**

#### 5. `max_off` et la boucle de rounds

`across.rs` relance a `2w` les jobs dont `max_off` depasse le test d'acceptation. Le GPU calcule `max_off`, le CPU decide. Si `max_off` differe d'une unite, ce n'est pas un score qui change, c'est **un job qui est rejoue ou non**, et donc potentiellement un `w` different dans le `reg.w` ecrit en sortie.

**Test :** compter, sur 200k paires reelles, le nombre de jobs requeues au round 1 et au round 2 en CPU et en GPU. Les trois nombres doivent etre egaux, pas proches.

#### 6. La divergence de longueur ne doit pas changer le resultat

Un noyau GPU va grouper les jobs par longueur (CUDASW++4.0 fait des buckets de 64). Le regroupement est legal (chaque job est une fonction pure) **a condition que le remplissage des voies mortes n'influence pas les voies vivantes** : pas de reduction inter-voies, pas de `simd_max` a travers le groupe, sauf a prouver que la voie morte porte un neutre.

**Test :** executer le meme lot dans dix ordres differents tires au hasard et exiger des resultats identiques job par job.

#### 7. Le partage CPU/GPU ne doit rien changer

C'est le test qui protege le co-ordonnancement.

**Test :** une variable `BWA4_GPU_SPLIT=0.0 | 0.25 | 0.5 | 0.75 | 1.0` qui fixe la fraction de jobs envoyee au GPU, et un gate qui exige **le meme md5 SAM pour les cinq valeurs** sur 200k paires reelles. Si ce test passe, le co-ordonnancement dynamique est sur par construction et n'a plus besoin d'etre re-prouve.

### Le gate d'integration

En plus des tests unitaires, ajouter a `scripts/` un `gpu_parity.sh` sur le modele de `oracle_diff.sh` :

1. meme binaire, `BWA4_GPU=off` puis `BWA4_GPU=metal` (ou `cuda`) ;
2. 200k paires GIAB reelles, PE, `-t8`, `-K` fixe ;
3. md5 des deux SAM, plus un `bwa-diff` en cas d'ecart pour localiser le premier enregistrement divergent.

Le gate doit tourner dans la CI sur la machine qui a le GPU, et bloquer la fusion.

### Ce que ce gate ne couvre pas

Rien de tout cela ne teste le **non-determinisme materiel** : un GPU qui a une erreur memoire non corrigee, ou un driver qui reordonne. C'est pour ca que `gpu_parity.sh` doit tourner **trois fois** et exiger trois md5 egaux, pas seulement un md5 egal a la reference.

Chapeau : #52.
## #55 : perf(gpu): backend Metal, Apple Silicon, memoire unifiee et zero copie

Le premier backend GPU a ecrire, et la raison est arithmetique, pas sentimentale : **le GPU integre du M4 Max suffit deja a prendre la totalite des 2,45x**, et il n'y a ni PCIe ni copie a payer.

### Le calcul qui justifie de commencer par la

Ce qu'il faut absorber : 130,4 s CPU de DP par million de paires, pendant les ~11,7 s de mur que le CPU passe sur le reste. Soit **33 a 116 Gcell/s soutenus** selon la convention de comptage (voir le chapeau).

Ce que la machine offre : M4 Max, **40 coeurs GPU, 5120 ALU**, 17,2 TFLOPS FP32 donc ~1,68 GHz, **546 GB/s** de bande passante unifiee.

- debit d'operations entieres : 5120 x 1,68 GHz = **~8,6 T op/s** ;
- notre cout par cellule en voies scalaires : ~15 operations ;
- plafond theorique : **~570 Gcell/s** ;
- en tablant sur 40 % d'efficacite reelle : **~230 Gcell/s**, soit **2 a 7x le besoin**.

Le CPU, lui, plafonne a ~125 Gcell/s (12 P-cores x 10,4). **Le GPU integre vaut donc environ deux fois le CPU entier sur cet etage**, et il est libre pendant que le CPU seede.

Bande passante : le DP est en registres, pas en memoire. Ce qui traverse la memoire, ce sont les sequences et les resultats, quelques centaines de Mo par lot au pire. 546 GB/s n'est pas la contrainte.

### L'avantage structurel : memoire unifiee

Sur un GPU PCIe, il faut copier query et target vers le GPU et les resultats en retour. GPU-BWA-MEM a du construire un systeme a deux etages (super-batches et mini-batches) avec des transferts asynchrones pour arriver a « zero overhead ».

Sur Apple Silicon **cette moitie du projet n'existe pas**. Un `MTLBuffer` en mode `.storageModeShared` est ecrit par le CPU et lu par le GPU sans copie. La mise a plat des jobs se fait directement dans le buffer final.

### Ce qu'il faut verifier avant d'ecrire une ligne (etape 0)

**Metal Shading Language expose-t-il l'arithmetique entiere saturee ?** Toute la conception en depend, et je ne l'affirme pas :

1. y a-t-il un equivalent de `add_sat` / `sub_sat` d'OpenCL en MSL ? Si oui, un `uchar4` donne **4 cellules par thread** et le cout tombe a ~4 operations par cellule, soit un plafond a ~2 Tcell/s ;
2. sinon, emuler par `min`/`max` sur `ushort4` (deux operations de plus par saturation) ou travailler en `int` a une cellule par voie ;
3. `simd_shuffle` et les operations de simdgroup existent, mais **on n'en veut pas** : notre disposition est inter-sequences, une alignement par thread, donc aucune communication entre voies. C'est precisement ce qui rend le portage simple et ce qui evite le piege de GPU-BWA-MEM, qui a mis un warp par read en front d'onde et n'a tire que **2x** sur le Smith-Waterman.

Ecrire un microbenchmark MSL autonome qui mesure ces trois variantes **avant** de toucher au pipeline. Le precedent est bon : le microbenchmark `ceiling` a corrige un plafond faux de 56 Gcell/s en 16 Gcell/s cote CPU.

### La disposition, et pourquoi elle est deja bonne

Notre noyau est **inter-sequences** : une voie = un alignement complet, aucune dependance entre voies. Sur GPU cela devient **un thread = un alignement**, ce qui est le mapping ideal :

- zero divergence a l'interieur d'un alignement ;
- la seule divergence est la difference de longueur entre les alignements d'un meme warp, et elle se traite par tri, exactement comme `BWA4_EXTEND_SHAPE` le mesure deja cote CPU ;
- aucune reduction inter-voies, donc aucun risque d'identite (voir le piege 6 du gate).

CUDASW++4.0 fait la meme chose avec des buckets de longueur de 64 et tire ses TCUPS de la. Nous avons deja l'infrastructure de tri.

### Le portage lui-meme

Deux noyaux, dans cet ordre :

1. **`matesw` (39,2 %)** : `fwd_local_sw_neon_u8` est la specification. La version quadruplet (`BWA4_RESCUE_ROWQUAD`, 4 lignes en vol) n'a pas de sens sur GPU, ou le parallelisme vient du nombre de threads ; **porter la version simple ligne**, avec la table de substitution XOR et `N_TARGET = 12` telle quelle. La table de 16 octets tient dans un registre ou en memoire constante.
2. **`batched_extend` (20,0 %)** : plus complexe (zdrop, `max_off`, `gscore`/`gtle`, les rounds). Le porter apres, une fois le rescue prouve identique.

### Etapes

- [ ] microbenchmark MSL : saturation entiere disponible ? cellules par thread ? debit reel en Gcell/s
- [ ] crate `bwa-metal`, feature Cargo, repli CPU silencieux
- [ ] noyau `matesw` u8, une ligne, un alignement par thread
- [ ] passer `matesw_equals_scalar` et le gate `gpu_parity.sh`
- [ ] mesurer le mur et le CPU sur 1 M paires, `-t8`, A/B entrelace contre `BWA4_GPU=off`
- [ ] puis seulement : `batched_extend`

### Attendu

Si le microbenchmark confirme >= 150 Gcell/s, le rescue seul (39,2 %) doit rendre **~1,6x en CPU** et davantage en mur puisque le GPU travaille pendant que le CPU seede. Les deux etages ensemble : **jusqu'a 2,45x**.

Si le microbenchmark rend moins de 60 Gcell/s, le calcul ci-dessus est faux et il faut le dire dans le ROADMAP avant d'aller plus loin.

Chapeau : #52. Couture : #53. Gate : #54.
## #56 : perf(gpu): backend CUDA, octets satures __vaddus4 puis DPX s16x2 sur Hopper

Le backend qui compte pour un serveur, et le seul qui existe la ou tourne la vraie production WGS.

### Le point de depart : CUDA a exactement ce qu'il faut, depuis Kepler

Notre noyau de rescue travaille en **u8 biaise sature**. CUDA expose depuis longtemps l'arithmetique SIMD-dans-le-registre sur quatre octets :

- `__vaddus4(a, b)` : quatre additions non signees **saturees** ;
- `__vsubus4(a, b)` : quatre soustractions non signees **saturees** ;
- `__vmaxu4(a, b)` : quatre maxima non signes ;
- `__vcmpgtu4(a, b)` : quatre comparaisons, masque par octet, ce qu'il faut pour l'argmax.

C'est **la transposition exacte** de `vqaddq_u8` / `vqsubq_u8` / `vmaxq_u8` / `vcgtq_u8`. Le portage de `fwd_local_sw_neon_u8` est donc mecanique, avec **4 cellules par thread** au lieu de 16 par vecteur.

Cout : ~15 operations pour 4 cellules, soit **~3,75 operations par cellule**. Sur un A100 (6912 coeurs CUDA a 1,41 GHz, ~9,7 T op entieres/s) cela donne un plafond de **~2,6 Tcell/s**, et sur H100 davantage. Le besoin est de 33 a 116 Gcell/s. **Marge d'un facteur 20 a 80.**

Ce que ces chiffres veulent dire concretement : sur une machine a GPU discret, **le DP cesse d'exister au profil**. L'interet du GPU n'est plus sa vitesse mais le fait qu'il rend 59,2 % du CPU au seeding, au chainage et au tri.

### DPX sur Hopper : utile, mais pas pour la raison qu'on croit

H100 ajoute les instructions DPX, dont `__vimax3_s16x2_relu` : maximum de trois valeurs, deux voies 16 bits, avec ReLU, en une instruction. C'est precisement le motif `max(diag, e, f)` puis clamp a zero de Smith-Waterman local. CUDASW++4.0 passe de 1,94 TCUPS (A100, half2) a **5,71 TCUPS (H100, s16x2 + DPX)**, et note que « DPX s16x2 doubles the performance compared to DPX s32 ».

Pour nous, DPX interesse **le noyau i16, pas le noyau u8** :

- le chemin u8 est deja couvert par `__v*u4`, avec 4 cellules par instruction contre 2 pour `s16x2` ;
- le chemin i16 est celui qu'on emprunte quand le u8 sature, et c'est la que `__vimax3_s16x2_relu` remplace deux `max` et un `max(0, .)`.

Attention : notre recurrence n'a **pas** de clamp a zero explicite dans le noyau u8, elle utilise la saturation basse a zero de l'arithmetique non signee. En s16 signe, le ReLU de DPX rend ce clamp explicite. **Verifier que les deux donnent le meme resultat sur les scores negatifs** avant de s'en servir : c'est le piege 3 du gate.

Interdit : `half2`. C'est ainsi que CUDASW++4.0 tire ses 5,01 TCUPS sur L40S, et c'est du flottant. Voir le gate.

### Ce qu'on ne refait pas

GPU-BWA-MEM (ICS 2023) a mis **un warp par read** et calcule la matrice en front d'onde diagonal : « in iteration i, thread t computes cell H[t, i - t] ». Resultat : **2x seulement** sur le Smith-Waterman, contre 3,7x sur le seeding.

Notre disposition inter-sequences donne **un thread par alignement**, sans communication entre threads, sans front d'onde, sans divergence a l'interieur d'un alignement. C'est structurellement meilleur pour des reads courts, et c'est deja ce que fait notre code CPU. Ne pas regresser vers le front d'onde.

### Les transferts

Contrairement au backend Metal, il y a un PCIe a traverser. Ce que dit le precedent : GPU-BWA-MEM atteint « zero overhead » avec deux etages (super-batches, mini-batches) et des transferts asynchrones recouvrant le calcul.

Pour nous, plus simple, parce que le lot est deja constitue par `across.rs` et `pe.rs` :

- buffers epingles (`cudaHostAlloc`), reutilises, dimensionnes sur le plus gros lot vu ;
- mise a plat des sequences en un buffer unique avec offsets, faite pendant que le lot precedent calcule ;
- deux flux CUDA au minimum, pour recouvrir copie et calcul ;
- la profondeur de file vient de `SwBackendAsync::queue_depth`.

### Etapes

- [ ] crate `bwa-cuda`, feature Cargo, detection a l'execution, repli CPU silencieux
- [ ] noyau `matesw` u8 en `__vaddus4` / `__vsubus4` / `__vmaxu4` / `__vcmpgtu4`, 4 cellules par thread
- [ ] tri par longueur et buckets, sur le modele des buckets de 64 de CUDASW++4.0
- [ ] gate `gpu_parity.sh` sur une machine x86 + NVIDIA
- [ ] mesure A/B contre `BWA4_GPU=off`, et contre bwa-mem2 sur la meme machine
- [ ] chemin i16 avec `__vimax3_s16x2_relu` si Hopper detecte, apres verification du ReLU
- [ ] `batched_extend`

### Materiel

Aucune de ces mesures n'est faisable ici. Meme situation que #44 : ce ticket demande une machine. Le workflow `bench-x86.yml` sait deja lancer un A/B de noyau sur un runner x86 ; il faudra un runner avec GPU.

Chapeau : #52. Gate : #54.

### Sources

- [CUDASW++4.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC11531700/)
- [GPU-BWA-MEM, ICS 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10425913/)
- [Hopper DPX](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
## #57 : perf(gpu): co-ordonnancement CPU+GPU, le partage des jobs est gratuit en risque

Le dernier etage, et celui qui transforme « le GPU calcule le DP » en « la machine entiere travaille en permanence ». Sans lui, un backend GPU donne un CPU qui attend puis un GPU qui attend, en alternance, et le gain reel est la moitie du gain theorique.

### Le theoreme qui rend tout ceci gratuit en risque

**Chaque job de DP est une fonction pure** de `(query, target, mat, o_del, e_del, o_ins, e_ins, w, end_bonus, zdrop, h0)`. Son resultat ne depend ni du thread qui l'execute, ni de l'ordre, ni de l'ISA. Donc :

> N'importe quelle repartition des jobs entre CPU et GPU, y compris une repartition qui change a chaque lot et qui depend de la charge de la machine, produit **exactement la meme sortie SAM**.

C'est ce qui distingue ce projet d'un ordonnancement classique. Il n'y a pas de compromis vitesse contre reproductibilite : le partage peut etre aussi opportuniste qu'on veut. Le gate `BWA4_GPU_SPLIT` de #54 existe pour prouver ce theoreme une fois, apres quoi il n'est plus a re-prouver.

### Le probleme a resoudre

Aujourd'hui, un thread traite un read de bout en bout : seed, chain, extend, tri, rescue, tri, SAM. Si le DP part au GPU, le thread **bloque** pendant que le GPU calcule, et l'accelerateur ne sert qu'a rendre une etape plus rapide, pas a doubler la machine.

Ce qu'on veut : pendant que le GPU aligne le lot N, les threads CPU seedent et chainent le lot N+1.

### Trois niveaux, par cout croissant

#### Niveau 1 : le partage statique (une journee, prend l'essentiel)

Chaque appel a `extend_batch` / `batched_ksw_align2` coupe le lot en deux : une fraction `f` au GPU, `1 - f` executee par le thread appelant sur le CPU pendant que le GPU calcule, puis on rejoint.

- pas de restructuration du pipeline ;
- le CPU ne dort jamais ;
- `f` est un reglage, mesure une fois par machine.

Optimum theorique : `f = D_gpu / (D_gpu + D_cpu_thread)`. Sur M4 Max, avec ~230 Gcell/s de GPU contre ~10,4 par thread, `f` vaut ~0,96 pour un seul thread appelant, mais **tous les threads appellent**, donc la file GPU voit 8 a 12 lots simultanes et `f` reel descend vers ~0,7. C'est un chiffre a mesurer, pas a deriver.

#### Niveau 2 : le partage adaptatif (petit, et il rend le reglage inutile)

Remplacer `f` fixe par une mesure glissante : chaque backend publie son debit observe en cellules par seconde, et le partage suit le ratio. Deux avantages :

- pas de reglage par machine, ce qui compte pour un binaire distribue ;
- robuste au partage du GPU avec un autre processus, cas courant sur un noeud de calcul.

Une moyenne mobile exponentielle sur les 32 derniers lots suffit. Le fait que le partage varie **ne change pas la sortie** (voir le theoreme).

#### Niveau 3 : le pipeline decale (gros, a faire seulement si le niveau 2 laisse le GPU sous-employe)

Decoupler la production de jobs de leur consommation : les threads CPU poussent les jobs de DP dans une file globale et **continuent** sur le read suivant sans attendre ; un collecteur reprend les reads quand leurs resultats arrivent.

Cout reel : `mem_align1_core` devient une machine a etats, ce qui touche l'ordre des enregistrements SAM. **L'ordre de sortie fait partie du contrat d'octet-identite**, donc il faut un tampon de reordonnancement par lot, ce qui augmente la RAM par lot, deja pointee comme 1,5x celle du fork a `-t16` (#25).

Ne pas commencer par la. Le niveau 2 suffit probablement, parce que la structure de `across.rs` (rounds `w` puis `2w`, requeue) donne deja deux vagues par lot.

### L'interaction avec la RAM (#25)

Un backend GPU ajoute des buffers plats persistants dimensionnes sur le plus gros lot. Sur une machine a memoire unifiee, cette memoire est **prise sur la meme reserve** que le reste. #25 dit que nous consommons deja 1,5x le fork a `-t16`. Mesurer la RAM du chemin GPU dans le meme harnais que #25, pas separement.

### Critere d'acceptation

- md5 SAM identique pour `BWA4_GPU_SPLIT` a 0,0 / 0,25 / 0,5 / 0,75 / 1,0, trois fois de suite ;
- occupation GPU mesuree pendant une course de 1 M paires (Instruments sur Apple, `nvidia-smi dmon` sur NVIDIA) : viser > 60 % ;
- gain en mur et en CPU mesure en A/B entrelace, medianes sur 5 courses, contre `BWA4_GPU=off` ;
- le tout consigne dans le ROADMAP avec les chiffres bruts, comme les autres campagnes.

### Ce que ce ticket ne fait pas

Il ne touche pas au tri des regions (15,1 %, permutation observable, voir #38), ni au seeding (6,7 %, deja 2,5x moins cher que le fork). Une fois le DP parti au GPU, **ces deux etages deviennent 37 % et 16 % du temps restant** et redeviennent les sujets principaux. C'est le bon moment pour reouvrir #38 et #50.

Chapeau : #52.
