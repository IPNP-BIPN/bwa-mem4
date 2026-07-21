# Divergences connues vs oracle bwa-mem2 2.3

Suivi de la traine de parite. Chaque entree : champ concerne, cause, statut, plan.

## Acceptees (par conception)

- **`@PG` : DECIDE (4.0.0). Nous emettons notre propre identite, definitivement.** Notre sortie
  emet `ID:bwa-mem4 PN:bwa-mem4 VN:<ver> CL:<notre argv>`, l'oracle emet `bwa-mem2`. Exclu du gate
  d'octet-identite (on compare `@SQ` + les lignes d'alignement).

  L'option alternative etait de se faire passer pour `bwa-mem2` dans le `@PG`, ce qui aurait rendu
  la sortie octet-identique **y compris l'en-tete**. Rejete : le `@PG` est le seul endroit d'un BAM
  qui enregistre quel binaire a produit les donnees, et c'est precisement ce que lisent les audits
  de provenance et les pipelines de reproductibilite. Usurper ce champ rendrait tout BAM produit
  ici intracable, pour gagner une ligne d'en-tete que le gate exclut de toute facon. Un aligneur
  qui ment sur sa propre identite est un probleme, pas une fonctionnalite.

  Consequence assumee : `diff` brut entre une sortie bwa-mem2 et une sortie bwa-mem4 montrera
  toujours cette ligne. Tous les scripts du depot filtrent `^@PG` pour cette raison.

## Résolu (phase 8, via oracle instrumenté)

- **Parité des régions sous-optimales (`XS`) + MAPQ, SE et PE : RÉSOLU.** Sur 5000 reads/paires
  wgsim (chr20 2 Mb) : **SE 5000/5000** (100%), **PE 9999/10000**. Deux causes racines, trouvées en
  comparant à un **oracle bit-identique instrumenté** (rebuild sse2neon v1.8.0 + safestringlib
  v1.2.0 + les 2 patches, recette Homebrew reproduite dans `scratchpad/oracle-build` ; sortie
  byte-identique au binaire installé hors `@PG`) qui dumpe SMEMs, chaînes pré/post-`mem_chain_flt`,
  régions post-`mark_primary` et les entrées de `mem_approx_mapq_se` :

  1. **Ordre des chaînes + tri instable dans `mem_chain_flt`** (résout 340 SE + 622 PE, tous des
     régions sous-optimales à un locus différent). bwa-mem2 stocke les chaînes dans un kbtree keyé
     par `pos` et les parcourt **in-order** (donc `pos` croissant) avant un `ks_introsort` **instable**
     (comparateur `flt_lt = a.w > b.w`). Pour deux chaînes de **poids égal** qui se chevauchent sur
     la query, le gagnant dépend (a) de l'ordre d'entrée et (b) de la permutation exacte de
     l'introsort. On construisait les chaînes en ordre d'occurrence + tri **stable** → on gardait
     l'autre locus. Fix : `build_chains` trie les chaînes par `pos`, et `mem_chain_flt` utilise un
     portage fidèle de `ks_introsort` (quicksort médiane-de-3 + fallback combsort + insertsort final)
     dans `crates/bwa-chain/src/lib.rs`. Le primaire (poids max) est toujours choisi identiquement.

  2. **`mapQ_coef_fac` est un `int` dans bwa-mem2** (résout 23 SE + 6 PE, MAPQ seule, régions
     identiques). `o->mapQ_coef_fac = (int)log(50) = 3`, pas `3.912`. On stockait le flottant
     `ln(50)` → MAPQ trop élevée sur les cas limites (ex. `_96f` : oracle 8, nous 15). Fix :
     `MemOpt::mapq_coef_fac = (50.0_f64.ln() as i32) as f64` dans `crates/bwa-core/src/opt.rs`.

  Vérifié sans régression, `-t8` == `-t1` byte-identique (hors `@PG`).

## En cours (résidu)

- **`-a` + contigs ALT : RESOLU.** Le residu de 23 enregistrements secondaires manquants sur
  200 026 est corrige. Cause racine : `mem_reg2sam` jetait **tous** les secondaires sans condition
  (`if p.secondary >= 0 { continue }`), alors que le C ne les jette que si `p->is_alt ||
  !(opt->flag&MEM_F_ALL)` (`bwamem.cpp:1541`). Le commentaire en place annoncait lui-meme que le
  support de `-a` restait a faire sur ce chemin.

  Ce n'etait donc **pas** un bug ALT : c'etait `-a` non supporte dans le chemin `no_pairing` du
  paired-end. Les contigs ALT n'ont fait que produire des reads qui empruntent ce chemin (mate non
  mappe) avec des regions secondaires, ce qu'aucun jeu de test precedent ne faisait.

  Le test de la drop-ratio a aussi recu la borne `p->secondary < INT_MAX` du C, qui n'est pas
  cosmetique : la branche ALT de `mem_mark_primary_se` gare les hits ALT a `secondary = INT_MAX`,
  et indexer `a[INT_MAX]` est un panic, pas un octet faux.

  Verifie : reproduction reduite a 4 paires (31 enregistrements de chaque cote), puis GRCh38 complet
  100 000 paires, **les trois cas byte-identiques** : defauts 200 003, `-j` 200 000, `-a` 200 026.

- **1 enregistrement PE (`XS` cosmétique).** `_f3e` mate-2 : oracle `XS:i:33`, nous `44` ; MAPQ=60,
  FLAG/POS/CIGAR/AS/MC identiques. La chaîne sous-optimale est au **même locus** (`rb=3189858`)
  mais avec une **composition de seeds différente** (`seedcov` oracle 50 vs nous 20) sur un locus
  répétitif, d'où une extension SW banded qui termine à un score différent. Cause **distincte** des
  deux ci-dessus (extension, pas chaînage/MAPQ). `XS` purement cosmétique ; non poursuivi pour ne
  pas risquer la parité SE 100 % / extension du primaire.

- **`XA:Z` (hits alternatifs, `mem_gen_alt`) : PORTÉ (phase 8, byte-identique).** `xa_group` +
  `mem_gen_alt` (`crates/bwa-mem/src/alt.rs`), champ `secondary_all` sur `MemAlnReg` (swap
  primaire/secondaire PE), tag émis après `SA` dans `mem_aln2sam`. Résout les 37 enregistrements
  `XA_tag` PE (et l'équivalent SE) sans régression. Les rares `XA` encore divergents sont **à
  l'intérieur** d'enregistrements déjà touchés par la cascade sous-optimale (jeu de régions
  différent), pas un bug de `mem_gen_alt`.

- **Mate rescue (`mem_matesw` / `ksw_align2`) : PORTÉ (phase 8, byte-identique).** `ksw_align2`
  scalaire (SW local avec coords début/fin via passe inverse + 2ᵉ meilleur score) dans
  `bwa-extend`, `mem_matesw` + `bns_fetch_seq` dans `pe.rs`, branché dans `mem_sam_pe` avant le
  marquage primaire. Sur paires concordantes toutes les orientations sont « skip » → no-op (pas de
  régression). Résout la paire `_822229` (mate `5S145M` réaligné + `MC`) : PE 9366 → **9369/10000**.
