# Divergences connues vs oracle bwa-mem2 2.3

Suivi de la traine de parite. Chaque entree : champ concerne, cause, statut, plan.

## Acceptees (par conception)

- **`@PG`** : notre sortie emet `ID:bwa-mem3 PN:bwa-mem3 VN:<ver> CL:<notre argv>`, l'oracle emet
  `bwa-mem2`. Exclu du gate d'octet-identite (on compare `@SQ` + lignes d'alignement). Decision finale
  (spoof eventuel de `@PG`) reportee en fin de projet.

## En cours

- **Parité des régions sous-optimales (`XS:i`, et indirectement `sub_n` donc parfois MAPQ), SE et
  PE.** Sur 5000 reads/paires wgsim (chr20 2 Mb), état actuel (après portage `XA`) :
  - **SE** : byte-identique sur **4660/5000** lignes ; tail = **317 `XS`-seul + 23 MAPQ**.
  - **PE** : byte-identique sur **9366/10000** enregistrements ; tail = **626 `XS`-seul + 6 MAPQ
    + 2 mate-rescue**. Cœur (FLAG/POS/CIGAR/RNEXT/PNEXT/TLEN/MC) quasi 100%, `mem_pestat` identique
    bit-à-bit, `mem_pair` + MAPQ combinée vérifiés.

  **Cause racine unique** (diagnostiquée, catégorisée) : notre ensemble de régions **sous-optimales
  chevauchant le primaire sur la query** diffère légèrement de celui de l'oracle. `XS = sub` =
  score de la meilleure telle région (`mem_mark_primary_se_core`), et `sub_n` = nombre de ces
  régions à ≤ 7 pts du primaire (d'où l'écart MAPQ). Les 626 `XS` + 6 MAPQ PE (632/634 du tail) en
  découlent tous. Le primaire (AS/POS/CIGAR) est **toujours** byte-identique ; seule la
  sous-optimale diffère. Exemple SE `_514496` : oracle `XS`=44, nous 26 (on ne produit qu'un seed
  26 bp à un autre locus). Écarts **dans les deux sens** (369 oracle>nous, 257 nous>oracle) → pas un
  biais de bande SW mais la **complétude/parité exacte de la cascade seed→chain→extension des
  sous-optimales**.
  **Impact** : `XS` purement cosmétique (aucun effet POS/CIGAR/MAPQ) ; les 6 MAPQ sont réels mais
  même cause. **Blocage diagnostic** : pinpointer exige la liste interne des régions de l'oracle
  (avant `mem_mark_primary_se`), donc un **build instrumenté bit-identique** de l'oracle (yak-shave
  sse2neon arm64). `bwa-mem2` n'a pas de sous-commande `fastmap` (le patch porte sur le driver
  `mem`), donc pas de dump SMEM direct.

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
