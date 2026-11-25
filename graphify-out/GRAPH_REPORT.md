# Graph Report - distracted-ramanujan-d9c924  (2026-11-25)

## Corpus Check
- 160 files · ~684,751 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 17 file(s) not represented in the graph (top: (none) 4, .toml 2, .log 2)

## Summary
- 3796 nodes · 6914 edges · 297 communities (257 shown, 40 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 224 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d604340a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- bwa-seed/src/lib.rs
- bwa-chain/src/lib.rs
- batched.rs
- shape_fuzz.py
- alloc_probe.rs
- Roadmap
- rescue_band.rs
- lisa.rs
- rescue_prune.rs
- pe.rs
- super
- cmd_longread.rs
- cmd_mem.rs
- fmindex.rs
- steps
- matesw.rs
- stage_time.rs
- steps
- bwa-mem/src/lib.rs
- bwa-extend/src/lib.rs
- primary.rs
- with
- steps
- cigar.rs
- with
- across.rs
- steps
- fwd_local_sw_batch
- FmIndex
- ExtendJob
- sw.rs
- bntseq.rs
- run
- fastq.rs
- align_read
- build.rs
- bwa-mem/src/meth.rs
- global_wave.rs
- emit.rs
- MemAlnReg
- Params
- bwa-neon/src/lib.rs
- __m256i
- mlp_probe.c
- prologue
- Output
- run_staged
- batched_align
- updates
- steps
- CpOcc
- FwdJob
- steps
- sais.rs
- Record
- A machine-independent model of this aligner's thread scaling
- opt.rs
- env
- bwa-mem4
- Le travail litteralement duplique, deux fois dans la meme journee (2026-08-19)
- ksw_align2
- path
- Prioritized levers
- inputs
- with
- bwa-core/src/meth.rs
- .l_pac
- The SIMD DP kernels: what is reachable, and what is already ruled out
- perf-profile-x86.yml
- steps
- require_avx512
- mem_sam_pe
- kbtree.rs
- .load
- PeStat
- Perf levers — measured (M-series, -t1, région 2 Mbp, 500k reads, median of 3)
- bench-x86.yml
- docker_gates.sh
- hugepage.rs
- OnceLock
- sais_rec
- sam.rs
- align_reads_batched_inner
- mem_gen_alt
- bench-arm.yml
- max_threads
- dry_run
- 5. Glossary
- bad_input.rs
- tiny
- seedflt.rs
- lockstep-sweep
- inputs
- steps
- unwrap_multiline_fastq
- Methylation mode (`--meth`)
- bwa-diff/src/lib.rs
- instant
- Deep research: is there an unexplored exact speedup? (2026-07-29)
- env
- with
- jobs
- README.md
- 3. The pipeline, end to end, following one read
- HtsWriter
- ksw_global2_scalar
- star_vs_fm.rs
- FastqReader
- #57 : perf(gpu): co-ordonnancement CPU+GPU, le partage des jobs est gratuit en risque
- Design: speed and RAM, measured against `fg-labs/bwa-mem3`
- inputs
- steps
- CLAUDE.md
- Contributing
- MethOpt
- open_reader
- .run_impl
- Global Constraints
- Catch fg-labs/bwa-mem3 0.16, then pass it
- with
- with
- steps
- with
- release.yml
- env
- Le GPU rouvert : le noyau tournait a 3,6 % de la machine, et la cause n'est pas le noyau (2026-08-29)
- opt_parity.sh
- 2. Why byte-identical output is the goal, and what it costs
- global_icount.rs
- batch_count.rs
- Error
- SuboptimalTracker
- ABuf<T>
- What fg-labs/bwa-mem3 does that we do not
- Les sept pieges specifiques au GPU
- #55 : perf(gpu): backend Metal, Apple Silicon, memoire unifiee et zero copie
- #56 : perf(gpu): backend CUDA, octets satures __vaddus4 puis DPX s16x2 sur Hopper
- Mate rescue: the next lever, priced before it is built
- steps
- endtoend-smallref
- with
- env
- run
- build.sh
- seq_emit_bench.rs
- Tier
- RescueTier
- #52 : meta: hybride CPU+GPU, le plafond est 2,45x et le GPU integre suffit deja
- Literature review: what is left to take, under byte-identity (2026-07-29)
- Where the real-data paired-end time actually goes (2026-07-29 profile)
- The mate-rescue kernel: padding-free column range (BANKED, -9.7% wall)
- What is left to make this faster, at 2026, under byte-identity
- head-to-head.yml
- head-to-head
- build
- check-version
- include
- giab32_headline.sh
- giab_publication.sh
- Apple Silicon memory-subsystem probes
- hybrid_bench.rs
- bwt_binning.rs
- ByteStr
- bench_batch.rs
- RowMax
- Adding a GPU, without giving up byte-identity
- #53 : perf(gpu): la couture asynchrone dans SwBackend, et mesurer la taille des lots
- Part two: the dead ends, and why (issue #49)
- Part one: the ceiling per CPU class, and the plan (issue #51)
- Three-arm head-to-head, genome scale, PE, `-t12`: wgsim says we win, real data says we lose
- runner
- on
- with
- La voie GPU : le plafond est 2,45x, et le GPU integre suffit deja (2026-08-08)
- Le mate rescue, six angles, et le seul qui paie (2026-08-11)
- Le mate rescue, le plus gros poste restant (2026-08-07)
- Le dossier x86 : ou nous perdons, mesure sur le meme hote (2026-08-18)
- Le lecteur n'est plus le goulot, et la sonde mentait (2026-08-11)
- run_in_order
- Chem
- Prune
- Scoring
- score_sort_bench.rs
- Stage
- PassHash
- piece16
- Divergences connues vs oracle bwa-mem2 2.3
- The first side-by-side x86 profile against fg-labs/bwa-mem3 (2026-08-29)
- graviton-selfhosted
- kernel_ab_only
- require_avx512
- require_avx512_e2e
- avx2_arm
- require_avx512
- require_avx512
- isa-and-tiers-x86
- real-reads-x86
- promote
- publish
- #53 etapes 1 et 2 : la couture asynchrone et le crate `bwa-gpu` (2026-08-09)
- #55 : le backend Metal existe, il est octet-identique, et il perd contre un coeur (2026-08-09)
- Les quatre backends de tableau des suffixes, mesures (2026-08-11)
- Le piege de jeu de donnees, et le classement refait (2026-08-11)
- Etat de l'art : personne ne fait mieux a contrat de sortie egal (2026-08-08)
- Le chemin critique du lecteur, et ce que rapidgzip y change (2026-08-11)
- Le regime de colonnes pleinement en bande, trouve par la mesure (2026-08-09)
- Phase 9e : la vague perf (tout octet-identique)
- fork_bench.sh
- giab_happy.sh
- graviton_setup.sh
- unpack_bench.rs
- Bases
- Mode
- Round 3: three structural ideas, all priced before building, all under the floor
- The `-t16` gap was our own P-core cap. Removing it: 1.10x against the fork, 6 of 6
- hugepage-probe
- read_pairs
- runner
- variant_cpu
- read_pairs
- extra_rustflags
- icount
- icount
- knob_sweep
- read_pairs
- read_set
- runner
- meth
- on
- Les chiffres x86, enfin mesures : +17 % sur le noyau de rescue (2026-08-09)
- #45 : `USQADD` et le `qsub(h, oe)` partage, +10,3 % de noyau, -2,7 % de CPU (2026-08-08)
- #46 : les deux passes scalaires ne coutent rien, et pourquoi le microbenchmark disait le contraire (2026-08-08)
- #47 : les colonnes de bourrage ont leur propre corps, et le gain depend de la longueur de lecture (2026-08-08)
- #53 etape 0 : la taille des lots, et elle repond deux choses differentes (2026-08-08)
- #54 : la barriere d'octet-identite GPU, ecrite avant le premier noyau (2026-08-08)
- L'extension sur GPU, branchee et mesuree : -22 % de CPU, et une bascule a -t8 (2026-08-09)
- Le noyau d'extension Metal : ecrit, octet-identique, 6,4x un coeur (2026-08-09)
- Recherche multi-agents sur les noyaux SIMD (2026-08-08)
- Le `uchar4` sur Metal : tente, non concluant, retire (2026-08-09)
- Indexeur : le tableau de suffixes est parallelisable, mais pas par le portage Rust (2026-08-05)
- Le chemin GPU est l'extension, pas le rescue, et les chiffres le disent (2026-08-09)
- Phase 0 de la campagne perf : mesure de reference contre le fork
- Trois questions fermees le meme jour : le gate GIAB, le PGO hors Apple, et l'indexeur parallele (2026-08-18)
- giab30x_bench.sh
- giab30x_pe.sh
- pe_diverge_hunt.sh
- pe_diverge_subset.sh
- pgo.sh
- fwd_local_sw_neon_u8
- AVX-512BW seed extension, and a NEON epilogue that measured null
- Measuring in Nils's actual regime: we are TIED at `-t16`, not ahead
- Vectorising what was left scalar: one kept, one reverted
- Where the `-t16` wall actually goes: `BWA4_STAGE_TIME=1`
- #25 tranche : le resident EST le vivant sous Linux, et le 2x est le double batch en vol (2026-08-18)
- #43 : la table de scores XOR portee sur AVX2 et SSE4.1, et le +18 a +50 % est a revoir (2026-08-09)
- #48B : la pre-passe de substitution repliee, +4 a +5 % de noyau, -0,9 % de CPU (2026-08-08)
- #48C : le masque de bande en espace signe, et ce que le desassemblage x86 dit vraiment (2026-08-09)
- #54 pieges 3 et 5 : la frontiere de saturation, et un `max_off` qui ne decide jamais rien (2026-08-08)
- #55 etape 0 : bloquee, il n'y a pas de compilateur Metal sur cette machine (2026-08-08)
- #55 etape 0 : debloquee sans Xcode, et le GPU tient 330 Gcell/s (2026-08-09)
- Campagne perf CPU et classement contre minibwa (2026-08-09)
- L'oracle tiers couvre enfin les penalites de gap asymetriques (2026-08-12)
- Ou le fork est reellement moins cher, profil contre profil (2026-08-07)
- Entree gzip : ou est le mur, et a combien de cœurs il tombe (2026-08-11)
- alt_parity.sh
- bench_in_container.sh
- fetch_giab30x.sh
- giab_bench.sh
- meth_oracle_diff.sh
- publication_bench.sh
- upstream_repros.sh
- vs_mem2.sh
- Linux arm64 at genome scale, without leaving the Mac
- PE profile on real GIAB: mate rescue is 47%, and it looks nothing like SE
- Two target rows per pass in the rescue kernel (kernel -8.9%, wall -2.2%)
- bioconda/README.md
- avx512_sde.sh
- bench.sh
- cc_no_profile.sh
- check.sh
- index_diff.sh
- longread_parity.sh
- make_testdata.sh
- oracle_diff.sh
- scale_test.sh
- setup_reference.sh
- bwa-mem4
- bwa-mem4-chain
- bwa-mem4-core
- bwa-mem4-diff
- bwa-mem4-extend
- bwa-mem4-index
- bwa-mem4-io
- bwa-mem4-mem
- bwa-mem4-neon
- bwa-mem4-seed

## God Nodes (most connected - your core abstractions)
1. `Roadmap` - 91 edges
2. `FmIndex` - 72 edges
3. `MemAlnReg` - 61 edges
4. `batched_align()` - 30 edges
5. `Params` - 30 edges
6. `mem_sam_pe()` - 28 edges
7. `run()` - 26 edges
8. `align_reads_batched_inner()` - 26 edges
9. `batch_mate_rescue()` - 26 edges
10. `run_pe()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `3.0 Index load` --references--> `run()`  [EXTRACTED]
  ARCHITECTURE.md → crates/bwa-cli/src/cmd_mem.rs
- `3.8 SAM output` --references--> `mem_gen_alt()`  [EXTRACTED]
  ARCHITECTURE.md → crates/bwa-mem/src/alt.rs
- `Where to look to change X` --references--> `build_opt()`  [EXTRACTED]
  ARCHITECTURE.md → crates/bwa-cli/src/cmd_mem.rs
- `Where to look to change X` --references--> `run()`  [EXTRACTED]
  ARCHITECTURE.md → crates/bwa-cli/src/cmd_mem.rs
- `The gates that exist` --references--> `defaults_match_bwa_mem2()`  [EXTRACTED]
  ARCHITECTURE.md → crates/bwa-core/src/opt.rs

## Import Cycles
- 1-file cycle: `crates/bwa-core/src/error.rs -> crates/bwa-core/src/error.rs`
- 1-file cycle: `crates/bwa-cli/src/stage_time.rs -> crates/bwa-cli/src/stage_time.rs`
- 1-file cycle: `crates/bwa-io/src/fastq.rs -> crates/bwa-io/src/fastq.rs`
- 1-file cycle: `crates/bwa-mem/src/pe.rs -> crates/bwa-mem/src/pe.rs`
- 1-file cycle: `crates/bwa-extend/src/global_wave.rs -> crates/bwa-extend/src/global_wave.rs`
- 1-file cycle: `crates/bwa-neon/src/rescue_prune.rs -> crates/bwa-neon/src/rescue_prune.rs`
- 1-file cycle: `crates/bwa-neon/src/batched.rs -> crates/bwa-neon/src/batched.rs`
- 1-file cycle: `crates/bwa-neon/src/rescue_band.rs -> crates/bwa-neon/src/rescue_band.rs`
- 1-file cycle: `crates/bwa-index/src/hugepage.rs -> crates/bwa-index/src/hugepage.rs`
- 1-file cycle: `crates/bwa-seed/src/lib.rs -> crates/bwa-seed/src/lib.rs`
- 2-file cycle: `crates/bwa-seed/src/lib.rs -> crates/bwa-seed/src/lisa_seed.rs -> crates/bwa-seed/src/lib.rs`
- 2-file cycle: `crates/bwa-mem/src/cigar.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/cigar.rs`
- 2-file cycle: `crates/bwa-mem/src/cigar.rs -> crates/bwa-mem/src/primary.rs -> crates/bwa-mem/src/cigar.rs`
- 2-file cycle: `crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/pe.rs -> crates/bwa-mem/src/lib.rs`
- 2-file cycle: `crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/primary.rs -> crates/bwa-mem/src/lib.rs`
- 2-file cycle: `crates/bwa-mem/src/across.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/across.rs`
- 3-file cycle: `crates/bwa-mem/src/cigar.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/pe.rs -> crates/bwa-mem/src/cigar.rs`
- 3-file cycle: `crates/bwa-mem/src/cigar.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/primary.rs -> crates/bwa-mem/src/cigar.rs`
- 3-file cycle: `crates/bwa-mem/src/cigar.rs -> crates/bwa-mem/src/primary.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/cigar.rs`
- 3-file cycle: `crates/bwa-mem/src/alt.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/pe.rs -> crates/bwa-mem/src/alt.rs`

## Communities (297 total, 40 thin omitted)

### Community 0 - "bwa-seed/src/lib.rs"
Cohesion: 0.05
Nodes (75): AMBIGUOUS_BASE, amd_zen5(), batched_full_seeding_equals_per_read(), batched_smems_equal_per_read(), BSD_LOCKSTEP_MIN, bsd_lockstep_width(), BWD_STEPS, bwt_seed_strategy() (+67 more)

### Community 1 - "bwa-chain/src/lib.rs"
Cohesion: 0.06
Nodes (71): T, build_chains(), build_chains_from_resolved(), build_chains_from_smems(), build_chains_meth(), build_filtered_chains(), chain_seeds(), ChainSeeds (+63 more)

### Community 2 - "batched.rs"
Cohesion: 0.04
Nodes (45): avx2_u8_and_i16_match_scalar(), avx512_u8_and_i16_match_scalar(), BITREV, block_mismatches(), BLOCK_SHIFT, block_walk_equals_the_byte_walk(), BUCKETS, CALL_HIST (+37 more)

### Community 3 - "shape_fuzz.py"
Cohesion: 0.06
Nodes (30): main(), parse(), Lcg, main(), mutate(), read_fasta(), convert(), main() (+22 more)

### Community 4 - "alloc_probe.rs"
Cohesion: 0.05
Nodes (47): allocations_land_in_the_current_stage_and_the_peak_is_the_live_maximum(), B_ALIGN, B_ALIGN_CHAIN, B_ALIGN_EXTEND, B_ALIGN_SA, B_ALIGN_SEED, B_READER, B_UNSTAGED (+39 more)

### Community 5 - "Roadmap"
Cohesion: 0.03
Nodes (60): #44 : la meme table sur AVX-512, plus l'echange de port gratuit (2026-08-09), #46C : le `rowmax` i32 des noyaux u8 x86 passe en u8 (2026-08-09), #48 : les deux leviers faisables ici etaient deja appliques par LLVM (2026-08-08), #50A : la passe inverse bornee par le lemme, 55 a 74 % de lignes en moins pour 0,8 % (2026-08-08), #50B : les fenetres de rescue ne se chevauchent pas, le 15 % n'existe pas (2026-08-08), Aucun aligneur de cette classe ne scale mieux, mesure (2026-08-09), AVX-512 : ce n'est pas une loterie, la fonctionnalite n'est pas exposee (2026-08-09), AVX-512 : correct, jamais chronometre, et pourquoi (2026-08-07) (+52 more)

### Community 6 - "rescue_band.rs"
Cohesion: 0.07
Nodes (31): ABuf, KswJob, u8_score_table(), avx512_band_kernel_matches_avx2(), avx512_bands_reproduce_ksw_align2(), Band, band_bench(), band_stats() (+23 more)

### Community 7 - "lisa.rs"
Cohesion: 0.07
Nodes (24): bidirectional_interval_matches_fmindex(), brute(), edge_cases(), exact_search_matches_bruteforce(), K, kmer_key(), lcg(), LearnedSa (+16 more)

### Community 8 - "rescue_prune.rs"
Cohesion: 0.08
Nodes (49): available(), band_dp(), band_of(), band_proto(), band_t1_bench(), batched_mate_rescue_keeps_every_consumed_field(), BComp, bcomps() (+41 more)

### Community 9 - "pe.rs"
Cohesion: 0.06
Nodes (49): Anchor, ANCHORS, batch_mate_rescue(), best_pair(), best_pair_matches_sorted_list(), bns_fetch_seq(), bns_fetch_seq_into(), DECIBANS_PER_NAT (+41 more)

### Community 10 - "super"
Cohesion: 0.09
Nodes (43): performance_core_count(), performance_core_count_is_never_zero(), comp2(), complements(), encodes_bases(), nt4(), NT4_COMPLEMENT_PIVOT, nt4_in_place() (+35 more)

### Community 11 - "cmd_longread.rs"
Cohesion: 0.06
Nodes (23): IndexArgs, resolve_mmi_presets(), run(), build_mmi(), load_or_build_index(), map_batch(), mmi_path(), preset_for() (+15 more)

### Community 12 - "cmd_mem.rs"
Cohesion: 0.06
Nodes (40): args_from(), BATCH_READAHEAD, BGZF_COMPRESSION_LEVEL, BGZF_SINK_BUFFER_BYTES, build_meth(), build_opt(), BWA_CIGAR_OP_HARD_CLIP, BWA_CIGAR_OP_SOFT_CLIP (+32 more)

### Community 13 - "fmindex.rs"
Cohesion: 0.06
Nodes (39): ANY_BASE, COUNT_SPAN_SHIFT, CP_BLOCK_SIZE, CP_MASK, CP_SHIFT, dump(), enabled(), EXT_CALLS (+31 more)

### Community 14 - "steps"
Cohesion: 0.05
Nodes (48): if, name, runs-on, steps, timeout-minutes, env, name, runs-on (+40 more)

### Community 15 - "matesw.rs"
Cohesion: 0.07
Nodes (45): ACCEPTED, BAND_CELLS, BAND_KERNEL_NS, BAND_OTHER_NS, BAND_R2, BUCKETS, CALL_HIST, CALL_MAX (+37 more)

### Community 16 - "stage_time.rs"
Cohesion: 0.07
Nodes (30): enter(), set_role_reader(), set_role_writer(), stage_discriminants_are_bucket_indices(), add(), BATCHES, BUSY, count_batch() (+22 more)

### Community 17 - "steps"
Cohesion: 0.05
Nodes (42): runs-on, steps, strategy, timeout-minutes, env, CARGO_TERM_COLOR, RUST_TOOLCHAIN, RUSTFLAGS (+34 more)

### Community 18 - "bwa-mem/src/lib.rs"
Cohesion: 0.08
Nodes (30): ALT_CALLS, ALT_NS, APPLY_DEDUP_CALLS, APPLY_DEDUP_NS, APPLY_DEDUP_ROWS, APPLY_HITS, APPLY_INSERT_NS, APPLY_NS (+22 more)

### Community 19 - "bwa-extend/src/lib.rs"
Cohesion: 0.13
Nodes (20): assert_backend_batch_matches_scalar(), assert_backend_batch_order_invariant(), assert_backend_matches_scalar(), assert_backend_saturation_boundary_matches_scalar(), assert_backend_tie_rule_matches_scalar(), BatchTicket, ExtendJob, fill_scmat_vec() (+12 more)

### Community 20 - "primary.rs"
Cohesion: 0.09
Nodes (33): dump(), EDGES, enabled(), hash_64(), hash_64_is_wang(), MAPQ_MAX, mapq_q0_uses_seedcov_branch(), MAPQ_SCORE_COEF (+25 more)

### Community 21 - "with"
Cohesion: 0.05
Nodes (38): concurrency, cancel-in-progress, group, run, RUSTFLAGS, defaults, name, runs-on (+30 more)

### Community 22 - "steps"
Cohesion: 0.05
Nodes (37): concurrency, cancel-in-progress, group, default, description, type, env, CARGO_REGISTRY_TOKEN (+29 more)

### Community 23 - "cigar.rs"
Cohesion: 0.09
Nodes (29): ALREADY_FAST, AMBIG, band_width_trace_enabled(), CALLS, CELLS, CELLS_PROVABLE, CIGAR_OP_DEL, CIGAR_OP_HARD_CLIP (+21 more)

### Community 24 - "with"
Cohesion: 0.05
Nodes (36): defaults, name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group (+28 more)

### Community 25 - "across.rs"
Cohesion: 0.09
Nodes (27): ALPHABET_SIZE, AMBIG, CELLS, CELLS_HIT, CHAINFLT_NS, COMPARABLE_LEN_FRAC, dump(), enabled() (+19 more)

### Community 26 - "steps"
Cohesion: 0.06
Nodes (34): default, description, required, runs-on, steps, timeout-minutes, env, BASES (+26 more)

### Community 27 - "fwd_local_sw_batch"
Cohesion: 0.15
Nodes (34): prefers_wide_x86(), avx2_matesw_i16_matches_scalar(), avx2_matesw_u8_matches_scalar(), avx512_avx2_matesw_uniform_qlen_pad_columns_equal_scalar(), avx512_matesw_i16_matches_scalar(), avx512_matesw_u8_matches_scalar(), batched_ksw_align2(), bench_jobs() (+26 more)

### Community 28 - "FmIndex"
Cohesion: 0.12
Nodes (3): FmIndex, SeedJump, SyncPtr

### Community 29 - "ExtendJob"
Cohesion: 0.15
Nodes (31): band_order(), band_split(), batched_extend(), batched_extend_scalar(), bucket(), cell_bound(), clamp_band(), dead_seed() (+23 more)

### Community 30 - "sw.rs"
Cohesion: 0.13
Nodes (29): call(), CANDS, diagonal_tied_by_a_gapped_path_is_still_the_cigar(), exact_match_scores_full_length(), EXC_MAX_V, excursions_cannot_gain(), global_exact_match_is_all_m(), global_single_deletion() (+21 more)

### Community 31 - "bntseq.rs"
Cohesion: 0.13
Nodes (14): Amb, BntSeq, comment_is_what_the_c_reads_back_from_ann(), Contig, fmt_err(), intv2rid_agrees_with_two_searches(), next_parse(), parse_amb() (+6 more)

### Community 32 - "run"
Cohesion: 0.22
Nodes (20): batch_overlap_enabled(), batched_regs(), Deferred, dump_regs_enabled(), emit_deferred(), emit_resolved(), finish_se(), finish_se_meth() (+12 more)

### Community 33 - "fastq.rs"
Cohesion: 0.11
Nodes (24): acgt_code(), fifo_attempt(), first_le_space(), first_le_space_matches_the_byte_scan(), INFLATE_BLOCK, INFLATE_DEPTH, is_acgt(), pack64() (+16 more)

### Community 34 - "align_read"
Cohesion: 0.17
Nodes (21): align_chains(), align_read(), align_read_dedup(), align_read_meth(), align_read_se(), best(), cal_max_gap(), dump_regs() (+13 more)

### Community 35 - "build.rs"
Cohesion: 0.13
Nodes (23): amb_holes_not_merged_across_contig_boundary(), AmbRec, AnnRec, build_bwt_and_samples(), build_index(), build_index_with_prefix(), CP_BLOCK_SIZE, DUMMY_CHAR (+15 more)

### Community 36 - "bwa-mem/src/meth.rs"
Cohesion: 0.14
Nodes (15): build_xm(), comment_tokens_reencode_like_htslib(), format_g(), leading_float(), leading_int(), MIN_LONGEST_M_PCT, op(), percent_g_matches_c() (+7 more)

### Community 37 - "global_wave.rs"
Cohesion: 0.12
Nodes (20): IOTA, load_table(), shim!, wave_fill!, max_lanes(), PAD, REV, safe() (+12 more)

### Community 38 - "emit.rs"
Cohesion: 0.12
Nodes (21): cigar_string(), cigar_string_which(), format_pa(), BLOCK, FWD16, FWD_TABLE, integers_match_to_string(), push_int() (+13 more)

### Community 39 - "MemAlnReg"
Cohesion: 0.18
Nodes (22): MemAlnReg, cal_sub(), grow_one(), infer_dir_orientations(), insert_settled_view(), insert_settled_view_equals_insert_into_settled(), matesw_apply(), mem_infer_dir() (+14 more)

### Community 40 - "Params"
Cohesion: 0.21
Nodes (20): Comp, components(), components_at(), decide_bands(), decide_counts(), decide_rel(), Decision, Fail (+12 more)

### Community 41 - "bwa-neon/src/lib.rs"
Cohesion: 0.12
Nodes (9): diag_score(), diag_score_matches_scalar(), diag_score_neon(), diag_score_ssse3(), neon_batch_matches_scalar_read_sized(), neon_ungapped_hit_matches_scalar(), NeonBackend, rewrite_jobs() (+1 more)

### Community 42 - "__m256i"
Cohesion: 0.19
Nodes (21): band_epi16(), band_epi8(), band_epu8(), bsl128(), bsl256(), bsl512(), ceq16(), ceq8() (+13 more)

### Community 43 - "mlp_probe.c"
Cohesion: 0.13
Nodes (9): now_ns(), pf(), run(), main(), try_alloc(), build(), chase(), main() (+1 more)

### Community 44 - "prologue"
Cohesion: 0.18
Nodes (19): band_s16(), band_u8(), band_u8_range_compare_is_exhaustively_the_two_compares(), bits(), bits16(), epilogue(), hmax(), hmin() (+11 more)

### Community 45 - "Output"
Cohesion: 0.13
Nodes (11): bam_header_only_still_produces_a_file(), bam_transcoder_reassembles_split_lines(), Output, Bam, Bgzf, Cram, Plain, queue_depth_defaults_to_double_buffering() (+3 more)

### Community 46 - "run_staged"
Cohesion: 0.23
Nodes (13): contains(), discard_contained(), interferes(), judge(), KeptIndex, .BUCKET_SHIFT, .HBITS, .MAX_BUCKET_WALK (+5 more)

### Community 47 - "batched_align"
Cohesion: 0.12
Nodes (20): batched_align(), batched_mate_rescue(), batched_mate_rescue_lazy(), bucket(), count_outcome(), dump(), enabled(), full_sw_cheaper() (+12 more)

### Community 48 - "updates"
Cohesion: 0.10
Nodes (20): patterns, patterns, prefix, actions, cargo, dependency-name, update-types, day (+12 more)

### Community 49 - "steps"
Cohesion: 0.10
Nodes (20): if, runs-on, steps, GH_TOKEN, PR_URL, jobs, auto-merge, name (+12 more)

### Community 51 - "FwdJob"
Cohesion: 0.28
Nodes (19): enc_bases(), extract_group(), fastcol_enabled(), fwd_local_sw_avx2_i16(), fwd_local_sw_avx2_u8_impl(), fwd_local_sw_avx512_i16(), fwd_local_sw_avx512_u8_impl(), fwd_local_sw_neon() (+11 more)

### Community 52 - "steps"
Cohesion: 0.10
Nodes (19): concurrency, cancel-in-progress, group, RUSTFLAGS, jobs, static-openmp, name, on (+11 more)

### Community 53 - "sais.rs"
Cohesion: 0.20
Nodes (18): bucket_ends(), bucket_starts(), check(), check_inplace(), EMPTY, IEMPTY, induce(), inplace_known_strings() (+10 more)

### Community 54 - "Record"
Cohesion: 0.20
Nodes (4): every_compression_yields_the_same_records(), PairedFastqReader, Record, RecordStream

### Community 55 - "A machine-independent model of this aligner's thread scaling"
Cohesion: 0.11
Nodes (16): 1. The model, 2.1 `ALU(N)`: the compute ceiling, 2.2 `RANDOM_BW(N)`: the random-access ceiling, 2.3 TLB reach: `entries x page_size`, 2. The three numbers to measure on a new machine, 3. What the model forbids, and what it permits, 4. The MLP corollary: why the lockstep width has a right answer, 5. Why "batch the accesses" cannot be applied twice (+8 more)

### Community 56 - "opt.rs"
Cohesion: 0.13
Nodes (15): ALL, ALPHABET_SIZE, AMBIGUOUS_BASE_SCORE, defaults_match_bwa_mem2(), fill_scmat(), KEEP_SUPP_MAPQ, MemOpt, NO_MULTI (+7 more)

### Community 57 - "env"
Cohesion: 0.21
Nodes (18): CFLAGS, CXXFLAGS, REQUIRE_AVX512, RUSTFLAGS, steps, release-pgo-x86, steps, steps (+10 more)

### Community 58 - "bwa-mem4"
Cohesion: 0.11
Nodes (18): A caveat on the word "identical", Against fg-labs/bwa-mem3, Building, bwa-mem4, Credits, GPU, Input formats, Installing (+10 more)

### Community 59 - "Le travail litteralement duplique, deux fois dans la meme journee (2026-08-19)"
Cohesion: 0.11
Nodes (18): 1. Les marches du tableau de suffixes, 2. Les jobs de programmation dynamique, #44 est fini : la partie A vaut +11,4 % mesures, la partie B ne vaut rien (2026-08-26), Ce que la meme recherche a elimine, L'ecart wgsim x86, enfin attribue etage par etage (2026-08-19), L'enquete est close : l'anomalie scalaire etait la plateforme (2026-08-20), L'enquete ksw_global2, hypothese par hypothese (2026-08-20), La bande resserree est le defaut, et le match contre le fork bascule (2026-08-20) (+10 more)

### Community 60 - "ksw_align2"
Cohesion: 0.21
Nodes (10): ksw_align2(), bwa_matrix(), hyalite_scoring(), LANES, local_sw_score_matches_independent_implementation(), oracle_score(), oracle_with_score2(), our_score() (+2 more)

### Community 61 - "path"
Cohesion: 0.18
Nodes (10): align(), paired_end_output_is_identical_with_and_without_the_overlap(), repo_root(), single_end_output_is_identical_with_and_without_the_overlap(), write_reads(), EXIT_USAGE_OR_IO, main(), MAX_EXAMPLES (+2 more)

### Community 62 - "Prioritized levers"
Cohesion: 0.12
Nodes (16): Closest analog, CPU optimization roadmap (bwa-mem4), Do NOT, Flat / denser suffix array: `-t1` +12%, `-t8` +0%. Not worth the RAM. (2026-07-16, measured), GPU line: at its ceiling, and the ceiling is 1.28x (2026-07-16, measured), Idea "port getScores16's exact band" — refuted by reading the reference (2026-07), Lever #1 result (2026-07): measured, negative — abandoned on NEON, Levier essaye a la suite de cette revue, et negatif (+8 more)

### Community 63 - "inputs"
Cohesion: 0.12
Nodes (17): default, description, required, default, description, required, default, description (+9 more)

### Community 64 - "with"
Cohesion: 0.19
Nodes (17): steps, steps, steps, steps, id, if, name, run (+9 more)

### Community 65 - "bwa-core/src/meth.rs"
Cohesion: 0.15
Nodes (14): conversion_follows_read_number(), convert_codes(), converted_fasta_is_uppercased_and_wrapped(), parse_tags(), PRUNE_KB, PRUNE_L, PRUNE_T, PRUNE_U (+6 more)

### Community 66 - ".l_pac"
Cohesion: 0.19
Nodes (8): rc_unpack_equals_base_by_base(), text_match_equals_base_by_base(), unpack_pac_base(), unpack_pac_range(), unpack_pac_range_append(), unpack_pac_range_append_rev(), unpack_pac_range_into(), unpack_pac_reproduces_0123()

### Community 67 - "The SIMD DP kernels: what is reachable, and what is already ruled out"
Cohesion: 0.12
Nodes (16): An `x86-64-v4` release binary, the AVX-512 tier build (2026-08-27), Issue #48 is finished, and the extension kernel's remaining gap is its recurrence (2026-08-29), Issue #48 lever A is already done, by LLVM, and the disassembly says so (2026-08-29), Part B of #44: the port arbitrage on the AVX-512 leaf maxes (2026-08-26), Part three: dead ends found after the file was written, The GPU probe reproduces the kernel's shape but not its cache, and that is why it stops here (2026-08-30), The memory-level-parallelism knobs are already right on x86. NULL., The mismatch-only shortcut for the CIGAR's global DP (2026-08-20) (+8 more)

### Community 68 - "perf-profile-x86.yml"
Cohesion: 0.12
Nodes (15): env, CARGO_TERM_COLOR, RUSTFLAGS, jobs, knobs, profile, if, runs-on (+7 more)

### Community 69 - "steps"
Cohesion: 0.23
Nodes (16): ALT_RUSTFLAGS, CG_FLAGS, CPU_MATCH, OURS_ENV, SET, steps, steps, steps (+8 more)

### Community 70 - "require_avx512"
Cohesion: 0.12
Nodes (16): read_pairs, require_avx512, runner, on, workflow_dispatch, default, description, required (+8 more)

### Community 71 - "mem_sam_pe"
Cohesion: 0.15
Nodes (14): 6. How to work on this safely, Diagnostic env vars, The build gotcha, before anything else, The gates that exist, The method that actually cracks a divergence, The standing lesson, Where to look to change X, id_shift_c() (+6 more)

### Community 72 - "kbtree.rs"
Cohesion: 0.28
Nodes (7): duplicate_pos_keeps_klib_order(), getp_aux(), KbTree, lower_picks_last_entry_below(), MAX_KEYS, Node, splits_preserve_order_and_lookup()

### Community 73 - ".load"
Cohesion: 0.23
Nodes (11): bwt_fingerprint(), load_refuses_truncated_or_mixed_files(), rd_i64(), read_at_parallel(), read_exact_at(), rebase_counts(), sibling(), sidecar_is_tied_to_its_index() (+3 more)

### Community 74 - "PeStat"
Cohesion: 0.24
Nodes (11): erfc(), mem_pair(), mem_pair_in(), mem_pestat_from(), memo_matches_direct_formula(), negative_score_end_pairs_at_zero(), PairPenalty, .SPAN (+3 more)

### Community 75 - "Perf levers — measured (M-series, -t1, région 2 Mbp, 500k reads, median of 3)"
Cohesion: 0.13
Nodes (15): Catching up with fg-labs/bwa-mem3 v0.15.0 and v0.15.1 (2026-11-01), Final standing on the gist's own benchmark (giab-4m, `-t16`, default `-K`), Genome scale, PE, 500k pairs, `-K` 10M: the stage table across `-t`, `ksw_global2`: a band bounded by a real alignment's score (-69% on wgsim, kept), ⚠️ Measurement trap: count your batches before concluding anything about I/O, Perf levers — measured (M-series, -t1, région 2 Mbp, 500k reads, median of 3), Real-world speed vs bwa-mem2 2.3 (2026-07-17, post-rebase `main`), Rescue chunking on REAL data: a null result, and two measurement traps (+7 more)

### Community 76 - "bench-x86.yml"
Cohesion: 0.13
Nodes (14): concurrency, cancel-in-progress, group, env, CARGO_TERM_COLOR, RUSTFLAGS, jobs, kernel-ab (+6 more)

### Community 77 - "docker_gates.sh"
Cohesion: 0.40
Nodes (14): cmd_bench(), cmd_build(), cmd_check(), cmd_fixture(), cmd_fork(), cmd_icount(), cmd_parity(), cmd_rss() (+6 more)

### Community 78 - "hugepage.rs"
Cohesion: 0.21
Nodes (9): seed_jump_level_off(), advise_hugepage(), disabled(), MADV_HUGEPAGE, madvise(), page_aligned_range(), page_aligned_range_rounds_inward(), SC_PAGESIZE (+1 more)

### Community 79 - "OnceLock"
Cohesion: 0.20
Nodes (9): append_comment(), append_rg_tag(), COPY_COMMENT, copy_comment(), escape(), MAX_RG_ID_BUFFER, RG_ID, rg_id() (+1 more)

### Community 80 - "sais_rec"
Cohesion: 0.35
Nodes (7): bucket_ends_i(), bucket_starts_i(), induce_i(), IntSlice, lms_eq(), sais_rec(), TypeBits

### Community 81 - "sam.rs"
Cohesion: 0.23
Nodes (10): alt_contig_sq_line_carries_ah(), header_has_no_hd_and_exact_sq(), header_inserts_rg_and_hdr_lines(), SAM_MISSING, SqRecord, unmapped_missing_qual_is_star(), unmapped_record_shape(), write_header() (+2 more)

### Community 82 - "align_reads_batched_inner"
Cohesion: 0.21
Nodes (11): align_reads_batched(), align_reads_batched_inner(), align_reads_batched_meth(), batched_across_reads_equals_per_read(), discard_enabled(), prefetch_smems(), repeat_skip_enabled(), seed_ext_redundant() (+3 more)

### Community 83 - "mem_gen_alt"
Cohesion: 0.30
Nodes (8): get_pri_idx(), hit_counts(), mem_gen_alt(), near_primary_alternate_grouped_distant_one_dropped(), reg(), too_many_hits_drops_all_of_that_primary(), xa_group(), XaTags

### Community 84 - "bench-arm.yml"
Cohesion: 0.14
Nodes (13): concurrency, cancel-in-progress, group, env, CARGO_TERM_COLOR, RUSTFLAGS, jobs, name (+5 more)

### Community 85 - "max_threads"
Cohesion: 0.14
Nodes (14): max_threads, read_pairs, run_graviton, default, description, required, workflow_dispatch, default (+6 more)

### Community 86 - "dry_run"
Cohesion: 0.14
Nodes (14): default, description, type, dry_run, tag, on, pull_request, push (+6 more)

### Community 87 - "5. Glossary"
Cohesion: 0.15
Nodes (13): 1. The problem, in plain English, 4. Crate map, 5. Glossary, ARCHITECTURE, Data and coordinates, Extension and scoring, Output, Paired end (+5 more)

### Community 88 - "bad_input.rs"
Cohesion: 0.45
Nodes (11): a_non_utf8_argument_is_an_error_not_a_panic(), assert_refused(), copy_index(), damaged_index_files_are_named(), long_read_mode_carries_the_read_group_and_extra_header(), long_read_mode_refuses_output_it_cannot_write(), mem(), scratch() (+3 more)

### Community 89 - "tiny"
Cohesion: 0.24
Nodes (10): backward_search_counts_match_naive(), forward_ext_equals_swapped_backward(), get_sa_batch_equals_get_sa(), get_sa_batch_runs_equal_get_sa(), get_sa_matches_sais_and_is_permutation(), konly_backward_step_equals_full(), sa_samples_match_the_file(), sa_tag() (+2 more)

### Community 90 - "seedflt.rs"
Cohesion: 0.26
Nodes (8): KSW_I16_LANES, mem_flt_chained_seeds(), MEM_HSP_COEF, MEM_MINSC_COEF, mem_seed_sw(), MEM_SEEDSW_COEF, MEM_SHORT_EXT, MEM_SHORT_LEN

### Community 91 - "lockstep-sweep"
Cohesion: 0.15
Nodes (13): run, defaults, lockstep-sweep, pgo-arm, defaults, name, runs-on, timeout-minutes (+5 more)

### Community 92 - "inputs"
Cohesion: 0.15
Nodes (13): default, description, required, default, description, required, base_ref, e2e_runner (+5 more)

### Community 93 - "steps"
Cohesion: 0.23
Nodes (13): steps, BASE_REF, READ_PAIRS, REQUIRE_AVX512_E2E, VARIANT_CPUS, steps, steps, env (+5 more)

### Community 94 - "unwrap_multiline_fastq"
Cohesion: 0.23
Nodes (6): needs_unwrapping(), unwrap_detection(), unwrap_multiline_fastq(), unwrap_rewrites_wrapped_records(), UnwrapFastq, UnwrapFastq<R>

### Community 95 - "Methylation mode (`--meth`)"
Cohesion: 0.17
Nodes (11): Differences from bwa-mem3 0.17, How it aligns, Index, Methylation mode (`--meth`), Options, Output, Run through Bismark, Speed (+3 more)

### Community 96 - "bwa-diff/src/lib.rs"
Cohesion: 0.25
Nodes (8): compare(), Divergence, FLAG_LAST_SEGMENT, FLAG_SECONDARY, FLAG_SUPPLEMENTARY, parse_primary(), Rec, Report

### Community 97 - "instant"
Cohesion: 0.31
Nodes (6): main(), report(), run_one(), with_sentinel(), main(), suffix_array_inplace()

### Community 98 - "Deep research: is there an unexplored exact speedup? (2026-07-29)"
Cohesion: 0.18
Nodes (10): Closing the q-gram door with a derivation instead of an assertion, Deep research: is there an unexplored exact speedup? (2026-07-29), Everything else, and why it is closed, Round 2: newest-first sweep, and the STAR / Salmon question (same day), Salmon is not an aligner, and that is the whole answer, Sources, STAR is fast for reasons we are forbidden from copying, The conclusion the review keeps arriving at (+2 more)

### Community 99 - "env"
Cohesion: 0.18
Nodes (10): env, BWA_MEM2_SHA256, BWA_MEM2_VERSION, CARGO_TERM_COLOR, RUST_TOOLCHAIN, RUSTFLAGS, jobs, name (+2 more)

### Community 100 - "with"
Cohesion: 0.18
Nodes (11): with, condarc, create-args, download-micromamba, environment-name, if-no-files-found, init-shell, micromamba-binary-path (+3 more)

### Community 101 - "jobs"
Cohesion: 0.18
Nodes (10): concurrency, cancel-in-progress, group, jobs, pgo-x86, name, defaults, name (+2 more)

### Community 103 - "3. The pipeline, end to end, following one read"
Cohesion: 0.20
Nodes (10): 3.0 Index load, 3.1 Read input and encoding, 3.2 Seeding: from a read to exact matches, 3.3 Chaining: from seeds to candidate loci, 3.4 Extension: from chains to scored regions, 3.5 Dedup, primary marking, MAPQ, 3.6 Paired-end: insert size, mate rescue, pairing, 3.7 CIGAR generation (+2 more)

### Community 105 - "ksw_global2_scalar"
Cohesion: 0.29
Nodes (7): wave16_matches_scalar(), global_cell(), global_mid_rows(), GlobalGaps, ksw_global2_scalar(), push_cigar(), RunCigar

### Community 106 - "star_vs_fm.rs"
Cohesion: 0.31
Nodes (7): fwd_ext(), K, main(), now(), prefetch(), QLEN, W

### Community 107 - "FastqReader"
Cohesion: 0.31
Nodes (4): empty_input_is_zero_records(), FastqReader, InterleavedFastqReader, non_utf8_header_keeps_its_bytes()

### Community 108 - "#57 : perf(gpu): co-ordonnancement CPU+GPU, le partage des jobs est gratuit en risque"
Cohesion: 0.20
Nodes (10): #57 : perf(gpu): co-ordonnancement CPU+GPU, le partage des jobs est gratuit en risque, Ce que ce ticket ne fait pas, Critere d'acceptation, L'interaction avec la RAM (#25), Le probleme a resoudre, Le theoreme qui rend tout ceci gratuit en risque, Niveau 1 : le partage statique (une journee, prend l'essentiel), Niveau 2 : le partage adaptatif (petit, et il rend le reglage inutile) (+2 more)

### Community 109 - "Design: speed and RAM, measured against `fg-labs/bwa-mem3`"
Cohesion: 0.20
Nodes (9): Design: speed and RAM, measured against `fg-labs/bwa-mem3`, Explicit non-goals, Gates, identical for every change in every phase, Phase 0 — the baseline we are allowed to trust, Phase A — the frames nobody has opened, Phase B — bwa-mem3's seeding, conditional on phase 0, Phase C — making `-t8` end-to-end structurally cheaper, The goal, stated so it can be falsified (+1 more)

### Community 110 - "inputs"
Cohesion: 0.20
Nodes (10): default, description, required, base_ref, runner, workflow_dispatch, default, description (+2 more)

### Community 111 - "steps"
Cohesion: 0.20
Nodes (10): AVX2_ARM, BASE_REF, REQUIRE_AVX512, steps, env, if, name, run (+2 more)

### Community 112 - "CLAUDE.md"
Cohesion: 0.22
Nodes (7): A. Setup, 🧩 Advanced Concepts & Libraries, B. Build & Deployment, C. Testing & Validation Gates, 🏗️ Code Architecture (Big Picture), 🎯 Core Purpose, 🛠️ Development Workflow & Commands

### Community 113 - "Contributing"
Cohesion: 0.22
Nodes (9): Architecture, Branches and releases, Build, test, gates, Contributing, Current status (parity), Dependency updates, What this is, Where the NEON work plugs in (phase 9a) (+1 more)

### Community 114 - "MethOpt"
Cohesion: 0.25
Nodes (4): matrices_free_only_the_chemistry(), MethOpt, pbat_swaps_which_read_is_ct(), spec30_prunes_short_repeats_only_when_a_long_seed_exists()

### Community 115 - "open_reader"
Cohesion: 0.33
Nodes (6): gzip_threads(), InflatedBlocks, open_reader(), parsed(), spawn_inflate(), spawn_plain()

### Community 116 - ".run_impl"
Cohesion: 0.25
Nodes (7): copy_fast(), fill_fast(), fit(), Lane, transpose16(), transpose_from(), transpose_to()

### Community 117 - "Global Constraints"
Cohesion: 0.22
Nodes (8): Global Constraints, Notes for whoever runs this, Phase 0: Fork Baseline Harness Implementation Plan, Task 1: bwa-mem4 reports its batch count, Task 2: `scripts/bench.sh` stops defaulting to the region index, Task 3: `scripts/fork_bench.sh`, the three-arm iteration harness, Task 4: the fork joins the 30x validation harness, Task 5: run Phase 0 and write down what it says

### Community 118 - "Catch fg-labs/bwa-mem3 0.16, then pass it"
Cohesion: 0.22
Nodes (8): Catch fg-labs/bwa-mem3 0.16, then pass it, Lever 1: banded rescue (the fork's `rescue_band`), priced GO, Lever 2: seeding and chaining (about 1.8x the fork's cost), Lever 3: idle samples, Measured against the fork (same 2M PE pairs, M4, `-t12`, CPU seconds), Order of work, revised, The round-1 rule is exact under our score2 too, Where we stand (6 Nov 2026)

### Community 119 - "with"
Cohesion: 0.22
Nodes (9): with, components, condarc, create-args, download-micromamba, environment-name, init-shell, micromamba-binary-path (+1 more)

### Community 120 - "with"
Cohesion: 0.22
Nodes (9): with, condarc, create-args, download-micromamba, environment-name, fetch-depth, init-shell, micromamba-binary-path (+1 more)

### Community 121 - "steps"
Cohesion: 0.22
Nodes (9): option-parity, runs-on, steps, timeout-minutes, continue-on-error, if, name, run (+1 more)

### Community 122 - "with"
Cohesion: 0.22
Nodes (9): with, components, condarc, create-args, download-micromamba, environment-name, init-shell, micromamba-binary-path (+1 more)

### Community 123 - "release.yml"
Cohesion: 0.22
Nodes (8): env, CARGO_TERM_COLOR, RUST_TOOLCHAIN, RUSTFLAGS, jobs, name, permissions, contents

### Community 124 - "env"
Cohesion: 0.22
Nodes (9): CFLAGS, CXXFLAGS, GH_TOKEN, GITHUB_REPOSITORY, MACOSX_DEPLOYMENT_TARGET, OPENMP_STATIC, SANITISE, TAG (+1 more)

### Community 125 - "Le GPU rouvert : le noyau tournait a 3,6 % de la machine, et la cause n'est pas le noyau (2026-08-29)"
Cohesion: 0.22
Nodes (9): Ce que cela dit de la suite, et cela reouvre le `uchar4` pour une autre raison que la premiere, Ce que le decoupage montre, et il corrige le diagnostic precedent, Ce qui borne vraiment le noyau GPU, mesure avec les deux noyaux (2026-08-29), Ce qui manquait : un gather par cellule, Et la correction ne transfere pas, ce qui est le resultat le plus utile des trois, La deuxieme cause, invisible jusqu'ici : `score2` sur l'hote, La vraie premiere cause : l'occupancy, et elle se corrige sans toucher au noyau, Le GPU rouvert : le noyau tournait a 3,6 % de la machine, et la cause n'est pas le noyau (2026-08-29) (+1 more)

### Community 126 - "opt_parity.sh"
Cohesion: 0.42
Nodes (8): check(), check_bgzf(), check_f(), check_longread(), check_o(), check_primary5(), check_shapes(), opt_parity.sh script

### Community 127 - "2. Why byte-identical output is the goal, and what it costs"
Cohesion: 0.25
Nodes (8): 2. Why byte-identical output is the goal, and what it costs, Example A: f32 versus f64, and the constants that must stay single-precision, Example B: `mapQ_coef_fac` is an `int` in the C, Example C: unstable sorts, where the instability is load-bearing, Example D: integer overflow and wraparound emulated on purpose, Example E: the order of random number draws is part of the file format, Example F: two Smith-Waterman kernels with deliberately opposite conventions, The general shape

### Community 128 - "global_icount.rs"
Cohesion: 0.32
Nodes (3): full_band(), mat(), tightened()

### Community 129 - "batch_count.rs"
Cohesion: 0.57
Nodes (7): batch_count_respects_verbosity(), repo_root(), reports_batch_count_on_stderr(), run_mem(), write_reads(), zero_k_and_zero_t_use_the_default_batch(), zero_threads_and_zero_k_use_the_default_batch_size()

### Community 130 - "Error"
Cohesion: 0.29
Nodes (6): Error, Fastq, IndexFormat, Io, Other, Result

### Community 131 - "SuboptimalTracker"
Cohesion: 0.32
Nodes (4): dump(), enabled(), ksw_local_fwd(), SuboptimalTracker

### Community 133 - "What fg-labs/bwa-mem3 does that we do not"
Cohesion: 0.25
Nodes (8): First, what their numbers mean, Ranked, what this teardown says to do, `src/read_arena.h` — measured, not worth it here, `src/sam_encode.cpp` — taken, in portable form, `src/smem_dedup.cpp` — worth testing, not yet tried, The finding that explains the x86 gap, The rest, briefly, What fg-labs/bwa-mem3 does that we do not

### Community 134 - "Les sept pieges specifiques au GPU"
Cohesion: 0.25
Nodes (8): 1. Aucun flottant, 2. L'egalite d'argmax : le premier maximum gagne, 3. La saturation a 255 et le repli i16, 4. Le codage de N, 5. `max_off` et la boucle de rounds, 6. La divergence de longueur ne doit pas changer le resultat, 7. Le partage CPU/GPU ne doit rien changer, Les sept pieges specifiques au GPU

### Community 135 - "#55 : perf(gpu): backend Metal, Apple Silicon, memoire unifiee et zero copie"
Cohesion: 0.25
Nodes (8): #55 : perf(gpu): backend Metal, Apple Silicon, memoire unifiee et zero copie, Attendu, Ce qu'il faut verifier avant d'ecrire une ligne (etape 0), Etapes, L'avantage structurel : memoire unifiee, La disposition, et pourquoi elle est deja bonne, Le calcul qui justifie de commencer par la, Le portage lui-meme

### Community 136 - "#56 : perf(gpu): backend CUDA, octets satures __vaddus4 puis DPX s16x2 sur Hopper"
Cohesion: 0.25
Nodes (8): #56 : perf(gpu): backend CUDA, octets satures __vaddus4 puis DPX s16x2 sur Hopper, Ce qu'on ne refait pas, DPX sur Hopper : utile, mais pas pour la raison qu'on croit, Etapes, Le point de depart : CUDA a exactement ce qu'il faut, depuis Kepler, Les transferts, Materiel, Sources

### Community 137 - "Mate rescue: the next lever, priced before it is built"
Cohesion: 0.25
Nodes (7): Mate rescue: the next lever, priced before it is built, Phase 0: re-measure on the current tree (~20 min, quiet host), Phase 1: price the component bands offline (no kernel change), Phase 2: build it (only on a go), Phase 3: gates and measurement, What is already closed (do not retry), Where the cells still are (hypothesis to verify first)

### Community 138 - "steps"
Cohesion: 0.46
Nodes (8): steps, steps, steps, if, name, run, shell, uses

### Community 139 - "endtoend-smallref"
Cohesion: 0.25
Nodes (8): run, defaults, if, name, runs-on, timeout-minutes, endtoend-smallref, shell

### Community 140 - "with"
Cohesion: 0.25
Nodes (8): with, condarc, create-args, download-micromamba, environment-name, init-shell, micromamba-binary-path, toolchain

### Community 141 - "env"
Cohesion: 0.25
Nodes (8): EVENT_NAME, IDX, KNOWN_X86_XS_DIVERGENCE, R1, R2, READS, REQUIRE_AVX512, env

### Community 142 - "run"
Cohesion: 0.25
Nodes (8): run, perf-x86, defaults, name, runs-on, timeout-minutes, defaults, shell

### Community 143 - "build.sh"
Cohesion: 0.25
Nodes (7): BINDGEN_EXTRA_CLANG_ARGS, CFLAGS, CPPFLAGS, LDFLAGS, LIBCLANG_PATH, RUSTFLAGS, build.sh script

### Community 144 - "seq_emit_bench.rs"
Cohesion: 0.38
Nodes (5): BLOCK, FWD, main(), push_as_chunks(), push_chunks_exact()

### Community 145 - "Tier"
Cohesion: 0.29
Nodes (7): calibrate(), Tier, Avx2, Avx512, Scalar, Sse41, time

### Community 146 - "RescueTier"
Cohesion: 0.29
Nodes (7): forced_rescue_tier(), RescueTier, Auto, Avx2, Avx512, Scalar, Sse41

### Community 147 - "#52 : meta: hybride CPU+GPU, le plafond est 2,45x et le GPU integre suffit deja"
Cohesion: 0.29
Nodes (7): 1. Le plafond d'Amdahl, calcule sur notre propre profil, 2. Combien de GPU faut-il pour atteindre ce plafond ? Bien moins qu'on ne croit, 3. Ce que l'octet-identite autorise et interdit sur GPU, 4. La concurrence, pour situer la cible, #52 : meta: hybride CPU+GPU, le plafond est 2,45x et le GPU integre suffit deja, 5. L'ordre des travaux, Sources

### Community 148 - "Literature review: what is left to take, under byte-identity (2026-07-29)"
Cohesion: 0.29
Nodes (6): 1. The paper that matters: minibwa (Li & Homer, arXiv 2606.15357, June 2026), 2. The other accelerators, and why each is closed, 3. What survives, ranked, 4. Two ideas killed by measurement today, 5. Sources, Literature review: what is left to take, under byte-identity (2026-07-29)

### Community 149 - "Where the real-data paired-end time actually goes (2026-07-29 profile)"
Cohesion: 0.29
Nodes (7): Attacking the 13.3%: two byte-identical changes, ~2% together, Attempted and null: hoisting the dead merge check out of the dedup scan, Flat arena for the mate-rescue reverse pass: NULL, Index-tag (indirect) introsort in the dedup sorts: REGRESSION, reverted, Inflate backend for gzipped FASTQ: real on the reader, ~1.2% ceiling on wall, `mem_sort_dedup_patch`: the shape of the 13.3%, measured, Where the real-data paired-end time actually goes (2026-07-29 profile)

### Community 150 - "The mate-rescue kernel: padding-free column range (BANKED, -9.7% wall)"
Cohesion: 0.29
Nodes (7): How the kernel was found to be the lever, Measured, On-demand argmax column in the rescue kernel: REGRESSION (+18% kernel CPU), reverted, Ported to every kernel, The change, The mate-rescue kernel: padding-free column range (BANKED, -9.7% wall), The same change under PGO, and the resulting standing against the fork

### Community 151 - "What is left to make this faster, at 2026, under byte-identity"
Cohesion: 0.29
Nodes (7): 1. The build, not the code (largest, and packaging-only), 2. Only 40% of the rescue kernel's own throughput reaches the pipeline on x86, 3. The extension kernel costs 2.0 operations per cell, the rescue kernel 0.95, Ranked, what to do, What 2026 published, and what of it survives the constraint, What is left to make this faster, at 2026, under byte-identity, Where our own numbers say the remaining upside is

### Community 152 - "head-to-head.yml"
Cohesion: 0.29
Nodes (6): concurrency, cancel-in-progress, group, jobs, name, on

### Community 153 - "head-to-head"
Cohesion: 0.29
Nodes (7): run, defaults, name, runs-on, timeout-minutes, head-to-head, shell

### Community 154 - "build"
Cohesion: 0.29
Nodes (7): needs, runs-on, strategy, timeout-minutes, build, fail-fast, matrix

### Community 155 - "check-version"
Cohesion: 0.29
Nodes (7): outputs, runs-on, timeout-minutes, check-version, publishing, tag, version

### Community 156 - "include"
Cohesion: 0.29
Nodes (7): deployment_target, omp, os, pgo, target_id, variant_cpu, include

### Community 157 - "giab32_headline.sh"
Cohesion: 0.57
Nodes (6): allowed(), note(), refuse(), run(), say(), giab32_headline.sh script

### Community 158 - "giab_publication.sh"
Cohesion: 0.62
Nodes (5): load1(), run(), say(), giab_publication.sh script, wait_quiet()

### Community 159 - "Apple Silicon memory-subsystem probes"
Cohesion: 0.33
Nodes (5): Apple Silicon memory-subsystem probes, `mlp_probe.c` — what the memory system actually delivers, `superpage_probe.c` — are huge pages available?, ⚠️ The trap this probe was built to catch, `tlb_probe.c` — the TLB hierarchy and its cost

### Community 160 - "hybrid_bench.rs"
Cohesion: 0.53
Nodes (4): load_sa(), main(), nt4(), read_fastq()

### Community 161 - "bwt_binning.rs"
Cohesion: 0.53
Nodes (4): bin_sort(), main(), pass(), W

### Community 164 - "RowMax"
Cohesion: 0.33
Nodes (3): i32, RowMax, u8

### Community 165 - "Adding a GPU, without giving up byte-identity"
Cohesion: 0.33
Nodes (5): #54 : test(gpu): la barriere d'octet-identite pour un backend GPU, a ecrire avant le premier noyau, Adding a GPU, without giving up byte-identity, Ce que ce gate ne couvre pas, Ce qui existe et qu'il suffit d'etendre, Le gate d'integration

### Community 166 - "#53 : perf(gpu): la couture asynchrone dans SwBackend, et mesurer la taille des lots"
Cohesion: 0.33
Nodes (6): #53 : perf(gpu): la couture asynchrone dans SwBackend, et mesurer la taille des lots, Ce qui existe deja, et qui est mieux que prevu, Critere d'acceptation, Etape 0 : mesurer la taille des lots (a faire en premier, c'est une demi-journee), Etape 1 : la couture asynchrone, Etape 2 : le crate

### Community 167 - "Part two: the dead ends, and why (issue #49)"
Cohesion: 0.33
Nodes (6): Already in the tree, Byte-identity fatal, Part two: the dead ends, and why (issue #49), Previously measured losses, restated, Refuted on the hardware evidence, The rule the research confirmed

### Community 168 - "Part one: the ceiling per CPU class, and the plan (issue #51)"
Cohesion: 0.33
Nodes (6): Caveat on completeness, First, the number that was wrong, Part one: the ceiling per CPU class, and the plan (issue #51), The ceiling per CPU class, The plan, The rule

### Community 169 - "Three-arm head-to-head, genome scale, PE, `-t12`: wgsim says we win, real data says we lose"
Cohesion: 0.33
Nodes (6): PGO closes the whole real-data gap, Revived and shipped: flattening the rescue rounds (the fork's shape), The question this hands to the Graviton benchmark, Three-arm head-to-head, genome scale, PE, `-t12`: wgsim says we win, real data says we lose, What this makes the next lever, Why this matters more than either number

### Community 170 - "runner"
Cohesion: 0.33
Nodes (6): runner, workflow_dispatch, default, description, required, inputs

### Community 171 - "on"
Cohesion: 0.33
Nodes (6): on, pull_request, push, schedule, branches, cron

### Community 172 - "with"
Cohesion: 0.33
Nodes (6): with, key, name, path, retention-days, toolchain

### Community 173 - "La voie GPU : le plafond est 2,45x, et le GPU integre suffit deja (2026-08-08)"
Cohesion: 0.33
Nodes (6): Ce que l'octet-identite autorise, et c'est plus large qu'attendu, Ce que la concurrence fait, et pourquoi son levier principal ne nous est pas offert, La couture existe deja, La voie GPU : le plafond est 2,45x, et le GPU integre suffit deja (2026-08-08), Le plafond, calcule sur notre propre profil, Le resultat qui n'etait pas attendu : il ne faut presque pas de GPU

### Community 174 - "Le mate rescue, six angles, et le seul qui paie (2026-08-11)"
Cohesion: 0.33
Nodes (6): Ce que la sonde `BWA4_MATESW_TIME` dit du noyau, Classement sur echantillon complet apres fastp, Le dedup incremental : prouve, octet-identique, et nul, Le mate rescue, six angles, et le seul qui paie (2026-08-11), Le tri par longueur du batch de rescue : +3,5 %, et pourquoi il etait nul le matin, Une anomalie non expliquee

### Community 175 - "Le mate rescue, le plus gros poste restant (2026-08-07)"
Cohesion: 0.33
Nodes (6): Ce qui a bien mieux marche : supprimer la reparation au lieu de la sauter (2026-08-08), Ce qui a marche : quatre lignes de cible a la fois, Ce qui n'a pas marche : sauter la reparation du N, La serie complete, un seul banc, Le DP principal : la meme forme biaisee, et un resultat decevant, Le mate rescue, le plus gros poste restant (2026-08-07)

### Community 176 - "Le dossier x86 : ou nous perdons, mesure sur le meme hote (2026-08-18)"
Cohesion: 0.33
Nodes (6): Deux corrections sur mes propres chiffres du jour (2026-08-18), L'indexeur parallele est livre, et l'objection tombe par la mesure (2026-08-19), Le dossier x86 : ou nous perdons, mesure sur le meme hote (2026-08-18), Le jeu d'instructions vaut 5,6 a 6,8 %, et il est livre (2026-08-18), Le PGO de la release etait un chiffre Apple generalise, et il coutait 3,3 % sur x86 (2026-08-18), Le profil x86, lu enfin, et il ne dit pas ce que les issues supposaient

### Community 177 - "Le lecteur n'est plus le goulot, et la sonde mentait (2026-08-11)"
Cohesion: 0.33
Nodes (6): Deux zeros, consignes, La sonde `stage_time` cachait 95 % du run, Le lecteur n'est plus le goulot, et la sonde mentait (2026-08-11), Ou est vraiment le temps, hors sonde, rapidgzip redevient utile apres coup, `Record` : trois allocations par read, devenues une

### Community 178 - "run_in_order"
Cohesion: 0.40
Nodes (3): guided_bounds(), guided_chunks_tile_the_batch_and_keep_their_order(), run_in_order()

### Community 179 - "Chem"
Cohesion: 0.40
Nodes (5): Chem, EmSeq, chem_names(), Taps, parse_chem()

### Community 180 - "Prune"
Cohesion: 0.40
Nodes (5): parse_prune(), Prune, Baseline, Off, Spec30

### Community 181 - "Scoring"
Cohesion: 0.40
Nodes (5): parse_scoring(), Scoring, Collapsed, Genomic, Neutral

### Community 182 - "score_sort_bench.rs"
Cohesion: 0.60
Nodes (3): key_u128(), key_u64x2(), Row

### Community 183 - "Stage"
Cohesion: 0.40
Nodes (5): Stage, Kept, Pending, Purged, Undecided

### Community 185 - "piece16"
Cohesion: 0.40
Nodes (5): fill_soa(), piece16(), piece16_is_the_tile_zero_padded(), t16(), t16v()

### Community 186 - "Divergences connues vs oracle bwa-mem2 2.3"
Cohesion: 0.40
Nodes (4): Acceptees (par conception), Divergences connues vs oracle bwa-mem2 2.3, En cours (résidu), Résolu (phase 8, via oracle instrumenté)

### Community 187 - "The first side-by-side x86 profile against fg-labs/bwa-mem3 (2026-08-29)"
Cohesion: 0.40
Nodes (4): Reading it in absolute time, which is the only way it means anything, The first side-by-side x86 profile against fg-labs/bwa-mem3 (2026-08-29), What the two profiles say, What this file is not

### Community 188 - "graviton-selfhosted"
Cohesion: 0.40
Nodes (5): if, name, runs-on, timeout-minutes, graviton-selfhosted

### Community 189 - "kernel_ab_only"
Cohesion: 0.40
Nodes (5): kernel_ab_only, default, description, required, type

### Community 190 - "require_avx512"
Cohesion: 0.40
Nodes (5): require_avx512, default, description, required, type

### Community 191 - "require_avx512_e2e"
Cohesion: 0.40
Nodes (5): require_avx512_e2e, default, description, required, type

### Community 192 - "avx2_arm"
Cohesion: 0.40
Nodes (5): default, description, required, type, avx2_arm

### Community 193 - "require_avx512"
Cohesion: 0.40
Nodes (5): require_avx512, default, description, required, type

### Community 194 - "require_avx512"
Cohesion: 0.40
Nodes (5): require_avx512, default, description, required, type

### Community 195 - "isa-and-tiers-x86"
Cohesion: 0.40
Nodes (5): defaults, name, runs-on, timeout-minutes, isa-and-tiers-x86

### Community 196 - "real-reads-x86"
Cohesion: 0.40
Nodes (5): real-reads-x86, defaults, name, runs-on, timeout-minutes

### Community 197 - "promote"
Cohesion: 0.40
Nodes (5): promote, if, needs, runs-on, timeout-minutes

### Community 198 - "publish"
Cohesion: 0.40
Nodes (5): publish, if, needs, runs-on, timeout-minutes

### Community 199 - "#53 etapes 1 et 2 : la couture asynchrone et le crate `bwa-gpu` (2026-08-09)"
Cohesion: 0.40
Nodes (5): #53 etapes 1 et 2 : la couture asynchrone et le crate `bwa-gpu` (2026-08-09), Etape 1, le contrat non bloquant, Etape 2, le crate, La limite, dite plutot que tue, Les trois criteres d'acceptation

### Community 200 - "#55 : le backend Metal existe, il est octet-identique, et il perd contre un coeur (2026-08-09)"
Cohesion: 0.40
Nodes (5): #55 : le backend Metal existe, il est octet-identique, et il perd contre un coeur (2026-08-09), Ce qu'il fait, Le chiffre, et il n'est pas bon, Les barrieres, appliquees au GPU, Pourquoi il n'est pas branche dans l'aligneur

### Community 201 - "Les quatre backends de tableau des suffixes, mesures (2026-08-11)"
Cohesion: 0.40
Nodes (5): CaPS-SA : inutilisable ici, pour une raison algorithmique, Ce qui est livre, Les quatre backends de tableau des suffixes, mesures (2026-08-11), Paralleliser divise le CPU total, et ce n'est pas une erreur de mesure, Un zero de plus

### Community 202 - "Le piege de jeu de donnees, et le classement refait (2026-08-11)"
Cohesion: 0.40
Nodes (5): Ce que ca dit, La discipline, d'abord, Le piege de jeu de donnees, et le classement refait (2026-08-11), `-t16`, meme jeu, meme discipline, 6 rondes, `-t4`, 500 k paires chr21, gzip, 6 rondes, medianes

### Community 203 - "Etat de l'art : personne ne fait mieux a contrat de sortie egal (2026-08-08)"
Cohesion: 0.40
Nodes (5): Ce que ca ferme, et ce que ca n'ouvre pas, Corroboration inattendue, vieille de quinze ans, Etat de l'art : personne ne fait mieux a contrat de sortie egal (2026-08-08), Le comparatif le plus direct qui existe : le BSW de bwa-mem2 lui-meme, Le seul qui nous depasse, et pourquoi ca ne compte pas tel quel

### Community 204 - "Le chemin critique du lecteur, et ce que rapidgzip y change (2026-08-11)"
Cohesion: 0.40
Nodes (5): Ce qui reste sur ce chemin, Deux correctifs, mesures, Le chemin critique du lecteur, et ce que rapidgzip y change (2026-08-11), Ou est le serie, mesure, Un bug trouve en chemin, et repare

### Community 205 - "Le regime de colonnes pleinement en bande, trouve par la mesure (2026-08-09)"
Cohesion: 0.40
Nodes (5): La preuve qui a triple sa portee, Le comptage qui a designe le levier, Le levier, Le profil qui l'a designe, Le regime de colonnes pleinement en bande, trouve par la mesure (2026-08-09)

### Community 206 - "Phase 9e : la vague perf (tout octet-identique)"
Cohesion: 0.40
Nodes (5): Le fait perf le plus structurant : le lockstep a mange le levier, Le mur du mate rescue : le nombre de cellules EST l'algorithme, Le profil reel, et pourquoi l'ancien etait faux, Phase 9e : la vague perf (tout octet-identique), Scaling en threads

### Community 207 - "fork_bench.sh"
Cohesion: 0.60
Nodes (3): med(), say(), fork_bench.sh script

### Community 208 - "giab_happy.sh"
Cohesion: 0.70
Nodes (4): align(), call(), score(), giab_happy.sh script

### Community 209 - "graviton_setup.sh"
Cohesion: 0.60
Nodes (4): have(), MAMBA_ROOT_PREFIX, say(), graviton_setup.sh script

### Community 210 - "unpack_bench.rs"
Cohesion: 1.00
Nodes (3): main(), unpack_scalar(), unpack_vector()

### Community 211 - "Bases"
Cohesion: 0.50
Nodes (4): Bases, Ascii, Codes, Packed

### Community 212 - "Mode"
Cohesion: 0.50
Nodes (4): Mode, Finish, Full, Staged

### Community 213 - "Round 3: three structural ideas, all priced before building, all under the floor"
Cohesion: 0.50
Nodes (4): Band-free column loop in the seed-extension kernel: fires on 15% of rows, Round 3: three structural ideas, all priced before building, all under the floor, The rescue window is bwa's, exactly, Union of one read's rescue windows: 1.03x redundancy

### Community 214 - "The `-t16` gap was our own P-core cap. Removing it: 1.10x against the fork, 6 of 6"
Cohesion: 0.50
Nodes (4): Final standing, in the gist's exact regime, The cap, and why it was wrong here, The `-t16` gap was our own P-core cap. Removing it: 1.10x against the fork, 6 of 6, What the barrier probe actually showed

### Community 215 - "hugepage-probe"
Cohesion: 0.50
Nodes (4): name, runs-on, timeout-minutes, hugepage-probe

### Community 216 - "read_pairs"
Cohesion: 0.50
Nodes (4): read_pairs, default, description, required

### Community 217 - "runner"
Cohesion: 0.50
Nodes (4): runner, default, description, required

### Community 218 - "variant_cpu"
Cohesion: 0.50
Nodes (4): variant_cpu, default, description, required

### Community 219 - "read_pairs"
Cohesion: 0.50
Nodes (4): read_pairs, default, description, required

### Community 220 - "extra_rustflags"
Cohesion: 0.50
Nodes (4): default, description, required, extra_rustflags

### Community 221 - "icount"
Cohesion: 0.50
Nodes (4): default, description, required, icount

### Community 222 - "icount"
Cohesion: 0.50
Nodes (4): if, runs-on, timeout-minutes, icount

### Community 223 - "knob_sweep"
Cohesion: 0.50
Nodes (4): knob_sweep, default, description, required

### Community 224 - "read_pairs"
Cohesion: 0.50
Nodes (4): read_pairs, default, description, required

### Community 225 - "read_set"
Cohesion: 0.50
Nodes (4): read_set, default, description, required

### Community 226 - "runner"
Cohesion: 0.50
Nodes (4): runner, default, description, required

### Community 227 - "meth"
Cohesion: 0.50
Nodes (4): meth, if, runs-on, timeout-minutes

### Community 228 - "on"
Cohesion: 0.50
Nodes (4): on, push, workflow_dispatch, branches

### Community 229 - "Les chiffres x86, enfin mesures : +17 % sur le noyau de rescue (2026-08-09)"
Cohesion: 0.50
Nodes (4): #44 reste non verifie, et ce n'est pas une loterie (2026-08-15), Les chiffres x86, enfin mesures : +17 % sur le noyau de rescue (2026-08-09), Noyau de rescue, `rescue_kernel_ab`, AVX2 u8, Noyaux d'extension, `x86_extend_ab`

### Community 230 - "#45 : `USQADD` et le `qsub(h, oe)` partage, +10,3 % de noyau, -2,7 % de CPU (2026-08-08)"
Cohesion: 0.50
Nodes (4): #45 : `USQADD` et le `qsub(h, oe)` partage, +10,3 % de noyau, -2,7 % de CPU (2026-08-08), Ce que la mesure donne, L'octet-identite, Le desassemblage, et les deversements qui disparaissent

### Community 231 - "#46 : les deux passes scalaires ne coutent rien, et pourquoi le microbenchmark disait le contraire (2026-08-08)"
Cohesion: 0.50
Nodes (4): #46 : les deux passes scalaires ne coutent rien, et pourquoi le microbenchmark disait le contraire (2026-08-08), A, la porte vectorielle sur `finish_row`, B, le pre-filtre vectoriel de `extract_group`, et le piege de mesure, Ce qui reste ouvert

### Community 232 - "#47 : les colonnes de bourrage ont leur propre corps, et le gain depend de la longueur de lecture (2026-08-08)"
Cohesion: 0.50
Nodes (4): #47 : les colonnes de bourrage ont leur propre corps, et le gain depend de la longueur de lecture (2026-08-08), Ce qui n'a pas ete fait, L'octet-identite, Le gain n'est pas un nombre, c'est une fonction de la fraction bourree

### Community 233 - "#53 etape 0 : la taille des lots, et elle repond deux choses differentes (2026-08-08)"
Cohesion: 0.50
Nodes (4): #53 etape 0 : la taille des lots, et elle repond deux choses differentes (2026-08-08), Ce que ce chiffre decide, Extension (`batched_extend`, la couture `SwBackend`), Mate rescue (`batched_ksw_align2`)

### Community 234 - "#54 : la barriere d'octet-identite GPU, ecrite avant le premier noyau (2026-08-08)"
Cohesion: 0.50
Nodes (4): #54 : la barriere d'octet-identite GPU, ecrite avant le premier noyau (2026-08-08), Ce qui reste des sept pieges, Deux barrieres generiques sur `SwBackend`, Le gate d'integration, `scripts/gpu_parity.sh`

### Community 235 - "L'extension sur GPU, branchee et mesuree : -22 % de CPU, et une bascule a -t8 (2026-08-09)"
Cohesion: 0.50
Nodes (4): Ce que cela dit de la suite, L'extension sur GPU, branchee et mesuree : -22 % de CPU, et une bascule a -t8 (2026-08-09), La mesure, et elle corrige ma propre prediction, Quand c'est utile aujourd'hui

### Community 236 - "Le noyau d'extension Metal : ecrit, octet-identique, 6,4x un coeur (2026-08-09)"
Cohesion: 0.50
Nodes (4): Ce que cela vaudrait, et ce qui manque, Ce qui a rendu la validation gratuite, Le noyau d'extension Metal : ecrit, octet-identique, 6,4x un coeur (2026-08-09), Les chiffres

### Community 237 - "Recherche multi-agents sur les noyaux SIMD (2026-08-08)"
Cohesion: 0.50
Nodes (4): Ce que la recherche a confirme, et qui clot un sujet, Honnetete sur la couverture, Le resultat qui explique le 1,25x de l'AVX-512, Recherche multi-agents sur les noyaux SIMD (2026-08-08)

### Community 238 - "Le `uchar4` sur Metal : tente, non concluant, retire (2026-08-09)"
Cohesion: 0.50
Nodes (4): Ce qui a ete appris en route, et qui reste vrai, Ce qui reste vrai pour la suite, Le `uchar4` sur Metal : tente, non concluant, retire (2026-08-09), Pourquoi c'est retire

### Community 239 - "Indexeur : le tableau de suffixes est parallelisable, mais pas par le portage Rust (2026-08-05)"
Cohesion: 0.50
Nodes (4): Ce qui a ete essaye et rejete, Indexeur : le tableau de suffixes est parallelisable, mais pas par le portage Rust (2026-08-05), Note historique, Pourquoi le portage pur Rust ne rattrape pas

### Community 240 - "Le chemin GPU est l'extension, pas le rescue, et les chiffres le disent (2026-08-09)"
Cohesion: 0.50
Nodes (4): Ce qui manque, et sa taille, Et l'agregation inter-threads y est deja prouvee sure, Le chemin GPU est l'extension, pas le rescue, et les chiffres le disent (2026-08-09), Les deux coutures, mesurees en production, meme course

### Community 241 - "Phase 0 de la campagne perf : mesure de reference contre le fork"
Cohesion: 0.50
Nodes (4): Historique phase 7-8 (traine PE), Phase 0 de la campagne perf : mesure de reference contre le fork, Phase A, levier RAM : pic 19,2 -> 11,2 Go (deux commits, octet-identiques), Phase B, le deficit PE = le kernel de sauvetage de mate

### Community 242 - "Trois questions fermees le meme jour : le gate GIAB, le PGO hors Apple, et l'indexeur parallele (2026-08-18)"
Cohesion: 0.50
Nodes (4): L'indexeur n'est pas mono-thread, il est mono-thread PAR DEFAUT (#37), Le PGO est un levier Apple Silicon, et rien d'autre (#33), Phase 11 : l'identite octet, dite en precision et rappel (#35), Trois questions fermees le meme jour : le gate GIAB, le PGO hors Apple, et l'indexeur parallele (2026-08-18)

### Community 243 - "giab30x_bench.sh"
Cohesion: 0.83
Nodes (3): run(), say(), giab30x_bench.sh script

### Community 244 - "giab30x_pe.sh"
Cohesion: 0.83
Nodes (3): run(), say(), giab30x_pe.sh script

### Community 245 - "pe_diverge_hunt.sh"
Cohesion: 1.00
Nodes (3): capture(), say(), pe_diverge_hunt.sh script

### Community 246 - "pe_diverge_subset.sh"
Cohesion: 0.83
Nodes (3): cap(), say(), pe_diverge_subset.sh script

### Community 247 - "pgo.sh"
Cohesion: 0.50
Nodes (3): CC, PATH, pgo.sh script

### Community 248 - "fwd_local_sw_neon_u8"
Cohesion: 0.67
Nodes (3): fwd_local_sw_neon_u8(), shareoe_enabled(), usqadd_enabled()

### Community 249 - "AVX-512BW seed extension, and a NEON epilogue that measured null"
Cohesion: 0.67
Nodes (3): AVX-512BW extension kernel (new, x86, unmeasured here), AVX-512BW seed extension, and a NEON epilogue that measured null, Row maxima published with one vector store: null on this host, kept

### Community 250 - "Measuring in Nils's actual regime: we are TIED at `-t16`, not ahead"
Cohesion: 0.67
Nodes (3): Measuring in Nils's actual regime: we are TIED at `-t16`, not ahead, Two hypotheses for the `-t16` loss, both tested, both dead tonight, Why this machine cannot settle it

### Community 251 - "Vectorising what was left scalar: one kept, one reverted"
Cohesion: 0.67
Nodes (3): `.pac` unpack: vectorised, kept (unmeasured on a drifting host), Row-major `score2` scan: REGRESSION (+7.3% kernel CPU), reverted, Vectorising what was left scalar: one kept, one reverted

### Community 252 - "Where the `-t16` wall actually goes: `BWA4_STAGE_TIME=1`"
Cohesion: 0.67
Nodes (3): Two theories this killed, What the chr21 measurement structurally cannot see, Where the `-t16` wall actually goes: `BWA4_STAGE_TIME=1`

### Community 253 - "#25 tranche : le resident EST le vivant sous Linux, et le 2x est le double batch en vol (2026-08-18)"
Cohesion: 0.67
Nodes (3): #25 tranche : le resident EST le vivant sous Linux, et le 2x est le double batch en vol (2026-08-18), Et le churn ne coute pas de wall non plus : l'arene par worker est morte des deux cotes, Le prix du recouvrement, mesure : 1,1 % de wall pour 41 % de la RSS

### Community 254 - "#43 : la table de scores XOR portee sur AVX2 et SSE4.1, et le +18 a +50 % est a revoir (2026-08-09)"
Cohesion: 0.67
Nodes (3): #43 : la table de scores XOR portee sur AVX2 et SSE4.1, et le +18 a +50 % est a revoir (2026-08-09), Ce que le desassemblage montre, et pourquoi il corrige la projection, Ce qui est verifie ici

### Community 255 - "#48B : la pre-passe de substitution repliee, +4 a +5 % de noyau, -0,9 % de CPU (2026-08-08)"
Cohesion: 0.67
Nodes (3): #48B : la pre-passe de substitution repliee, +4 a +5 % de noyau, -0,9 % de CPU (2026-08-08), La difference avec #48A et #48D se voit dans les symboles, Mesure

### Community 256 - "#48C : le masque de bande en espace signe, et ce que le desassemblage x86 dit vraiment (2026-08-09)"
Cohesion: 0.67
Nodes (3): #48C : le masque de bande en espace signe, et ce que le desassemblage x86 dit vraiment (2026-08-09), Ce qui n'est PAS mesure, et il faut le dire, L'octet-identite, elle, est verifiee ici

### Community 257 - "#54 pieges 3 et 5 : la frontiere de saturation, et un `max_off` qui ne decide jamais rien (2026-08-08)"
Cohesion: 0.67
Nodes (3): #54 pieges 3 et 5 : la frontiere de saturation, et un `max_off` qui ne decide jamais rien (2026-08-08), Piege 3, la frontiere u8 / i16, Piege 5, `max_off` et la boucle de rounds : il ne se declenche jamais

### Community 258 - "#55 etape 0 : bloquee, il n'y a pas de compilateur Metal sur cette machine (2026-08-08)"
Cohesion: 0.67
Nodes (3): #55 etape 0 : bloquee, il n'y a pas de compilateur Metal sur cette machine (2026-08-08), Ce qui change dans le calcul si `subsat` manque, Le protocole exact a executer quand Xcode sera la

### Community 259 - "#55 etape 0 : debloquee sans Xcode, et le GPU tient 330 Gcell/s (2026-08-09)"
Cohesion: 0.67
Nodes (3): #55 etape 0 : debloquee sans Xcode, et le GPU tient 330 Gcell/s (2026-08-09), Ce que MSL expose reellement, Le plafond, mesure au lieu d'etre projete

### Community 260 - "Campagne perf CPU et classement contre minibwa (2026-08-09)"
Cohesion: 0.67
Nodes (3): Campagne perf CPU et classement contre minibwa (2026-08-09), Contre bwa-mem2 2.3 et contre minibwa, Contre le fork fg-labs/bwa-mem3 v0.9.0 (sorti le 2026-08-07)

### Community 261 - "L'oracle tiers couvre enfin les penalites de gap asymetriques (2026-08-12)"
Cohesion: 0.67
Nodes (3): Ce que l'elargissement a trouve, et qui n'est pas un bug, L'oracle tiers couvre enfin les penalites de gap asymetriques (2026-08-12), Verifie que le nouveau bras est vivant

### Community 262 - "Ou le fork est reellement moins cher, profil contre profil (2026-08-07)"
Cohesion: 0.67
Nodes (3): Ce qui etait recuperable dans ce plafond, et l'a ete, Le tri : ce que l'octet-identite coute exactement, chiffre, Ou le fork est reellement moins cher, profil contre profil (2026-08-07)

### Community 263 - "Entree gzip : ou est le mur, et a combien de cœurs il tombe (2026-08-11)"
Cohesion: 0.67
Nodes (3): Correction, apres avoir applique la loi de `thread-broker` a nos chiffres, Entree gzip : ou est le mur, et a combien de cœurs il tombe (2026-08-11), hyalite 0.4.0

## Knowledge Gaps
- **89 isolated node(s):** `bwa-mem4-chain`, `Refused`, `Contained`, `Appended`, `bwa-mem4` (+84 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1543 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **40 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ARCHITECTURE` connect `5. Glossary` to `3. The pipeline, end to end, following one read`, `mem_sam_pe`, `README.md`, `2. Why byte-identical output is the goal, and what it costs`?**
  _High betweenness centrality (0.137) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `batched_align()` (e.g. with `Params` and `bands_reproduce_ksw_align2()`) actually correct?**
  _`batched_align()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bwa-mem4-chain`, `Refused`, `Contained` to the rest of the system?**
  _89 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `bwa-seed/src/lib.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.05212716222533895 - nodes in this community are weakly interconnected._
- **Why does `6. How to work on this safely` connect `mem_sam_pe` to `5. Glossary`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Should `bwa-chain/src/lib.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.06073871409028728 - nodes in this community are weakly interconnected._
- **Why does `Where to look to change X` connect `mem_sam_pe` to `run`, `bwa-seed/src/lib.rs`, `align_read`, `emit.rs`, `cmd_mem.rs`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._