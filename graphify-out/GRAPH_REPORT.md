# Graph Report - wt-graphify-bwa-mem4  (2026-11-27)

## Corpus Check
- 163 files · ~691,282 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 17 file(s) not represented in the graph (top: (none) 4, .toml 2, .log 2)

## Summary
- 3858 nodes · 6981 edges · 324 communities (270 shown, 54 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 236 edges (avg confidence: 0.84)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6ceda635`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- bwa-seed/src/lib.rs
- bwa-chain/src/lib.rs
- pe.rs
- batched.rs
- shape_fuzz.py
- alloc_probe.rs
- Batch
- lisa.rs
- rescue_prune.rs
- Roadmap
- cmd_longread.rs
- across.rs
- cmd_mem.rs
- path
- steps
- super
- stage_alloc.rs
- matesw.rs
- primary.rs
- steps
- bwa-mem/src/lib.rs
- with
- steps
- with
- fmindex.rs
- cigar.rs
- steps
- FmIndex
- 2. Why byte-identical output is the goal, and what it costs
- Perf levers — measured (M-series, -t1, région 2 Mbp, 500k reads, median of 3)
- emit.rs
- sw.rs
- ExtendJob
- The SIMD DP kernels: what is reachable, and what is already ruled out
- run
- fastq.rs
- build.rs
- bwa-mem/src/meth.rs
- global_wave.rs
- bwa-extend/src/lib.rs
- __m256i
- Params
- bwa-neon/src/lib.rs
- align_read
- mlp_probe.c
- Output
- prologue
- scmat
- batched_align
- updates
- steps
- tiny
- steps
- sais.rs
- Record
- run_staged
- FwdJob
- MemAlnReg
- env
- bwa-mem4
- Le travail litteralement duplique, deux fois dans la meme journee (2026-08-19)
- ksw_align2
- README.md
- opt.rs
- Prioritized levers
- inputs
- with
- bwa-core/src/meth.rs
- bwa-mem/src/alt_lift.rs
- Who else is making BWA-MEM fast, and what of it applies here
- perf-profile-x86.yml
- steps
- require_avx512
- kbtree.rs
- fwd_local_sw_batch
- bench-x86.yml
- docker_gates.sh
- mem_sam_pe
- OnceLock
- sais_rec
- sam.rs
- mem_gen_alt
- bench-arm.yml
- max_threads
- dry_run
- bntseq.rs
- seedflt.rs
- lockstep-sweep
- inputs
- steps
- hugepage.rs
- mem_sam_pe
- unwrap_multiline_fastq
- Methylation mode (`--meth`)
- instant
- Deep research: is there an unexplored exact speedup? (2026-07-29)
- A machine-independent model of this aligner's thread scaling
- env
- with
- jobs
- HtsWriter
- cmd_longread.rs
- star_vs_fm.rs
- .load
- FastqReader
- #57 : perf(gpu): co-ordonnancement CPU+GPU, le partage des jobs est gratuit en risque
- Design: speed and RAM, measured against `fg-labs/bwa-mem3`
- inputs
- steps
- CLAUDE.md
- MethOpt
- .backward_ext
- rand48.rs
- open_reader
- rescue_band.rs
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
- global_icount.rs
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
- unpack16
- seq_emit_bench.rs
- Tier
- RescueTier
- #52 : meta: hybride CPU+GPU, le plafond est 2,45x et le GPU integre suffit deja
- Literature review: what is left to take, under byte-identity (2026-07-29)
- Where the real-data paired-end time actually goes (2026-07-29 profile)
- The mate-rescue kernel: padding-free column range (BANKED, -9.7% wall)
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
- bwa-index/src/lib.rs
- ByteStr
- bench_batch.rs
- RowMax
- Adding a GPU, without giving up byte-identity
- #53 : perf(gpu): la couture asynchrone dans SwBackend, et mesurer la taille des lots
- Three-arm head-to-head, genome scale, PE, `-t12`: wgsim says we win, real data says we lose
- runner
- on
- with
- La voie GPU : le plafond est 2,45x, et le GPU integre suffit deja (2026-08-08)
- Le mate rescue, six angles, et le seul qui paie (2026-08-11)
- Le mate rescue, le plus gros poste restant (2026-08-07)
- Le dossier x86 : ou nous perdons, mesure sur le meme hote (2026-08-18)
- Le lecteur n'est plus le goulot, et la sonde mentait (2026-08-11)
- run_pe
- align_reads_batched_inner
- Prune
- Scoring
- score_sort_bench.rs
- run
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
- score2_lanes
- Round 3: three structural ideas, all priced before building, all under the floor
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
- stage_time.rs
- bioconda/README.md
- RecordStream
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
- bounds
- build_opt
- dna.rs
- mem_reg2sam
- README.md
- 3. The pipeline, end to end, following one read
- super
- Stage
- enabled
- MemOpt
- 5. Glossary
- What is left to make this faster, at 2026, under byte-identity
- Part two: the dead ends, and why (issue #49)
- Part one: the ceiling per CPU class, and the plan (issue #51)
- alt_project_diff.sh
- MdMode
- The `-t16` gap was our own P-core cap. Removing it: 1.10x against the fork, 6 of 6
- AVX-512BW seed extension, and a NEON epilogue that measured null
- Measuring in Nils's actual regime: we are TIED at `-t16`, not ahead
- Real-world speed vs bwa-mem2 2.3 (2026-07-17, post-rebase `main`)
- PE profile on real GIAB: mate rescue is 47%, and it looks nothing like SE
- Two target rows per pass in the rescue kernel (kernel -8.9%, wall -2.2%)

## God Nodes (most connected - your core abstractions)
1. `Roadmap` - 91 edges
2. `MemAlnReg` - 69 edges
3. `FmIndex` - 53 edges
4. `Params` - 30 edges
5. `batched_align()` - 29 edges
6. `mem_sam_pe()` - 27 edges
7. `batch_mate_rescue()` - 26 edges
8. `run()` - 25 edges
9. `run_pe()` - 25 edges
10. `align_reads_batched_inner()` - 25 edges

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
- 1-file cycle: `crates/bwa-cli/src/stage_time.rs -> crates/bwa-cli/src/stage_time.rs`
- 1-file cycle: `crates/bwa-mem/src/pe.rs -> crates/bwa-mem/src/pe.rs`
- 1-file cycle: `crates/bwa-core/src/error.rs -> crates/bwa-core/src/error.rs`
- 1-file cycle: `crates/bwa-neon/src/rescue_prune.rs -> crates/bwa-neon/src/rescue_prune.rs`
- 1-file cycle: `crates/bwa-neon/src/rescue_band.rs -> crates/bwa-neon/src/rescue_band.rs`
- 1-file cycle: `crates/bwa-extend/src/global_wave.rs -> crates/bwa-extend/src/global_wave.rs`
- 1-file cycle: `crates/bwa-io/src/fastq.rs -> crates/bwa-io/src/fastq.rs`
- 1-file cycle: `crates/bwa-seed/src/lib.rs -> crates/bwa-seed/src/lib.rs`
- 1-file cycle: `crates/bwa-neon/src/batched.rs -> crates/bwa-neon/src/batched.rs`
- 1-file cycle: `crates/bwa-index/src/hugepage.rs -> crates/bwa-index/src/hugepage.rs`
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

## Communities (324 total, 54 thin omitted)

### Community 0 - "bwa-seed/src/lib.rs"
Cohesion: 0.05
Nodes (75): AMBIGUOUS_BASE, amd_zen5(), batched_full_seeding_equals_per_read(), batched_smems_equal_per_read(), BSD_LOCKSTEP_MIN, bsd_lockstep_width(), BWD_STEPS, bwt_seed_strategy() (+67 more)

### Community 1 - "bwa-chain/src/lib.rs"
Cohesion: 0.06
Nodes (70): T, build_chains(), build_chains_from_resolved(), build_chains_from_smems(), build_chains_meth(), build_filtered_chains(), chain_seeds(), ChainSeeds (+62 more)

### Community 2 - "pe.rs"
Cohesion: 0.06
Nodes (43): ANCHORS, batch_mate_rescue(), best_pair(), best_pair_matches_sorted_list(), DECIBANS_PER_NAT, dump(), enabled(), GROUPS (+35 more)

### Community 3 - "batched.rs"
Cohesion: 0.04
Nodes (45): avx2_u8_and_i16_match_scalar(), avx512_u8_and_i16_match_scalar(), BITREV, block_mismatches(), BLOCK_SHIFT, block_walk_equals_the_byte_walk(), BUCKETS, CALL_HIST (+37 more)

### Community 4 - "shape_fuzz.py"
Cohesion: 0.06
Nodes (30): main(), parse(), Lcg, main(), mutate(), read_fasta(), convert(), main() (+22 more)

### Community 5 - "alloc_probe.rs"
Cohesion: 0.05
Nodes (47): allocations_land_in_the_current_stage_and_the_peak_is_the_live_maximum(), B_ALIGN, B_ALIGN_CHAIN, B_ALIGN_EXTEND, B_ALIGN_SA, B_ALIGN_SEED, B_READER, B_UNSTAGED (+39 more)

### Community 6 - "Batch"
Cohesion: 0.11
Nodes (21): ABuf, rescue_avx512_ok(), u8_score_table(), avx512_band_kernel_matches_avx2(), Band, band_bench(), band_stats(), Batch (+13 more)

### Community 7 - "lisa.rs"
Cohesion: 0.07
Nodes (24): bidirectional_interval_matches_fmindex(), brute(), edge_cases(), exact_search_matches_bruteforce(), K, kmer_key(), lcg(), LearnedSa (+16 more)

### Community 8 - "rescue_prune.rs"
Cohesion: 0.10
Nodes (41): available(), band_dp(), band_of(), band_proto(), band_t1_bench(), batched_mate_rescue_keeps_every_consumed_field(), BComp, bcomps() (+33 more)

### Community 9 - "Roadmap"
Cohesion: 0.03
Nodes (60): #44 : la meme table sur AVX-512, plus l'echange de port gratuit (2026-08-09), #46C : le `rowmax` i32 des noyaux u8 x86 passe en u8 (2026-08-09), #48 : les deux leviers faisables ici etaient deja appliques par LLVM (2026-08-08), #50A : la passe inverse bornee par le lemme, 55 a 74 % de lignes en moins pour 0,8 % (2026-08-08), #50B : les fenetres de rescue ne se chevauchent pas, le 15 % n'existe pas (2026-08-08), Aucun aligneur de cette classe ne scale mieux, mesure (2026-08-09), AVX-512 : ce n'est pas une loterie, la fonctionnalite n'est pas exposee (2026-08-09), AVX-512 : correct, jamais chronometre, et pourquoi (2026-08-07) (+52 more)

### Community 10 - "cmd_longread.rs"
Cohesion: 0.10
Nodes (14): build_meth(), IndexArgs, resolve_mmi_presets(), run(), Cli, Cmd, Index, Mem (+6 more)

### Community 11 - "across.rs"
Cohesion: 0.10
Nodes (25): ALPHABET_SIZE, AMBIG, CELLS, CELLS_HIT, CHAINFLT_NS, COMPARABLE_LEN_FRAC, EXTEND_NS, HIT_THRESH (+17 more)

### Community 12 - "cmd_mem.rs"
Cohesion: 0.05
Nodes (29): ALT_DEMOTED, BATCH_READAHEAD, BGZF_COMPRESSION_LEVEL, BGZF_SINK_BUFFER_BYTES, BWA_CIGAR_OP_HARD_CLIP, BWA_CIGAR_OP_SOFT_CLIP, C_ROUND_BIAS, CIGAR_LEN_SHIFT (+21 more)

### Community 13 - "path"
Cohesion: 0.08
Nodes (36): a_non_utf8_argument_is_an_error_not_a_panic(), assert_refused(), copy_index(), damaged_index_files_are_named(), long_read_mode_carries_the_read_group_and_extra_header(), long_read_mode_refuses_output_it_cannot_write(), mem(), scratch() (+28 more)

### Community 14 - "steps"
Cohesion: 0.05
Nodes (48): if, name, runs-on, steps, timeout-minutes, env, name, runs-on (+40 more)

### Community 15 - "super"
Cohesion: 0.28
Nodes (22): MemSeed, backward_lem(), bwt_seed_strategy_lsa(), bwt_seed_strategy_lsa_fast(), collect_smems_lsa(), collect_smems_lsa_zigzag(), fast_full_seedset_concordant_with_fmindex(), forward_lem() (+14 more)

### Community 16 - "stage_alloc.rs"
Cohesion: 0.14
Nodes (3): enter(), stage_discriminants_are_bucket_indices(), names()

### Community 17 - "matesw.rs"
Cohesion: 0.07
Nodes (41): ACCEPTED, BAND_CELLS, BAND_KERNEL_NS, BAND_OTHER_NS, BAND_R2, BUCKETS, CALL_HIST, CALL_MAX (+33 more)

### Community 18 - "primary.rs"
Cohesion: 0.08
Nodes (38): dump(), EDGES, enabled(), hash_64(), hash_64_is_wang(), ln_int(), MAPQ_MAX, mapq_q0_uses_seedcov_branch() (+30 more)

### Community 19 - "steps"
Cohesion: 0.05
Nodes (42): runs-on, steps, strategy, timeout-minutes, env, CARGO_TERM_COLOR, RUST_TOOLCHAIN, RUSTFLAGS (+34 more)

### Community 20 - "bwa-mem/src/lib.rs"
Cohesion: 0.08
Nodes (30): ALT_CALLS, ALT_NS, APPLY_DEDUP_CALLS, APPLY_DEDUP_NS, APPLY_DEDUP_ROWS, APPLY_HITS, APPLY_INSERT_NS, APPLY_NS (+22 more)

### Community 21 - "with"
Cohesion: 0.05
Nodes (38): concurrency, cancel-in-progress, group, run, RUSTFLAGS, defaults, name, runs-on (+30 more)

### Community 22 - "steps"
Cohesion: 0.05
Nodes (37): concurrency, cancel-in-progress, group, default, description, type, env, CARGO_REGISTRY_TOKEN (+29 more)

### Community 23 - "with"
Cohesion: 0.05
Nodes (36): defaults, name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group (+28 more)

### Community 24 - "fmindex.rs"
Cohesion: 0.08
Nodes (29): ANY_BASE, COUNT_SPAN_SHIFT, CP_BLOCK_SIZE, CP_MASK, CP_SHIFT, EXT_CALLS, EXT_LINES, fields_x86() (+21 more)

### Community 25 - "cigar.rs"
Cohesion: 0.09
Nodes (31): ALREADY_FAST, AMBIG, band_width_trace_enabled(), CALLS, CELLS, CELLS_PROVABLE, CIGAR_OP_DEL, CIGAR_OP_HARD_CLIP (+23 more)

### Community 26 - "steps"
Cohesion: 0.06
Nodes (34): default, description, required, runs-on, steps, timeout-minutes, env, BASES (+26 more)

### Community 27 - "FmIndex"
Cohesion: 0.16
Nodes (4): CpOcc, FmIndex, rows_before(), Smem

### Community 28 - "2. Why byte-identical output is the goal, and what it costs"
Cohesion: 0.14
Nodes (14): 1. The problem, in plain English, 2. Why byte-identical output is the goal, and what it costs, 4. Crate map, ARCHITECTURE, Example A: f32 versus f64, and the constants that must stay single-precision, Example B: `mapQ_coef_fac` is an `int` in the C, Example C: unstable sorts, where the instability is load-bearing, Example D: integer overflow and wraparound emulated on purpose (+6 more)

### Community 29 - "Perf levers — measured (M-series, -t1, région 2 Mbp, 500k reads, median of 3)"
Cohesion: 0.12
Nodes (16): bwa-mem3 0.17 perf PRs against this tree (2026-11-17, at the 0.18 catch-up), Catching up with fg-labs/bwa-mem3 v0.15.0 and v0.15.1 (2026-11-01), Final standing on the gist's own benchmark (giab-4m, `-t16`, default `-K`), Genome scale, PE, 500k pairs, `-K` 10M: the stage table across `-t`, `ksw_global2`: a band bounded by a real alignment's score (-69% on wgsim, kept), Linux arm64 at genome scale, without leaving the Mac, Perf levers — measured (M-series, -t1, région 2 Mbp, 500k reads, median of 3), Rescue chunking on REAL data: a null result, and two measurement traps (+8 more)

### Community 30 - "emit.rs"
Cohesion: 0.16
Nodes (13): BLOCK, FWD16, FWD_TABLE, integers_match_to_string(), push_int(), push_int_str(), push_qual_rev(), push_seq_fwd() (+5 more)

### Community 31 - "sw.rs"
Cohesion: 0.12
Nodes (32): call(), CANDS, diagonal_tied_by_a_gapped_path_is_still_the_cigar(), exact_match_scores_full_length(), EXC_MAX_V, excursions_cannot_gain(), global_cell(), global_exact_match_is_all_m() (+24 more)

### Community 32 - "ExtendJob"
Cohesion: 0.15
Nodes (31): band_order(), band_split(), batched_extend(), batched_extend_scalar(), bucket(), cell_bound(), clamp_band(), dead_seed() (+23 more)

### Community 33 - "The SIMD DP kernels: what is reachable, and what is already ruled out"
Cohesion: 0.12
Nodes (16): An `x86-64-v4` release binary, the AVX-512 tier build (2026-08-27), Issue #48 is finished, and the extension kernel's remaining gap is its recurrence (2026-08-29), Issue #48 lever A is already done, by LLVM, and the disassembly says so (2026-08-29), Part B of #44: the port arbitrage on the AVX-512 leaf maxes (2026-08-26), Part three: dead ends found after the file was written, The GPU probe reproduces the kernel's shape but not its cache, and that is why it stops here (2026-08-30), The memory-level-parallelism knobs are already right on x86. NULL., The mismatch-only shortcut for the CIGAR's global DP (2026-08-20) (+8 more)

### Community 34 - "run"
Cohesion: 0.29
Nodes (13): batched_regs(), Deferred, dump_regs_enabled(), emit_deferred(), emit_resolved(), finish_se(), finish_se_meth(), forget_index() (+5 more)

### Community 35 - "fastq.rs"
Cohesion: 0.13
Nodes (19): acgt_code(), fifo_attempt(), first_le_space(), first_le_space_matches_the_byte_scan(), INFLATE_BLOCK, INFLATE_DEPTH, is_acgt(), pack64() (+11 more)

### Community 36 - "build.rs"
Cohesion: 0.13
Nodes (24): amb_holes_not_merged_across_contig_boundary(), AmbRec, AnnRec, build_bwt_and_samples(), build_index(), build_index_parts(), build_index_with_prefix(), CP_BLOCK_SIZE (+16 more)

### Community 37 - "bwa-mem/src/meth.rs"
Cohesion: 0.14
Nodes (15): build_xm(), comment_tokens_reencode_like_htslib(), format_g(), leading_float(), leading_int(), MIN_LONGEST_M_PCT, op(), percent_g_matches_c() (+7 more)

### Community 38 - "global_wave.rs"
Cohesion: 0.11
Nodes (20): IOTA, load_table(), shim!, wave_fill!, max_lanes(), PAD, REV, safe() (+12 more)

### Community 39 - "bwa-extend/src/lib.rs"
Cohesion: 0.13
Nodes (20): assert_backend_batch_matches_scalar(), assert_backend_batch_order_invariant(), assert_backend_matches_scalar(), assert_backend_saturation_boundary_matches_scalar(), assert_backend_tie_rule_matches_scalar(), BatchTicket, ExtendJob, fill_scmat_vec() (+12 more)

### Community 40 - "__m256i"
Cohesion: 0.19
Nodes (21): band_epi16(), band_epi8(), band_epu8(), bsl128(), bsl256(), bsl512(), ceq16(), ceq8() (+13 more)

### Community 41 - "Params"
Cohesion: 0.33
Nodes (12): components(), components_at(), decide_counts(), decide_rel(), dense_bound(), dense_decision(), dense_scan_equals_sparse_scan(), extend() (+4 more)

### Community 42 - "bwa-neon/src/lib.rs"
Cohesion: 0.12
Nodes (9): diag_score(), diag_score_matches_scalar(), diag_score_neon(), diag_score_ssse3(), neon_batch_matches_scalar_read_sized(), neon_ungapped_hit_matches_scalar(), NeonBackend, rewrite_jobs() (+1 more)

### Community 43 - "align_read"
Cohesion: 0.19
Nodes (20): discard_enabled(), skip_contained_enabled(), align_chains(), align_read(), align_read_dedup(), align_read_meth(), align_read_se(), best() (+12 more)

### Community 44 - "mlp_probe.c"
Cohesion: 0.13
Nodes (9): now_ns(), pf(), run(), main(), try_alloc(), build(), chase(), main() (+1 more)

### Community 45 - "Output"
Cohesion: 0.24
Nodes (7): bam_header_only_still_produces_a_file(), bam_transcoder_reassembles_split_lines(), Output, Bam, Bgzf, Cram, Plain

### Community 46 - "prologue"
Cohesion: 0.18
Nodes (19): band_s16(), band_u8(), band_u8_range_compare_is_exhaustively_the_two_compares(), bits(), bits16(), epilogue(), hmax(), hmin() (+11 more)

### Community 47 - "scmat"
Cohesion: 0.20
Nodes (18): avx2_matesw_i16_matches_scalar(), avx512_avx2_matesw_uniform_qlen_pad_columns_equal_scalar(), avx512_matesw_i16_matches_scalar(), avx512_matesw_u8_matches_scalar(), fwd_local_sw_scalar(), mat5_add_code_covers_meth_matrices(), mate_rescue_long_mate_start_is_exact(), matesw_equals_scalar() (+10 more)

### Community 48 - "batched_align"
Cohesion: 0.14
Nodes (21): batched_align(), batched_ksw_align2(), batched_mate_rescue(), batched_mate_rescue_lazy(), bucket(), count_outcome(), dump(), enabled() (+13 more)

### Community 49 - "updates"
Cohesion: 0.10
Nodes (20): patterns, patterns, prefix, actions, cargo, dependency-name, update-types, day (+12 more)

### Community 50 - "steps"
Cohesion: 0.10
Nodes (20): if, runs-on, steps, GH_TOKEN, PR_URL, jobs, auto-merge, name (+12 more)

### Community 51 - "tiny"
Cohesion: 0.16
Nodes (12): backward_search_counts_match_naive(), get_sa_matches_sais_and_is_permutation(), rc_unpack_equals_base_by_base(), sa_samples_match_the_file(), text_match_equals_base_by_base(), tiny(), unpack_pac_base(), unpack_pac_range() (+4 more)

### Community 52 - "steps"
Cohesion: 0.10
Nodes (19): concurrency, cancel-in-progress, group, RUSTFLAGS, jobs, static-openmp, name, on (+11 more)

### Community 53 - "sais.rs"
Cohesion: 0.20
Nodes (18): bucket_ends(), bucket_starts(), check(), check_inplace(), EMPTY, IEMPTY, induce(), inplace_known_strings() (+10 more)

### Community 55 - "run_staged"
Cohesion: 0.15
Nodes (17): contains(), discard_contained(), interferes(), judge(), KeptIndex, .BUCKET_SHIFT, .HBITS, .MAX_BUCKET_WALK (+9 more)

### Community 56 - "FwdJob"
Cohesion: 0.31
Nodes (19): enc_bases(), extract_group(), fastcol_enabled(), fwd_local_sw_avx2_i16(), fwd_local_sw_avx2_u8_impl(), fwd_local_sw_avx512_i16(), fwd_local_sw_avx512_u8_impl(), fwd_local_sw_neon() (+11 more)

### Community 57 - "MemAlnReg"
Cohesion: 0.21
Nodes (15): grow_one(), insert_settled_view(), insert_settled_view_equals_insert_into_settled(), matesw_apply(), region(), RegionView, settle_order(), skip_from_view() (+7 more)

### Community 58 - "env"
Cohesion: 0.21
Nodes (18): CFLAGS, CXXFLAGS, REQUIRE_AVX512, RUSTFLAGS, steps, release-pgo-x86, steps, steps (+10 more)

### Community 59 - "bwa-mem4"
Cohesion: 0.11
Nodes (19): A caveat on the word "identical", Against fg-labs/bwa-mem3, ALT projection (`--alt-project`), Building, bwa-mem4, Credits, GPU, Input formats (+11 more)

### Community 60 - "Le travail litteralement duplique, deux fois dans la meme journee (2026-08-19)"
Cohesion: 0.11
Nodes (18): 1. Les marches du tableau de suffixes, 2. Les jobs de programmation dynamique, #44 est fini : la partie A vaut +11,4 % mesures, la partie B ne vaut rien (2026-08-26), Ce que la meme recherche a elimine, L'ecart wgsim x86, enfin attribue etage par etage (2026-08-19), L'enquete est close : l'anomalie scalaire etait la plateforme (2026-08-20), L'enquete ksw_global2, hypothese par hypothese (2026-08-20), La bande resserree est le defaut, et le match contre le fork bascule (2026-08-20) (+10 more)

### Community 61 - "ksw_align2"
Cohesion: 0.21
Nodes (10): ksw_align2(), bwa_matrix(), hyalite_scoring(), LANES, local_sw_score_matches_independent_implementation(), oracle_score(), oracle_with_score2(), our_score() (+2 more)

### Community 62 - "README.md"
Cohesion: 0.22
Nodes (9): Architecture, Branches and releases, Build, test, gates, Contributing, Current status (parity), Dependency updates, What this is, Where the NEON work plugs in (phase 9a) (+1 more)

### Community 63 - "opt.rs"
Cohesion: 0.15
Nodes (12): ALL, ALPHABET_SIZE, AMBIGUOUS_BASE_SCORE, KEEP_SUPP_MAPQ, NO_MULTI, NO_RESCUE, NOPAIRING, PE (+4 more)

### Community 64 - "Prioritized levers"
Cohesion: 0.12
Nodes (16): Closest analog, CPU optimization roadmap (bwa-mem4), Do NOT, Flat / denser suffix array: `-t1` +12%, `-t8` +0%. Not worth the RAM. (2026-07-16, measured), GPU line: at its ceiling, and the ceiling is 1.28x (2026-07-16, measured), Idea "port getScores16's exact band" — refuted by reading the reference (2026-07), Lever #1 result (2026-07): measured, negative — abandoned on NEON, Levier essaye a la suite de cette revue, et negatif (+8 more)

### Community 65 - "inputs"
Cohesion: 0.12
Nodes (17): default, description, required, default, description, required, default, description (+9 more)

### Community 66 - "with"
Cohesion: 0.19
Nodes (17): steps, steps, steps, steps, id, if, name, run (+9 more)

### Community 67 - "bwa-core/src/meth.rs"
Cohesion: 0.14
Nodes (16): chem_names(), conversion_follows_read_number(), convert_codes(), converted_fasta_is_uppercased_and_wrapped(), parse_chem(), parse_tags(), PRUNE_KB, PRUNE_L (+8 more)

### Community 68 - "bwa-mem/src/alt_lift.rs"
Cohesion: 0.10
Nodes (39): ALT_BW_MAX, bed_footprint_overlap(), CAN_REPLACE, CANNOT_REPLACE, cigar_parse_rules(), drop_unlifted(), ExcludeBed, excluded_footprint() (+31 more)

### Community 69 - "Who else is making BWA-MEM fast, and what of it applies here"
Cohesion: 0.25
Nodes (5): Implementation attempt: QuadRank against `cp_occ`, measured (2026-08-09), The landscape, What the survey rules out, Who else is making BWA-MEM fast, and what of it applies here, Why QuadRank is the one that matters

### Community 70 - "perf-profile-x86.yml"
Cohesion: 0.12
Nodes (15): env, CARGO_TERM_COLOR, RUSTFLAGS, jobs, knobs, profile, if, runs-on (+7 more)

### Community 71 - "steps"
Cohesion: 0.23
Nodes (16): ALT_RUSTFLAGS, CG_FLAGS, CPU_MATCH, OURS_ENV, SET, steps, steps, steps (+8 more)

### Community 72 - "require_avx512"
Cohesion: 0.12
Nodes (16): read_pairs, require_avx512, runner, on, workflow_dispatch, default, description, required (+8 more)

### Community 73 - "kbtree.rs"
Cohesion: 0.28
Nodes (7): duplicate_pos_keeps_klib_order(), getp_aux(), KbTree, lower_picks_last_entry_below(), MAX_KEYS, Node, splits_preserve_order_and_lookup()

### Community 74 - "fwd_local_sw_batch"
Cohesion: 0.25
Nodes (17): prefers_wide_x86(), avx2_matesw_u8_matches_scalar(), bench_jobs(), fwd_local_sw_avx2_u8(), fwd_local_sw_avx512_u8(), fwd_local_sw_batch(), fwd_local_sw_neon_u8(), fwd_local_sw_sse41_u8() (+9 more)

### Community 75 - "bench-x86.yml"
Cohesion: 0.13
Nodes (14): concurrency, cancel-in-progress, group, env, CARGO_TERM_COLOR, RUSTFLAGS, jobs, kernel-ab (+6 more)

### Community 76 - "docker_gates.sh"
Cohesion: 0.40
Nodes (14): cmd_bench(), cmd_build(), cmd_check(), cmd_fixture(), cmd_fork(), cmd_icount(), cmd_parity(), cmd_rss() (+6 more)

### Community 77 - "mem_sam_pe"
Cohesion: 0.20
Nodes (10): 6. How to work on this safely, Diagnostic env vars, The build gotcha, before anything else, The gates that exist, The method that actually cracks a divergence, The standing lesson, Where to look to change X, id_shift_c() (+2 more)

### Community 78 - "OnceLock"
Cohesion: 0.20
Nodes (9): append_comment(), append_rg_tag(), COPY_COMMENT, copy_comment(), escape(), MAX_RG_ID_BUFFER, RG_ID, rg_id() (+1 more)

### Community 79 - "sais_rec"
Cohesion: 0.35
Nodes (7): bucket_ends_i(), bucket_starts_i(), induce_i(), IntSlice, lms_eq(), sais_rec(), TypeBits

### Community 80 - "sam.rs"
Cohesion: 0.26
Nodes (10): alt_contig_sq_line_carries_ah(), header_has_no_hd_and_exact_sq(), header_inserts_rg_and_hdr_lines(), SAM_MISSING, SqRecord, unmapped_missing_qual_is_star(), unmapped_record_shape(), write_header() (+2 more)

### Community 81 - "mem_gen_alt"
Cohesion: 0.31
Nodes (9): get_pri_idx(), hit_counts(), mem_gen_alt(), near_primary_alternate_grouped_distant_one_dropped(), reg(), too_many_hits_drops_all_of_that_primary(), xa_group(), XaTags (+1 more)

### Community 82 - "bench-arm.yml"
Cohesion: 0.14
Nodes (13): concurrency, cancel-in-progress, group, env, CARGO_TERM_COLOR, RUSTFLAGS, jobs, name (+5 more)

### Community 83 - "max_threads"
Cohesion: 0.14
Nodes (14): max_threads, read_pairs, run_graviton, default, description, required, workflow_dispatch, default (+6 more)

### Community 84 - "dry_run"
Cohesion: 0.14
Nodes (14): default, description, type, dry_run, tag, on, pull_request, push (+6 more)

### Community 85 - "bntseq.rs"
Cohesion: 0.17
Nodes (11): BntSeq, comment_is_what_the_c_reads_back_from_ann(), fmt_err(), intv2rid_agrees_with_two_searches(), next_parse(), parse_amb(), parse_ann(), parses_committed_tiny_index() (+3 more)

### Community 86 - "seedflt.rs"
Cohesion: 0.26
Nodes (8): KSW_I16_LANES, mem_flt_chained_seeds(), MEM_HSP_COEF, MEM_MINSC_COEF, mem_seed_sw(), MEM_SEEDSW_COEF, MEM_SHORT_EXT, MEM_SHORT_LEN

### Community 87 - "lockstep-sweep"
Cohesion: 0.15
Nodes (13): run, defaults, lockstep-sweep, pgo-arm, defaults, name, runs-on, timeout-minutes (+5 more)

### Community 88 - "inputs"
Cohesion: 0.15
Nodes (13): default, description, required, default, description, required, base_ref, e2e_runner (+5 more)

### Community 89 - "steps"
Cohesion: 0.23
Nodes (13): steps, BASE_REF, READ_PAIRS, REQUIRE_AVX512_E2E, VARIANT_CPUS, steps, steps, env (+5 more)

### Community 90 - "hugepage.rs"
Cohesion: 0.26
Nodes (8): advise_hugepage(), disabled(), MADV_HUGEPAGE, madvise(), page_aligned_range(), page_aligned_range_rounds_inward(), SC_PAGESIZE, sysconf()

### Community 91 - "mem_sam_pe"
Cohesion: 0.13
Nodes (25): Anchor, bns_fetch_seq(), bns_fetch_seq_into(), cal_sub(), dump_pestat(), erfc(), matesw_collect(), mem_infer_dir() (+17 more)

### Community 92 - "unwrap_multiline_fastq"
Cohesion: 0.23
Nodes (6): needs_unwrapping(), unwrap_detection(), unwrap_multiline_fastq(), unwrap_rewrites_wrapped_records(), UnwrapFastq, UnwrapFastq<R>

### Community 93 - "Methylation mode (`--meth`)"
Cohesion: 0.17
Nodes (11): Differences from bwa-mem3 0.18, How it aligns, Index, Methylation mode (`--meth`), Options, Output, Run through Bismark, Speed (+3 more)

### Community 94 - "instant"
Cohesion: 0.31
Nodes (6): main(), report(), run_one(), with_sentinel(), main(), suffix_array_inplace()

### Community 95 - "Deep research: is there an unexplored exact speedup? (2026-07-29)"
Cohesion: 0.18
Nodes (10): Closing the q-gram door with a derivation instead of an assertion, Deep research: is there an unexplored exact speedup? (2026-07-29), Everything else, and why it is closed, Round 2: newest-first sweep, and the STAR / Salmon question (same day), Salmon is not an aligner, and that is the whole answer, Sources, STAR is fast for reasons we are forbidden from copying, The conclusion the review keeps arriving at (+2 more)

### Community 96 - "A machine-independent model of this aligner's thread scaling"
Cohesion: 0.18
Nodes (11): 1. The model, 2.1 `ALU(N)`: the compute ceiling, 2.2 `RANDOM_BW(N)`: the random-access ceiling, 2.3 TLB reach: `entries x page_size`, 2. The three numbers to measure on a new machine, 3. What the model forbids, and what it permits, 4. The MLP corollary: why the lockstep width has a right answer, 5. Why "batch the accesses" cannot be applied twice (+3 more)

### Community 97 - "env"
Cohesion: 0.18
Nodes (10): env, BWA_MEM2_SHA256, BWA_MEM2_VERSION, CARGO_TERM_COLOR, RUST_TOOLCHAIN, RUSTFLAGS, jobs, name (+2 more)

### Community 98 - "with"
Cohesion: 0.18
Nodes (11): with, condarc, create-args, download-micromamba, environment-name, if-no-files-found, init-shell, micromamba-binary-path (+3 more)

### Community 99 - "jobs"
Cohesion: 0.18
Nodes (10): concurrency, cancel-in-progress, group, jobs, pgo-x86, name, defaults, name (+2 more)

### Community 101 - "cmd_longread.rs"
Cohesion: 0.16
Nodes (10): build_mmi(), load_or_build_index(), map_batch(), mmi_path(), preset_for(), preset_slug(), rammap_version(), reference_fasta() (+2 more)

### Community 102 - "star_vs_fm.rs"
Cohesion: 0.31
Nodes (7): fwd_ext(), K, main(), now(), prefetch(), QLEN, W

### Community 103 - ".load"
Cohesion: 0.27
Nodes (6): load_refuses_truncated_or_mixed_files(), rd_i64(), read_at_parallel(), read_exact_at(), seed_jump_level_off(), sibling()

### Community 104 - "FastqReader"
Cohesion: 0.33
Nodes (4): empty_input_is_zero_records(), FastqReader, InterleavedFastqReader, non_utf8_header_keeps_its_bytes()

### Community 105 - "#57 : perf(gpu): co-ordonnancement CPU+GPU, le partage des jobs est gratuit en risque"
Cohesion: 0.20
Nodes (10): #57 : perf(gpu): co-ordonnancement CPU+GPU, le partage des jobs est gratuit en risque, Ce que ce ticket ne fait pas, Critere d'acceptation, L'interaction avec la RAM (#25), Le probleme a resoudre, Le theoreme qui rend tout ceci gratuit en risque, Niveau 1 : le partage statique (une journee, prend l'essentiel), Niveau 2 : le partage adaptatif (petit, et il rend le reglage inutile) (+2 more)

### Community 106 - "Design: speed and RAM, measured against `fg-labs/bwa-mem3`"
Cohesion: 0.20
Nodes (9): Design: speed and RAM, measured against `fg-labs/bwa-mem3`, Explicit non-goals, Gates, identical for every change in every phase, Phase 0 — the baseline we are allowed to trust, Phase A — the frames nobody has opened, Phase B — bwa-mem3's seeding, conditional on phase 0, Phase C — making `-t8` end-to-end structurally cheaper, The goal, stated so it can be falsified (+1 more)

### Community 107 - "inputs"
Cohesion: 0.20
Nodes (10): default, description, required, base_ref, runner, workflow_dispatch, default, description (+2 more)

### Community 108 - "steps"
Cohesion: 0.20
Nodes (10): AVX2_ARM, BASE_REF, REQUIRE_AVX512, steps, env, if, name, run (+2 more)

### Community 109 - "CLAUDE.md"
Cohesion: 0.22
Nodes (7): A. Setup, 🧩 Advanced Concepts & Libraries, B. Build & Deployment, C. Testing & Validation Gates, 🏗️ Code Architecture (Big Picture), 🎯 Core Purpose, 🛠️ Development Workflow & Commands

### Community 110 - "MethOpt"
Cohesion: 0.20
Nodes (7): Chem, EmSeq, Taps, matrices_free_only_the_chemistry(), MethOpt, pbat_swaps_which_read_is_ct(), spec30_prunes_short_repeats_only_when_a_long_seed_exists()

### Community 111 - ".backward_ext"
Cohesion: 0.28
Nodes (6): dump(), enabled(), forward_ext_equals_swapped_backward(), konly_backward_step_equals_full(), seed_jump_table_equals_the_forward_walk(), SeedJump

### Community 112 - "rand48.rs"
Cohesion: 0.31
Nodes (5): ADD, MASK48, matches_libc_lrand48_seed11(), MULT, Rand48

### Community 113 - "open_reader"
Cohesion: 0.33
Nodes (6): gzip_threads(), InflatedBlocks, open_reader(), parsed(), spawn_inflate(), spawn_plain()

### Community 114 - "rescue_band.rs"
Cohesion: 0.08
Nodes (25): copy_fast(), fill_fast(), avx512_bands_reproduce_ksw_align2(), bands_reproduce_ksw_align2(), COMP_CAP, DEAD, enabled(), fit() (+17 more)

### Community 115 - "Global Constraints"
Cohesion: 0.22
Nodes (8): Global Constraints, Notes for whoever runs this, Phase 0: Fork Baseline Harness Implementation Plan, Task 1: bwa-mem4 reports its batch count, Task 2: `scripts/bench.sh` stops defaulting to the region index, Task 3: `scripts/fork_bench.sh`, the three-arm iteration harness, Task 4: the fork joins the 30x validation harness, Task 5: run Phase 0 and write down what it says

### Community 116 - "Catch fg-labs/bwa-mem3 0.16, then pass it"
Cohesion: 0.22
Nodes (8): Catch fg-labs/bwa-mem3 0.16, then pass it, Lever 1: banded rescue (the fork's `rescue_band`), priced GO, Lever 2: seeding and chaining (about 1.8x the fork's cost), Lever 3: idle samples, Measured against the fork (same 2M PE pairs, M4, `-t12`, CPU seconds), Order of work, revised, The round-1 rule is exact under our score2 too, Where we stand (6 Nov 2026)

### Community 117 - "with"
Cohesion: 0.22
Nodes (9): with, components, condarc, create-args, download-micromamba, environment-name, init-shell, micromamba-binary-path (+1 more)

### Community 118 - "with"
Cohesion: 0.22
Nodes (9): with, condarc, create-args, download-micromamba, environment-name, fetch-depth, init-shell, micromamba-binary-path (+1 more)

### Community 119 - "steps"
Cohesion: 0.22
Nodes (9): option-parity, runs-on, steps, timeout-minutes, continue-on-error, if, name, run (+1 more)

### Community 120 - "with"
Cohesion: 0.22
Nodes (9): with, components, condarc, create-args, download-micromamba, environment-name, init-shell, micromamba-binary-path (+1 more)

### Community 121 - "release.yml"
Cohesion: 0.22
Nodes (8): env, CARGO_TERM_COLOR, RUST_TOOLCHAIN, RUSTFLAGS, jobs, name, permissions, contents

### Community 122 - "env"
Cohesion: 0.22
Nodes (9): CFLAGS, CXXFLAGS, GH_TOKEN, GITHUB_REPOSITORY, MACOSX_DEPLOYMENT_TARGET, OPENMP_STATIC, SANITISE, TAG (+1 more)

### Community 123 - "Le GPU rouvert : le noyau tournait a 3,6 % de la machine, et la cause n'est pas le noyau (2026-08-29)"
Cohesion: 0.22
Nodes (9): Ce que cela dit de la suite, et cela reouvre le `uchar4` pour une autre raison que la premiere, Ce que le decoupage montre, et il corrige le diagnostic precedent, Ce qui borne vraiment le noyau GPU, mesure avec les deux noyaux (2026-08-29), Ce qui manquait : un gather par cellule, Et la correction ne transfere pas, ce qui est le resultat le plus utile des trois, La deuxieme cause, invisible jusqu'ici : `score2` sur l'hote, La vraie premiere cause : l'occupancy, et elle se corrige sans toucher au noyau, Le GPU rouvert : le noyau tournait a 3,6 % de la machine, et la cause n'est pas le noyau (2026-08-29) (+1 more)

### Community 124 - "opt_parity.sh"
Cohesion: 0.42
Nodes (8): check(), check_bgzf(), check_f(), check_longread(), check_o(), check_primary5(), check_shapes(), opt_parity.sh script

### Community 125 - "global_icount.rs"
Cohesion: 0.32
Nodes (3): full_band(), mat(), tightened()

### Community 126 - "Error"
Cohesion: 0.29
Nodes (6): Error, Fastq, IndexFormat, Io, Other, Result

### Community 127 - "SuboptimalTracker"
Cohesion: 0.18
Nodes (8): wave16_matches_scalar(), dump(), enabled(), ksw_global2_scalar(), ksw_local_fwd(), push_cigar(), RunCigar, SuboptimalTracker

### Community 129 - "What fg-labs/bwa-mem3 does that we do not"
Cohesion: 0.25
Nodes (8): First, what their numbers mean, Ranked, what this teardown says to do, `src/read_arena.h` — measured, not worth it here, `src/sam_encode.cpp` — taken, in portable form, `src/smem_dedup.cpp` — worth testing, not yet tried, The finding that explains the x86 gap, The rest, briefly, What fg-labs/bwa-mem3 does that we do not

### Community 130 - "Les sept pieges specifiques au GPU"
Cohesion: 0.25
Nodes (8): 1. Aucun flottant, 2. L'egalite d'argmax : le premier maximum gagne, 3. La saturation a 255 et le repli i16, 4. Le codage de N, 5. `max_off` et la boucle de rounds, 6. La divergence de longueur ne doit pas changer le resultat, 7. Le partage CPU/GPU ne doit rien changer, Les sept pieges specifiques au GPU

### Community 131 - "#55 : perf(gpu): backend Metal, Apple Silicon, memoire unifiee et zero copie"
Cohesion: 0.25
Nodes (8): #55 : perf(gpu): backend Metal, Apple Silicon, memoire unifiee et zero copie, Attendu, Ce qu'il faut verifier avant d'ecrire une ligne (etape 0), Etapes, L'avantage structurel : memoire unifiee, La disposition, et pourquoi elle est deja bonne, Le calcul qui justifie de commencer par la, Le portage lui-meme

### Community 132 - "#56 : perf(gpu): backend CUDA, octets satures __vaddus4 puis DPX s16x2 sur Hopper"
Cohesion: 0.25
Nodes (8): #56 : perf(gpu): backend CUDA, octets satures __vaddus4 puis DPX s16x2 sur Hopper, Ce qu'on ne refait pas, DPX sur Hopper : utile, mais pas pour la raison qu'on croit, Etapes, Le point de depart : CUDA a exactement ce qu'il faut, depuis Kepler, Les transferts, Materiel, Sources

### Community 133 - "Mate rescue: the next lever, priced before it is built"
Cohesion: 0.25
Nodes (7): Mate rescue: the next lever, priced before it is built, Phase 0: re-measure on the current tree (~20 min, quiet host), Phase 1: price the component bands offline (no kernel change), Phase 2: build it (only on a go), Phase 3: gates and measurement, What is already closed (do not retry), Where the cells still are (hypothesis to verify first)

### Community 134 - "steps"
Cohesion: 0.46
Nodes (8): steps, steps, steps, if, name, run, shell, uses

### Community 135 - "endtoend-smallref"
Cohesion: 0.25
Nodes (8): run, defaults, if, name, runs-on, timeout-minutes, endtoend-smallref, shell

### Community 136 - "with"
Cohesion: 0.25
Nodes (8): with, condarc, create-args, download-micromamba, environment-name, init-shell, micromamba-binary-path, toolchain

### Community 137 - "env"
Cohesion: 0.25
Nodes (8): EVENT_NAME, IDX, KNOWN_X86_XS_DIVERGENCE, R1, R2, READS, REQUIRE_AVX512, env

### Community 138 - "run"
Cohesion: 0.25
Nodes (8): run, perf-x86, defaults, name, runs-on, timeout-minutes, defaults, shell

### Community 139 - "build.sh"
Cohesion: 0.25
Nodes (7): BINDGEN_EXTRA_CLANG_ARGS, CFLAGS, CPPFLAGS, LDFLAGS, LIBCLANG_PATH, RUSTFLAGS, build.sh script

### Community 140 - "unpack16"
Cohesion: 0.33
Nodes (4): unpack16(), unpack64(), unpack_fwd_neon(), unpack_pac_rev_neon()

### Community 141 - "seq_emit_bench.rs"
Cohesion: 0.38
Nodes (5): BLOCK, FWD, main(), push_as_chunks(), push_chunks_exact()

### Community 142 - "Tier"
Cohesion: 0.29
Nodes (7): calibrate(), Tier, Avx2, Avx512, Scalar, Sse41, time

### Community 143 - "RescueTier"
Cohesion: 0.29
Nodes (7): forced_rescue_tier(), RescueTier, Auto, Avx2, Avx512, Scalar, Sse41

### Community 144 - "#52 : meta: hybride CPU+GPU, le plafond est 2,45x et le GPU integre suffit deja"
Cohesion: 0.29
Nodes (7): 1. Le plafond d'Amdahl, calcule sur notre propre profil, 2. Combien de GPU faut-il pour atteindre ce plafond ? Bien moins qu'on ne croit, 3. Ce que l'octet-identite autorise et interdit sur GPU, 4. La concurrence, pour situer la cible, #52 : meta: hybride CPU+GPU, le plafond est 2,45x et le GPU integre suffit deja, 5. L'ordre des travaux, Sources

### Community 145 - "Literature review: what is left to take, under byte-identity (2026-07-29)"
Cohesion: 0.29
Nodes (6): 1. The paper that matters: minibwa (Li & Homer, arXiv 2606.15357, June 2026), 2. The other accelerators, and why each is closed, 3. What survives, ranked, 4. Two ideas killed by measurement today, 5. Sources, Literature review: what is left to take, under byte-identity (2026-07-29)

### Community 146 - "Where the real-data paired-end time actually goes (2026-07-29 profile)"
Cohesion: 0.29
Nodes (7): Attacking the 13.3%: two byte-identical changes, ~2% together, Attempted and null: hoisting the dead merge check out of the dedup scan, Flat arena for the mate-rescue reverse pass: NULL, Index-tag (indirect) introsort in the dedup sorts: REGRESSION, reverted, Inflate backend for gzipped FASTQ: real on the reader, ~1.2% ceiling on wall, `mem_sort_dedup_patch`: the shape of the 13.3%, measured, Where the real-data paired-end time actually goes (2026-07-29 profile)

### Community 147 - "The mate-rescue kernel: padding-free column range (BANKED, -9.7% wall)"
Cohesion: 0.29
Nodes (7): How the kernel was found to be the lever, Measured, On-demand argmax column in the rescue kernel: REGRESSION (+18% kernel CPU), reverted, Ported to every kernel, The change, The mate-rescue kernel: padding-free column range (BANKED, -9.7% wall), The same change under PGO, and the resulting standing against the fork

### Community 148 - "head-to-head.yml"
Cohesion: 0.29
Nodes (6): concurrency, cancel-in-progress, group, jobs, name, on

### Community 149 - "head-to-head"
Cohesion: 0.29
Nodes (7): run, defaults, name, runs-on, timeout-minutes, head-to-head, shell

### Community 150 - "build"
Cohesion: 0.29
Nodes (7): needs, runs-on, strategy, timeout-minutes, build, fail-fast, matrix

### Community 151 - "check-version"
Cohesion: 0.29
Nodes (7): outputs, runs-on, timeout-minutes, check-version, publishing, tag, version

### Community 152 - "include"
Cohesion: 0.29
Nodes (7): deployment_target, omp, os, pgo, target_id, variant_cpu, include

### Community 153 - "giab32_headline.sh"
Cohesion: 0.57
Nodes (6): allowed(), note(), refuse(), run(), say(), giab32_headline.sh script

### Community 154 - "giab_publication.sh"
Cohesion: 0.62
Nodes (5): load1(), run(), say(), giab_publication.sh script, wait_quiet()

### Community 155 - "Apple Silicon memory-subsystem probes"
Cohesion: 0.33
Nodes (5): Apple Silicon memory-subsystem probes, `mlp_probe.c` — what the memory system actually delivers, `superpage_probe.c` — are huge pages available?, ⚠️ The trap this probe was built to catch, `tlb_probe.c` — the TLB hierarchy and its cost

### Community 156 - "hybrid_bench.rs"
Cohesion: 0.53
Nodes (4): load_sa(), main(), nt4(), read_fastq()

### Community 157 - "bwt_binning.rs"
Cohesion: 0.53
Nodes (4): bin_sort(), main(), pass(), W

### Community 158 - "bwa-index/src/lib.rs"
Cohesion: 0.33
Nodes (3): Amb, Contig, SEED_JUMP_DEPTH

### Community 161 - "RowMax"
Cohesion: 0.33
Nodes (3): i32, RowMax, u8

### Community 162 - "Adding a GPU, without giving up byte-identity"
Cohesion: 0.33
Nodes (5): #54 : test(gpu): la barriere d'octet-identite pour un backend GPU, a ecrire avant le premier noyau, Adding a GPU, without giving up byte-identity, Ce que ce gate ne couvre pas, Ce qui existe et qu'il suffit d'etendre, Le gate d'integration

### Community 163 - "#53 : perf(gpu): la couture asynchrone dans SwBackend, et mesurer la taille des lots"
Cohesion: 0.33
Nodes (6): #53 : perf(gpu): la couture asynchrone dans SwBackend, et mesurer la taille des lots, Ce qui existe deja, et qui est mieux que prevu, Critere d'acceptation, Etape 0 : mesurer la taille des lots (a faire en premier, c'est une demi-journee), Etape 1 : la couture asynchrone, Etape 2 : le crate

### Community 164 - "Three-arm head-to-head, genome scale, PE, `-t12`: wgsim says we win, real data says we lose"
Cohesion: 0.33
Nodes (6): PGO closes the whole real-data gap, Revived and shipped: flattening the rescue rounds (the fork's shape), The question this hands to the Graviton benchmark, Three-arm head-to-head, genome scale, PE, `-t12`: wgsim says we win, real data says we lose, What this makes the next lever, Why this matters more than either number

### Community 165 - "runner"
Cohesion: 0.33
Nodes (6): runner, workflow_dispatch, default, description, required, inputs

### Community 166 - "on"
Cohesion: 0.33
Nodes (6): on, pull_request, push, schedule, branches, cron

### Community 167 - "with"
Cohesion: 0.33
Nodes (6): with, key, name, path, retention-days, toolchain

### Community 168 - "La voie GPU : le plafond est 2,45x, et le GPU integre suffit deja (2026-08-08)"
Cohesion: 0.33
Nodes (6): Ce que l'octet-identite autorise, et c'est plus large qu'attendu, Ce que la concurrence fait, et pourquoi son levier principal ne nous est pas offert, La couture existe deja, La voie GPU : le plafond est 2,45x, et le GPU integre suffit deja (2026-08-08), Le plafond, calcule sur notre propre profil, Le resultat qui n'etait pas attendu : il ne faut presque pas de GPU

### Community 169 - "Le mate rescue, six angles, et le seul qui paie (2026-08-11)"
Cohesion: 0.33
Nodes (6): Ce que la sonde `BWA4_MATESW_TIME` dit du noyau, Classement sur echantillon complet apres fastp, Le dedup incremental : prouve, octet-identique, et nul, Le mate rescue, six angles, et le seul qui paie (2026-08-11), Le tri par longueur du batch de rescue : +3,5 %, et pourquoi il etait nul le matin, Une anomalie non expliquee

### Community 170 - "Le mate rescue, le plus gros poste restant (2026-08-07)"
Cohesion: 0.33
Nodes (6): Ce qui a bien mieux marche : supprimer la reparation au lieu de la sauter (2026-08-08), Ce qui a marche : quatre lignes de cible a la fois, Ce qui n'a pas marche : sauter la reparation du N, La serie complete, un seul banc, Le DP principal : la meme forme biaisee, et un resultat decevant, Le mate rescue, le plus gros poste restant (2026-08-07)

### Community 171 - "Le dossier x86 : ou nous perdons, mesure sur le meme hote (2026-08-18)"
Cohesion: 0.33
Nodes (6): Deux corrections sur mes propres chiffres du jour (2026-08-18), L'indexeur parallele est livre, et l'objection tombe par la mesure (2026-08-19), Le dossier x86 : ou nous perdons, mesure sur le meme hote (2026-08-18), Le jeu d'instructions vaut 5,6 a 6,8 %, et il est livre (2026-08-18), Le PGO de la release etait un chiffre Apple generalise, et il coutait 3,3 % sur x86 (2026-08-18), Le profil x86, lu enfin, et il ne dit pas ce que les issues supposaient

### Community 172 - "Le lecteur n'est plus le goulot, et la sonde mentait (2026-08-11)"
Cohesion: 0.33
Nodes (6): Deux zeros, consignes, La sonde `stage_time` cachait 95 % du run, Le lecteur n'est plus le goulot, et la sonde mentait (2026-08-11), Ou est vraiment le temps, hors sonde, rapidgzip redevient utile apres coup, `Record` : trois allocations par read, devenues une

### Community 173 - "run_pe"
Cohesion: 0.20
Nodes (10): batch_overlap_enabled(), count_demoted(), guided_bounds(), guided_chunks_tile_the_batch_and_keep_their_order(), pair_oa(), PeBatch, rescue_pairs_override(), rescue_pairs_per_chunk() (+2 more)

### Community 174 - "align_reads_batched_inner"
Cohesion: 0.25
Nodes (14): align_reads_batched(), align_reads_batched_inner(), align_reads_batched_meth(), batched_across_reads_equals_per_read(), dump(), enabled(), measure(), meth_seed_score() (+6 more)

### Community 175 - "Prune"
Cohesion: 0.40
Nodes (5): parse_prune(), Prune, Baseline, Off, Spec30

### Community 176 - "Scoring"
Cohesion: 0.40
Nodes (5): parse_scoring(), Scoring, Collapsed, Genomic, Neutral

### Community 177 - "score_sort_bench.rs"
Cohesion: 0.47
Nodes (4): key_u128(), key_u64x2(), main(), Row

### Community 178 - "run"
Cohesion: 0.18
Nodes (11): check_alt_project(), extra_header_lines(), load_alt_project(), MemArgs, queue_depth_defaults_to_double_buffering(), queue_depths(), report_demoted(), run() (+3 more)

### Community 180 - "piece16"
Cohesion: 0.40
Nodes (5): fill_soa(), piece16(), piece16_is_the_tile_zero_padded(), t16(), t16v()

### Community 181 - "Divergences connues vs oracle bwa-mem2 2.3"
Cohesion: 0.40
Nodes (4): Acceptees (par conception), Divergences connues vs oracle bwa-mem2 2.3, En cours (résidu), Résolu (phase 8, via oracle instrumenté)

### Community 182 - "The first side-by-side x86 profile against fg-labs/bwa-mem3 (2026-08-29)"
Cohesion: 0.40
Nodes (4): Reading it in absolute time, which is the only way it means anything, The first side-by-side x86 profile against fg-labs/bwa-mem3 (2026-08-29), What the two profiles say, What this file is not

### Community 183 - "graviton-selfhosted"
Cohesion: 0.40
Nodes (5): if, name, runs-on, timeout-minutes, graviton-selfhosted

### Community 184 - "kernel_ab_only"
Cohesion: 0.40
Nodes (5): kernel_ab_only, default, description, required, type

### Community 185 - "require_avx512"
Cohesion: 0.40
Nodes (5): require_avx512, default, description, required, type

### Community 186 - "require_avx512_e2e"
Cohesion: 0.40
Nodes (5): require_avx512_e2e, default, description, required, type

### Community 187 - "avx2_arm"
Cohesion: 0.40
Nodes (5): default, description, required, type, avx2_arm

### Community 188 - "require_avx512"
Cohesion: 0.40
Nodes (5): require_avx512, default, description, required, type

### Community 189 - "require_avx512"
Cohesion: 0.40
Nodes (5): require_avx512, default, description, required, type

### Community 190 - "isa-and-tiers-x86"
Cohesion: 0.40
Nodes (5): defaults, name, runs-on, timeout-minutes, isa-and-tiers-x86

### Community 191 - "real-reads-x86"
Cohesion: 0.40
Nodes (5): real-reads-x86, defaults, name, runs-on, timeout-minutes

### Community 192 - "promote"
Cohesion: 0.40
Nodes (5): promote, if, needs, runs-on, timeout-minutes

### Community 193 - "publish"
Cohesion: 0.40
Nodes (5): publish, if, needs, runs-on, timeout-minutes

### Community 194 - "#53 etapes 1 et 2 : la couture asynchrone et le crate `bwa-gpu` (2026-08-09)"
Cohesion: 0.40
Nodes (5): #53 etapes 1 et 2 : la couture asynchrone et le crate `bwa-gpu` (2026-08-09), Etape 1, le contrat non bloquant, Etape 2, le crate, La limite, dite plutot que tue, Les trois criteres d'acceptation

### Community 195 - "#55 : le backend Metal existe, il est octet-identique, et il perd contre un coeur (2026-08-09)"
Cohesion: 0.40
Nodes (5): #55 : le backend Metal existe, il est octet-identique, et il perd contre un coeur (2026-08-09), Ce qu'il fait, Le chiffre, et il n'est pas bon, Les barrieres, appliquees au GPU, Pourquoi il n'est pas branche dans l'aligneur

### Community 196 - "Les quatre backends de tableau des suffixes, mesures (2026-08-11)"
Cohesion: 0.40
Nodes (5): CaPS-SA : inutilisable ici, pour une raison algorithmique, Ce qui est livre, Les quatre backends de tableau des suffixes, mesures (2026-08-11), Paralleliser divise le CPU total, et ce n'est pas une erreur de mesure, Un zero de plus

### Community 197 - "Le piege de jeu de donnees, et le classement refait (2026-08-11)"
Cohesion: 0.40
Nodes (5): Ce que ca dit, La discipline, d'abord, Le piege de jeu de donnees, et le classement refait (2026-08-11), `-t16`, meme jeu, meme discipline, 6 rondes, `-t4`, 500 k paires chr21, gzip, 6 rondes, medianes

### Community 198 - "Etat de l'art : personne ne fait mieux a contrat de sortie egal (2026-08-08)"
Cohesion: 0.40
Nodes (5): Ce que ca ferme, et ce que ca n'ouvre pas, Corroboration inattendue, vieille de quinze ans, Etat de l'art : personne ne fait mieux a contrat de sortie egal (2026-08-08), Le comparatif le plus direct qui existe : le BSW de bwa-mem2 lui-meme, Le seul qui nous depasse, et pourquoi ca ne compte pas tel quel

### Community 199 - "Le chemin critique du lecteur, et ce que rapidgzip y change (2026-08-11)"
Cohesion: 0.40
Nodes (5): Ce qui reste sur ce chemin, Deux correctifs, mesures, Le chemin critique du lecteur, et ce que rapidgzip y change (2026-08-11), Ou est le serie, mesure, Un bug trouve en chemin, et repare

### Community 200 - "Le regime de colonnes pleinement en bande, trouve par la mesure (2026-08-09)"
Cohesion: 0.40
Nodes (5): La preuve qui a triple sa portee, Le comptage qui a designe le levier, Le levier, Le profil qui l'a designe, Le regime de colonnes pleinement en bande, trouve par la mesure (2026-08-09)

### Community 201 - "Phase 9e : la vague perf (tout octet-identique)"
Cohesion: 0.40
Nodes (5): Le fait perf le plus structurant : le lockstep a mange le levier, Le mur du mate rescue : le nombre de cellules EST l'algorithme, Le profil reel, et pourquoi l'ancien etait faux, Phase 9e : la vague perf (tout octet-identique), Scaling en threads

### Community 202 - "fork_bench.sh"
Cohesion: 0.60
Nodes (3): med(), say(), fork_bench.sh script

### Community 203 - "giab_happy.sh"
Cohesion: 0.70
Nodes (4): align(), call(), score(), giab_happy.sh script

### Community 204 - "graviton_setup.sh"
Cohesion: 0.60
Nodes (4): have(), MAMBA_ROOT_PREFIX, say(), graviton_setup.sh script

### Community 205 - "unpack_bench.rs"
Cohesion: 1.00
Nodes (3): main(), unpack_scalar(), unpack_vector()

### Community 206 - "Bases"
Cohesion: 0.50
Nodes (4): Bases, Ascii, Codes, Packed

### Community 207 - "Mode"
Cohesion: 0.50
Nodes (4): Mode, Finish, Full, Staged

### Community 208 - "score2_lanes"
Cohesion: 0.50
Nodes (3): s2_rows_neon16(), score2_lanes(), Score2Args

### Community 209 - "Round 3: three structural ideas, all priced before building, all under the floor"
Cohesion: 0.50
Nodes (4): Band-free column loop in the seed-extension kernel: fires on 15% of rows, Round 3: three structural ideas, all priced before building, all under the floor, The rescue window is bwa's, exactly, Union of one read's rescue windows: 1.03x redundancy

### Community 210 - "hugepage-probe"
Cohesion: 0.50
Nodes (4): name, runs-on, timeout-minutes, hugepage-probe

### Community 211 - "read_pairs"
Cohesion: 0.50
Nodes (4): read_pairs, default, description, required

### Community 212 - "runner"
Cohesion: 0.50
Nodes (4): runner, default, description, required

### Community 213 - "variant_cpu"
Cohesion: 0.50
Nodes (4): variant_cpu, default, description, required

### Community 214 - "read_pairs"
Cohesion: 0.50
Nodes (4): read_pairs, default, description, required

### Community 215 - "extra_rustflags"
Cohesion: 0.50
Nodes (4): default, description, required, extra_rustflags

### Community 216 - "icount"
Cohesion: 0.50
Nodes (4): default, description, required, icount

### Community 217 - "icount"
Cohesion: 0.50
Nodes (4): if, runs-on, timeout-minutes, icount

### Community 218 - "knob_sweep"
Cohesion: 0.50
Nodes (4): knob_sweep, default, description, required

### Community 219 - "read_pairs"
Cohesion: 0.50
Nodes (4): read_pairs, default, description, required

### Community 220 - "read_set"
Cohesion: 0.50
Nodes (4): read_set, default, description, required

### Community 221 - "runner"
Cohesion: 0.50
Nodes (4): runner, default, description, required

### Community 222 - "meth"
Cohesion: 0.50
Nodes (4): meth, if, runs-on, timeout-minutes

### Community 223 - "on"
Cohesion: 0.50
Nodes (4): on, push, workflow_dispatch, branches

### Community 224 - "Les chiffres x86, enfin mesures : +17 % sur le noyau de rescue (2026-08-09)"
Cohesion: 0.50
Nodes (4): #44 reste non verifie, et ce n'est pas une loterie (2026-08-15), Les chiffres x86, enfin mesures : +17 % sur le noyau de rescue (2026-08-09), Noyau de rescue, `rescue_kernel_ab`, AVX2 u8, Noyaux d'extension, `x86_extend_ab`

### Community 225 - "#45 : `USQADD` et le `qsub(h, oe)` partage, +10,3 % de noyau, -2,7 % de CPU (2026-08-08)"
Cohesion: 0.50
Nodes (4): #45 : `USQADD` et le `qsub(h, oe)` partage, +10,3 % de noyau, -2,7 % de CPU (2026-08-08), Ce que la mesure donne, L'octet-identite, Le desassemblage, et les deversements qui disparaissent

### Community 226 - "#46 : les deux passes scalaires ne coutent rien, et pourquoi le microbenchmark disait le contraire (2026-08-08)"
Cohesion: 0.50
Nodes (4): #46 : les deux passes scalaires ne coutent rien, et pourquoi le microbenchmark disait le contraire (2026-08-08), A, la porte vectorielle sur `finish_row`, B, le pre-filtre vectoriel de `extract_group`, et le piege de mesure, Ce qui reste ouvert

### Community 227 - "#47 : les colonnes de bourrage ont leur propre corps, et le gain depend de la longueur de lecture (2026-08-08)"
Cohesion: 0.50
Nodes (4): #47 : les colonnes de bourrage ont leur propre corps, et le gain depend de la longueur de lecture (2026-08-08), Ce qui n'a pas ete fait, L'octet-identite, Le gain n'est pas un nombre, c'est une fonction de la fraction bourree

### Community 228 - "#53 etape 0 : la taille des lots, et elle repond deux choses differentes (2026-08-08)"
Cohesion: 0.50
Nodes (4): #53 etape 0 : la taille des lots, et elle repond deux choses differentes (2026-08-08), Ce que ce chiffre decide, Extension (`batched_extend`, la couture `SwBackend`), Mate rescue (`batched_ksw_align2`)

### Community 229 - "#54 : la barriere d'octet-identite GPU, ecrite avant le premier noyau (2026-08-08)"
Cohesion: 0.50
Nodes (4): #54 : la barriere d'octet-identite GPU, ecrite avant le premier noyau (2026-08-08), Ce qui reste des sept pieges, Deux barrieres generiques sur `SwBackend`, Le gate d'integration, `scripts/gpu_parity.sh`

### Community 230 - "L'extension sur GPU, branchee et mesuree : -22 % de CPU, et une bascule a -t8 (2026-08-09)"
Cohesion: 0.50
Nodes (4): Ce que cela dit de la suite, L'extension sur GPU, branchee et mesuree : -22 % de CPU, et une bascule a -t8 (2026-08-09), La mesure, et elle corrige ma propre prediction, Quand c'est utile aujourd'hui

### Community 231 - "Le noyau d'extension Metal : ecrit, octet-identique, 6,4x un coeur (2026-08-09)"
Cohesion: 0.50
Nodes (4): Ce que cela vaudrait, et ce qui manque, Ce qui a rendu la validation gratuite, Le noyau d'extension Metal : ecrit, octet-identique, 6,4x un coeur (2026-08-09), Les chiffres

### Community 232 - "Recherche multi-agents sur les noyaux SIMD (2026-08-08)"
Cohesion: 0.50
Nodes (4): Ce que la recherche a confirme, et qui clot un sujet, Honnetete sur la couverture, Le resultat qui explique le 1,25x de l'AVX-512, Recherche multi-agents sur les noyaux SIMD (2026-08-08)

### Community 233 - "Le `uchar4` sur Metal : tente, non concluant, retire (2026-08-09)"
Cohesion: 0.50
Nodes (4): Ce qui a ete appris en route, et qui reste vrai, Ce qui reste vrai pour la suite, Le `uchar4` sur Metal : tente, non concluant, retire (2026-08-09), Pourquoi c'est retire

### Community 234 - "Indexeur : le tableau de suffixes est parallelisable, mais pas par le portage Rust (2026-08-05)"
Cohesion: 0.50
Nodes (4): Ce qui a ete essaye et rejete, Indexeur : le tableau de suffixes est parallelisable, mais pas par le portage Rust (2026-08-05), Note historique, Pourquoi le portage pur Rust ne rattrape pas

### Community 235 - "Le chemin GPU est l'extension, pas le rescue, et les chiffres le disent (2026-08-09)"
Cohesion: 0.50
Nodes (4): Ce qui manque, et sa taille, Et l'agregation inter-threads y est deja prouvee sure, Le chemin GPU est l'extension, pas le rescue, et les chiffres le disent (2026-08-09), Les deux coutures, mesurees en production, meme course

### Community 236 - "Phase 0 de la campagne perf : mesure de reference contre le fork"
Cohesion: 0.50
Nodes (4): Historique phase 7-8 (traine PE), Phase 0 de la campagne perf : mesure de reference contre le fork, Phase A, levier RAM : pic 19,2 -> 11,2 Go (deux commits, octet-identiques), Phase B, le deficit PE = le kernel de sauvetage de mate

### Community 237 - "Trois questions fermees le meme jour : le gate GIAB, le PGO hors Apple, et l'indexeur parallele (2026-08-18)"
Cohesion: 0.50
Nodes (4): L'indexeur n'est pas mono-thread, il est mono-thread PAR DEFAUT (#37), Le PGO est un levier Apple Silicon, et rien d'autre (#33), Phase 11 : l'identite octet, dite en precision et rappel (#35), Trois questions fermees le meme jour : le gate GIAB, le PGO hors Apple, et l'indexeur parallele (2026-08-18)

### Community 238 - "giab30x_bench.sh"
Cohesion: 0.83
Nodes (3): run(), say(), giab30x_bench.sh script

### Community 239 - "giab30x_pe.sh"
Cohesion: 0.83
Nodes (3): run(), say(), giab30x_pe.sh script

### Community 240 - "pe_diverge_hunt.sh"
Cohesion: 1.00
Nodes (3): capture(), say(), pe_diverge_hunt.sh script

### Community 241 - "pe_diverge_subset.sh"
Cohesion: 0.83
Nodes (3): cap(), say(), pe_diverge_subset.sh script

### Community 242 - "pgo.sh"
Cohesion: 0.50
Nodes (3): CC, PATH, pgo.sh script

### Community 243 - "Vectorising what was left scalar: one kept, one reverted"
Cohesion: 0.67
Nodes (3): `.pac` unpack: vectorised, kept (unmeasured on a drifting host), Row-major `score2` scan: REGRESSION (+7.3% kernel CPU), reverted, Vectorising what was left scalar: one kept, one reverted

### Community 244 - "Where the `-t16` wall actually goes: `BWA4_STAGE_TIME=1`"
Cohesion: 0.67
Nodes (3): Two theories this killed, What the chr21 measurement structurally cannot see, Where the `-t16` wall actually goes: `BWA4_STAGE_TIME=1`

### Community 245 - "#25 tranche : le resident EST le vivant sous Linux, et le 2x est le double batch en vol (2026-08-18)"
Cohesion: 0.67
Nodes (3): #25 tranche : le resident EST le vivant sous Linux, et le 2x est le double batch en vol (2026-08-18), Et le churn ne coute pas de wall non plus : l'arene par worker est morte des deux cotes, Le prix du recouvrement, mesure : 1,1 % de wall pour 41 % de la RSS

### Community 246 - "#43 : la table de scores XOR portee sur AVX2 et SSE4.1, et le +18 a +50 % est a revoir (2026-08-09)"
Cohesion: 0.67
Nodes (3): #43 : la table de scores XOR portee sur AVX2 et SSE4.1, et le +18 a +50 % est a revoir (2026-08-09), Ce que le desassemblage montre, et pourquoi il corrige la projection, Ce qui est verifie ici

### Community 247 - "#48B : la pre-passe de substitution repliee, +4 a +5 % de noyau, -0,9 % de CPU (2026-08-08)"
Cohesion: 0.67
Nodes (3): #48B : la pre-passe de substitution repliee, +4 a +5 % de noyau, -0,9 % de CPU (2026-08-08), La difference avec #48A et #48D se voit dans les symboles, Mesure

### Community 248 - "#48C : le masque de bande en espace signe, et ce que le desassemblage x86 dit vraiment (2026-08-09)"
Cohesion: 0.67
Nodes (3): #48C : le masque de bande en espace signe, et ce que le desassemblage x86 dit vraiment (2026-08-09), Ce qui n'est PAS mesure, et il faut le dire, L'octet-identite, elle, est verifiee ici

### Community 249 - "#54 pieges 3 et 5 : la frontiere de saturation, et un `max_off` qui ne decide jamais rien (2026-08-08)"
Cohesion: 0.67
Nodes (3): #54 pieges 3 et 5 : la frontiere de saturation, et un `max_off` qui ne decide jamais rien (2026-08-08), Piege 3, la frontiere u8 / i16, Piege 5, `max_off` et la boucle de rounds : il ne se declenche jamais

### Community 250 - "#55 etape 0 : bloquee, il n'y a pas de compilateur Metal sur cette machine (2026-08-08)"
Cohesion: 0.67
Nodes (3): #55 etape 0 : bloquee, il n'y a pas de compilateur Metal sur cette machine (2026-08-08), Ce qui change dans le calcul si `subsat` manque, Le protocole exact a executer quand Xcode sera la

### Community 251 - "#55 etape 0 : debloquee sans Xcode, et le GPU tient 330 Gcell/s (2026-08-09)"
Cohesion: 0.67
Nodes (3): #55 etape 0 : debloquee sans Xcode, et le GPU tient 330 Gcell/s (2026-08-09), Ce que MSL expose reellement, Le plafond, mesure au lieu d'etre projete

### Community 252 - "Campagne perf CPU et classement contre minibwa (2026-08-09)"
Cohesion: 0.67
Nodes (3): Campagne perf CPU et classement contre minibwa (2026-08-09), Contre bwa-mem2 2.3 et contre minibwa, Contre le fork fg-labs/bwa-mem3 v0.9.0 (sorti le 2026-08-07)

### Community 253 - "L'oracle tiers couvre enfin les penalites de gap asymetriques (2026-08-12)"
Cohesion: 0.67
Nodes (3): Ce que l'elargissement a trouve, et qui n'est pas un bug, L'oracle tiers couvre enfin les penalites de gap asymetriques (2026-08-12), Verifie que le nouveau bras est vivant

### Community 254 - "Ou le fork est reellement moins cher, profil contre profil (2026-08-07)"
Cohesion: 0.67
Nodes (3): Ce qui etait recuperable dans ce plafond, et l'a ete, Le tri : ce que l'octet-identite coute exactement, chiffre, Ou le fork est reellement moins cher, profil contre profil (2026-08-07)

### Community 255 - "Entree gzip : ou est le mur, et a combien de cœurs il tombe (2026-08-11)"
Cohesion: 0.67
Nodes (3): Correction, apres avoir applique la loi de `thread-broker` a nos chiffres, Entree gzip : ou est le mur, et a combien de cœurs il tombe (2026-08-11), hyalite 0.4.0

### Community 264 - "stage_time.rs"
Cohesion: 0.23
Nodes (9): BATCHES, BUSY, MAX, MAX_WORKERS, N_STAGES, NAMES, NS, SUM (+1 more)

### Community 266 - "RecordStream"
Cohesion: 0.19
Nodes (7): paired_reader_preserves_order_and_batch_boundaries(), paired_reader_refuses_different_names(), paired_reader_refuses_unequal_files(), paired_reader_reports_a_missing_file(), PairedFastqReader, RecordStream, write_fastq()

### Community 288 - "bounds"
Cohesion: 0.19
Nodes (10): bounds(), cge(), from2(), hmax(), ld(), ext!, pack(), shift_in() (+2 more)

### Community 289 - "build_opt"
Cohesion: 0.24
Nodes (10): args_from(), build_meth(), build_opt(), f_aliases_o_and_minus_one_parses(), preset_defers_to_explicit_options_and_suppresses_rescaling(), presets_match_bwa_intractg(), presets_match_bwa_long_read(), scoring_validation_refuses_unrunnable_penalties() (+2 more)

### Community 290 - "dna.rs"
Cohesion: 0.27
Nodes (9): comp2(), complements(), encodes_bases(), nt4(), NT4_COMPLEMENT_PIVOT, nt4_in_place(), nt4_in_place_matches_the_table_on_every_byte(), NT4_N (+1 more)

### Community 291 - "mem_reg2sam"
Cohesion: 0.25
Nodes (7): format_pa(), MemAln, MethEmit, add_cigar(), get_rlen(), mem_aln2sam(), mem_reg2sam()

### Community 293 - "3. The pipeline, end to end, following one read"
Cohesion: 0.20
Nodes (10): 3.0 Index load, 3.1 Read input and encoding, 3.2 Seeding: from a read to exact matches, 3.3 Chaining: from seeds to candidate loci, 3.4 Extension: from chains to scored regions, 3.5 Dedup, primary marking, MAPQ, 3.6 Paired-end: insert size, mate rescue, pairing, 3.7 CIGAR generation (+2 more)

### Community 294 - "super"
Cohesion: 0.29
Nodes (7): performance_core_count(), performance_core_count_is_never_zero(), BYTES_PER_GB, learned_index_fits(), ram_detection_plausible(), RAM_HEADROOM_FACTOR, total_ram_bytes()

### Community 295 - "Stage"
Cohesion: 0.22
Nodes (9): Stage, Align, DedupPrep, Deinterleave, Pestat, Rescue, SamEmit, WaitRead (+1 more)

### Community 296 - "enabled"
Cohesion: 0.36
Nodes (7): add(), count_batch(), dump(), enabled(), measure(), region(), worker()

### Community 297 - "MemOpt"
Cohesion: 0.32
Nodes (5): AltLift, AltLift1, defaults_match_bwa_mem2(), fill_scmat(), MemOpt

### Community 298 - "5. Glossary"
Cohesion: 0.29
Nodes (7): 5. Glossary, Data and coordinates, Extension and scoring, Output, Paired end, Seeding and chaining, The index

### Community 299 - "What is left to make this faster, at 2026, under byte-identity"
Cohesion: 0.29
Nodes (7): 1. The build, not the code (largest, and packaging-only), 2. Only 40% of the rescue kernel's own throughput reaches the pipeline on x86, 3. The extension kernel costs 2.0 operations per cell, the rescue kernel 0.95, Ranked, what to do, What 2026 published, and what of it survives the constraint, What is left to make this faster, at 2026, under byte-identity, Where our own numbers say the remaining upside is

### Community 300 - "Part two: the dead ends, and why (issue #49)"
Cohesion: 0.33
Nodes (6): Already in the tree, Byte-identity fatal, Part two: the dead ends, and why (issue #49), Previously measured losses, restated, Refuted on the hardware evidence, The rule the research confirmed

### Community 301 - "Part one: the ceiling per CPU class, and the plan (issue #51)"
Cohesion: 0.33
Nodes (6): Caveat on completeness, First, the number that was wrong, Part one: the ceiling per CPU class, and the plan (issue #51), The ceiling per CPU class, The plan, The rule

### Community 302 - "alt_project_diff.sh"
Cohesion: 0.60
Nodes (3): heredoc(), run(), alt_project_diff.sh script

### Community 303 - "MdMode"
Cohesion: 0.50
Nodes (4): MdMode, Full, None, Short

### Community 304 - "The `-t16` gap was our own P-core cap. Removing it: 1.10x against the fork, 6 of 6"
Cohesion: 0.50
Nodes (4): Final standing, in the gist's exact regime, The cap, and why it was wrong here, The `-t16` gap was our own P-core cap. Removing it: 1.10x against the fork, 6 of 6, What the barrier probe actually showed

### Community 305 - "AVX-512BW seed extension, and a NEON epilogue that measured null"
Cohesion: 0.67
Nodes (3): AVX-512BW extension kernel (new, x86, unmeasured here), AVX-512BW seed extension, and a NEON epilogue that measured null, Row maxima published with one vector store: null on this host, kept

### Community 306 - "Measuring in Nils's actual regime: we are TIED at `-t16`, not ahead"
Cohesion: 0.67
Nodes (3): Measuring in Nils's actual regime: we are TIED at `-t16`, not ahead, Two hypotheses for the `-t16` loss, both tested, both dead tonight, Why this machine cannot settle it

## Knowledge Gaps
- **94 isolated node(s):** `Full`, `Short`, `None`, `bwa-mem4-mem`, `Staged` (+89 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1579 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **54 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ARCHITECTURE` connect `2. Why byte-identical output is the goal, and what it costs` to `5. Glossary`, `mem_sam_pe`, `README.md`, `3. The pipeline, end to end, following one read`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Params` (e.g. with `batched_align()` and `prune_params()`) actually correct?**
  _`Params` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Full`, `Short`, `None` to the rest of the system?**
  _94 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `bwa-seed/src/lib.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.05212716222533895 - nodes in this community are weakly interconnected._
- **Why does `6. How to work on this safely` connect `mem_sam_pe` to `2. Why byte-identical output is the goal, and what it costs`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Should `bwa-chain/src/lib.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.06190476190476191 - nodes in this community are weakly interconnected._
- **Why does `Roadmap` connect `Roadmap` to `La voie GPU : le plafond est 2,45x, et le GPU integre suffit deja (2026-08-08)`, `Le mate rescue, six angles, et le seul qui paie (2026-08-11)`, `Le mate rescue, le plus gros poste restant (2026-08-07)`, `Le dossier x86 : ou nous perdons, mesure sur le meme hote (2026-08-18)`, `Le lecteur n'est plus le goulot, et la sonde mentait (2026-08-11)`, `Le travail litteralement duplique, deux fois dans la meme journee (2026-08-19)`, `#53 etapes 1 et 2 : la couture asynchrone et le crate `bwa-gpu` (2026-08-09)`, `#55 : le backend Metal existe, il est octet-identique, et il perd contre un coeur (2026-08-09)`, `Les quatre backends de tableau des suffixes, mesures (2026-08-11)`, `Who else is making BWA-MEM fast, and what of it applies here`, `Etat de l'art : personne ne fait mieux a contrat de sortie egal (2026-08-08)`, `Le chemin critique du lecteur, et ce que rapidgzip y change (2026-08-11)`, `Le piege de jeu de donnees, et le classement refait (2026-08-11)`, `Le regime de colonnes pleinement en bande, trouve par la mesure (2026-08-09)`, `Phase 9e : la vague perf (tout octet-identique)`, `Les chiffres x86, enfin mesures : +17 % sur le noyau de rescue (2026-08-09)`, `#45 : `USQADD` et le `qsub(h, oe)` partage, +10,3 % de noyau, -2,7 % de CPU (2026-08-08)`, `#46 : les deux passes scalaires ne coutent rien, et pourquoi le microbenchmark disait le contraire (2026-08-08)`, `#47 : les colonnes de bourrage ont leur propre corps, et le gain depend de la longueur de lecture (2026-08-08)`, `#53 etape 0 : la taille des lots, et elle repond deux choses differentes (2026-08-08)`, `#54 : la barriere d'octet-identite GPU, ecrite avant le premier noyau (2026-08-08)`, `L'extension sur GPU, branchee et mesuree : -22 % de CPU, et une bascule a -t8 (2026-08-09)`, `Le noyau d'extension Metal : ecrit, octet-identique, 6,4x un coeur (2026-08-09)`, `Recherche multi-agents sur les noyaux SIMD (2026-08-08)`, `Le `uchar4` sur Metal : tente, non concluant, retire (2026-08-09)`, `Indexeur : le tableau de suffixes est parallelisable, mais pas par le portage Rust (2026-08-05)`, `Le chemin GPU est l'extension, pas le rescue, et les chiffres le disent (2026-08-09)`, `Le GPU rouvert : le noyau tournait a 3,6 % de la machine, et la cause n'est pas le noyau (2026-08-29)`, `Phase 0 de la campagne perf : mesure de reference contre le fork`, `Trois questions fermees le meme jour : le gate GIAB, le PGO hors Apple, et l'indexeur parallele (2026-08-18)`, `#25 tranche : le resident EST le vivant sous Linux, et le 2x est le double batch en vol (2026-08-18)`, `#43 : la table de scores XOR portee sur AVX2 et SSE4.1, et le +18 a +50 % est a revoir (2026-08-09)`, `#48B : la pre-passe de substitution repliee, +4 a +5 % de noyau, -0,9 % de CPU (2026-08-08)`, `#48C : le masque de bande en espace signe, et ce que le desassemblage x86 dit vraiment (2026-08-09)`, `#54 pieges 3 et 5 : la frontiere de saturation, et un `max_off` qui ne decide jamais rien (2026-08-08)`, `#55 etape 0 : bloquee, il n'y a pas de compilateur Metal sur cette machine (2026-08-08)`, `#55 etape 0 : debloquee sans Xcode, et le GPU tient 330 Gcell/s (2026-08-09)`, `Campagne perf CPU et classement contre minibwa (2026-08-09)`, `L'oracle tiers couvre enfin les penalites de gap asymetriques (2026-08-12)`, `Ou le fork est reellement moins cher, profil contre profil (2026-08-07)`, `Entree gzip : ou est le mur, et a combien de cœurs il tombe (2026-08-11)`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._