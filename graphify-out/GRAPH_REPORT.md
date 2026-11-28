# Graph Report - distracted-ramanujan-d9c924  (2026-11-26)

## Corpus Check
- Large corpus: 160 files · ~684,751 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 3574 nodes · 6862 edges · 234 communities (193 shown, 41 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 312 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Chain Building and Filtering
- SMEM Seeding (FM-index)
- Batched Extension SIMD Tests
- Python Fuzz and Meth Simulators
- Rescue Band Benchmark
- Allocation Probe
- Learned SA (LISA) Index
- LISA Zigzag Seeding
- Batched Mate Rescue (PE)
- Rescue Pruning Hits
- CLI Commands and Presets
- Primary Marking and MAPQ
- mem Option Parsing
- FM-index Occ Unpacking
- AVX-512 Check Workflow
- Stage Timing and Alloc Stats
- Mate SW Band Counters
- Core Design Decisions (Docs)
- CI Lint/Build/Test
- mem Stage Counters
- Methylation Options
- GIAB Gate Workflow
- crates.io Publish Workflow
- CIGAR Generation
- Alloc Probe Workflow
- SA Backends Workflow
- PE Mate SW and SAM Pairing
- Text-Sampled SA (TSA)
- Across-Read Batching Stats
- Reference Annotations (BntSeq)
- SwBackend Scalar Oracle
- Divergences and Byte-Identity
- Global SW Band Tightening
- Forward Local SW Kernels
- Global Wavefront Kernel
- mem Run SE/PE Driver
- FASTQ Packing and Pairing
- Extension Job Dispatch
- Measured Perf Levers
- Kernel Ceiling and Dead Ends
- Band Prologue/Epilogue NEON
- Batched KSW Align and Rescue
- Architecture Pipeline Docs
- Index Construction
- Methylation SAM Tags
- Rescue Band Decisions
- NEON Backend
- SAM Record Emission
- Per-Read Chain Alignment
- x86 SIMD Intrinsics
- Staged Across-Read Dedup
- README and Credits
- Reader/Writer Pipeline
- Literature and State of the Art
- Dependabot Config
- Dependabot Auto-Merge
- Third-Party Oracle Tests
- Bidirectional FM Extension
- Index OpenMP Workflow
- SwBackend Async Trait
- SA-IS Suffix Array
- FASTQ Record Stream
- NEON u8 Forward SW
- Apple Silicon uarch Probes
- Default Options (bwa-mem2 parity)
- GPU Plan
- Perf x86 Workflow Steps
- Contributing and Release Process
- ALT Contigs and XA Tags
- Methylation Mode Design
- Perf Profile x86 Inputs
- Release Workflow Steps
- Packed Reference Access
- Perf Profile x86 Jobs
- Perf Profile x86 Steps
- Perf x86 Inputs
- Phase 0 Comparative Baseline
- Index Loading and Mapping
- Across-Read Batched Align
- Bench x86 Workflow
- Docker Gates Script
- Huge Pages
- Read Group and Comments
- SA-IS Recursion
- SAM Header Writer
- Bench ARM Workflow
- Bench ARM Inputs
- Release Triggers
- Bad Input Tests
- FM-index Search Tests
- Pair Penalty Scoring
- Seed Filtering
- Seeding Levers (minibwa, Prefetch)
- Memory-System Scaling Limits
- Fork Teardown and x86 Profile
- ARM Lockstep Sweep Jobs
- Bench x86 Inputs
- Bench x86 Steps
- Hybrid Bench and SAM Diff
- Multiline FASTQ Unwrap
- bwa-diff Comparator
- SA Parity Tools
- Parity Workflow
- Profile Micromamba Setup
- PGO x86 Job
- HTS Writer/Transcoder
- Integration Byte-Identity Tests
- STAR vs FM Benchmark
- FASTQ Reader Variants
- Head-to-Head Inputs
- Head-to-Head Steps
- Global SW Instruction Count
- Batch Count Tests
- Scalar Global SW and CIGAR
- Decompression Reader
- Aligned Buffers
- Optimization Roadmap
- Bench ARM Setup
- Bench x86 Setup
- Option Parity Job
- Perf x86 Setup
- Release Env
- Release Build Env
- Option Parity Script
- Superpage Probe
- Core Error Types
- Suboptimal Score Tracker
- Alignment Algorithm Alternatives
- Rescue Lever Plans
- Bench ARM Steps
- x86 End-to-End Small Ref
- Head-to-Head Setup
- Parity Env
- Perf x86 Job
- Bioconda Build Script
- SIMD Kernel Search Levers
- Sequence Emit Benchmark
- Extension SIMD Tiers
- Rescue SIMD Tiers
- head-to-head.yml head-to-head.yml
- head-to-head.yml head-to-head
- release.yml build
- release.yml check-version
- release.yml include
- RAM Reduction Levers
- giab32_headline.sh Script
- giab_publication.sh Script
- BWT Binning Benchmark
- SA-IS String Views
- Score Sort Benchmark
- Batch Extension Benchmark
- Rescue Row Max Clamp
- Rank Structures (QuadRank)
- parity.yml runner
- parity.yml on
- parity.yml with
- Lockstep Seeding Scaling
- Guided Batch Chunking
- Across-Read Stage States
- PE Pass Hash
- Tile Padding (piece16)
- bench-arm.yml graviton-selfhosted
- bench-x86.yml kernel_ab_only
- bench-x86.yml require_avx512
- bench-x86.yml require_avx512_e2e
- head-to-head.yml avx2_arm
- head-to-head.yml require_avx512
- parity.yml require_avx512
- perf-x86.yml isa-and-tiers-x86
- perf-x86.yml real-reads-x86
- release.yml promote
- release.yml publish
- fork_bench.sh Script
- giab_happy.sh Script
- graviton_setup.sh Script
- Unpack Benchmark
- Base Encodings
- Across-Read Modes
- bench-x86.yml hugepage-probe
- bench-x86.yml read_pairs
- bench-x86.yml runner
- bench-x86.yml variant_cpu
- head-to-head.yml read_pairs
- perf-profile-x86.yml extra_rustflags
- perf-profile-x86.yml icount
- perf-profile-x86.yml icount
- perf-profile-x86.yml knob_sweep
- perf-profile-x86.yml read_pairs
- perf-profile-x86.yml read_set
- perf-profile-x86.yml runner
- perf-profile-x86.yml meth
- perf-profile-x86.yml on
- Suffix-Array Indexer Backends
- FASTQ Reader Critical Path
- giab30x_bench.sh Script
- giab30x_pe.sh Script
- pe_diverge_hunt.sh Script
- pe_diverge_subset.sh Script
- pgo.sh Script
- NEON SW Feature Flags
- alt_parity.sh Script
- bench_in_container.sh Script
- fetch_giab30x.sh Script
- giab_bench.sh Script
- meth_oracle_diff.sh Script
- publication_bench.sh Script
- upstream_repros.sh Script
- vs_mem2.sh Script
- Full-Band Gate Flag
- BWA-MEME / ERT
- Compat Output Shaping
- BAM Writer Options
- avx512_sde.sh Script
- bench.sh Script
- cc_no_profile.sh Script
- check.sh Script
- index_diff.sh Script
- longread_parity.sh Script
- make_testdata.sh Script
- oracle_diff.sh Script
- scale_test.sh Script
- setup_reference.sh Script
- bwa-mem4 Manifest
- bwa-mem4-chain Manifest
- bwa-mem4-core Manifest
- bwa-mem4-diff Manifest
- bwa-mem4-extend Manifest
- bwa-mem4-index Manifest
- bwa-mem4-io Manifest
- bwa-mem4-mem Manifest
- bwa-mem4-neon Manifest
- bwa-mem4-seed Manifest

## God Nodes (most connected - your core abstractions)
1. `FmIndex` - 72 edges
2. `MemAlnReg` - 61 edges
3. `Perf levers measured` - 35 edges
4. `batched_align()` - 30 edges
5. `Params` - 30 edges
6. `mem_sam_pe()` - 28 edges
7. `align_reads_batched_inner()` - 26 edges
8. `batch_mate_rescue()` - 26 edges
9. `Kernel ceiling and dead ends` - 26 edges
10. `mem_sort_dedup_patch()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `Async SwBackendAsync seam, batch-size measurement (#53)` --references--> `SwBackend`  [EXTRACTED]
  docs/gpu-plan.md → crates/bwa-extend/src/lib.rs
- `Band-retry loop stays on CPU, GPU computes` --references--> `align_reads_batched()`  [EXTRACTED]
  docs/gpu-plan.md → crates/bwa-mem/src/across.rs
- `Mate rescue ksw_align2 / mem_matesw ported` --references--> `mem_matesw()`  [EXTRACTED]
  DIVERGENCES.md → crates/bwa-mem/src/pe.rs
- `Rescue pruning under genomic/neutral relation` --references--> `mem_matesw()`  [INFERRED]
  docs/meth-mode.md → crates/bwa-mem/src/pe.rs
- `Chaining (greedy, kbtree)` --references--> `ks_introsort_by()`  [EXTRACTED]
  ARCHITECTURE.md → crates/bwa-chain/src/lib.rs

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
- 2-file cycle: `crates/bwa-mem/src/across.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/across.rs`
- 2-file cycle: `crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/pe.rs -> crates/bwa-mem/src/lib.rs`
- 2-file cycle: `crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/primary.rs -> crates/bwa-mem/src/lib.rs`
- 3-file cycle: `crates/bwa-mem/src/cigar.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/pe.rs -> crates/bwa-mem/src/cigar.rs`
- 3-file cycle: `crates/bwa-mem/src/cigar.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/primary.rs -> crates/bwa-mem/src/cigar.rs`
- 3-file cycle: `crates/bwa-mem/src/cigar.rs -> crates/bwa-mem/src/primary.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/cigar.rs`
- 3-file cycle: `crates/bwa-mem/src/alt.rs -> crates/bwa-mem/src/lib.rs -> crates/bwa-mem/src/pe.rs -> crates/bwa-mem/src/alt.rs`

## Hyperedges (group relationships)
- **SIMD kernel issue campaign (#43-#50)** — roadmap_issue_43_avx2_xor, roadmap_issue_44_avx512, roadmap_issue_45_usqadd, roadmap_issue_46_scalar_passes, roadmap_issue_47_pad_columns, roadmap_issue_48_batched, roadmap_issue_50_span_lemma [EXTRACTED 1.00]
- **Issue #25 RAM-per-batch investigation chain** — roadmap_macos_rss_accounting, roadmap_stage_alloc_probe, roadmap_mimalloc_kept, roadmap_batch_overlap, roadmap_issue_25_ram_per_batch [EXTRACTED 1.00]
- **GPU backend lifecycle (seam, Metal, abandon, reopen)** — roadmap_gpu_swbackend_seam, roadmap_gpu_metal_backend, roadmap_gpu_abandoned, roadmap_gpu_reopened_probes [EXTRACTED 1.00]
- **Read pipeline stages seed to SAM** — architecture_seeding, architecture_chaining, architecture_extension, architecture_dedup_primary_mapq, architecture_cigar_gen, architecture_sam_output [EXTRACTED 1.00]
- **Faithfully reproduced C quirks** — architecture_f32_threshold, architecture_mapq_coef_fac_int, architecture_unstable_sort, architecture_hash64_wraparound, architecture_lrand48_n_replacement, architecture_gap_open_conventions [EXTRACTED 1.00]
- **GPU backend plan components** — docs_gpu_plan_async_seam, docs_gpu_plan_identity_gate, docs_gpu_plan_metal_backend, docs_gpu_plan_cuda_backend, docs_gpu_plan_coscheduling [EXTRACTED 1.00]
- **Memory-access levers killed by lockstep prefetch** — docs_optimization_roadmap_sa_prefetch, docs_optimization_roadmap_ten_mer_cache, docs_optimization_roadmap_flat_sa, docs_optimization_roadmap_sa_colocate, docs_optimization_roadmap_madvise_hugepage, docs_optimization_roadmap_zhang_binning [EXTRACTED 1.00]
- **Causes of the x86 deficit vs fork** — docs_fork_teardown_avx2_baseline, docs_x86_side_by_side_profile_grouped_deficit, docs_whats_left_2026_delivered_40pct, docs_x86_side_by_side_profile_get_sa_prefetch_gap [INFERRED 0.85]
- **Literature levers closed by byte-identity** — docs_deep_research_2026_07_29_bitpal_bgsa, docs_deep_research_2026_07_29_monge_seaweed, docs_deep_research_2026_07_29_seedex, docs_deep_research_2026_07_29_qgram_proof, docs_deep_research_2026_07_29_star_mmp [INFERRED 0.85]

## Communities (234 total, 41 thin omitted)

### Community 0 - "Chain Building and Filtering"
Cohesion: 0.05
Nodes (78): Chaining (greedy, kbtree), duplicate_pos_keeps_klib_order(), getp_aux(), KbTree, lower_picks_last_entry_below(), MAX_KEYS, Node, splits_preserve_order_and_lookup() (+70 more)

### Community 1 - "SMEM Seeding (FM-index)"
Cohesion: 0.05
Nodes (75): AMBIGUOUS_BASE, amd_zen5(), batched_full_seeding_equals_per_read(), batched_smems_equal_per_read(), BSD_LOCKSTEP_MIN, bsd_lockstep_width(), BWD_STEPS, bwt_seed_strategy() (+67 more)

### Community 2 - "Batched Extension SIMD Tests"
Cohesion: 0.04
Nodes (49): avx2_u8_and_i16_match_scalar(), avx512_u8_and_i16_match_scalar(), BITREV, block_mismatches(), BLOCK_SHIFT, block_walk_equals_the_byte_walk(), bucket(), BUCKETS (+41 more)

### Community 3 - "Python Fuzz and Meth Simulators"
Cohesion: 0.06
Nodes (30): main(), parse(), Lcg, main(), mutate(), read_fasta(), convert(), main() (+22 more)

### Community 4 - "Rescue Band Benchmark"
Cohesion: 0.06
Nodes (37): copy_fast(), fill_fast(), rescue_avx512_ok(), u8_score_table(), avx512_band_kernel_matches_avx2(), avx512_bands_reproduce_ksw_align2(), Band, band_bench() (+29 more)

### Community 5 - "Allocation Probe"
Cohesion: 0.05
Nodes (47): allocations_land_in_the_current_stage_and_the_peak_is_the_live_maximum(), B_ALIGN, B_ALIGN_CHAIN, B_ALIGN_EXTEND, B_ALIGN_SA, B_ALIGN_SEED, B_READER, B_UNSTAGED (+39 more)

### Community 6 - "Learned SA (LISA) Index"
Cohesion: 0.07
Nodes (24): bidirectional_interval_matches_fmindex(), brute(), edge_cases(), exact_search_matches_bruteforce(), K, kmer_key(), lcg(), LearnedSa (+16 more)

### Community 7 - "LISA Zigzag Seeding"
Cohesion: 0.09
Nodes (43): performance_core_count(), performance_core_count_is_never_zero(), comp2(), complements(), encodes_bases(), nt4(), NT4_COMPLEMENT_PIVOT, nt4_in_place() (+35 more)

### Community 8 - "Batched Mate Rescue (PE)"
Cohesion: 0.06
Nodes (49): Anchor, ANCHORS, batch_mate_rescue(), best_pair(), best_pair_matches_sorted_list(), DECIBANS_PER_NAT, dump(), enabled() (+41 more)

### Community 9 - "Rescue Pruning Hits"
Cohesion: 0.08
Nodes (45): available(), band_dp(), band_of(), band_proto(), band_t1_bench(), BComp, bcomps(), bounds() (+37 more)

### Community 10 - "CLI Commands and Presets"
Cohesion: 0.06
Nodes (23): IndexArgs, resolve_mmi_presets(), run(), build_mmi(), load_or_build_index(), map_batch(), mmi_path(), preset_for() (+15 more)

### Community 11 - "Primary Marking and MAPQ"
Cohesion: 0.09
Nodes (43): Dedup, primary marking, MAPQ, align_read_se(), MemAlnReg, dump(), EDGES, enabled(), hash_64(), hash_64_is_wang() (+35 more)

### Community 12 - "mem Option Parsing"
Cohesion: 0.06
Nodes (40): args_from(), BATCH_READAHEAD, BGZF_COMPRESSION_LEVEL, BGZF_SINK_BUFFER_BYTES, build_meth(), build_opt(), BWA_CIGAR_OP_HARD_CLIP, BWA_CIGAR_OP_SOFT_CLIP (+32 more)

### Community 13 - "FM-index Occ Unpacking"
Cohesion: 0.06
Nodes (39): ANY_BASE, COUNT_SPAN_SHIFT, CP_BLOCK_SIZE, CP_MASK, CP_SHIFT, dump(), enabled(), EXT_CALLS (+31 more)

### Community 14 - "AVX-512 Check Workflow"
Cohesion: 0.05
Nodes (48): if, name, runs-on, steps, timeout-minutes, env, name, runs-on (+40 more)

### Community 15 - "Stage Timing and Alloc Stats"
Cohesion: 0.07
Nodes (30): enter(), set_role_reader(), set_role_writer(), stage_discriminants_are_bucket_indices(), add(), BATCHES, BUSY, count_batch() (+22 more)

### Community 16 - "Mate SW Band Counters"
Cohesion: 0.07
Nodes (41): ACCEPTED, BAND_CELLS, BAND_KERNEL_NS, BAND_OTHER_NS, BAND_R2, BUCKETS, CALL_HIST, CALL_MAX (+33 more)

### Community 17 - "Core Design Decisions (Docs)"
Cohesion: 0.05
Nodes (37): ROADMAP.md (bwa-mem4 perf and parity journal), Asymmetric gap-penalty coverage in oracle (hyalite 0.4), AVX-512 on Golden Cove is port-0 bound, AVX-512 kernels verified under Intel SDE (-skx, -spr), AVX-512 tier value depends on microarchitecture (Zen4 0.7% vs Granite Rapids 6.8%), bwa-mem2 -A 2 x86 vs arm64 disagreement (bwa-mem2#297), scripts/docker_gates.sh container gates, fg-labs/bwa-mem3 fork (nh13) comparator (+29 more)

### Community 18 - "CI Lint/Build/Test"
Cohesion: 0.05
Nodes (42): runs-on, steps, strategy, timeout-minutes, env, CARGO_TERM_COLOR, RUST_TOOLCHAIN, RUSTFLAGS (+34 more)

### Community 19 - "mem Stage Counters"
Cohesion: 0.08
Nodes (30): ALT_CALLS, ALT_NS, APPLY_DEDUP_CALLS, APPLY_DEDUP_NS, APPLY_DEDUP_ROWS, APPLY_HITS, APPLY_INSERT_NS, APPLY_NS (+22 more)

### Community 20 - "Methylation Options"
Cohesion: 0.07
Nodes (33): Chem, EmSeq, chem_names(), Taps, conversion_follows_read_number(), convert_codes(), converted_fasta_is_uppercased_and_wrapped(), matrices_free_only_the_chemistry() (+25 more)

### Community 21 - "GIAB Gate Workflow"
Cohesion: 0.05
Nodes (38): concurrency, cancel-in-progress, group, run, RUSTFLAGS, defaults, name, runs-on (+30 more)

### Community 22 - "crates.io Publish Workflow"
Cohesion: 0.05
Nodes (37): concurrency, cancel-in-progress, group, default, description, type, env, CARGO_REGISTRY_TOKEN (+29 more)

### Community 23 - "CIGAR Generation"
Cohesion: 0.09
Nodes (29): ALREADY_FAST, AMBIG, band_width_trace_enabled(), CALLS, CELLS, CELLS_PROVABLE, CIGAR_OP_DEL, CIGAR_OP_HARD_CLIP (+21 more)

### Community 24 - "Alloc Probe Workflow"
Cohesion: 0.05
Nodes (36): defaults, name, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group (+28 more)

### Community 25 - "SA Backends Workflow"
Cohesion: 0.06
Nodes (34): default, description, required, runs-on, steps, timeout-minutes, env, BASES (+26 more)

### Community 26 - "PE Mate SW and SAM Pairing"
Cohesion: 0.14
Nodes (25): Paired-end: pestat, mate rescue, mem_pair, Read pipeline: FASTQ to SAM, bns_fetch_seq(), bns_fetch_seq_into(), cal_sub(), dump_pestat(), grow_one(), infer_dir_orientations() (+17 more)

### Community 27 - "Text-Sampled SA (TSA)"
Cohesion: 0.12
Nodes (3): FmIndex, SeedJump, SyncPtr

### Community 28 - "Across-Read Batching Stats"
Cohesion: 0.10
Nodes (26): ALPHABET_SIZE, AMBIG, CELLS, CELLS_HIT, CHAINFLT_NS, COMPARABLE_LEN_FRAC, dump(), enabled() (+18 more)

### Community 29 - "Reference Annotations (BntSeq)"
Cohesion: 0.13
Nodes (14): Amb, BntSeq, comment_is_what_the_c_reads_back_from_ann(), Contig, fmt_err(), intv2rid_agrees_with_two_searches(), next_parse(), parse_amb() (+6 more)

### Community 30 - "SwBackend Scalar Oracle"
Cohesion: 0.17
Nodes (17): Backend acceptance harness, assert_backend_batch_matches_scalar(), assert_backend_batch_order_invariant(), assert_backend_matches_scalar(), assert_backend_saturation_boundary_matches_scalar(), assert_backend_tie_rule_matches_scalar(), fill_scmat_vec(), LCG_INCREMENT (+9 more)

### Community 31 - "Divergences and Byte-Identity"
Cohesion: 0.09
Nodes (15): Bit-identical instrumented oracle method, bwa-mem2 v2.3 patched oracle, sam-diff field-level concordance tool, SAM output and FLAG bits, rand48 lrand48 port, sse2neon used only on the oracle side, Resolved: chain order and unstable introsort in mem_chain_flt, Resolved: mapQ_coef_fac int truncation (+7 more)

### Community 32 - "Global SW Band Tightening"
Cohesion: 0.14
Nodes (27): CANDS, diagonal_tied_by_a_gapped_path_is_still_the_cigar(), exact_match_scores_full_length(), EXC_MAX_V, excursions_cannot_gain(), global_cell(), global_exact_match_is_all_m(), global_mid_rows() (+19 more)

### Community 33 - "Forward Local SW Kernels"
Cohesion: 0.18
Nodes (29): prefers_wide_x86(), avx2_matesw_i16_matches_scalar(), avx2_matesw_u8_matches_scalar(), avx512_avx2_matesw_uniform_qlen_pad_columns_equal_scalar(), avx512_matesw_i16_matches_scalar(), avx512_matesw_u8_matches_scalar(), bench_jobs(), fwd_local_sw_avx2_u8() (+21 more)

### Community 34 - "Global Wavefront Kernel"
Cohesion: 0.11
Nodes (20): IOTA, load_table(), shim!, wave_fill!, max_lanes(), PAD, REV, safe() (+12 more)

### Community 35 - "mem Run SE/PE Driver"
Cohesion: 0.22
Nodes (20): batch_overlap_enabled(), batched_regs(), Deferred, dump_regs_enabled(), emit_deferred(), emit_resolved(), finish_se(), finish_se_meth() (+12 more)

### Community 36 - "FASTQ Packing and Pairing"
Cohesion: 0.11
Nodes (24): acgt_code(), fifo_attempt(), first_le_space(), first_le_space_matches_the_byte_scan(), INFLATE_BLOCK, INFLATE_DEPTH, is_acgt(), pack64() (+16 more)

### Community 37 - "Extension Job Dispatch"
Cohesion: 0.20
Nodes (26): band_order(), band_split(), batched_extend(), batched_extend_scalar(), cell_bound(), clamp_band(), dead_seed(), dedup_jobs() (+18 more)

### Community 38 - "Measured Perf Levers"
Cohesion: 0.11
Nodes (24): Row pairing in batched.rs is byte-identity fatal, Band-free extension column loop fires on 15% of rows, BWA4_BARRIER_TIME occupancy probe, Rescue component bands: NO-GO, mem_sort_dedup_patch cost shape (BWA4_DEDUP_SHAPE), Perf levers measured, zlib-rs inflate default, insert_into_settled avoids rescue re-sorts (+16 more)

### Community 39 - "Kernel Ceiling and Dead Ends"
Cohesion: 0.11
Nodes (20): CIGAR generation (banded global DP), Extension (banded Smith-Waterman, band retry, z-drop), 4-row body on AVX2 refuted (register pressure), AVX-512 only 1.25x AVX2 (p0-bound), Replacing vpblendvb with andn+max: 0 percent, Achievable Gcell/s per core by CPU class, Sixteen dead ends and what would be new, Kernel ceiling and dead ends (+12 more)

### Community 40 - "Band Prologue/Epilogue NEON"
Cohesion: 0.14
Nodes (22): band_s16(), band_u8(), band_u8_range_compare_is_exhaustively_the_two_compares(), bits(), bits16(), epilogue(), hmax(), hmin() (+14 more)

### Community 41 - "Batched KSW Align and Rescue"
Cohesion: 0.11
Nodes (26): batched_align(), batched_ksw_align2(), batched_mate_rescue(), batched_mate_rescue_lazy(), bucket(), count_outcome(), dump(), enabled() (+18 more)

### Community 42 - "Architecture Pipeline Docs"
Cohesion: 0.09
Nodes (22): 2L coordinate space, ALT contigs handling, Crate map (twelve crates), BWA4_* diagnostic env vars, FM-index (bidirectional, k l s interval), Glossary of bwa terms, LISA/RMI learned-index dead end, scripts/opt_parity.sh per-option parity gate (+14 more)

### Community 43 - "Index Construction"
Cohesion: 0.13
Nodes (23): amb_holes_not_merged_across_contig_boundary(), AmbRec, AnnRec, build_bwt_and_samples(), build_index(), build_index_with_prefix(), CP_BLOCK_SIZE, DUMMY_CHAR (+15 more)

### Community 44 - "Methylation SAM Tags"
Cohesion: 0.14
Nodes (15): build_xm(), comment_tokens_reencode_like_htslib(), format_g(), leading_float(), leading_int(), MIN_LONGEST_M_PCT, op(), percent_g_matches_c() (+7 more)

### Community 45 - "Rescue Band Decisions"
Cohesion: 0.23
Nodes (21): Comp, components(), components_at(), decide_bands(), decide_counts(), decide_rel(), Decision, Fail (+13 more)

### Community 46 - "NEON Backend"
Cohesion: 0.12
Nodes (9): diag_score(), diag_score_matches_scalar(), diag_score_neon(), diag_score_ssse3(), neon_batch_matches_scalar_read_sized(), neon_ungapped_hit_matches_scalar(), NeonBackend, rewrite_jobs() (+1 more)

### Community 47 - "SAM Record Emission"
Cohesion: 0.13
Nodes (20): cigar_string(), cigar_string_which(), format_pa(), BLOCK, FWD16, FWD_TABLE, integers_match_to_string(), push_int() (+12 more)

### Community 48 - "Per-Read Chain Alignment"
Cohesion: 0.20
Nodes (19): align_chains(), align_read(), align_read_dedup(), align_read_meth(), best(), cal_max_gap(), dump_regs(), dump_regs_enabled() (+11 more)

### Community 49 - "x86 SIMD Intrinsics"
Cohesion: 0.19
Nodes (21): band_epi16(), band_epi8(), band_epu8(), bsl128(), bsl256(), bsl512(), ceq16(), ceq8() (+13 more)

### Community 50 - "Staged Across-Read Dedup"
Cohesion: 0.21
Nodes (14): contains(), discard_contained(), interferes(), judge(), KeptIndex, .BUCKET_SHIFT, .HBITS, .MAX_BUCKET_WALK (+6 more)

### Community 51 - "README and Credits"
Cohesion: 0.12
Nodes (20): Runtime dependency table (clap, rayon, htslib, mimalloc, ...), Dev-only dependencies (noodles, gungraun), longread_parity.sh gate, rammap-core long-read delegation, Catching up with bwa-mem3 0.15.0/0.15.1 levers, Batch overlap off by default (BWA4_BATCH_OVERLAP), bwa-mem2 (reference implementation), fg-labs/bwa-mem3 (fork by Nils Homer) (+12 more)

### Community 52 - "Reader/Writer Pipeline"
Cohesion: 0.13
Nodes (11): bam_header_only_still_produces_a_file(), bam_transcoder_reassembles_split_lines(), Output, Bam, Bgzf, Cram, Plain, queue_depth_defaults_to_double_buffering() (+3 more)

### Community 53 - "Literature and State of the Art"
Cohesion: 0.10
Nodes (19): ALAE (arXiv:1208.0274), Deep research 2026-07-29, Mate rescue DP, BWA-MEM-SCALE (Kim ICPP 2022), BWA-MEME (Jung & Han 2022), Literature review 2026-07, ERT (Subramaniyan ISCA 2021), Prousalis survey 2025 (+11 more)

### Community 54 - "Dependabot Config"
Cohesion: 0.10
Nodes (20): patterns, patterns, prefix, actions, cargo, dependency-name, update-types, day (+12 more)

### Community 55 - "Dependabot Auto-Merge"
Cohesion: 0.10
Nodes (20): if, runs-on, steps, GH_TOKEN, PR_URL, jobs, auto-merge, name (+12 more)

### Community 56 - "Third-Party Oracle Tests"
Cohesion: 0.17
Nodes (13): ksw_align2(), bwa_matrix(), hyalite_scoring(), LANES, local_sw_score_matches_independent_implementation(), oracle_score(), oracle_with_score2(), our_score() (+5 more)

### Community 58 - "Index OpenMP Workflow"
Cohesion: 0.10
Nodes (19): concurrency, cancel-in-progress, group, RUSTFLAGS, jobs, static-openmp, name, on (+11 more)

### Community 59 - "SwBackend Async Trait"
Cohesion: 0.14
Nodes (9): SwBackend trait, scalar as source of truth, NEON backend collaboration with nh13, BatchTicket, ExtendJob, ScalarBackend, SwBackend, SwBackendAsync, SyncAsAsync<B> (+1 more)

### Community 60 - "SA-IS Suffix Array"
Cohesion: 0.20
Nodes (18): bucket_ends(), bucket_starts(), check(), check_inplace(), EMPTY, IEMPTY, induce(), inplace_known_strings() (+10 more)

### Community 61 - "FASTQ Record Stream"
Cohesion: 0.20
Nodes (4): every_compression_yields_the_same_records(), PairedFastqReader, Record, RecordStream

### Community 62 - "NEON u8 Forward SW"
Cohesion: 0.31
Nodes (19): enc_bases(), extract_group(), fastcol_enabled(), fwd_local_sw_avx2_i16(), fwd_local_sw_avx2_u8_impl(), fwd_local_sw_avx512_i16(), fwd_local_sw_avx512_u8_impl(), fwd_local_sw_neon() (+11 more)

### Community 63 - "Apple Silicon uarch Probes"
Cohesion: 0.17
Nodes (10): now_ns(), pf(), run(), Apple Silicon uarch probes, superpage_probe.c (no huge pages on arm64 macOS), build(), chase(), main() (+2 more)

### Community 64 - "Default Options (bwa-mem2 parity)"
Cohesion: 0.13
Nodes (15): ALL, ALPHABET_SIZE, AMBIGUOUS_BASE_SCORE, defaults_match_bwa_mem2(), fill_scmat(), KEEP_SUPP_MAPQ, MemOpt, NO_MULTI (+7 more)

### Community 65 - "GPU Plan"
Cohesion: 0.20
Nodes (14): Amdahl ceiling: 2.45x if DP became free, Async SwBackendAsync seam, batch-size measurement (#53), CPU+GPU co-scheduling in three levels (#57), CUDA backend with __vaddus4 and DPX (#56), CUDASW++4.0 throughput reference, GPU plan, GPU-BWA-MEM (ICS 2023) competitor, BWA4_GPU_SPLIT md5 gate (+6 more)

### Community 66 - "Perf x86 Workflow Steps"
Cohesion: 0.21
Nodes (18): CFLAGS, CXXFLAGS, REQUIRE_AVX512, RUSTFLAGS, steps, release-pgo-x86, steps, steps (+10 more)

### Community 67 - "Contributing and Release Process"
Cohesion: 0.19
Nodes (11): AVX-512 verification with Intel SDE, Branch model: dev default, main fast-forwarded on release, crates.io publishing (bwa-mem4-* prefix), Dependabot grouped PR and auto-merge policy, Manual gates alt_parity.sh and giab30x_pe.sh, Binary architecture: SIMD kernels chosen at runtime, 58-case parity gate, Release notes conventions (+3 more)

### Community 68 - "ALT Contigs and XA Tags"
Cohesion: 0.23
Nodes (10): get_pri_idx(), hit_counts(), mem_gen_alt(), near_primary_alternate_grouped_distant_one_dropped(), reg(), too_many_hits_drops_all_of_that_primary(), xa_group(), XaTags (+2 more)

### Community 69 - "Methylation Mode Design"
Cohesion: 0.18
Nodes (16): Converted-read seeding, original-genome extension, Bismark #1209 compatibility, Differences from bwa-mem3 0.17, Methylation mode doc, Converted-genome index (.meth.fa and .meth.*), Methylation mode --meth (bisulfite, EM-seq, TAPS, 5-base), --meth-* options and defaults, bwa-mem3 0.17 meth oracle (+8 more)

### Community 70 - "Perf Profile x86 Inputs"
Cohesion: 0.12
Nodes (17): default, description, required, default, description, required, default, description (+9 more)

### Community 71 - "Release Workflow Steps"
Cohesion: 0.19
Nodes (17): steps, steps, steps, steps, id, if, name, run (+9 more)

### Community 72 - "Packed Reference Access"
Cohesion: 0.19
Nodes (8): rc_unpack_equals_base_by_base(), text_match_equals_base_by_base(), unpack_pac_base(), unpack_pac_range(), unpack_pac_range_append(), unpack_pac_range_append_rev(), unpack_pac_range_into(), unpack_pac_reproduces_0123()

### Community 73 - "Perf Profile x86 Jobs"
Cohesion: 0.12
Nodes (15): env, CARGO_TERM_COLOR, RUSTFLAGS, jobs, knobs, profile, if, runs-on (+7 more)

### Community 74 - "Perf Profile x86 Steps"
Cohesion: 0.23
Nodes (16): ALT_RUSTFLAGS, CG_FLAGS, CPU_MATCH, OURS_ENV, SET, steps, steps, steps (+8 more)

### Community 75 - "Perf x86 Inputs"
Cohesion: 0.12
Nodes (16): read_pairs, require_avx512, runner, on, workflow_dispatch, default, description, required (+8 more)

### Community 76 - "Phase 0 Comparative Baseline"
Cohesion: 0.13
Nodes (12): Batch count stderr line, Phase 0 comparative baseline plan, scripts/fork_bench.sh three-arm harness, scripts/giab30x_bench.sh third arm, run_pipeline in cmd_mem.rs, Seeding and chaining ~1.8x fork cost, Speed and RAM comparative design, Speed and RAM goal at -t8 end-to-end (+4 more)

### Community 77 - "Index Loading and Mapping"
Cohesion: 0.23
Nodes (11): bwt_fingerprint(), load_refuses_truncated_or_mixed_files(), rd_i64(), read_at_parallel(), read_exact_at(), rebase_counts(), sibling(), sidecar_is_tied_to_its_index() (+3 more)

### Community 78 - "Across-Read Batched Align"
Cohesion: 0.21
Nodes (11): align_reads_batched(), align_reads_batched_inner(), align_reads_batched_meth(), batched_across_reads_equals_per_read(), discard_enabled(), prefetch_smems(), repeat_skip_enabled(), seed_ext_redundant() (+3 more)

### Community 79 - "Bench x86 Workflow"
Cohesion: 0.13
Nodes (14): concurrency, cancel-in-progress, group, env, CARGO_TERM_COLOR, RUSTFLAGS, jobs, kernel-ab (+6 more)

### Community 80 - "Docker Gates Script"
Cohesion: 0.40
Nodes (14): cmd_bench(), cmd_build(), cmd_check(), cmd_fixture(), cmd_fork(), cmd_icount(), cmd_parity(), cmd_rss() (+6 more)

### Community 81 - "Huge Pages"
Cohesion: 0.21
Nodes (9): seed_jump_level_off(), advise_hugepage(), disabled(), MADV_HUGEPAGE, madvise(), page_aligned_range(), page_aligned_range_rounds_inward(), SC_PAGESIZE (+1 more)

### Community 82 - "Read Group and Comments"
Cohesion: 0.20
Nodes (9): append_comment(), append_rg_tag(), COPY_COMMENT, copy_comment(), escape(), MAX_RG_ID_BUFFER, RG_ID, rg_id() (+1 more)

### Community 83 - "SA-IS Recursion"
Cohesion: 0.35
Nodes (7): bucket_ends_i(), bucket_starts_i(), induce_i(), IntSlice, lms_eq(), sais_rec(), TypeBits

### Community 84 - "SAM Header Writer"
Cohesion: 0.23
Nodes (10): alt_contig_sq_line_carries_ah(), header_has_no_hd_and_exact_sq(), header_inserts_rg_and_hdr_lines(), SAM_MISSING, SqRecord, unmapped_missing_qual_is_star(), unmapped_record_shape(), write_header() (+2 more)

### Community 85 - "Bench ARM Workflow"
Cohesion: 0.14
Nodes (13): concurrency, cancel-in-progress, group, env, CARGO_TERM_COLOR, RUSTFLAGS, jobs, name (+5 more)

### Community 86 - "Bench ARM Inputs"
Cohesion: 0.14
Nodes (14): max_threads, read_pairs, run_graviton, default, description, required, workflow_dispatch, default (+6 more)

### Community 87 - "Release Triggers"
Cohesion: 0.14
Nodes (14): default, description, type, dry_run, tag, on, pull_request, push (+6 more)

### Community 88 - "Bad Input Tests"
Cohesion: 0.45
Nodes (11): a_non_utf8_argument_is_an_error_not_a_panic(), assert_refused(), copy_index(), damaged_index_files_are_named(), long_read_mode_carries_the_read_group_and_extra_header(), long_read_mode_refuses_output_it_cannot_write(), mem(), scratch() (+3 more)

### Community 89 - "FM-index Search Tests"
Cohesion: 0.24
Nodes (10): backward_search_counts_match_naive(), forward_ext_equals_swapped_backward(), get_sa_batch_equals_get_sa(), get_sa_batch_runs_equal_get_sa(), get_sa_matches_sais_and_is_permutation(), konly_backward_step_equals_full(), sa_samples_match_the_file(), sa_tag() (+2 more)

### Community 90 - "Pair Penalty Scoring"
Cohesion: 0.23
Nodes (9): erfc(), mem_pair(), mem_pair_in(), memo_matches_direct_formula(), negative_score_end_pairs_at_zero(), PairPenalty, .SPAN, PairPenaltyKey (+1 more)

### Community 91 - "Seed Filtering"
Cohesion: 0.26
Nodes (8): KSW_I16_LANES, mem_flt_chained_seeds(), MEM_HSP_COEF, MEM_MINSC_COEF, mem_seed_sw(), MEM_SEEDSW_COEF, MEM_SHORT_EXT, MEM_SHORT_LEN

### Community 92 - "Seeding Levers (minibwa, Prefetch)"
Cohesion: 0.15
Nodes (7): minibwa (Li & Homer 2026), strobealign, smem_dedup adjacent SMEM dedup (untested), minibwa (Li & Homer 2026), Lockstep prefetch batched driver, Ungapped-first / q-mer prefilter lever, get_sa_batch vs get_sa_entries_prefetch 4.5x gap

### Community 93 - "Memory-System Scaling Limits"
Cohesion: 0.17
Nodes (7): STAR maximal mappable prefix + uncompressed SA, TLB / page-walk wall, ALU(N) compute ceiling, crates/bwa-index/src/hugepage.rs THP, Random-access bandwidth ceiling, efficiency(N)=ALU(N) x RANDOM_BW(N), TLB reach (entries x page size)

### Community 94 - "Fork Teardown and x86 Profile"
Cohesion: 0.17
Nodes (9): Fork teardown, Ship per-ISA x86 artefacts (v2/v3), sam_encode SIMD SEQ/QUAL emit (taken), Rescue kernel delivered throughput gap (40% x86 vs 63% ARM), x86 side-by-side profile, mem_sort_dedup_patch sort/dedup cost, Zen 3 perf profile vs fork, Bioconda packaging (+1 more)

### Community 95 - "ARM Lockstep Sweep Jobs"
Cohesion: 0.15
Nodes (13): run, defaults, lockstep-sweep, pgo-arm, defaults, name, runs-on, timeout-minutes (+5 more)

### Community 96 - "Bench x86 Inputs"
Cohesion: 0.15
Nodes (13): default, description, required, default, description, required, base_ref, e2e_runner (+5 more)

### Community 97 - "Bench x86 Steps"
Cohesion: 0.23
Nodes (13): steps, BASE_REF, READ_PAIRS, REQUIRE_AVX512_E2E, VARIANT_CPUS, steps, steps, env (+5 more)

### Community 98 - "Hybrid Bench and SAM Diff"
Cohesion: 0.23
Nodes (7): EXIT_USAGE_OR_IO, main(), MAX_EXAMPLES, load_sa(), main(), nt4(), read_fastq()

### Community 99 - "Multiline FASTQ Unwrap"
Cohesion: 0.23
Nodes (6): needs_unwrapping(), unwrap_detection(), unwrap_multiline_fastq(), unwrap_rewrites_wrapped_records(), UnwrapFastq, UnwrapFastq<R>

### Community 100 - "bwa-diff Comparator"
Cohesion: 0.25
Nodes (8): compare(), Divergence, FLAG_LAST_SEGMENT, FLAG_SECONDARY, FLAG_SUPPLEMENTARY, parse_primary(), Rec, Report

### Community 101 - "SA Parity Tools"
Cohesion: 0.31
Nodes (6): main(), report(), run_one(), with_sentinel(), main(), suffix_array_inplace()

### Community 102 - "Parity Workflow"
Cohesion: 0.18
Nodes (10): env, BWA_MEM2_SHA256, BWA_MEM2_VERSION, CARGO_TERM_COLOR, RUST_TOOLCHAIN, RUSTFLAGS, jobs, name (+2 more)

### Community 103 - "Profile Micromamba Setup"
Cohesion: 0.18
Nodes (11): with, condarc, create-args, download-micromamba, environment-name, if-no-files-found, init-shell, micromamba-binary-path (+3 more)

### Community 104 - "PGO x86 Job"
Cohesion: 0.18
Nodes (10): concurrency, cancel-in-progress, group, jobs, pgo-x86, name, defaults, name (+2 more)

### Community 106 - "Integration Byte-Identity Tests"
Cohesion: 0.36
Nodes (7): align(), paired_end_output_is_identical_with_and_without_the_overlap(), repo_root(), single_end_output_is_identical_with_and_without_the_overlap(), write_reads(), tiny_dir(), tiny_index_is_byte_identical()

### Community 107 - "STAR vs FM Benchmark"
Cohesion: 0.31
Nodes (7): fwd_ext(), K, main(), now(), prefetch(), QLEN, W

### Community 108 - "FASTQ Reader Variants"
Cohesion: 0.31
Nodes (4): empty_input_is_zero_records(), FastqReader, InterleavedFastqReader, non_utf8_header_keeps_its_bytes()

### Community 109 - "Head-to-Head Inputs"
Cohesion: 0.20
Nodes (10): default, description, required, base_ref, runner, workflow_dispatch, default, description (+2 more)

### Community 110 - "Head-to-Head Steps"
Cohesion: 0.20
Nodes (10): AVX2_ARM, BASE_REF, REQUIRE_AVX512, steps, env, if, name, run (+2 more)

### Community 111 - "Global SW Instruction Count"
Cohesion: 0.33
Nodes (4): full_band(), mat(), tightened(), ksw_global2()

### Community 112 - "Batch Count Tests"
Cohesion: 0.47
Nodes (7): batch_count_respects_verbosity(), repo_root(), reports_batch_count_on_stderr(), run_mem(), write_reads(), zero_k_and_zero_t_use_the_default_batch(), zero_threads_and_zero_k_use_the_default_batch_size()

### Community 113 - "Scalar Global SW and CIGAR"
Cohesion: 0.31
Nodes (6): wave16_matches_scalar(), ksw_global2_scalar(), push_cigar(), query_profile_row(), query_profile_ssse3(), RunCigar

### Community 114 - "Decompression Reader"
Cohesion: 0.33
Nodes (6): gzip_threads(), InflatedBlocks, open_reader(), parsed(), spawn_inflate(), spawn_plain()

### Community 116 - "Optimization Roadmap"
Cohesion: 0.22
Nodes (6): Byte-identity to bwa-mem2 output contract, Optimization roadmap, PGO build (+3.1% SE / +4.5% PE), AMAC / Cimple batching, MLP limit via MSHR / Little's law, scripts/bench.sh region-index guard

### Community 117 - "Bench ARM Setup"
Cohesion: 0.22
Nodes (9): with, components, condarc, create-args, download-micromamba, environment-name, init-shell, micromamba-binary-path (+1 more)

### Community 118 - "Bench x86 Setup"
Cohesion: 0.22
Nodes (9): with, condarc, create-args, download-micromamba, environment-name, fetch-depth, init-shell, micromamba-binary-path (+1 more)

### Community 119 - "Option Parity Job"
Cohesion: 0.22
Nodes (9): option-parity, runs-on, steps, timeout-minutes, continue-on-error, if, name, run (+1 more)

### Community 120 - "Perf x86 Setup"
Cohesion: 0.22
Nodes (9): with, components, condarc, create-args, download-micromamba, environment-name, init-shell, micromamba-binary-path (+1 more)

### Community 121 - "Release Env"
Cohesion: 0.22
Nodes (8): env, CARGO_TERM_COLOR, RUST_TOOLCHAIN, RUSTFLAGS, jobs, name, permissions, contents

### Community 122 - "Release Build Env"
Cohesion: 0.22
Nodes (9): CFLAGS, CXXFLAGS, GH_TOKEN, GITHUB_REPOSITORY, MACOSX_DEPLOYMENT_TARGET, OPENMP_STATIC, SANITISE, TAG (+1 more)

### Community 123 - "Option Parity Script"
Cohesion: 0.42
Nodes (8): check(), check_bgzf(), check_f(), check_longread(), check_o(), check_primary5(), check_shapes(), opt_parity.sh script

### Community 125 - "Core Error Types"
Cohesion: 0.29
Nodes (6): Error, Fastq, IndexFormat, Io, Other, Result

### Community 126 - "Suboptimal Score Tracker"
Cohesion: 0.32
Nodes (4): dump(), enabled(), ksw_local_fwd(), SuboptimalTracker

### Community 127 - "Alignment Algorithm Alternatives"
Cohesion: 0.25
Nodes (5): BitPAl / BGSA bit-parallel alignment, fg-labs/bwa-mem3 fork, Monge/seaweed sub-quadratic alignment, SeedEx speculate-and-certify, Suzuki-Kasahara difference recurrences

### Community 128 - "Rescue Lever Plans"
Cohesion: 0.32
Nodes (6): Rescue next lever plan, rescue_prune Hull decision, K-mer pruning rescue_prune.rs, Fork-style banded rescue at T1 (GO), Catch the fork plan, Idle sample share 27%

### Community 129 - "Bench ARM Steps"
Cohesion: 0.46
Nodes (8): steps, steps, steps, if, name, run, shell, uses

### Community 130 - "x86 End-to-End Small Ref"
Cohesion: 0.25
Nodes (8): run, defaults, if, name, runs-on, timeout-minutes, endtoend-smallref, shell

### Community 131 - "Head-to-Head Setup"
Cohesion: 0.25
Nodes (8): with, condarc, create-args, download-micromamba, environment-name, init-shell, micromamba-binary-path, toolchain

### Community 132 - "Parity Env"
Cohesion: 0.25
Nodes (8): EVENT_NAME, IDX, KNOWN_X86_XS_DIVERGENCE, R1, R2, READS, REQUIRE_AVX512, env

### Community 133 - "Perf x86 Job"
Cohesion: 0.25
Nodes (8): run, perf-x86, defaults, name, runs-on, timeout-minutes, defaults, shell

### Community 134 - "Bioconda Build Script"
Cohesion: 0.25
Nodes (7): BINDGEN_EXTRA_CLANG_ARGS, CFLAGS, CPPFLAGS, LDFLAGS, LIBCLANG_PATH, RUSTFLAGS, build.sh script

### Community 135 - "SIMD Kernel Search Levers"
Cohesion: 0.29
Nodes (7): Fully-in-band column regime in extension kernel (-5.07% kernel), #45 USQADD and shared qsub(h,oe) (+10.3% kernel), #46 scalar passes (finish_row, extract_group): negative; 46C rowmax u8 shipped, #47 padding-column regime (gain scales with padded fraction), #48 batched.rs levers (A/D already applied by LLVM; B fold prepass +4-5%; C x86 band mask), #50 alignment span lemma (50B no overlap; 50A reverse-pass bound), Multi-agent SIMD kernel search (32 agents, 8 survivors, index #51)

### Community 136 - "Sequence Emit Benchmark"
Cohesion: 0.38
Nodes (5): BLOCK, FWD, main(), push_as_chunks(), push_chunks_exact()

### Community 137 - "Extension SIMD Tiers"
Cohesion: 0.29
Nodes (7): calibrate(), Tier, Avx2, Avx512, Scalar, Sse41, time

### Community 138 - "Rescue SIMD Tiers"
Cohesion: 0.29
Nodes (7): forced_rescue_tier(), RescueTier, Auto, Avx2, Avx512, Scalar, Sse41

### Community 139 - "head-to-head.yml head-to-head.yml"
Cohesion: 0.29
Nodes (6): concurrency, cancel-in-progress, group, jobs, name, on

### Community 140 - "head-to-head.yml head-to-head"
Cohesion: 0.29
Nodes (7): run, defaults, name, runs-on, timeout-minutes, head-to-head, shell

### Community 141 - "release.yml build"
Cohesion: 0.29
Nodes (7): needs, runs-on, strategy, timeout-minutes, build, fail-fast, matrix

### Community 142 - "release.yml check-version"
Cohesion: 0.29
Nodes (7): outputs, runs-on, timeout-minutes, check-version, publishing, tag, version

### Community 143 - "release.yml include"
Cohesion: 0.29
Nodes (7): deployment_target, omp, os, pgo, target_id, variant_cpu, include

### Community 144 - "RAM Reduction Levers"
Cohesion: 0.29
Nodes (5): Two-batch overlap (BWA4_NO_BATCH_OVERLAP: +1.1% wall for -41% RSS), Compact FM-index in memory (32-byte checkpoints, sa_ms_byte as bit), #25 RAM per batch: resident = live on Linux; 2x is double in-flight batch, RAM reduction 19.2 to 11.2 GB (pac reference, read() BWT), BWA4_STAGE_ALLOC allocator probe (feature stage-alloc)

### Community 145 - "giab32_headline.sh Script"
Cohesion: 0.57
Nodes (6): allowed(), note(), refuse(), run(), say(), giab32_headline.sh script

### Community 146 - "giab_publication.sh Script"
Cohesion: 0.62
Nodes (5): load1(), run(), say(), giab_publication.sh script, wait_quiet()

### Community 147 - "BWT Binning Benchmark"
Cohesion: 0.53
Nodes (4): bin_sort(), main(), pass(), W

### Community 149 - "Score Sort Benchmark"
Cohesion: 0.47
Nodes (4): key_u128(), key_u64x2(), main(), Row

### Community 151 - "Rescue Row Max Clamp"
Cohesion: 0.33
Nodes (3): i32, RowMax, u8

### Community 152 - "Rank Structures (QuadRank)"
Cohesion: 0.40
Nodes (6): cp_occ FM checkpoint structure, QuadRank rank structure (2.29 bits/base), QuadRank vs cp_occ Rosetta benchmark, ropebwt3 / BML, Singletrack gap-affine backtrace, 2026 survey vs byte-identity

### Community 153 - "parity.yml runner"
Cohesion: 0.33
Nodes (6): runner, workflow_dispatch, default, description, required, inputs

### Community 154 - "parity.yml on"
Cohesion: 0.33
Nodes (6): on, pull_request, push, schedule, branches, cron

### Community 155 - "parity.yml with"
Cohesion: 0.33
Nodes (6): with, key, name, path, retention-days, toolchain

### Community 156 - "Lockstep Seeding Scaling"
Cohesion: 0.33
Nodes (6): Dead levers list (LISA, ILP interleave, vector occ, SBT prepass, STAR prefix, SME), 1 GB hugepages: reachable but not measurable on hosted runner, Lockstep seeding width measured at startup (MLP probe in cp_occ), get_sa_batch prefetch window 32 to 128, Scaling model: efficiency = ALU(N) x random-memory throughput(N), Lockstep seeding plus prefetch hides FM-index latency

### Community 157 - "Guided Batch Chunking"
Cohesion: 0.40
Nodes (3): guided_bounds(), guided_chunks_tile_the_batch_and_keep_their_order(), run_in_order()

### Community 158 - "Across-Read Stage States"
Cohesion: 0.40
Nodes (5): Stage, Kept, Pending, Purged, Undecided

### Community 160 - "Tile Padding (piece16)"
Cohesion: 0.40
Nodes (5): fill_soa(), piece16(), piece16_is_the_tile_zero_padded(), t16(), t16v()

### Community 161 - "bench-arm.yml graviton-selfhosted"
Cohesion: 0.40
Nodes (5): if, name, runs-on, timeout-minutes, graviton-selfhosted

### Community 162 - "bench-x86.yml kernel_ab_only"
Cohesion: 0.40
Nodes (5): kernel_ab_only, default, description, required, type

### Community 163 - "bench-x86.yml require_avx512"
Cohesion: 0.40
Nodes (5): require_avx512, default, description, required, type

### Community 164 - "bench-x86.yml require_avx512_e2e"
Cohesion: 0.40
Nodes (5): require_avx512_e2e, default, description, required, type

### Community 165 - "head-to-head.yml avx2_arm"
Cohesion: 0.40
Nodes (5): default, description, required, type, avx2_arm

### Community 166 - "head-to-head.yml require_avx512"
Cohesion: 0.40
Nodes (5): require_avx512, default, description, required, type

### Community 167 - "parity.yml require_avx512"
Cohesion: 0.40
Nodes (5): require_avx512, default, description, required, type

### Community 168 - "perf-x86.yml isa-and-tiers-x86"
Cohesion: 0.40
Nodes (5): defaults, name, runs-on, timeout-minutes, isa-and-tiers-x86

### Community 169 - "perf-x86.yml real-reads-x86"
Cohesion: 0.40
Nodes (5): real-reads-x86, defaults, name, runs-on, timeout-minutes

### Community 170 - "release.yml promote"
Cohesion: 0.40
Nodes (5): promote, if, needs, runs-on, timeout-minutes

### Community 171 - "release.yml publish"
Cohesion: 0.40
Nodes (5): publish, if, needs, runs-on, timeout-minutes

### Community 172 - "fork_bench.sh Script"
Cohesion: 0.60
Nodes (3): med(), say(), fork_bench.sh script

### Community 173 - "giab_happy.sh Script"
Cohesion: 0.70
Nodes (4): align(), call(), score(), giab_happy.sh script

### Community 174 - "graviton_setup.sh Script"
Cohesion: 0.60
Nodes (4): have(), MAMBA_ROOT_PREFIX, say(), graviton_setup.sh script

### Community 175 - "Unpack Benchmark"
Cohesion: 1.00
Nodes (3): main(), unpack_scalar(), unpack_vector()

### Community 176 - "Base Encodings"
Cohesion: 0.50
Nodes (4): Bases, Ascii, Codes, Packed

### Community 177 - "Across-Read Modes"
Cohesion: 0.50
Nodes (4): Mode, Finish, Full, Staged

### Community 178 - "bench-x86.yml hugepage-probe"
Cohesion: 0.50
Nodes (4): name, runs-on, timeout-minutes, hugepage-probe

### Community 179 - "bench-x86.yml read_pairs"
Cohesion: 0.50
Nodes (4): read_pairs, default, description, required

### Community 180 - "bench-x86.yml runner"
Cohesion: 0.50
Nodes (4): runner, default, description, required

### Community 181 - "bench-x86.yml variant_cpu"
Cohesion: 0.50
Nodes (4): variant_cpu, default, description, required

### Community 182 - "head-to-head.yml read_pairs"
Cohesion: 0.50
Nodes (4): read_pairs, default, description, required

### Community 183 - "perf-profile-x86.yml extra_rustflags"
Cohesion: 0.50
Nodes (4): default, description, required, extra_rustflags

### Community 184 - "perf-profile-x86.yml icount"
Cohesion: 0.50
Nodes (4): default, description, required, icount

### Community 185 - "perf-profile-x86.yml icount"
Cohesion: 0.50
Nodes (4): if, runs-on, timeout-minutes, icount

### Community 186 - "perf-profile-x86.yml knob_sweep"
Cohesion: 0.50
Nodes (4): knob_sweep, default, description, required

### Community 187 - "perf-profile-x86.yml read_pairs"
Cohesion: 0.50
Nodes (4): read_pairs, default, description, required

### Community 188 - "perf-profile-x86.yml read_set"
Cohesion: 0.50
Nodes (4): read_set, default, description, required

### Community 189 - "perf-profile-x86.yml runner"
Cohesion: 0.50
Nodes (4): runner, default, description, required

### Community 190 - "perf-profile-x86.yml meth"
Cohesion: 0.50
Nodes (4): meth, if, runs-on, timeout-minutes

### Community 191 - "perf-profile-x86.yml on"
Cohesion: 0.50
Nodes (4): on, push, workflow_dispatch, branches

### Community 192 - "Suffix-Array Indexer Backends"
Cohesion: 0.50
Nodes (3): libsais-c-omp with static OpenMP (ldd-asserted release), libsais-rs threading: +1.17x arm64, 0.69x x86, Suffix-array indexer backends (libsais C, libsais-rs, OpenMP, CaPS-SA, sufr)

### Community 193 - "FASTQ Reader Critical Path"
Cohesion: 0.50
Nodes (4): open_reader FIFO reopen bug (zero records, no error), FASTQ reader critical path (parallel pair read, inflate thread, rapidgzip), Record single-buffer representation (3 allocs to 1), thread-broker control law d* applied (0.09 thread; threshold N>=177)

### Community 194 - "giab30x_bench.sh Script"
Cohesion: 0.83
Nodes (3): run(), say(), giab30x_bench.sh script

### Community 195 - "giab30x_pe.sh Script"
Cohesion: 0.83
Nodes (3): run(), say(), giab30x_pe.sh script

### Community 196 - "pe_diverge_hunt.sh Script"
Cohesion: 1.00
Nodes (3): capture(), say(), pe_diverge_hunt.sh script

### Community 197 - "pe_diverge_subset.sh Script"
Cohesion: 0.83
Nodes (3): cap(), say(), pe_diverge_subset.sh script

### Community 198 - "pgo.sh Script"
Cohesion: 0.50
Nodes (3): CC, PATH, pgo.sh script

### Community 199 - "NEON SW Feature Flags"
Cohesion: 0.67
Nodes (3): fwd_local_sw_neon_u8(), shareoe_enabled(), usqadd_enabled()

## Knowledge Gaps
- **183 isolated node(s):** `bwa-mem4-chain`, `Refused`, `Contained`, `Appended`, `bwa-mem4` (+178 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1193 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **41 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ksw_align2()` connect `Third-Party Oracle Tests` to `Global SW Band Tightening`, `Rescue Band Benchmark`, `Batched Mate Rescue (PE)`, `Batched KSW Align and Rescue`, `Rescue Pruning Hits`, `Mate SW Band Counters`, `SwBackend Scalar Oracle`, `PE Mate SW and SAM Pairing`, `Seed Filtering`, `Suboptimal Score Tracker`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `batched_align()` (e.g. with `Params` and `bands_reproduce_ksw_align2()`) actually correct?**
  _`batched_align()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bwa-mem4-chain`, `Refused`, `Contained` to the rest of the system?**
  _183 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Chain Building and Filtering` be split into smaller, more focused modules?**
  _Cohesion score 0.0500990099009901 - nodes in this community are weakly interconnected._
- **Why does `mem_sort_dedup_patch()` connect `Primary Marking and MAPQ` to `Chain Building and Filtering`, `mem Run SE/PE Driver`, `Measured Perf Levers`, `Batched Mate Rescue (PE)`, `mem Option Parsing`, `Per-Read Chain Alignment`, `mem Stage Counters`, `PE Mate SW and SAM Pairing`, `Divergences and Byte-Identity`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Should `SMEM Seeding (FM-index)` be split into smaller, more focused modules?**
  _Cohesion score 0.05212716222533895 - nodes in this community are weakly interconnected._
- **Why does `ksw_extend2()` connect `SwBackend Scalar Oracle` to `Global SW Band Tightening`, `Batched Extension SIMD Tests`, `Kernel Ceiling and Dead Ends`, `NEON Backend`, `Per-Read Chain Alignment`, `mem Stage Counters`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._