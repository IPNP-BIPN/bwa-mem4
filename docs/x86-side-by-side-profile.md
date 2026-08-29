# The first side-by-side x86 profile against fg-labs/bwa-mem3 (2026-08-29)

Every previous statement about the wgsim deficit on x86 was inferred from stage timers. Ours report
our stage shares; the fork reports its own, in its own units, per thread; the two cannot be
subtracted. `perf-profile-x86.yml` records BOTH binaries under `perf` on ONE host with ONE read set,
which turns "their scalar glue is faster" into a ranked list of functions.

AMD EPYC 7763 (Zen 3, no AVX-512), 1M simulated pairs on chr21, `-t4`, our `x86-64-v3` build with
frame pointers, their bioconda binary. Self time, symbols above 0.4%.

The part matters and this file nearly got it wrong: the first draft of this paragraph said "Intel
Xeon Platinum 8573C" because that is the part the recent measurements were taken on, and the gate
that checks this document against the profile is what caught it. `ubuntu-22.04` draws EPYC 7763
uniformly. The 1.21x deficit used for the scaling below was itself replicated on this part among
others, so the arithmetic stands, but every line here is a Zen 3 line.

## What the two profiles say

| ours | % | theirs | % |
|---|---|---|---|
| `batched_extend_avx2_u8` | 24.16 | `smithWaterman256_8` | 23.18 |
| `ksw_global2` | 11.87 | `ksw_global2_avx2` | 11.77 |
| `LsSlot::step` | 11.25 | `ls_advance_backward_step` | 7.05 |
| `fwd_local_sw_batch` | 9.55 | `kswv256_u8_impl` | 9.39 |
| `mem_sort_dedup_patch` | 4.65 | `bwtSeedStrategyAllPosOneThread` | 7.24 |
| `unpack_pac_fwd` | 3.50 | `mem_chain2aln_across_reads_V2` | 5.85 |
| `get_sa_batch` | 3.39 | `getSMEMsOnePosOneThread_lockstep` | 5.33 |
| `build_chains_from_resolved` | 2.74 | `call_one_step` | 3.72 |
| `LocalKey<T>::with` (the two sorts) | 2.60 | `chain_add_one_seed` | 2.94 |
| `mem_chain_flt` | 2.52 | `pdqsort_loop<dedup_pair_sc*` | 2.50 |
| `discard_contained` | 2.38 | `ls_advance_forward_step` | 2.29 |
| `align_reads_batched` | 2.33 | `mem_sort_dedup_patch` | 1.75 |
| `mem_collect_smem_batched` | 2.12 | `smithWatermanBatchWrapper8` | 1.72 |
| `batch_mate_rescue` | 2.04 | `pdqsort_loop<dedup_pair_re*` | 1.48 |
| `run_side` | 1.85 | `bns_get_seq_into` | 1.39 |
| `extract_group` | 1.40 | `ungapped_analyze` | 1.04 |
| `dispatch_bins` | 1.04 | `get_sa_entries_prefetch` | 0.91 |

## Reading it in absolute time, which is the only way it means anything

Percentages are of different totals: our run is the longer one by 1.21x, so an equal share is 1.21x
the seconds. Scaling ours by 1.21 and grouping by what the code does, in units where the fork's
whole run is 100:

| what | ours | theirs | difference |
|---|---|---|---|
| seeding and SMEM collection | 13.6 | 25.6 | **-12.0, we are far ahead** |
| extension kernel | 29.2 | 23.2 | +6.0 |
| scalar glue (chains, containment, batching, dispatch) | 19.2 | 12.4 | +6.8 |
| SA resolution and reference fetch | 8.3 | 2.3 | +6.0 |
| CIGAR (`ksw_global2`) | 14.4 | 11.8 | +2.6 |
| mate-rescue kernel | 11.6 | 9.4 | +2.2 |
| sort and dedup | 8.8 | 6.7 | +2.1 |

Four things this changes.

**The deficit is not one thing.** It is three roughly equal pieces (extension kernel, scalar glue,
SA and reference fetch) plus two smaller ones, against one large advantage of our own. Any single
fix is worth at most a quarter of it.

**Our seeding is the best part of this aligner and nobody knew.** LISA does in 13.6 units what their
five SMEM functions do in 25.6. That is the largest single line in the table and it runs the other
way. It also explains why the gap closes on real reads, where seeding is a bigger share of the run.

**Their dedup is not the advantage the teardown assumed.** They replaced klib's introsort with
pdqsort, and their sort-and-dedup total (`mem_sort_dedup_patch` 1.75 + two `pdqsort_loop`
instantiations 3.98 + `dedup_perm_sort_by_score` 0.52 plus `dedup_perm_sort_by_re` 0.40 = 6.65) is close to ours (`mem_sort_dedup_patch` 4.65 plus `LocalKey<T>::with` 2.60 = 7.25) and
was 8.8 against 6.7 in absolute terms before the u128 key landed. Worth a few units, not the story.

**`get_sa_entries_prefetch` against `get_sa_batch`: 0.9 against 4.1.** The largest RELATIVE gap in
the profile, 4.5x, on the one function whose name in both codebases says the same thing. Our window
sweep found the prefetch width flat from 16 to 128 on three machines, but every one of those
machines was Apple Silicon or a hosted ARM part, where a DRAM miss costs a fraction of what it costs
on this Xeon. The sweep answered "does the width matter on this machine" and was read as "does
prefetching matter". Those are different questions and the second one is now open again, on x86.

## What this file is not

It is a single host, a single read set, and a `perf` sample at 499 Hz with frame-pointer unwinding.
The groupings above are judgement calls about which symbol does what, made from the names and from
reading both sources; a symbol that belongs in two groups is put in one. Nothing here is a
measurement of a change, because no change was made between the two recordings. It is a map, and the
next optimisation should be chosen from it rather than from the stage timers.
