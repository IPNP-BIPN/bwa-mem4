# A machine-independent model of this aligner's thread scaling

Written 2026-08-09, after a campaign in which seven candidate causes of "we only reach 64% efficiency
at `-t16`" were tested and eliminated one by one. The conclusion is not a bug and not a lever: the
scaling of an FM-index aligner is **set by two hardware quantities**, both measurable in about two
minutes on any machine, and the code's job is only to not fall below them.

This document is the model, the measurement recipe, and the validation. It replaces "why don't we
scale" as a recurring question with a number anyone can compute for their own hardware.

## 1. The model

For a workload dominated by dependent random lookups into a structure far larger than cache:

```
efficiency(N) = ALU(N) x RANDOM_BW(N)
```

- `ALU(N)`: what N threads of pure computation achieve on this machine, relative to one thread. It is
  **not** N: it captures all-core clock behaviour and, on hybrid CPUs, the fact that efficiency cores
  are not performance cores.
- `RANDOM_BW(N)`: what the memory system delivers on **random** reads with N threads, relative to one
  thread. It is **not** the STREAM/sequential number, and the gap between the two is large.

Both are properties of the machine. Neither can be improved by the aligner. Their product is the
ceiling the aligner is allowed to approach.

### Validation on an M4 Max (12 P cores + 4 E cores), GRCh38, 2 M read pairs

| threads | random read | `RANDOM_BW` | `ALU` | **predicted** | **measured** | error |
|---|---|---|---|---|---|---|
| 4 | 23,3 GB/s | 97 % | 99 % | **96 %** | 94 % | 2 pts |
| 8 | 43,8 GB/s | 91 % | 100 % | **91 %** | 86 % | 5 pts |
| 12 | 59,9 GB/s | 83 % | 97,5 % | **81 %** | 76 % | 5 pts |
| 16 | 69,7 GB/s | 73 % | 82 % | **60 %** | **59 %** | 1 pt |

The model predicts the whole curve within 5 points and hits the `-t16` point exactly. Note what the
two factors do at the ends: at `-t8` the machine is still fully compute-parallel and the loss is
entirely memory; at `-t16` the memory loss (73 %) and the E-core loss (82 %) multiply, which is why
that point falls so much faster than the ones before it.

## 2. The three numbers to measure on a new machine

### 2.1 `ALU(N)`: the compute ceiling

A loop with no memory traffic at all, N threads, throughput relative to one thread. Anything it loses
is clock or core heterogeneity, never cache or DRAM. On the M4 Max: 100 % at 8, 97,5 % at 12, **82 %
at 16**, because the four E cores are worth 1,4x between them.

A homogeneous server should return ~100 % up to its core count; if it does not, the machine is
thermally or power limited and no software will recover it.

### 2.2 `RANDOM_BW(N)`: the random-access ceiling

**Sequential bandwidth is the wrong number** and using it is the single most common error in this
area. On the M4 Max at 12 threads:

| pattern | read |
|---|---|
| sequential | 284,6 GB/s |
| stride 64 B | 143,7 GB/s |
| stride 4 KiB | 49,9 GB/s |
| **random uniform** | **59,9 GB/s (-78,9 %)** |

Random is **4,75x below** sequential. An index walk is random by construction: the FM recurrence
*computes* its next address from a popcount, so no prefetcher can anticipate it.

And random bandwidth **does not scale linearly with cores**: 5,99 / 23,3 / 43,8 / 59,9 / 69,7 GB/s at
1 / 4 / 8 / 12 / 16 threads, i.e. 97 / 91 / 83 / 73 % efficiency. That curve is the aligner's curve.

### 2.3 TLB reach: `entries x page_size`

Measured on this machine rather than quoted: L1 TLB ~192 entries (4 MiB), **L2 TLB ~3200 entries**,
page size 16 KiB, so a reach of **56 MiB**. The FM index's `cp_occ` is **9,4 GB, or 168x that reach**,
so essentially every access misses the L2 TLB and pays a page walk. Measured cost, cache-hot so it is
pure translation and not DRAM: **+10,7 ns per access** (16,81 ns spread against 6,14 ns packed).

Reach is the one term software can move, and only by changing the page size:

| page size | reach with 3200 entries | covers a 9,4 GB index? |
|---|---|---|
| 4 KiB (Linux default) | 12,5 MiB | no, 750x short |
| 16 KiB (Apple Silicon) | 56 MiB | no, 168x short |
| 2 MiB (Linux THP) | **6,4 GiB** | nearly |
| 1 GiB (explicit hugetlbfs) | 3,1 TiB | yes, entirely |

This is why `crates/bwa-index/src/hugepage.rs` measured **1,7x on the seeding stage** on Linux between
`THP=never` and `THP=madvise`, and why macOS cannot reach it: 16 KiB pages, no THP, no superpage API.

## 3. What the model forbids, and what it permits

**Forbidden.** Expecting any code change to lift `efficiency(N)` above `ALU(N) x RANDOM_BW(N)`. Load
balancing, batch sizes, allocator tuning, thread affinity and scheduling all move a workload *toward*
that product; none moves the product. This project measured all five and they returned 0 %, which is
the correct result and not a failure of the attempts.

**Permitted, in the order the model ranks them:**

1. **Raise the page size.** It is the only term that changes `RANDOM_BW` for a given workload, because
   a page walk is itself memory traffic. 2 MiB is free on Linux where a THP-hinting allocator is in
   use; 1 GiB needs an explicit hugetlbfs and a host with the RAM to reserve it.
2. **Reduce the number of accesses**, i.e. the algorithm. This is where minibwa's lead comes from
   (38,6 CPU-s against our 46,8 on the same data): a different SMEM algorithm and heuristics, not
   better scaling. Its efficiency at `-t16` is 62 %, below ours.
3. **Keep per-thread MLP at the hardware's limit, and not above.** See below.

## 4. The MLP corollary: why the lockstep width has a right answer

Cimple (Kiriansky et al., PACT'18) states the rule:

> *"The primary MLP limit for single threaded execution is the number of Miss Status Holding Registers
> (MSHR) [...] Modern processors typically have 6-10 L1 cache MSHRs [...] By Little's law, the
> achievable bandwidth equals the number of MSHR entries divided by the average memory latency."*

So the software's lockstep width `W` should be set to the hardware's outstanding-miss capacity. Below
it, latency is exposed; above it, the extra slots only add working set and scheduling cost.

Measured values: **~28 concurrent lanes on an Apple M4** (Lemire, 2025, pointer-chase lane sweep,
"both visibly sustain 28 lanes"), 10 L1 MSHRs on Intel Haswell (Cimple).

This is exactly what our own sweep found without knowing the number: 16 -> 32 slots is worth **1,27 %**
at `-t4`, and at `-t12`/`-t16` the knee is flat (N16 and N32 tie), because at those thread counts the
shared term, not the per-core term, is binding. `BWA4_LOCKSTEP_N` exists to re-sweep it; the default
is 32 on aarch64 and 16 elsewhere, and neither should be raised without measuring.

## 5. Why "batch the accesses" cannot be applied twice

A recurring proposal is to import a latency-hiding technique from the database literature: AMAC
(Kocberber et al., VLDB'16), coroutine-to-transaction (CoroBase, MosaicDB, VLDB'24), software
pipelining of index lookups. They report large gains, up to 2,3x for AMAC on irregular lookups.

**We already do this.** The lockstep driver is structurally AMAC: N independent FM walks in flight,
each with its own state, advanced one step per round, with the checkpoint block prefetched a round
ahead. That is why six separate "fewer memory accesses" levers have measured 0 % here (LISA's learned
index, a flat 49,6 GB suffix array, minibwa's 10-mer cache, an extra `get_sa_batch` prefetch, a THP
madvise hint, and co-locating the two sampled-SA arrays): each removed accesses that the batching
already overlapped, or added a structure whose own footprint cost more TLB than it saved.

The rule this project now applies before building any memory-system lever: **check whether the
accesses it removes are ones already overlapped**. If they are, its measured gain will be zero
regardless of how sound the reasoning looks.

## 6. Sources

- Kiriansky, Xu, Rinard, Devadas, *Cimple: Instruction and Memory Level Parallelism*, PACT 2018 —
  MSHR as the MLP ceiling, Little's law framing. <https://people.csail.mit.edu/rinard/paper/pact18.pdf>
- Kocberber, Falsafi, Grot, *Asynchronous Memory Access Chaining*, VLDB 2016 — the batching structure
  our lockstep implements. <http://www.vldb.org/pvldb/vol9/p252-kocberber.pdf>
- Huang, Wang, Zhou, Meng, *The Art of Latency Hiding in Modern Database Engines*, VLDB 2024 —
  coroutine-to-transaction, and the point that avoiding oversubscription matters as much as
  prefetching. <https://www.vldb.org/pvldb/vol17/p577-huang.pdf>
- Kim, Kim et al., *BWA-MEM-SCALE*, ICPP 2022 — 1 GiB HugeTLB worth +10,9 points; 1,97-2,03x from
  pipeline and I/O rework alone. <https://dl.acm.org/doi/fullHtml/10.1145/3545008.3545033>
- Vasimuddin, Misra, Li, Aluru, *Efficient Architecture-Aware Acceleration of BWA-MEM for Multicore
  Systems*, IPDPS 2019 — 3,5x single-thread against 2,4x single-socket, i.e. this same wall reported
  in the paper that founded bwa-mem2. <https://arxiv.org/abs/1907.12931>
- Langmead, Wilks, Antonescu, Charles, *Scaling read aligners to hundreds of threads*, Bioinformatics
  2019 — parser locks, lock type, output contention, allocation, NUMA.
  <https://academic.oup.com/bioinformatics/article/35/3/421/5055585>
- Gunther, *Universal Scalability Law* — the contention and coherence terms, and why a scaling curve
  has a ceiling rather than an asymptote. <https://www.perfdynamics.com/Manifesto/USLscalability.pdf>
- Lemire, *Memory-level parallelism: Apple M2 vs Apple M4*, 2025 — the 28-lane measurement.
  <https://lemire.me/blog/2025/07/09/memory-level-parallelism-apple-m2-vs-apple-m4/>
- 7-cpu, *Apple M1* — TLB sizes and the 4 ns/line parallel random read figure.
  <https://www.7-cpu.com/cpu/Apple_M1.html>
- Local measurements: `timoheimonen/macOS-memory-benchmark` for the TLB reach and the random-access
  bandwidth curve; this repository's own ALU probe and `BWA4_TRAFFIC` counters for the rest.
