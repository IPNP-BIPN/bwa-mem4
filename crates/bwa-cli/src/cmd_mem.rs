//! `bwa-mem3 mem` subcommand.
//!
//! Phase 6: seed -> chain -> extend -> best region -> `reg2aln` (exact CIGAR + NM/MD). MAPQ and
//! secondary/XA handling follow.

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Args;
use rayon::prelude::*;

use bwa_core::{dna, MemOpt};
use bwa_index::{BntSeq, FmIndex};
use bwa_io::{sam, FastqReader, PairedFastqReader, Record, SqRecord};
use bwa_mem::{
    align_reads_batched, alt::mem_gen_alt, cigar_string, mem_approx_mapq_se, mem_mark_primary_se,
    mem_pestat, mem_sam_pe, mem_sort_dedup_patch, reg2aln, MemAlnReg,
};
use bwa_neon::{NeonBackend, NeonFwd};

#[derive(Args)]
pub struct MemArgs {
    /// Number of threads (single-threaded for now; accepted but ignored).
    #[arg(short = 't', default_value_t = 1)]
    pub threads: i32,
    /// Process INT input bases per batch (fixes batch boundaries for reproducibility).
    #[arg(short = 'K')]
    pub k_batch: Option<i64>,
    /// Index prefix: the FASTA path that was indexed.
    pub index_prefix: PathBuf,
    /// Reads in FASTQ (R1, or the only file for single-end).
    pub reads: PathBuf,
    /// Optional mate reads (R2): triggers paired-end mode.
    pub reads2: Option<PathBuf>,
    /// Write SAM to PATH instead of stdout. A `.gz`/`.bgz` suffix selects BGZF (block-gzip) output,
    /// compressed in parallel on `-t` worker threads (readable by samtools/bgzip/tabix).
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,
    /// Route seed extension through the Metal GPU backend (macOS only; opt-in, byte-identical to the
    /// CPU path). Ignored on other platforms.
    #[arg(long)]
    pub gpu: bool,
}

/// The SAM output sink: plain (stdout or an uncompressed file) or parallel BGZF (block-gzip). Both
/// expose `std::io::Write` so the formatting/writing paths stay generic; `finish` drains the BGZF
/// worker pool and writes the EOF marker (surfacing errors the `Drop` path would swallow).
enum Output {
    Plain(Box<dyn Write + Send>),
    Bgzf(Box<bgzf::MultithreadedWriter<Box<dyn Write + Send>>>),
}

impl Write for Output {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Output::Plain(w) => w.write(buf),
            Output::Bgzf(w) => w.write(buf),
        }
    }
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Output::Plain(w) => w.flush(),
            Output::Bgzf(w) => w.flush(),
        }
    }
}

impl Output {
    /// Open the output sink. `None` writes uncompressed SAM to stdout; a path with a `.gz`/`.bgz`
    /// suffix writes BGZF compressed in parallel on `threads` workers; any other path writes an
    /// uncompressed file.
    fn open(path: Option<&Path>, threads: usize) -> anyhow::Result<Self> {
        match path {
            None => Ok(Output::Plain(Box::new(BufWriter::new(std::io::stdout())))),
            Some(p) => {
                let bgzf = p
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("gz") || e.eq_ignore_ascii_case("bgz"));
                let file = std::fs::File::create(p)?;
                if bgzf {
                    let sink: Box<dyn Write + Send> =
                        Box::new(BufWriter::with_capacity(1 << 20, file));
                    let level = bgzf::CompressionLevel::new(6)
                        .map_err(|e| anyhow::anyhow!("bgzf compression level: {e}"))?;
                    let workers = std::num::NonZero::new(threads.max(1))
                        .unwrap_or(std::num::NonZero::<usize>::MIN);
                    Ok(Output::Bgzf(Box::new(
                        bgzf::MultithreadedWriter::with_worker_count(workers, sink, level),
                    )))
                } else {
                    Ok(Output::Plain(Box::new(BufWriter::new(file))))
                }
            }
        }
    }

    /// Flush and finalize (BGZF: drain workers + write the EOF marker). Consumes the sink.
    fn finish(self) -> anyhow::Result<()> {
        match self {
            Output::Plain(mut w) => w.flush()?,
            Output::Bgzf(mut w) => {
                w.finish()?;
            }
        }
        Ok(())
    }
}

/// Overlap I/O with compute. A **reader** thread produces batches (opening the FASTQ *inside* the
/// thread, so the reader never crosses a thread boundary — only the `Send` record batches do), the
/// main thread aligns+formats each batch (internally parallel across the rayon pool) into one byte
/// buffer, and a **writer** thread drains those buffers. Bounded channels cap the batches in flight.
///
/// Output order equals read order (one reader → sequential main → one writer), and batch boundaries
/// are fixed by `-K`, so the output is byte-identical to the old serial read→align→write loop — the
/// serial read and write are simply hidden behind the next batch's compute.
fn run_pipeline<B: Send>(
    out: Output,
    read_batches: impl FnOnce(std::sync::mpsc::SyncSender<(B, u64)>) -> anyhow::Result<()> + Send,
    process: impl Fn(B, u64) -> Vec<u8>,
) -> anyhow::Result<()> {
    std::thread::scope(|scope| -> anyhow::Result<()> {
        // A couple of batches read-ahead / write-behind is enough to hide I/O behind compute.
        let (batch_tx, batch_rx) = std::sync::mpsc::sync_channel::<(B, u64)>(2);
        let (line_tx, line_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(3);

        let reader = scope.spawn(move || read_batches(batch_tx));
        let writer = scope.spawn(move || -> anyhow::Result<()> {
            let mut out = out;
            for buf in line_rx {
                out.write_all(&buf)?;
            }
            out.finish()?;
            Ok(())
        });

        for (batch, base_id) in batch_rx {
            let buf = process(batch, base_id);
            if line_tx.send(buf).is_err() {
                break; // writer exited; its error surfaces on join below
            }
        }
        drop(line_tx);

        reader.join().expect("reader thread panicked")?;
        writer.join().expect("writer thread panicked")?;
        Ok(())
    })
}

pub fn run(args: MemArgs, argv: &[String]) -> anyhow::Result<()> {
    let opt = MemOpt::default();
    let n_threads = args.threads.max(1) as usize;
    // Fixed-size rayon pool. Output order and global read ids are independent of thread count, so
    // byte-identity holds at any `-t` once `-K` fixes the batch boundaries.
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok();
    let k_batch = args
        .k_batch
        .unwrap_or(opt.chunk_size * i64::from(args.threads))
        .max(1) as usize;

    let fm = FmIndex::load(&args.index_prefix)?;
    let bns = BntSeq::load(&args.index_prefix)?;
    let sqs: Vec<SqRecord> = bns
        .contigs
        .iter()
        .map(|c| SqRecord {
            name: c.name.clone(),
            len: i64::from(c.len),
        })
        .collect();

    let mut out = Output::open(args.output.as_deref(), n_threads)?;
    let cl = argv.join(" ");
    sam::write_header(
        &mut out,
        &sqs,
        "bwa-mem3",
        "bwa-mem3",
        env!("CARGO_PKG_VERSION"),
        &cl,
    )?;

    let backend = Backend::select(args.gpu);

    if let Some(reads2) = args.reads2.clone() {
        return run_pe(&fm, &bns, &opt, &args.reads, &reads2, k_batch, backend, out);
    }

    // Reader thread: open the FASTQ here and stream fixed-`-K` batches with their cumulative base id.
    let reads_path = args.reads.clone();
    let read_batches =
        move |tx: std::sync::mpsc::SyncSender<(Vec<Record>, u64)>| -> anyhow::Result<()> {
            let mut reader = FastqReader::from_path(&reads_path)?;
            let mut base_id = 0u64;
            loop {
                let batch = reader.next_batch(k_batch)?;
                if batch.is_empty() {
                    break;
                }
                let n = batch.len() as u64;
                if tx.send((batch, base_id)).is_err() {
                    break;
                }
                base_id += n;
            }
            Ok(())
        };

    // Main: seed -> chain -> BATCHED seed extension across the whole read batch (NEON backend),
    // mirroring bwa-mem2's mem_chain2aln_across_reads_V2. Chunked so extension parallelizes; each
    // read's regions are independent of chunk composition, so output stays byte-identical at any
    // thread count once -K fixes the batch boundaries.
    let process = |batch: Vec<Record>, base_id: u64| -> Vec<u8> {
        let all_codes: Vec<Vec<u8>> = batch
            .iter()
            .map(|rec| rec.seq.iter().map(|&b| dna::nt4(b)).collect())
            .collect();
        let regs_all = batched_regs(&fm, &bns, &opt, &all_codes, backend);
        let lines: Vec<Vec<u8>> = batch
            .par_iter()
            .enumerate()
            .map(|(i, rec)| {
                finish_se(
                    &fm,
                    &bns,
                    &opt,
                    rec,
                    &all_codes[i],
                    regs_all[i].clone(),
                    base_id + i as u64,
                )
            })
            .collect();
        let mut buf = Vec::with_capacity(lines.iter().map(Vec::len).sum());
        for l in &lines {
            buf.extend_from_slice(l);
        }
        buf
    };

    run_pipeline(out, read_batches, process)
}

/// Seed + chain + batched extension for a whole read batch, returning each read's pre-dedup regions
/// (byte-identical to `align_read` per read). Chunked across the rayon pool so extension batches run
/// in parallel; per-read results are independent of the chunking.
fn batched_regs(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[Vec<u8>],
    backend: Backend,
) -> Vec<Vec<MemAlnReg>> {
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = codes.len().div_ceil(nthreads).max(1);
    codes
        .par_chunks(chunk)
        .flat_map(|c| backend.align(fm, bns, opt, c))
        .collect()
}

/// Seed-extension backend chosen at startup. `Metal` only exists on macOS with a GPU; everywhere
/// else `--gpu` degrades to the SIMD CPU backend. The choice is per-process, so byte-identity of the
/// output is unaffected either way (both are byte-identical to the oracle).
#[derive(Clone, Copy)]
enum Backend {
    Cpu,
    #[cfg(target_os = "macos")]
    Metal,
}

impl Backend {
    fn select(want_gpu: bool) -> Self {
        if !want_gpu {
            return Backend::Cpu;
        }
        #[cfg(target_os = "macos")]
        {
            if bwa_gpu::metal_available() {
                eprintln!("[bwa-mem3] seed extension: Metal GPU backend");
                return Backend::Metal;
            }
            eprintln!("[bwa-mem3] --gpu requested but no Metal device; using CPU backend");
            Backend::Cpu
        }
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("[bwa-mem3] --gpu is macOS-only; using CPU backend");
            Backend::Cpu
        }
    }

    fn align(
        self,
        fm: &FmIndex,
        bns: &BntSeq,
        opt: &MemOpt,
        codes: &[Vec<u8>],
    ) -> Vec<Vec<MemAlnReg>> {
        match self {
            Backend::Cpu => align_reads_batched(fm, bns, opt, codes, &NeonBackend),
            #[cfg(target_os = "macos")]
            Backend::Metal => align_reads_batched(fm, bns, opt, codes, &bwa_gpu::MetalBackend),
        }
    }
}

/// Deduplicate + primary-mark a read's batched regions, then format its SAM record. Pure (no shared
/// state beyond the immutable index/options), so it is safe across rayon workers.
fn finish_se(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    rec: &Record,
    codes: &[u8],
    regs_pre: Vec<MemAlnReg>,
    read_id: u64,
) -> Vec<u8> {
    let mut regs = mem_sort_dedup_patch(fm, opt, codes, regs_pre);
    mem_mark_primary_se(opt, &mut regs, read_id);
    // After marking, regs[0] is the highest-scoring primary region.
    let xa = mem_gen_alt(fm, bns, opt, &regs, codes.len() as i32, codes);
    let best = regs.first().filter(|r| r.score >= opt.t);
    let mut buf = Vec::new();
    match best {
        Some(r) => {
            let aln = reg2aln(fm, bns, opt, codes.len() as i32, codes, r);
            let mapq = mem_approx_mapq_se(opt, r);
            let rname = &bns.contigs[aln.rid as usize].name;
            let flag = if aln.is_rev { 16 } else { 0 };
            let cigar = cigar_string(&aln.cigar);
            let mut tags = format!(
                "NM:i:{}\tMD:Z:{}\tAS:i:{}\tXS:i:{}",
                aln.nm, aln.md, aln.score, aln.sub
            );
            if let Some(xa0) = &xa[0] {
                tags.push_str("\tXA:Z:");
                tags.push_str(xa0);
            }
            let pos = aln.pos + 1;
            if aln.is_rev {
                let seq = dna::revcomp_ascii(&rec.seq);
                let qual = rec.qual.as_ref().map(|q| {
                    let mut v = q.clone();
                    v.reverse();
                    v
                });
                sam::write_mapped_se(
                    &mut buf,
                    &rec.name,
                    flag,
                    rname,
                    pos,
                    mapq,
                    &cigar,
                    &seq,
                    qual.as_deref(),
                    &tags,
                )
                .expect("write to Vec");
            } else {
                sam::write_mapped_se(
                    &mut buf,
                    &rec.name,
                    flag,
                    rname,
                    pos,
                    mapq,
                    &cigar,
                    &rec.seq,
                    rec.qual.as_deref(),
                    &tags,
                )
                .expect("write to Vec");
            }
        }
        None => {
            sam::write_unmapped(&mut buf, &rec.name, &rec.seq, rec.qual.as_deref())
                .expect("write to Vec");
        }
    }
    buf
}

/// Paired-end driver: per batch, align+dedup both ends of every pair, estimate insert sizes
/// (`mem_pestat`), then emit paired SAM (`mem_sam_pe`). The pair index is global across batches (for
/// the `hash` tie-break), matching bwa-mem2's `(n_processed>>1)+i`.
#[allow(clippy::too_many_arguments)]
fn run_pe(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    reads1: &std::path::Path,
    reads2: &std::path::Path,
    k_batch: usize,
    backend: Backend,
    out: Output,
) -> anyhow::Result<()> {
    // Reader thread: open the mate files here and stream fixed-`-K` pair batches with the cumulative
    // pair id (global across batches for the `hash` tie-break, matching bwa-mem2's `(n_processed>>1)+i`).
    let (reads1, reads2) = (reads1.to_owned(), reads2.to_owned());
    let read_batches =
        move |tx: std::sync::mpsc::SyncSender<(Vec<(Record, Record)>, u64)>| -> anyhow::Result<()> {
            let mut reader = PairedFastqReader::from_paths(&reads1, &reads2)?;
            let mut base_pair = 0u64;
            loop {
                let batch = reader.next_batch(k_batch)?;
                if batch.is_empty() {
                    break;
                }
                let n = batch.len() as u64;
                if tx.send((batch, base_pair)).is_err() {
                    break;
                }
                base_pair += n;
            }
            Ok(())
        };

    // Seed -> chain -> BATCHED extension over both ends of every pair (interleaved c1,c2,...), then
    // per-read dedup. Regions are per-read independent, so this is byte-identical to the per-read
    // path; primary marking and pairing happen later, per bwa-mem2.
    let process = |batch: Vec<(Record, Record)>, base_pair: u64| -> Vec<u8> {
        let all_codes: Vec<Vec<u8>> = batch
            .iter()
            .flat_map(|(r1, r2)| {
                [
                    r1.seq.iter().map(|&b| dna::nt4(b)).collect::<Vec<u8>>(),
                    r2.seq.iter().map(|&b| dna::nt4(b)).collect::<Vec<u8>>(),
                ]
            })
            .collect();
        let regs_all = batched_regs(fm, bns, opt, &all_codes, backend);
        let mut prepared: Vec<PrepPair> = batch
            .par_iter()
            .enumerate()
            .map(|(i, (r1, r2))| {
                let c1 = all_codes[2 * i].clone();
                let c2 = all_codes[2 * i + 1].clone();
                let a1 = mem_sort_dedup_patch(fm, opt, &c1, regs_all[2 * i].clone());
                let a2 = mem_sort_dedup_patch(fm, opt, &c2, regs_all[2 * i + 1].clone());
                PrepPair {
                    c1,
                    c2,
                    a1,
                    a2,
                    name1: r1.name.clone(),
                    name2: r2.name.clone(),
                    q1: r1.qual.clone(),
                    q2: r2.qual.clone(),
                }
            })
            .collect();

        // Insert-size stats over the whole batch (interleaved region slices, no copy).
        let regs_ref: Vec<&[MemAlnReg]> = prepared
            .iter()
            .flat_map(|p| [p.a1.as_slice(), p.a2.as_slice()])
            .collect();
        let pes = mem_pestat(opt, bns.l_pac, &regs_ref);

        // Emit paired SAM in parallel (each pair owns its regions; global pair id fixes hashes).
        let bufs: Vec<Vec<u8>> = prepared
            .par_iter_mut()
            .enumerate()
            .map(|(i, p)| {
                let names = [p.name1.clone(), p.name2.clone()];
                let seqs = [p.c1.as_slice(), p.c2.as_slice()];
                let quals = [p.q1.as_deref(), p.q2.as_deref()];
                let mut buf = Vec::new();
                mem_sam_pe(
                    fm,
                    bns,
                    opt,
                    &pes,
                    base_pair + i as u64,
                    &names,
                    &seqs,
                    &quals,
                    &mut p.a1,
                    &mut p.a2,
                    &mut buf,
                    &NeonFwd,
                )
                .expect("write to Vec");
                buf
            })
            .collect();
        let mut buf = Vec::with_capacity(bufs.iter().map(Vec::len).sum());
        for b in &bufs {
            buf.extend_from_slice(b);
        }
        buf
    };

    run_pipeline(out, read_batches, process)
}

/// One read pair prepared for the pairing/output stage: nt4 codes, dedup'd regions, names, quals.
struct PrepPair {
    c1: Vec<u8>,
    c2: Vec<u8>,
    a1: Vec<MemAlnReg>,
    a2: Vec<MemAlnReg>,
    name1: String,
    name2: String,
    q1: Option<Vec<u8>>,
    q2: Option<Vec<u8>>,
}
