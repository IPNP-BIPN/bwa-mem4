//! `bwa-mem3 mem` subcommand.
//!
//! Phase 6: seed -> chain -> extend -> best region -> `reg2aln` (exact CIGAR + NM/MD). MAPQ and
//! secondary/XA handling follow.

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Args;
use rayon::prelude::*;

use bwa_core::opt::flags;
use bwa_core::{dna, MemOpt};
use bwa_index::{BntSeq, FmIndex};
use bwa_io::{sam, FastqReader, PairedFastqReader, Record, SqRecord};
use bwa_mem::{
    align_reads_batched, alt::mem_gen_alt, batch_mate_rescue, cigar::cigar_string_which,
    cigar::MemAln, cigar_string, mem_approx_mapq_se, mem_mark_primary_se, mem_pestat, mem_sam_pe,
    mem_sort_dedup_patch, reg2aln, MemAlnReg, PairRescueData,
};
use bwa_neon::NeonBackend;

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
                    let sink: Box<dyn Write + Send> = Box::new(BufWriter::with_capacity(1 << 20, file));
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
        run_pe(&fm, &bns, &opt, &args.reads, &reads2, k_batch, backend, out)?;
        bwa_gpu::dump_stats();
        return Ok(());
    }

    // Reader thread: open the FASTQ here and stream fixed-`-K` batches with their cumulative base id.
    let reads_path = args.reads.clone();
    let read_batches = move |tx: std::sync::mpsc::SyncSender<(Vec<Record>, u64)>| -> anyhow::Result<()> {
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
        // Move each read's regions out of `regs_all` (consumed by `finish_se`) instead of cloning:
        // `into_par_iter` yields the owned `Vec<MemAlnReg>`, dropping a per-read Vec allocation+copy.
        let lines: Vec<Vec<u8>> = batch
            .par_iter()
            .zip(all_codes.par_iter())
            .zip(regs_all.into_par_iter())
            .enumerate()
            .map(|(i, ((rec, codes), regs_pre))| {
                finish_se(&fm, &bns, &opt, rec, codes, regs_pre, base_id + i as u64)
            })
            .collect();
        let mut buf = Vec::with_capacity(lines.iter().map(Vec::len).sum());
        for l in &lines {
            buf.extend_from_slice(l);
        }
        buf
    };

    run_pipeline(out, read_batches, process)?;
    bwa_gpu::dump_stats();
    Ok(())
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
            Backend::Metal => {
                align_reads_batched(fm, bns, opt, codes, &bwa_gpu::MetalBackend)
            }
        }
    }
}

/// Env-gated (`BWA3_DUMP_REGS`) region dump; cached, since `finish_se` runs per read.
fn dump_regs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA3_DUMP_REGS").is_some())
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
    if dump_regs_enabled() {
        eprintln!("=== read {} ===", rec.name);
        bwa_mem::dump_regs(bns, "pre-dedup", &regs_pre);
    }
    let mut regs = mem_sort_dedup_patch(fm, opt, codes, regs_pre);
    mem_mark_primary_se(opt, &mut regs, read_id);
    if dump_regs_enabled() {
        bwa_mem::dump_regs(bns, "post-dedup+mark", &regs);
    }
    // After marking, regs[0] is the highest-scoring primary region.
    let xa = mem_gen_alt(fm, bns, opt, &regs, codes.len() as i32, codes);

    // `mem_reg2sam`: emit every region clearing -T that is not shadowed by a better overlapping one
    // (`secondary < 0`). The first survivor is the primary; the others are chimeric hits, emitted as
    // supplementary records whose MAPQ cannot exceed the primary's. Shadowed regions are never
    // emitted here -- they surface in the primary's XA:Z tag instead (`mem_gen_alt`).
    let mut alns: Vec<EmitAln> = Vec::new();
    for (k, p) in regs.iter().enumerate() {
        if p.score < opt.t || p.secondary >= 0 {
            continue;
        }
        let aln = reg2aln(fm, bns, opt, codes.len() as i32, codes, p);
        let mut mapq = mem_approx_mapq_se(opt, p);
        if let Some(first) = alns.first() {
            if !p.is_alt && mapq > first.mapq {
                mapq = first.mapq;
            }
        }
        alns.push(EmitAln { aln, mapq, is_alt: p.is_alt, xa: xa[k].clone() });
    }

    let mut buf = Vec::new();
    if alns.is_empty() {
        sam::write_unmapped(&mut buf, &rec.name, &rec.seq, rec.qual.as_deref())
            .expect("write to Vec");
        return buf;
    }
    for which in 0..alns.len() {
        write_aln_se(&mut buf, bns, opt, rec, &alns, which);
    }
    buf
}

/// One emitted alignment plus the per-record state `mem_aln2sam` needs (`mem_aln_t` + its MAPQ,
/// which we compute outside `reg2aln`).
struct EmitAln {
    aln: MemAln,
    mapq: u32,
    is_alt: bool,
    xa: Option<String>,
}

/// `mem_aln2sam` for one of a read's alignments. `which` indexes `alns`: 0 is the primary, any other
/// is a supplementary record (FLAG 0x800), which hard-clips and carries only its own slice of
/// SEQ/QUAL so the read's bases are stored exactly once across the records.
fn write_aln_se(
    buf: &mut Vec<u8>,
    bns: &BntSeq,
    opt: &MemOpt,
    rec: &Record,
    alns: &[EmitAln],
    which: usize,
) {
    let e = &alns[which];
    let aln = &e.aln;
    let softclip = opt.flag & flags::SOFTCLIP != 0;
    let clip_ends = which != 0 && !softclip && !e.is_alt && !aln.cigar.is_empty();

    let mut flag = aln.flag;
    if aln.is_rev {
        flag |= 0x10;
    }
    if which != 0 {
        flag |= if opt.flag & flags::NO_MULTI != 0 { 0x100 } else { 0x800 };
    }

    // Hard-clipped ends are not in this record's SEQ/QUAL. bwa reads the clip lengths off both
    // cigar ends (which coincide for a 1-op cigar) and maps them onto the *forward* read.
    let (mut qb, mut qe) = (0usize, rec.seq.len());
    if clip_ends {
        let (first, last) = (aln.cigar[0], aln.cigar[aln.cigar.len() - 1]);
        let is_clip = |c: u32| (c & 0xf) == 3 || (c & 0xf) == 4;
        let (flen, llen) = ((first >> 4) as usize, (last >> 4) as usize);
        if aln.is_rev {
            if is_clip(first) {
                qe -= flen;
            }
            if is_clip(last) {
                qb += llen;
            }
        } else {
            if is_clip(first) {
                qb += flen;
            }
            if is_clip(last) {
                qe -= llen;
            }
        }
    }

    let mut tags = String::new();
    if !aln.cigar.is_empty() {
        tags.push_str(&format!("NM:i:{}\tMD:Z:{}", aln.nm, aln.md));
    }
    if aln.score >= 0 {
        tags.push_str(&format!("\tAS:i:{}", aln.score));
    }
    if aln.sub >= 0 {
        tags.push_str(&format!("\tXS:i:{}", aln.sub));
    }
    // SA:Z lists this read's *other* emitted alignments, with their raw (unconverted) CIGARs.
    if alns.len() > 1 {
        tags.push_str("\tSA:Z:");
        for (i, o) in alns.iter().enumerate() {
            if i == which {
                continue;
            }
            tags.push_str(&format!(
                "{},{},{},{},{},{};",
                bns.contigs[o.aln.rid as usize].name,
                o.aln.pos + 1,
                if o.aln.is_rev { '-' } else { '+' },
                cigar_string(&o.aln.cigar),
                o.mapq,
                o.aln.nm,
            ));
        }
    }
    if let Some(x) = &e.xa {
        tags.push_str("\tXA:Z:");
        tags.push_str(x);
    }

    let cigar = cigar_string_which(&aln.cigar, which, e.is_alt, softclip);
    let rname = &bns.contigs[aln.rid as usize].name;
    let (seq, qual) = if aln.is_rev {
        let q = rec.qual.as_ref().map(|q| {
            let mut v = q[qb..qe].to_vec();
            v.reverse();
            v
        });
        (dna::revcomp_ascii(&rec.seq[qb..qe]), q)
    } else {
        (rec.seq[qb..qe].to_vec(), rec.qual.as_ref().map(|q| q[qb..qe].to_vec()))
    };
    sam::write_mapped_se(
        buf,
        &rec.name,
        flag,
        rname,
        aln.pos + 1,
        e.mapq,
        &cigar,
        &seq,
        qual.as_deref(),
        &tags,
    )
    .expect("write to Vec");
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
        // Pair up each mate's owned codes + regions by sequential moves (no content copy), so the
        // parallel prep can consume them instead of cloning `all_codes[2i]`/`regs_all[2i]` per pair.
        let mut code_it = all_codes.into_iter();
        let mut reg_it = regs_all.into_iter();
        #[allow(clippy::type_complexity)]
        let paired: Vec<((Vec<u8>, Vec<u8>), (Vec<MemAlnReg>, Vec<MemAlnReg>))> = (0..batch.len())
            .map(|_| {
                let c1 = code_it.next().unwrap();
                let c2 = code_it.next().unwrap();
                let r1 = reg_it.next().unwrap();
                let r2 = reg_it.next().unwrap();
                ((c1, c2), (r1, r2))
            })
            .collect();
        let mut prepared: Vec<PrepPair> = batch
            .par_iter()
            .zip(paired.into_par_iter())
            .map(|((r1, r2), ((c1, c2), (rg1, rg2)))| {
                let a1 = mem_sort_dedup_patch(fm, opt, &c1, rg1);
                let a2 = mem_sort_dedup_patch(fm, opt, &c2, rg2);
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

        // Mate rescue, batched across the whole pair batch so the per-anchor insert-window SW fills
        // the SIMD lanes. Byte-identical to the per-pair rescue in `mem_sam_pe` (which is then told to
        // skip it). `BWA3_SCALAR_RESCUE` keeps the per-pair path for A/B verification.
        let scalar_rescue = std::env::var_os("BWA3_SCALAR_RESCUE").is_some();
        if !scalar_rescue {
            let mut rd: Vec<PairRescueData> = prepared
                .iter_mut()
                .map(|p| PairRescueData {
                    seq0: p.c1.as_slice(),
                    seq1: p.c2.as_slice(),
                    a0: &mut p.a1,
                    a1: &mut p.a2,
                })
                .collect();
            // Each pair's rescue is independent, so run chunks in parallel; a chunk of a few hundred
            // pairs still has enough rescue jobs to fill the SIMD lanes. Keeps -t8 scaling (the rescue
            // is otherwise a serial section) while byte-identical to the per-pair path.
            rd.par_chunks_mut(512)
                .for_each(|chunk| batch_mate_rescue(fm, bns, opt, &pes, chunk));
        }

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
                    !scalar_rescue,
                    &mut buf,
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
