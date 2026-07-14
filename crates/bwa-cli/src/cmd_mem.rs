//! `bwa-mem3 mem` subcommand.
//!
//! Phase 6: seed -> chain -> extend -> best region -> `reg2aln` (exact CIGAR + NM/MD). MAPQ and
//! secondary/XA handling follow.

use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Args;
use rayon::prelude::*;

use bwa_core::{dna, MemOpt};
use bwa_index::{BntSeq, FmIndex};
use bwa_io::{sam, FastqReader, PairedFastqReader, Record, SqRecord};
use bwa_mem::{
    align_read_dedup, align_read_se, cigar_string, mem_approx_mapq_se, mem_pestat, mem_sam_pe,
    reg2aln, MemAlnReg,
};

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

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let cl = argv.join(" ");
    sam::write_header(
        &mut out,
        &sqs,
        "bwa-mem3",
        "bwa-mem3",
        env!("CARGO_PKG_VERSION"),
        &cl,
    )?;

    if let Some(reads2) = args.reads2.clone() {
        run_pe(&fm, &bns, &opt, &args.reads, &reads2, k_batch, &mut out)?;
        out.flush()?;
        return Ok(());
    }

    let mut reader = FastqReader::from_path(&args.reads)?;
    let mut base_id = 0u64;
    loop {
        let batch = reader.next_batch(k_batch)?;
        if batch.is_empty() {
            break;
        }
        // Align reads in parallel; each read's global id (base_id + i) and the output order are
        // fixed by batch position, so the result is byte-identical to single-threaded.
        let lines: Vec<Vec<u8>> = batch
            .par_iter()
            .enumerate()
            .map(|(i, rec)| format_se(&fm, &bns, &opt, rec, base_id + i as u64))
            .collect();
        for l in &lines {
            out.write_all(l)?;
        }
        base_id += batch.len() as u64;
    }
    out.flush()?;
    Ok(())
}

/// Align one read single-end and format its SAM record into a fresh buffer. Pure (no shared state
/// beyond the immutable index/options), so it is safe to run across rayon workers.
fn format_se(fm: &FmIndex, bns: &BntSeq, opt: &MemOpt, rec: &Record, read_id: u64) -> Vec<u8> {
    let codes: Vec<u8> = rec.seq.iter().map(|&b| dna::nt4(b)).collect();
    let regs = align_read_se(fm, bns, opt, &codes, read_id);
    // After marking, regs[0] is the highest-scoring primary region.
    let best = regs.first().filter(|r| r.score >= opt.t);
    let mut buf = Vec::new();
    match best {
        Some(r) => {
            let aln = reg2aln(fm, bns, opt, codes.len() as i32, &codes, r);
            let mapq = mem_approx_mapq_se(opt, r);
            let rname = &bns.contigs[aln.rid as usize].name;
            let flag = if aln.is_rev { 16 } else { 0 };
            let cigar = cigar_string(&aln.cigar);
            let tags = format!(
                "NM:i:{}\tMD:Z:{}\tAS:i:{}\tXS:i:{}",
                aln.nm, aln.md, aln.score, aln.sub
            );
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
fn run_pe<W: std::io::Write>(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    reads1: &std::path::Path,
    reads2: &std::path::Path,
    k_batch: usize,
    out: &mut W,
) -> anyhow::Result<()> {
    let mut reader = PairedFastqReader::from_paths(reads1, reads2)?;
    let mut base_pair = 0u64;
    loop {
        let batch = reader.next_batch(k_batch)?;
        if batch.is_empty() {
            break;
        }
        // Align + dedup both ends of every pair in parallel (regions only; primary marking and
        // pairing happen later, per bwa-mem2). No cross-pair state, so order is preserved.
        let mut prepared: Vec<PrepPair> = batch
            .par_iter()
            .map(|(r1, r2)| {
                let c1: Vec<u8> = r1.seq.iter().map(|&b| dna::nt4(b)).collect();
                let c2: Vec<u8> = r2.seq.iter().map(|&b| dna::nt4(b)).collect();
                let a1 = align_read_dedup(fm, bns, opt, &c1);
                let a2 = align_read_dedup(fm, bns, opt, &c2);
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
                )
                .expect("write to Vec");
                buf
            })
            .collect();
        for b in &bufs {
            out.write_all(b)?;
        }
        base_pair += batch.len() as u64;
    }
    Ok(())
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
