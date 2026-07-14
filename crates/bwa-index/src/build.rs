//! FMD index construction, byte-identical to `bwa-mem2 index`.
//!
//! Mirrors `bns_fasta2bntseq` (`.pac`/`.ann`/`.amb`) and `FMI_search::build_index`
//! (`.0123`/`.bwt.2bit.64`) from `reference/bwa-mem2/src`. See `docs`/the plan for the exact
//! byte layout. The suffix array is built with our own SA-IS (`crate::sais`).

use std::ffi::OsString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use bwa_core::{dna, Error, Result};

use crate::rand48::Rand48;
use crate::sais::suffix_array_inplace;

/// bwa-mem2's fixed RNG seed for ambiguous-base randomization (`.ann` third field).
const SEED: u32 = 11;

struct FastaContig {
    name: String,
    comment: String,
    seq: Vec<u8>,
}

struct AnnRec {
    name: String,
    anno: String,
    offset: i64,
    len: i32,
    n_ambs: i32,
}

struct AmbRec {
    offset: i64,
    len: i32,
    amb: char,
}

/// Build the five index files (`.pac`, `.ann`, `.amb`, `.0123`, `.bwt.2bit.64`) next to `fasta`.
pub fn build_index(fasta: &Path) -> Result<()> {
    let contigs = read_fasta(fasta)?;
    if contigs.is_empty() {
        return Err(Error::Other("empty FASTA".into()));
    }

    // 1. Encode the forward reference to 2-bit codes, randomizing N via lrand48, and collect the
    //    contig annotations and ambiguous-base holes (as bwa-mem2's add1 does).
    let mut rng = Rand48::srand48(SEED as i64);
    let mut forward: Vec<u8> = Vec::new();
    let mut anns: Vec<AnnRec> = Vec::new();
    let mut ambs: Vec<AmbRec> = Vec::new();

    for c in &contigs {
        let start = forward.len();
        let mut n_ambs = 0i32;
        // `lasts` resets per contig (bwa-mem2's `add1` does `for (i = lasts = 0; ...)`), so an
        // N-run at a contig boundary is split into one hole per contig, never merged across.
        let mut lasts: u8 = 0;
        for &raw in &c.seq {
            let mut code = dna::nt4(raw);
            if code >= 4 {
                let pos = forward.len() as i64;
                if raw != lasts {
                    ambs.push(AmbRec {
                        offset: pos,
                        len: 1,
                        amb: raw as char,
                    });
                    n_ambs += 1;
                } else if let Some(last) = ambs.last_mut() {
                    last.len += 1;
                }
                code = (rng.lrand48() & 3) as u8;
            }
            forward.push(code);
            lasts = raw;
        }
        let len = (forward.len() - start) as i32;
        let anno = if c.comment.is_empty() {
            "(null)".to_string()
        } else {
            c.comment.clone()
        };
        anns.push(AnnRec {
            name: c.name.clone(),
            anno,
            offset: start as i64,
            len,
            n_ambs,
        });
    }

    let l_pac = forward.len() as i64;
    let n_seqs = anns.len() as i32;

    // 2. Write .pac (2-bit packed forward reference).
    write_pac(&sibling(fasta, "pac"), &forward)?;

    // 3. Write .ann and .amb.
    write_ann(&sibling(fasta, "ann"), l_pac, n_seqs, &anns)?;
    write_amb(&sibling(fasta, "amb"), l_pac, n_seqs, &ambs)?;

    // 4. Build the forward++reverse-complement binary reference and write .0123.
    let mut bref = Vec::with_capacity(2 * forward.len());
    bref.extend_from_slice(&forward);
    for &c in forward.iter().rev() {
        bref.push(if c < 4 { 3 - c } else { c });
    }
    File::create(sibling(fasta, "0123")).and_then(|f| BufWriter::new(f).write_all(&bref))?;

    // 5. Build and write the FM-index (.bwt.2bit.64).
    write_fm_index(&sibling(fasta, "bwt.2bit.64"), &bref)?;

    Ok(())
}

fn sibling(prefix: &Path, ext: &str) -> PathBuf {
    let mut s: OsString = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

/// Minimal plain-text FASTA reader (name, comment, concatenated sequence).
fn read_fasta(path: &Path) -> Result<Vec<FastaContig>> {
    let text = std::fs::read_to_string(path)?;
    let mut contigs: Vec<FastaContig> = Vec::new();
    for line in text.lines() {
        if let Some(header) = line.strip_prefix('>') {
            let (name, comment) = match header.find([' ', '\t']) {
                Some(i) => (header[..i].to_string(), header[i + 1..].to_string()),
                None => (header.to_string(), String::new()),
            };
            contigs.push(FastaContig {
                name,
                comment,
                seq: Vec::new(),
            });
        } else if let Some(cur) = contigs.last_mut() {
            cur.seq
                .extend(line.bytes().filter(|b| !b.is_ascii_whitespace()));
        }
    }
    Ok(contigs)
}

/// `.pac`: 2-bit packed forward reference (first base in the high bits of each byte); the file is
/// always `floor(L/4)+2` bytes, and the final byte encodes `L mod 4`.
fn write_pac(path: &Path, forward: &[u8]) -> Result<()> {
    let l = forward.len();
    let mut pac = vec![0u8; (l >> 2) + 1];
    for (i, &c) in forward.iter().enumerate() {
        pac[i >> 2] |= c << ((3 - (i & 3)) << 1);
    }
    let body = (l >> 2) + usize::from(!l.is_multiple_of(4));
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&pac[..body])?;
    if l.is_multiple_of(4) {
        w.write_all(&[0u8])?;
    }
    w.write_all(&[(l % 4) as u8])?;
    w.flush()?;
    Ok(())
}

/// `.ann` (text): `l_pac n_seqs seed`, then per contig `gi name anno` and `offset len n_ambs`.
fn write_ann(path: &Path, l_pac: i64, n_seqs: i32, anns: &[AnnRec]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "{l_pac} {n_seqs} {SEED}")?;
    for a in anns {
        // gi is always 0; anno is always non-empty ("(null)" when absent) so it is always printed.
        writeln!(w, "0 {} {}", a.name, a.anno)?;
        writeln!(w, "{} {} {}", a.offset, a.len, a.n_ambs)?;
    }
    w.flush()?;
    Ok(())
}

/// `.amb` (text): `l_pac n_seqs n_holes`, then per hole `offset len amb_char`.
fn write_amb(path: &Path, l_pac: i64, n_seqs: i32, ambs: &[AmbRec]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "{} {} {}", l_pac, n_seqs, ambs.len())?;
    for a in ambs {
        writeln!(w, "{} {} {}", a.offset, a.len, a.amb)?;
    }
    w.flush()?;
    Ok(())
}

/// `.bwt.2bit.64`: `[ref_seq_len:i64 | count[5]:i64 | CP_OCC[] | sa_ms_byte[]:i8 | sa_ls_word[]:u32
/// | sentinel_index:i64]`, all little-endian. `bref` is the 2L forward++RC binary reference.
fn write_fm_index(path: &Path, bref: &[u8]) -> Result<()> {
    let two_l = bref.len(); // 2L
    let sa = suffix_array_inplace(bref); // length N = 2L+1, sa[0] = 2L (memory-efficient SA-IS)
    let n = sa.len(); // reference_seq_len = 2L + 1

    // count[5] = {0, #A, #A+#C, #A+#C+#G, 2L} over the 2L binary bases.
    let mut hist = [0i64; 4];
    for &c in bref {
        hist[c as usize] += 1;
    }
    let count: [i64; 5] = [
        0,
        hist[0],
        hist[0] + hist[1],
        hist[0] + hist[1] + hist[2],
        two_l as i64,
    ];

    // BWT: bwt[i] = 4 (sentinel) if sa[i]==0 else bref[sa[i]-1].
    let mut bwt = vec![6u8; n];
    let mut sentinel_index = 0i64;
    for (i, &sai) in sa.iter().enumerate() {
        if sai == 0 {
            bwt[i] = 4;
            sentinel_index = i as i64;
        } else {
            bwt[i] = bref[(sai - 1) as usize];
        }
    }

    // CP_OCC checkpoints, one per 64-base block.
    let n_blocks = (n >> 6) + 1;
    let mut cp_count = vec![[0i64; 4]; n_blocks];
    let mut one_hot = vec![[0u64; 4]; n_blocks];
    let mut running = [0i64; 4];
    for (bi, (cc, oh)) in cp_count.iter_mut().zip(one_hot.iter_mut()).enumerate() {
        *cc = running; // counts of each base in bwt[0..bi*64)
        let i0 = bi * 64;
        for j in 0..64 {
            let idx = i0 + j;
            let c = if idx < n { bwt[idx] } else { 6 };
            for w in oh.iter_mut() {
                *w <<= 1;
            }
            if (c as usize) < 4 {
                oh[c as usize] |= 1;
                running[c as usize] += 1;
            }
        }
    }

    // Compressed suffix array: sample every 8th entry (N is odd, so exactly (N>>3)+1 samples).
    let sa_count = (n >> 3) + 1;
    let mut sa_ms_byte = Vec::with_capacity(sa_count);
    let mut sa_ls_word = Vec::with_capacity(sa_count);
    let mut i = 0usize;
    while i < n {
        let v = sa[i] as u64;
        sa_ls_word.push((v & 0xFFFF_FFFF) as u32);
        sa_ms_byte.push(((v >> 32) & 0xFF) as u8);
        i += 8;
    }
    debug_assert_eq!(sa_ms_byte.len(), sa_count);

    // Serialize.
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(n as i64).to_le_bytes())?;
    for c in count {
        w.write_all(&c.to_le_bytes())?;
    }
    for (cc, oh) in cp_count.iter().zip(one_hot.iter()) {
        for &v in cc {
            w.write_all(&v.to_le_bytes())?;
        }
        for &v in oh {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.write_all(&sa_ms_byte)?;
    for &v in &sa_ls_word {
        w.write_all(&v.to_le_bytes())?;
    }
    w.write_all(&sentinel_index.to_le_bytes())?;
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An N-run must NOT be merged across a contig boundary: bwa-mem2's `add1` resets its
    /// `lasts` tracker to 0 at the start of every contig, so N at the end of one contig and N at
    /// the start of the next are recorded as two separate `.amb` holes, not one.
    #[test]
    fn amb_holes_not_merged_across_contig_boundary() {
        let dir = std::env::temp_dir().join(format!(
            "bwa3_amb_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let fasta = dir.join("t.fa");
        // c1 ends with NN (pos 4,5); c2 starts with NN (pos 6,7). l_pac = 12.
        std::fs::write(&fasta, ">c1\nACGTNN\n>c2\nNNACGT\n").unwrap();

        build_index(&fasta).unwrap();
        let amb = std::fs::read_to_string(dir.join("t.fa.amb")).unwrap();
        let lines: Vec<&str> = amb.lines().collect();

        assert_eq!(lines[0], "12 2 2", "header l_pac n_seqs n_holes");
        assert_eq!(lines[1], "4 2 N", "chr1 trailing NN, its own hole");
        assert_eq!(lines[2], "6 2 N", "chr2 leading NN, a separate hole");

        std::fs::remove_dir_all(&dir).ok();
    }
}
