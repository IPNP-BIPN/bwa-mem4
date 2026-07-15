//! FM-index loading and traversal, mirroring bwa-mem2's `FMI_search`.
//!
//! Loads `.bwt.2bit.64` (checkpointed occurrences + compressed suffix array) and `.0123` (the
//! forward++reverse-complement binary reference), and provides the primitives seeding needs:
//! [`FmIndex::get_occ`], [`FmIndex::backward_ext`] and [`FmIndex::get_sa`]. See
//! `reference/bwa-mem2/src/FMI_search.cpp` (`load_index`, `GET_OCC`, `backwardExt`,
//! `get_sa_entry_compressed`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use bwa_core::Result;

/// A bidirectional FM-index interval, mirroring bwa-mem2's `SMEM`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Smem {
    pub rid: u32,
    pub m: u32,
    pub n: u32,
    /// Forward interval start.
    pub k: i64,
    /// Reverse-complement interval start.
    pub l: i64,
    /// Interval size (occurrence count).
    pub s: i64,
}

/// One 64-base checkpoint block, interleaved exactly as bwa-mem2's `CP_OCC`: the cumulative counts
/// and the one-hot BWT bitvectors for the 4 bases sit together in a single 64-byte, cache-line-aligned
/// record, so a single occ lookup touches one cache line instead of two.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct CpOcc {
    cp_count: [i64; 4],
    one_hot: [u64; 4],
}

/// Loaded FM-index plus the binary reference for O(1) base access.
pub struct FmIndex {
    /// `reference_seq_len` = 2L + 1.
    pub ref_seq_len: i64,
    /// Cumulative base counts, already incremented by 1 as bwa-mem2's `load_index` does.
    count: [i64; 5],
    cp_occ: Vec<CpOcc>,
    sa_ms_byte: Vec<i8>,
    sa_ls_word: Vec<u32>,
    sentinel_index: i64,
    one_hot_mask: [u64; 64],
    /// The `.0123` reference: forward then reverse-complement, one byte/base (2L bytes).
    reference: Vec<u8>,
}

fn sibling(prefix: &Path, ext: &str) -> PathBuf {
    let mut s: OsString = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn rd_i64(d: &[u8], p: &mut usize) -> i64 {
    let v = i64::from_le_bytes(d[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}
fn rd_u64(d: &[u8], p: &mut usize) -> u64 {
    let v = u64::from_le_bytes(d[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}
fn rd_u32(d: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes(d[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}

impl FmIndex {
    /// Load `<prefix>.bwt.2bit.64` and `<prefix>.0123`.
    pub fn load(prefix: &Path) -> Result<Self> {
        let d = std::fs::read(sibling(prefix, "bwt.2bit.64"))?;
        let mut p = 0usize;
        let ref_seq_len = rd_i64(&d, &mut p);
        let mut count = [0i64; 5];
        for c in &mut count {
            *c = rd_i64(&d, &mut p);
        }
        for c in &mut count {
            *c += 1; // as load_index does
        }
        let cp_size = ((ref_seq_len >> 6) + 1) as usize;
        let mut cp_occ = Vec::with_capacity(cp_size);
        for _ in 0..cp_size {
            let cp_count = [
                rd_i64(&d, &mut p),
                rd_i64(&d, &mut p),
                rd_i64(&d, &mut p),
                rd_i64(&d, &mut p),
            ];
            let one_hot = [
                rd_u64(&d, &mut p),
                rd_u64(&d, &mut p),
                rd_u64(&d, &mut p),
                rd_u64(&d, &mut p),
            ];
            cp_occ.push(CpOcc { cp_count, one_hot });
        }
        let sa_size = ((ref_seq_len >> 3) + 1) as usize;
        let mut sa_ms_byte = Vec::with_capacity(sa_size);
        for _ in 0..sa_size {
            sa_ms_byte.push(d[p] as i8);
            p += 1;
        }
        let mut sa_ls_word = Vec::with_capacity(sa_size);
        for _ in 0..sa_size {
            sa_ls_word.push(rd_u32(&d, &mut p));
        }
        let sentinel_index = rd_i64(&d, &mut p);

        let mut one_hot_mask = [0u64; 64];
        let base = 0x8000_0000_0000_0000u64;
        one_hot_mask[1] = base;
        for i in 2..64 {
            one_hot_mask[i] = (one_hot_mask[i - 1] >> 1) | base;
        }

        let reference = std::fs::read(sibling(prefix, "0123"))?;

        Ok(FmIndex {
            ref_seq_len,
            count,
            cp_occ,
            sa_ms_byte,
            sa_ls_word,
            sentinel_index,
            one_hot_mask,
            reference,
        })
    }

    /// Prefetch the two checkpoint blocks a future [`backward_ext`] on an interval `[sp, ep)` will
    /// touch (`cp_occ[sp>>6]` and `cp_occ[ep>>6]`). Issued one SMEM step ahead in the seeding walk to
    /// hide the DRAM latency of the data-dependent block loads, exactly as bwa-mem2's / nh13's
    /// `ENABLE_PREFETCH` does. A pure hint: results are unchanged. No-op off AArch64/x86_64.
    #[inline]
    pub fn prefetch_occ(&self, sp: i64, ep: i64) {
        // SAFETY (both arches): forming the block pointers is in-bounds (`sp`/`ep` are valid BWT
        // positions, so `>>6` indexes an allocated `cp_occ` slot); the prefetch is a hint that never
        // faults or writes.
        #[cfg(target_arch = "aarch64")]
        {
            let base = self.cp_occ.as_ptr();
            // `pldl1keep` = prefetch-for-load into L1, keep (AArch64 equivalent of `_MM_HINT_T0`).
            unsafe {
                let p_sp = base.add((sp >> 6) as usize);
                let p_ep = base.add((ep >> 6) as usize);
                std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p_sp, options(nostack, readonly, preserves_flags));
                std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p_ep, options(nostack, readonly, preserves_flags));
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let base = self.cp_occ.as_ptr();
            unsafe {
                _mm_prefetch(base.add((sp >> 6) as usize) as *const i8, _MM_HINT_T0);
                _mm_prefetch(base.add((ep >> 6) as usize) as *const i8, _MM_HINT_T0);
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let _ = (sp, ep);
        }
    }

    /// Occurrences of base `c` in `bwt[0..pp)`, i.e. bwa-mem2's `GET_OCC`.
    #[inline]
    pub fn get_occ(&self, pp: i64, c: usize) -> i64 {
        let block = &self.cp_occ[(pp >> 6) as usize];
        let y = (pp & 63) as usize;
        block.cp_count[c] + (block.one_hot[c] & self.one_hot_mask[y]).count_ones() as i64
    }

    /// The full interval `[0, ref_seq_len)` (the empty match), for starting a backward search.
    #[inline]
    pub fn full_interval(&self) -> Smem {
        Smem {
            rid: 0,
            m: 0,
            n: 0,
            k: 0,
            l: 0,
            s: self.ref_seq_len,
        }
    }

    /// Extend `smem` by one base `a` on the left, i.e. bwa-mem2's `backwardExt`.
    pub fn backward_ext(&self, smem: Smem, a: usize) -> Smem {
        // Load the sp/ep checkpoint blocks once (all 4 bases share them), rather than re-deriving
        // the block index and re-indexing per base as `get_occ` would. Values are identical.
        let sp = smem.k;
        let ep = smem.k + smem.s;
        let blk_sp = &self.cp_occ[(sp >> 6) as usize];
        let blk_ep = &self.cp_occ[(ep >> 6) as usize];
        let msk_sp = self.one_hot_mask[(sp & 63) as usize];
        let msk_ep = self.one_hot_mask[(ep & 63) as usize];
        let mut k = [0i64; 4];
        let mut s = [0i64; 4];
        for b in 0..4 {
            let occ_sp = blk_sp.cp_count[b] + (blk_sp.one_hot[b] & msk_sp).count_ones() as i64;
            let occ_ep = blk_ep.cp_count[b] + (blk_ep.one_hot[b] & msk_ep).count_ones() as i64;
            k[b] = self.count[b] + occ_sp;
            s[b] = occ_ep - occ_sp;
        }
        let sentinel_offset =
            i64::from(smem.k <= self.sentinel_index && smem.k + smem.s > self.sentinel_index);
        let mut l = [0i64; 4];
        l[3] = smem.l + sentinel_offset;
        l[2] = l[3] + s[3];
        l[1] = l[2] + s[2];
        l[0] = l[1] + s[1];
        Smem {
            k: k[a],
            l: l[a],
            s: s[a],
            ..smem
        }
    }

    /// Suffix-array value at BWT row `pos`, decompressing via LF-walk to the nearest sample
    /// (bwa-mem2's `get_sa_entry_compressed`).
    pub fn get_sa(&self, pos: i64) -> i64 {
        if pos & 7 == 0 {
            let idx = (pos >> 3) as usize;
            return (i64::from(self.sa_ms_byte[idx]) << 32) + i64::from(self.sa_ls_word[idx]);
        }
        let mut offset = 0i64;
        let mut sp = pos;
        loop {
            let occ_id = (sp >> 6) as usize;
            let y = 63 - (sp & 63);
            let oh = self.cp_occ[occ_id].one_hot;
            let b = if (oh[0] >> y) & 1 == 1 {
                0
            } else if (oh[1] >> y) & 1 == 1 {
                1
            } else if (oh[2] >> y) & 1 == 1 {
                2
            } else if (oh[3] >> y) & 1 == 1 {
                3
            } else {
                4
            };
            if b == 4 {
                return offset;
            }
            let occ_sp = self.get_occ(sp, b);
            sp = self.count[b] + occ_sp;
            offset += 1;
            if sp & 7 == 0 {
                break;
            }
        }
        let idx = (sp >> 3) as usize;
        let sa_entry = (i64::from(self.sa_ms_byte[idx]) << 32) + i64::from(self.sa_ls_word[idx]);
        sa_entry + offset
    }

    /// Prefetch the single `cp_occ` checkpoint block that the LF-walk step at `pos` will touch. A
    /// pure hint (never faults or writes); no-op off AArch64/x86_64.
    #[inline]
    fn prefetch_cp(&self, pos: i64) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let p = self.cp_occ.as_ptr().add((pos >> 6) as usize);
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(self.cp_occ.as_ptr().add((pos >> 6) as usize) as *const i8, _MM_HINT_T0);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = pos;
    }

    /// [`get_sa`] for many independent positions at once. Each `get_sa` is a data-dependent LF-walk
    /// (each step a random `cp_occ` block load), but distinct positions are independent, so running a
    /// **window** of them in lockstep — prefetch every active slot's next block, then advance all one
    /// step — keeps many DRAM misses in flight and hides the latency. Result-identical to calling
    /// [`get_sa`] per position; `out[i]` receives `get_sa(positions[i])`.
    pub fn get_sa_batch(&self, positions: &[i64], out: &mut [i64]) {
        debug_assert_eq!(positions.len(), out.len());
        const W: usize = 32;
        let sa = |p: i64| -> i64 {
            let idx = (p >> 3) as usize;
            (i64::from(self.sa_ms_byte[idx]) << 32) + i64::from(self.sa_ls_word[idx])
        };
        let mut base = 0usize;
        while base < positions.len() {
            let w = (positions.len() - base).min(W);
            let mut sp = [0i64; W];
            let mut off = [0i32; W];
            let mut slot = [0u8; W]; // still-walking local indices [0, w)
            let mut nact = 0usize;
            for j in 0..w {
                let pos = positions[base + j];
                if pos & 7 == 0 {
                    out[base + j] = sa(pos);
                } else {
                    sp[j] = pos;
                    off[j] = 0;
                    slot[nact] = j as u8;
                    nact += 1;
                }
            }
            while nact > 0 {
                for a in 0..nact {
                    self.prefetch_cp(sp[slot[a] as usize]);
                }
                let mut wr = 0usize;
                for a in 0..nact {
                    let j = slot[a] as usize;
                    let spi = sp[j];
                    let occ_id = (spi >> 6) as usize;
                    let y = 63 - (spi & 63);
                    let oh = self.cp_occ[occ_id].one_hot;
                    let b = if (oh[0] >> y) & 1 == 1 {
                        0
                    } else if (oh[1] >> y) & 1 == 1 {
                        1
                    } else if (oh[2] >> y) & 1 == 1 {
                        2
                    } else if (oh[3] >> y) & 1 == 1 {
                        3
                    } else {
                        4
                    };
                    if b == 4 {
                        out[base + j] = i64::from(off[j]);
                        continue;
                    }
                    let nsp = self.count[b] + self.get_occ(spi, b);
                    off[j] += 1;
                    if nsp & 7 == 0 {
                        out[base + j] = sa(nsp) + i64::from(off[j]);
                    } else {
                        sp[j] = nsp;
                        slot[wr] = j as u8;
                        wr += 1;
                    }
                }
                nact = wr;
            }
            base += w;
        }
    }

    /// The binary reference base (0-3) at position `pos` in `[0, 2L)`.
    #[inline]
    pub fn base(&self, pos: i64) -> u8 {
        self.reference[pos as usize]
    }

    /// The loaded cumulative base counts (already `+1`, as bwa-mem2's `load_index`).
    #[inline]
    pub fn counts(&self) -> [i64; 5] {
        self.count
    }

    /// The `.0123` binary reference (forward ++ reverse-complement, 2L bytes).
    #[inline]
    pub fn reference(&self) -> &[u8] {
        &self.reference
    }

    /// Length of the forward reference `L` (`ref_seq_len` is `2L + 1`).
    #[inline]
    pub fn l_pac(&self) -> i64 {
        (self.ref_seq_len - 1) / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sais::suffix_array_with_sentinel;

    fn tiny() -> FmIndex {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        FmIndex::load(Path::new(prefix)).unwrap()
    }

    #[test]
    fn get_sa_matches_sais_and_is_permutation() {
        let fm = tiny();
        let two_l = fm.reference.len();
        let n = fm.ref_seq_len;
        assert_eq!(n, two_l as i64 + 1);
        let sa = suffix_array_with_sentinel(&fm.reference);
        let mut seen = vec![false; two_l + 1];
        for i in 0..n {
            let v = fm.get_sa(i);
            assert_eq!(v, sa[i as usize], "get_sa mismatch at row {i}");
            assert!(!seen[v as usize], "get_sa not a permutation at {i}");
            seen[v as usize] = true;
        }
    }

    #[test]
    fn backward_search_counts_match_naive() {
        let fm = tiny();
        let bref = &fm.reference;
        for &(start, len) in &[(100usize, 20usize), (5000, 15), (123, 31), (77_777, 25)] {
            let pat = &bref[start..start + len];
            let mut sm = fm.full_interval();
            for &c in pat.iter().rev() {
                sm = fm.backward_ext(sm, c as usize);
            }
            let naive = bref.windows(len).filter(|w| *w == pat).count() as i64;
            assert_eq!(sm.s, naive, "occurrence mismatch for pattern at {start}");
        }
    }
}
