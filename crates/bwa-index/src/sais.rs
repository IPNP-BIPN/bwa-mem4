//! Suffix array construction by induced sorting (SA-IS), pure Rust.
//!
//! [`suffix_array_with_sentinel`] returns the suffix array of a byte string with an implicit `$`
//! sentinel that sorts before every real symbol. This matches bwa-mem2's convention exactly: the
//! returned vector has length `n+1`, element `0` is `n` (the sentinel suffix), and `[1..]` is the
//! suffix array of the real string (equivalent to `saisxx(s, sa+1, n)` followed by `sa[0]=n`).

const EMPTY: usize = usize::MAX;

/// Suffix array of `s` including the sentinel row first.
///
/// Result length is `s.len()+1`; `result[0] == s.len()` and `result[1..]` is the suffix array of
/// `s` (shorter suffixes sort first on ties, i.e. the standard suffix array).
pub fn suffix_array_with_sentinel(s: &[u8]) -> Vec<i64> {
    let n = s.len();
    // Map bytes to 1..=256 and append the 0 sentinel, so 0 is uniquely smallest.
    let mut t: Vec<usize> = Vec::with_capacity(n + 1);
    for &b in s {
        t.push(b as usize + 1);
    }
    t.push(0);
    sa_is(&t, 257).into_iter().map(|x| x as i64).collect()
}

#[inline]
fn is_lms(is_s: &[bool], i: usize) -> bool {
    i > 0 && is_s[i] && !is_s[i - 1]
}

/// Bucket start offsets (`out[c]` = first index of symbol `c`'s bucket).
fn bucket_starts(s: &[usize], k: usize) -> Vec<usize> {
    let mut c = vec![0usize; k];
    for &x in s {
        c[x] += 1;
    }
    let mut sum = 0;
    for slot in c.iter_mut() {
        let cnt = *slot;
        *slot = sum;
        sum += cnt;
    }
    c
}

/// Bucket end offsets (`out[c]` = one past the last index of symbol `c`'s bucket).
fn bucket_ends(s: &[usize], k: usize) -> Vec<usize> {
    let mut c = vec![0usize; k];
    for &x in s {
        c[x] += 1;
    }
    let mut sum = 0;
    for slot in c.iter_mut() {
        sum += *slot;
        *slot = sum;
    }
    c
}

/// Induce L-type then S-type suffixes from the LMS suffixes already placed in `sa`.
fn induce(sa: &mut [usize], s: &[usize], is_s: &[bool], k: usize) {
    let n = s.len();
    let mut starts = bucket_starts(s, k);
    for i in 0..n {
        let j = sa[i];
        if j != EMPTY && j != 0 && !is_s[j - 1] {
            let c = s[j - 1];
            sa[starts[c]] = j - 1;
            starts[c] += 1;
        }
    }
    let mut ends = bucket_ends(s, k);
    for i in (0..n).rev() {
        let j = sa[i];
        if j != EMPTY && j != 0 && is_s[j - 1] {
            let c = s[j - 1];
            ends[c] -= 1;
            sa[ends[c]] = j - 1;
        }
    }
}

fn lms_substr_equal(s: &[usize], is_s: &[bool], a: usize, b: usize) -> bool {
    if a == b {
        return true;
    }
    let n = s.len();
    let mut i = 0usize;
    loop {
        let pa = a + i;
        let pb = b + i;
        if pa >= n || pb >= n {
            return pa >= n && pb >= n;
        }
        if s[pa] != s[pb] || is_s[pa] != is_s[pb] {
            return false;
        }
        if i > 0 {
            let al = is_lms(is_s, pa);
            let bl = is_lms(is_s, pb);
            if al || bl {
                return al && bl;
            }
        }
        i += 1;
    }
}

fn sa_is(s: &[usize], k: usize) -> Vec<usize> {
    let n = s.len();
    let mut sa = vec![EMPTY; n];
    if n == 1 {
        sa[0] = 0;
        return sa;
    }

    // Classify S-type (true) / L-type positions.
    let mut is_s = vec![false; n];
    is_s[n - 1] = true;
    for i in (0..n - 1).rev() {
        is_s[i] = s[i] < s[i + 1] || (s[i] == s[i + 1] && is_s[i + 1]);
    }

    // Step 1: place LMS suffixes at bucket ends, then induce.
    let mut ends = bucket_ends(s, k);
    for i in (0..n).rev() {
        if is_lms(&is_s, i) {
            let c = s[i];
            ends[c] -= 1;
            sa[ends[c]] = i;
        }
    }
    induce(&mut sa, s, &is_s, k);

    // Step 2: name LMS substrings in sorted order.
    let mut lms_sorted = Vec::new();
    for &p in sa.iter() {
        if p != EMPTY && is_lms(&is_s, p) {
            lms_sorted.push(p);
        }
    }
    let mut names = vec![EMPTY; n];
    let mut cur = 0usize;
    names[lms_sorted[0]] = 0;
    let mut prev = lms_sorted[0];
    for &p in lms_sorted.iter().skip(1) {
        if !lms_substr_equal(s, &is_s, prev, p) {
            cur += 1;
        }
        names[p] = cur;
        prev = p;
    }
    let num_names = cur + 1;

    // Reduced string in position order.
    let mut lms_positions = Vec::new();
    let mut reduced = Vec::new();
    for (i, &name) in names.iter().enumerate() {
        if is_lms(&is_s, i) {
            lms_positions.push(i);
            reduced.push(name);
        }
    }

    // Step 3: suffix array of the reduced string.
    let reduced_sa = if num_names == reduced.len() {
        let mut rsa = vec![0usize; reduced.len()];
        for (i, &name) in reduced.iter().enumerate() {
            rsa[name] = i;
        }
        rsa
    } else {
        sa_is(&reduced, num_names)
    };

    // Step 4: re-place LMS suffixes in sorted order, then induce the final SA.
    for x in sa.iter_mut() {
        *x = EMPTY;
    }
    let mut ends = bucket_ends(s, k);
    for &idx in reduced_sa.iter().rev() {
        let p = lms_positions[idx];
        let c = s[p];
        ends[c] -= 1;
        sa[ends[c]] = p;
    }
    induce(&mut sa, s, &is_s, k);

    sa
}

/// Memory-efficient in-place SA-IS: same result as [`suffix_array_with_sentinel`] but built with
/// ~8-9 bytes/base peak (the SA array plus a type bitvector and per-level bucket arrays), instead
/// of the ~50 bytes/base of the array-heavy version. This is what lets the full genome index fit in
/// RAM. The SA is unique, so the output is byte-identical to any correct construction.
pub fn suffix_array_inplace(s: &[u8]) -> Vec<i64> {
    // Map bytes to 1..=256 with a 0 sentinel appended (uniquely smallest), then run SA-IS on the
    // full length-(n+1) string; the sentinel suffix sorts first, so sa[0] == n automatically.
    let big_n = s.len() + 1;
    let mut t: Vec<i64> = Vec::with_capacity(big_n);
    for &b in s {
        t.push(i64::from(b) + 1);
    }
    t.push(0);
    let mut sa = vec![0i64; big_n];
    sais_rec(&t, &mut sa, 257);
    sa
}

/// EMPTY marker for the in-place SA array (positions are always >= 0).
const IEMPTY: i64 = -1;

/// S-type classification as a bitvector (one bit per position).
struct TypeBits(Vec<u64>);
impl TypeBits {
    fn new(n: usize) -> Self {
        TypeBits(vec![0u64; n.div_ceil(64)])
    }
    #[inline]
    fn set_s(&mut self, i: usize) {
        self.0[i >> 6] |= 1u64 << (i & 63);
    }
    #[inline]
    fn is_s(&self, i: usize) -> bool {
        (self.0[i >> 6] >> (i & 63)) & 1 != 0
    }
    /// LMS = S-type preceded by an L-type.
    #[inline]
    fn is_lms(&self, i: usize) -> bool {
        i > 0 && self.is_s(i) && !self.is_s(i - 1)
    }
}

/// Bucket end offsets for an integer string (`out[c]` = one past symbol `c`'s bucket).
fn bucket_ends_i(s: &[i64], k: usize) -> Vec<i64> {
    let mut c = vec![0i64; k];
    for &x in s {
        c[x as usize] += 1;
    }
    let mut sum = 0;
    for slot in c.iter_mut() {
        sum += *slot;
        *slot = sum;
    }
    c
}

/// Bucket start offsets for an integer string.
fn bucket_starts_i(s: &[i64], k: usize) -> Vec<i64> {
    let mut c = vec![0i64; k];
    for &x in s {
        c[x as usize] += 1;
    }
    let mut sum = 0;
    for slot in c.iter_mut() {
        let cnt = *slot;
        *slot = sum;
        sum += cnt;
    }
    c
}

/// Induce L-type then S-type suffixes from the LMS suffixes already placed in `sa`.
fn induce_i(sa: &mut [i64], s: &[i64], ty: &TypeBits, k: usize) {
    let n = s.len();
    let mut starts = bucket_starts_i(s, k);
    for i in 0..n {
        let j = sa[i];
        if j > 0 && !ty.is_s((j - 1) as usize) {
            let c = s[(j - 1) as usize] as usize;
            sa[starts[c] as usize] = j - 1;
            starts[c] += 1;
        }
    }
    let mut ends = bucket_ends_i(s, k);
    for i in (0..n).rev() {
        let j = sa[i];
        if j > 0 && ty.is_s((j - 1) as usize) {
            let c = s[(j - 1) as usize] as usize;
            ends[c] -= 1;
            sa[ends[c] as usize] = j - 1;
        }
    }
}

/// Whether the LMS substrings starting at `a` and `b` are equal (same symbols and S/L types up to
/// and including the next LMS boundary).
fn lms_eq(s: &[i64], ty: &TypeBits, a: usize, b: usize) -> bool {
    if a == b {
        return true;
    }
    let n = s.len();
    let mut i = 0usize;
    loop {
        let (pa, pb) = (a + i, b + i);
        if pa >= n || pb >= n {
            return pa >= n && pb >= n;
        }
        if s[pa] != s[pb] || ty.is_s(pa) != ty.is_s(pb) {
            return false;
        }
        if i > 0 {
            let (al, bl) = (ty.is_lms(pa), ty.is_lms(pb));
            if al || bl {
                return al && bl;
            }
        }
        i += 1;
    }
}

/// In-place SA-IS on integer string `s` (values in `0..k`, `s[n-1]` the unique smallest sentinel),
/// writing the suffix array of `s` into `sa` (length `n`). The reduced sub-problem is packed into
/// `sa` itself, so no per-level O(n) array is allocated (only a type bitvector and one stage-3 temp).
fn sais_rec(s: &[i64], sa: &mut [i64], k: usize) {
    let n = s.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        sa[0] = 0;
        return;
    }

    // Classify S/L types.
    let mut ty = TypeBits::new(n);
    ty.set_s(n - 1);
    for i in (0..n - 1).rev() {
        if s[i] < s[i + 1] || (s[i] == s[i + 1] && ty.is_s(i + 1)) {
            ty.set_s(i);
        }
    }

    // Stage 1: bucket LMS suffixes at their bucket ends, then induce to sort LMS substrings.
    sa.fill(IEMPTY);
    let mut ends = bucket_ends_i(s, k);
    for i in (1..n).rev() {
        if ty.is_lms(i) {
            let c = s[i] as usize;
            ends[c] -= 1;
            sa[ends[c] as usize] = i as i64;
        }
    }
    induce_i(sa, s, &ty, k);

    // Compact the sorted LMS positions into sa[0..m].
    let mut m = 0usize;
    for i in 0..n {
        let p = sa[i];
        if p != IEMPTY && ty.is_lms(p as usize) {
            sa[m] = p;
            m += 1;
        }
    }

    // Name LMS substrings: store name of the LMS at position p into sa[m + p/2] (LMS positions are
    // >= 2 apart, so p/2 are distinct and fit in [m, m + n/2) <= n).
    for x in sa[m..n].iter_mut() {
        *x = IEMPTY;
    }
    let mut name: i64 = 0;
    let mut prev: i64 = -1;
    for idx in 0..m {
        let p = sa[idx] as usize;
        if prev >= 0 && !lms_eq(s, &ty, prev as usize, p) {
            name += 1;
        }
        sa[m + (p >> 1)] = name;
        prev = p as i64;
    }
    let num_names = if m == 0 { 0 } else { (name + 1) as usize };

    // Build the reduced string RA (length m, values 0..num_names) into the tail sa[n-m..n],
    // reading names down from sa[m..m+n/2] (safe: write index >= read index throughout).
    {
        // Name slots span sa[m ..= m + (n-1)/2] (the max LMS position is the sentinel at n-1).
        let mut j = m as i64 - 1;
        let mut i = (m + (n - 1) / 2) as i64;
        while i >= m as i64 {
            if sa[i as usize] != IEMPTY {
                sa[(n - m) + j as usize] = sa[i as usize];
                j -= 1;
            }
            i -= 1;
        }
    }

    // Solve the reduced problem: SA1 into sa[0..m].
    if num_names < m {
        let (head, tail) = sa.split_at_mut(n - m);
        sais_rec(tail, &mut head[..m], num_names);
    } else {
        // All names distinct: SA1 is the inverse permutation of RA.
        for i in 0..m {
            let r = sa[n - m + i] as usize;
            sa[r] = i as i64;
        }
    }

    // Map SA1 (ranks in sa[0..m]) back to text positions, then re-induce the final SA.
    let mut sorted_lms = vec![0i64; m];
    {
        // LMS positions in text order into sa[n-m..n].
        let mut j = 0usize;
        for i in 1..n {
            if ty.is_lms(i) {
                sa[n - m + j] = i as i64;
                j += 1;
            }
        }
        for (i, out) in sorted_lms.iter_mut().enumerate() {
            *out = sa[n - m + sa[i] as usize];
        }
    }
    sa.fill(IEMPTY);
    let mut ends = bucket_ends_i(s, k);
    for &p in sorted_lms.iter().rev() {
        let c = s[p as usize] as usize;
        ends[c] -= 1;
        sa[ends[c] as usize] = p;
    }
    induce_i(sa, s, &ty, k);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive suffix array (with sentinel-first), for validation.
    fn naive(s: &[u8]) -> Vec<i64> {
        let n = s.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| s[a..].cmp(&s[b..]));
        let mut out = vec![n as i64];
        out.extend(idx.into_iter().map(|x| x as i64));
        out
    }

    fn check(s: &[u8]) {
        assert_eq!(
            suffix_array_with_sentinel(s),
            naive(s),
            "mismatch for {s:?}"
        );
    }

    #[test]
    fn known_strings() {
        check(b"");
        check(b"a");
        check(b"aa");
        check(b"banana");
        check(b"mississippi");
        check(b"abracadabra");
        check(&[0, 0, 0, 1, 0, 1, 1]);
        check(&[3, 2, 1, 0, 3, 2, 1, 0]);
    }

    #[test]
    fn random_small_alphabet() {
        // Deterministic pseudo-random strings over {0,1,2,3} (the DNA alphabet), various lengths.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..300 {
            let len = (next() % 200) as usize;
            let s: Vec<u8> = (0..len).map(|_| (next() % 4) as u8).collect();
            check(&s);
        }
    }

    #[test]
    fn random_byte_alphabet() {
        let mut state: u64 = 0xdead_beef_cafe_babe;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..100 {
            let len = (next() % 150) as usize;
            let s: Vec<u8> = (0..len).map(|_| (next() % 256) as u8).collect();
            check(&s);
        }
    }

    /// The in-place SA-IS must produce exactly the same SA as the reference implementation.
    fn check_inplace(s: &[u8]) {
        assert_eq!(
            suffix_array_inplace(s),
            suffix_array_with_sentinel(s),
            "in-place mismatch for {s:?}"
        );
    }

    #[test]
    fn inplace_known_strings() {
        check_inplace(b"");
        check_inplace(b"a");
        check_inplace(b"aa");
        check_inplace(b"banana");
        check_inplace(b"mississippi");
        check_inplace(b"abracadabra");
        check_inplace(&[0, 0, 0, 1, 0, 1, 1]);
        check_inplace(&[3, 2, 1, 0, 3, 2, 1, 0]);
    }

    #[test]
    fn inplace_random_dna_and_bytes() {
        let mut state: u64 = 0x0f1e_2d3c_4b5a_6978;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..400 {
            let len = (next() % 300) as usize;
            let alpha = if next() & 1 == 0 { 4 } else { 256 };
            let s: Vec<u8> = (0..len).map(|_| (next() % alpha) as u8).collect();
            check_inplace(&s);
        }
    }

    #[test]
    fn inplace_large_and_repetitive() {
        let mut state: u64 = 0xa5a5_1234_9999_0f0f;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Larger DNA strings (deep recursion) and highly repetitive inputs (small name counts).
        for &len in &[1000usize, 5000, 20000] {
            let s: Vec<u8> = (0..len).map(|_| (next() % 4) as u8).collect();
            check_inplace(&s);
        }
        check_inplace(&vec![2u8; 4096]); // all identical -> maximal recursion depth
        let periodic: Vec<u8> = (0..8192).map(|i| (i % 3) as u8).collect();
        check_inplace(&periodic);
    }
}
