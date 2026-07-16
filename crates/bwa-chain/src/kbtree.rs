//! A faithful port of klib's kbtree (`reference/bwa-mem2/src/kbtree.h`), specialized to the
//! position-keyed chain index bwa's `mem_chain` builds (`KBTREE_INIT(chn, mem_chain_t, chain_cmp)`).
//!
//! This exists only because the tree's *shape* is observable. `chain_cmp` compares `pos` alone, so
//! the tree holds duplicate keys, and `mem_chain` offers each seed to exactly one chain: the one
//! `kb_intervalp` returns as `lower`. Which chain that is depends on node layout and split history,
//! so an ordered map cannot stand in for it -- picking a different chain makes bwa merge seeds we
//! keep separate, or lose chains outright.
//!
//! Only `putp` and `intervalp` are ported; `mem_chain` never deletes. Nodes hold up to `2t-1 = 9`
//! keys: klib derives `t` from `KB_DEFAULT_SIZE` (512) and `sizeof(mem_chain_t)` (40 bytes), giving
//! `t = ((512 - 4 - 8) / (8 + 40) + 1) >> 1 = 5`. Reads with more than 9 chains are routine, so the
//! multi-node behaviour (and hence the splits) is load-bearing, not a corner case.

/// klib's `b->t` for `mem_chain_t` under `KB_DEFAULT_SIZE`. See the module docs.
const T: usize = 5;
/// Maximum keys per node (`2t - 1`); a node this full is split before being descended into.
const MAX_KEYS: usize = 2 * T - 1;

struct Node {
    is_internal: bool,
    /// `(pos, chain index)`. Ordering compares `pos` only, mirroring `chain_cmp`, so equal-`pos`
    /// entries are duplicates rather than replacements.
    keys: Vec<(i64, usize)>,
    /// Child node indices; `keys.len() + 1` entries when internal.
    ptrs: Vec<usize>,
}

/// Position-keyed chain tree with klib's exact insert/search behaviour.
pub(crate) struct KbTree {
    nodes: Vec<Node>,
    root: usize,
}

/// `__kb_getp_aux`: binary search for `k`, returning `(i, r)`.
///
/// `i` is the index the caller descends or inserts around and `r` the comparison against it: `r == 0`
/// means an exact hit, and because the search is a lower bound, that hit is the *first* of any run of
/// equal keys. `r != 0` leaves `i` on the last key below `k` (possibly `-1`).
fn getp_aux(node: &Node, k: i64) -> (isize, i32) {
    if node.keys.is_empty() {
        return (-1, 0);
    }
    let (mut begin, mut end) = (0usize, node.keys.len());
    while begin < end {
        let mid = (begin + end) >> 1;
        if node.keys[mid].0 < k {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    if begin == node.keys.len() {
        return ((node.keys.len() - 1) as isize, 1);
    }
    // The lower bound guarantees `k <= keys[begin]`, so `r` is 0 or negative.
    let r = if k < node.keys[begin].0 { -1 } else { 0 };
    if r < 0 {
        return (begin as isize - 1, r);
    }
    (begin as isize, r)
}

impl KbTree {
    pub(crate) fn new() -> Self {
        KbTree {
            nodes: vec![Node { is_internal: false, keys: Vec::new(), ptrs: Vec::new() }],
            root: 0,
        }
    }

    /// `kb_intervalp`'s `lower`: the first chain at `k` if one exists, else the last one below it.
    /// `None` when the tree is empty or every key is above `k`.
    pub(crate) fn lower(&self, k: i64) -> Option<usize> {
        let mut x = self.root;
        let mut lower = None;
        loop {
            let node = &self.nodes[x];
            let (i, r) = getp_aux(node, k);
            if i >= 0 && r == 0 {
                return Some(node.keys[i as usize].1);
            }
            if i >= 0 {
                lower = Some(node.keys[i as usize].1);
            }
            if !node.is_internal {
                return lower;
            }
            x = node.ptrs[(i + 1) as usize];
        }
    }

    /// `__kb_split`: `y = x.ptrs[i]` is full; move its upper half into a fresh sibling and lift its
    /// median key into `x` at `i`.
    fn split(&mut self, x: usize, i: usize, y: usize) {
        let (z_internal, z_keys, z_ptrs, median) = {
            let yn = &mut self.nodes[y];
            let z_keys: Vec<(i64, usize)> = yn.keys[T..MAX_KEYS].to_vec();
            let z_ptrs: Vec<usize> =
                if yn.is_internal { yn.ptrs[T..=MAX_KEYS].to_vec() } else { Vec::new() };
            let median = yn.keys[T - 1];
            yn.keys.truncate(T - 1);
            if yn.is_internal {
                yn.ptrs.truncate(T);
            }
            (yn.is_internal, z_keys, z_ptrs, median)
        };
        let z = self.nodes.len();
        self.nodes.push(Node { is_internal: z_internal, keys: z_keys, ptrs: z_ptrs });
        let xn = &mut self.nodes[x];
        xn.ptrs.insert(i + 1, z);
        xn.keys.insert(i, median);
    }

    /// `__kb_putp_aux`, with klib's preemptive top-down splitting.
    fn putp_aux(&mut self, x: usize, k: (i64, usize)) {
        if !self.nodes[x].is_internal {
            let (i, _) = getp_aux(&self.nodes[x], k.0);
            // Insert *after* `i`: on an exact hit `i` is the first duplicate, so a newcomer lands
            // directly behind it, and inserting A, B, C at one pos leaves [A, C, B].
            self.nodes[x].keys.insert((i + 1) as usize, k);
            return;
        }
        let mut i = (getp_aux(&self.nodes[x], k.0).0 + 1) as usize;
        let child = self.nodes[x].ptrs[i];
        if self.nodes[child].keys.len() == MAX_KEYS {
            self.split(x, i, child);
            if k.0 > self.nodes[x].keys[i].0 {
                i += 1;
            }
        }
        let next = self.nodes[x].ptrs[i];
        self.putp_aux(next, k);
    }

    /// `kb_putp`: split a full root first (growing the tree by one level), then descend.
    pub(crate) fn put(&mut self, pos: i64, idx: usize) {
        if self.nodes[self.root].keys.len() == MAX_KEYS {
            let old = self.root;
            let s = self.nodes.len();
            self.nodes.push(Node { is_internal: true, keys: Vec::new(), ptrs: vec![old] });
            self.root = s;
            self.split(s, 0, old);
        }
        let r = self.root;
        self.putp_aux(r, (pos, idx));
    }

    /// In-order traversal, the order `mem_chain` emits chains in (and therefore the order
    /// `mem_chain_flt` sees, which drives its unstable equal-weight tie-break).
    pub(crate) fn in_order(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.walk(self.root, &mut out);
        out
    }

    fn walk(&self, x: usize, out: &mut Vec<usize>) {
        let node = &self.nodes[x];
        if !node.is_internal {
            out.extend(node.keys.iter().map(|&(_, i)| i));
            return;
        }
        for (i, &(_, ci)) in node.keys.iter().enumerate() {
            self.walk(node.ptrs[i], out);
            out.push(ci);
        }
        self.walk(node.ptrs[node.keys.len()], out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Duplicates must survive and come back in klib's array order: the first insert stays first and
    /// later ones stack up behind it, so A, B, C at one pos read back as [A, C, B].
    #[test]
    fn duplicate_pos_keeps_klib_order() {
        let mut t = KbTree::new();
        t.put(100, 0);
        t.put(100, 1);
        t.put(100, 2);
        assert_eq!(t.in_order(), vec![0, 2, 1]);
        // `lower` on an exact hit is the first duplicate.
        assert_eq!(t.lower(100), Some(0));
    }

    /// Below an exact hit, `lower` is the last entry under the key; above every key, the last key.
    #[test]
    fn lower_picks_last_entry_below() {
        let mut t = KbTree::new();
        t.put(10, 0);
        t.put(20, 1);
        t.put(20, 2);
        assert_eq!(t.lower(5), None);
        assert_eq!(t.lower(10), Some(0));
        assert_eq!(t.lower(15), Some(0));
        assert_eq!(t.lower(20), Some(1)); // first duplicate at 20
        assert_eq!(t.lower(25), Some(2)); // last array entry below 25
    }

    /// Past one node (9 keys) the tree splits; in-order traversal must still be sorted by pos, and
    /// `lower` must still find the right neighbour across nodes.
    #[test]
    fn splits_preserve_order_and_lookup() {
        let mut t = KbTree::new();
        for i in 0..50i64 {
            t.put(i * 10, i as usize);
        }
        let order = t.in_order();
        assert_eq!(order.len(), 50);
        assert_eq!(order, (0..50).collect::<Vec<usize>>());
        assert_eq!(t.lower(0), Some(0));
        assert_eq!(t.lower(495), Some(49));
        assert_eq!(t.lower(255), Some(25));
        assert_eq!(t.lower(-1), None);
    }
}
