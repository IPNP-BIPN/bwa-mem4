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
//! keys. klib derives the branching factor in `kb_init` (`kbtree.h:64`):
//!
//! ```text
//! b->t = ((size - 4 - sizeof(void*)) / (sizeof(void*) + sizeof(key_t)) + 1) >> 1
//! b->n = 2 * b->t - 1
//! ```
//!
//! with `size` the byte budget for one node. `mem_chain` passes `KB_DEFAULT_SIZE + 8` = **520**, not
//! 512 (`bwamem.cpp:845`, commented "+8, due to addition of counters in chain"), and
//! `sizeof(mem_chain_t)` is **48** bytes on LP64, not 40: `seqid, cseed, n, m, first, rid` (6 x
//! int32 = 24), the `w:29/kept:2/is_alt:1` bitfield word (4), `frac_rep` (4), `pos` (8), `seeds`
//! pointer (8). So `t = ((520 - 4 - 8) / (8 + 48) + 1) >> 1 = (9 + 1) >> 1 = 5` and `2t - 1 = 9`.
//! (An earlier version of this comment said 512 and 40 bytes; those two errors happened to cancel
//! and still give 5. The numbers above are measured, not assumed.)
//!
//! Reads with more than 9 chains are routine, so the multi-node behaviour (and hence the splits) is
//! load-bearing, not a corner case.
//!
//! # What a B-tree is, in the three lines needed to read this file
//!
//! A node holds a sorted run of keys plus one more child pointer than it has keys: child `i` holds
//! everything between key `i-1` and key `i`. A search walks down, at each node binary-searching the
//! keys to pick a child. An insert that would overflow a node splits it, pushing the middle key up
//! into the parent. klib splits *preemptively* on the way down, which is why no split here ever has
//! to propagate back upward, and why the tree's final shape depends on the insertion ORDER and not
//! only on the set of keys inserted.
//!
//! # Glossary: names kept identical to klib
//!
//! | name | klib origin | meaning |
//! |---|---|---|
//! | `t` | `b->t` | The B-tree's branching parameter. A node holds between `t-1` and `2t-1` keys. |
//! | `x` | `x` | The node currently being visited, as an index into `nodes`. |
//! | `y`, `z` | `y`, `z` | In a split: `y` is the full node being split, `z` the fresh right-hand sibling that takes its upper half. |
//! | `i` | `i` | A key slot within a node; the child to descend into is always `ptrs[i + 1]`. |
//! | `r` | `*rr` | The comparison result from `getp_aux`: 0 = exact hit, -1 = `k` sits below key `i+1`, 1 = `k` is above every key. |
//! | `k` | `*k` | The key being searched for or inserted, i.e. a chain's reference start position. |
//!
//! # Reading order for this file
//!
//! 1. [`getp_aux`]: the per-node binary search, and its exact return contract. Everything else
//!    depends on it, including the `-1` and the off-by-one it applies internally.
//! 2. [`KbTree::lower`]: the only lookup `mem_chain` performs.
//! 3. [`KbTree::put`] then [`KbTree::putp_aux`] then [`KbTree::split`]: the insert path.
//! 4. [`KbTree::in_order`] / [`KbTree::walk`]: how chains are handed to `mem_chain_flt`.

/// klib's `b->t` for `mem_chain_t` under `KB_DEFAULT_SIZE`. See the module docs for the derivation
/// (`t = ((520 - 4 - 8) / (8 + 48) + 1) >> 1 = 5`).
///
/// This is NOT a tuning knob. It fixes where [`KbTree::split`] cuts a full node and therefore the
/// whole tree shape; changing it changes which chain [`KbTree::lower`] returns for some seeds, which
/// changes the chains, which changes the SAM. It must equal what the C computes for
/// `sizeof(mem_chain_t)`, or byte-identity is lost.
const T: usize = 5;
/// Maximum keys per node (`2t - 1` = 9); a node this full is split before being descended into.
/// Derived from [`T`], so the same warning applies: 9 is what the C node layout yields, not a choice.
const MAX_KEYS: usize = 2 * T - 1;

/// One B-tree node. Kept in an arena (`KbTree::nodes`) and referred to by index, so "node id" below
/// always means a position in that vector, never a slot within a node.
struct Node {
    /// False for a leaf, in which case `ptrs` is empty and the node is a terminus of any descent.
    is_internal: bool,
    /// `(pos, chain index)`. `pos` is a chain's first seed's REFERENCE start in 2L space (bases,
    /// `>= l_pac` means reverse strand); the `usize` is an opaque payload, an index into the caller's
    /// `chains` vector, and is never compared. Ordering compares `pos` only, mirroring `chain_cmp`,
    /// so equal-`pos` entries are duplicates rather than replacements.
    ///
    /// Invariants relied on everywhere below: `pos` is non-decreasing across the vector (duplicates
    /// allowed), and `keys.len() <= MAX_KEYS`. A node is split the moment it reaches `MAX_KEYS`, so
    /// no node is ever observed over-full. The usual B-tree lower bound of `T - 1` keys holds for
    /// every node except the root, which may hold as few as 1 (or 0, when the tree is a fresh empty
    /// leaf); since nothing is ever deleted, no other underflow is reachable.
    keys: Vec<(i64, usize)>,
    /// Child NODE IDS (indices into `KbTree::nodes`), not key slots. Exactly `keys.len() + 1` entries
    /// when `is_internal`, empty otherwise. `ptrs[i]` roots the subtree holding everything ordered
    /// before `keys[i]`, and `ptrs[keys.len()]` everything after the last key.
    ptrs: Vec<usize>,
}

/// Position-keyed chain tree with klib's exact insert/search behaviour.
pub(crate) struct KbTree {
    /// Arena of every node ever allocated, including the old root after a root split. Indices into it
    /// are stable for the tree's lifetime (nodes are only pushed, never removed or reordered), which
    /// is what makes storing child links as plain `usize` sound.
    nodes: Vec<Node>,
    /// Node id of the current root. Changes only in [`KbTree::put`], when a full root forces a new
    /// level; the old root then becomes `nodes[root].ptrs[0]`.
    root: usize,
}

/// `__kb_getp_aux` (`kbtree.h:125-138`): binary search for `k` within one node, returning `(i, r)`.
///
/// `i` is the index the caller descends or inserts around and `r` the comparison against it: `r == 0`
/// means an exact hit, and because the search is a lower bound, that hit is the *first* of any run of
/// equal keys. `r != 0` leaves `i` on the last key below `k` (possibly `-1`).
///
/// The `-1` return is why `i` is `isize`: it means "before every key in this node", and callers use
/// `ptrs[i + 1]` (so `ptrs[0]`) to descend, which is exactly why the child index is always `i + 1`.
///
/// Return contract, restated because everything below depends on it:
/// - `(-1, 0)` for an empty node.
/// - `(len - 1, 1)` when `k` is greater than every key (`r = 1`, `i` on the last key).
/// - `(idx, 0)` on an exact hit, `idx` = the FIRST key equal to `k`.
/// - `(idx - 1, -1)` on a miss inside the range: the decrement happens HERE, in `getp_aux`, matching
///   the C's `if ((*rr = __cmp(...)) < 0) --begin;` at `kbtree.h:136`. That is the reason
///   `kb_intervalp` can then write `*lower = &key[i]` rather than `&key[i - 1]`
///   (`kbtree.h:170`): the off-by-one has already been applied.
///
/// # Parameters
/// - `node`: the single node to search. Its `keys` must be non-decreasing in `pos` (the B-tree
///   invariant); nothing below the node is consulted, so this says nothing about subtrees.
/// - `k`: the reference position being looked for, in 2L space, same units as `Node::keys.0`.
///
/// # Returns
/// `(i, r)` where `i` is a KEY SLOT within this node in `-1 ..= keys.len() - 1` (never a node id and
/// never a child index; the child to descend into is `ptrs[i + 1]`), and `r` is -1, 0 or 1 per the
/// contract above.
fn getp_aux(node: &Node, k: i64) -> (isize, i32) {
    if node.keys.is_empty() {
        return (-1, 0);
    }
    // Lower-bound binary search over this node's key slots: on exit `begin` is the first slot whose
    // `pos >= k`, or `keys.len()` if every key is below `k`. `end` is exclusive throughout.
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
    // The lower bound guarantees `k <= keys[begin]`, so `r` is 0 or negative. `r == 0` is an exact
    // key match at slot `begin`; `r == -1` means `k` falls strictly between slots `begin - 1` and
    // `begin`, and the caller wants the slot BELOW, hence the decrement.
    let r = if k < node.keys[begin].0 { -1 } else { 0 };
    if r < 0 {
        return (begin as isize - 1, r);
    }
    (begin as isize, r)
}

impl KbTree {
    /// An empty tree: one leaf node with no keys, which is also the root.
    ///
    /// # Returns
    /// A tree for which `lower` always yields `None` and `in_order` an empty vector.
    pub(crate) fn new() -> Self {
        KbTree {
            nodes: vec![Node { is_internal: false, keys: Vec::new(), ptrs: Vec::new() }],
            root: 0,
        }
    }

    /// `kb_intervalp`'s `lower` (`kbtree.h:159-176`): the first chain at `k` if one exists, else the
    /// last one below it. `None` when the tree is empty or every key is above `k`.
    ///
    /// This is the single lookup `mem_chain` uses to decide which chain a seed is offered to, so its
    /// answer is directly observable in the SAM output. `upper` is computed by the C but never read
    /// at the call site (`bwamem.cpp:924` uses only `lower`), so it is not ported.
    ///
    /// The descent keeps improving `lower` on the way down: an internal node's key at `i` is a
    /// better (larger) lower bound than anything found higher up, and the search then continues into
    /// the subtree at `ptrs[i + 1]`, which may hold a better one still. An exact hit short-circuits
    /// immediately, which is what makes the answer the FIRST duplicate rather than the last.
    ///
    /// # Parameters
    /// - `k`: the seed's REFERENCE start in 2L space (bases). Any `i64`; values below every key and
    ///   above every key are both handled.
    ///
    /// # Returns
    /// The stored payload (the caller's index into its `chains` vector), not a key, not a node id.
    /// `None` when no chain starts at or below `k`.
    pub(crate) fn lower(&self, k: i64) -> Option<usize> {
        // Node id currently being visited; the descent moves it strictly downward, so it terminates.
        let mut x = self.root;
        // Best candidate found so far: the payload of the largest key `<= k` seen on the path. Loop
        // invariant at the top of each iteration: no key outside the subtree rooted at `x` is both
        // `<= k` and greater than `lower`'s key, so anything better can only lie below `x`.
        let mut lower = None;
        loop {
            let node = &self.nodes[x];
            // `i` is a key SLOT inside `node`; see `getp_aux`'s return contract.
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
            // Descend. `i + 1` converts a key slot to the child slot on its right, and is in
            // `0 ..= keys.len()`, exactly the valid range of `ptrs` for an internal node.
            x = node.ptrs[(i + 1) as usize];
        }
    }

    /// `__kb_split` (`kbtree.h:183-198`): `y = x.ptrs[i]` is full (`MAX_KEYS` keys); move its upper
    /// half into a fresh sibling `z` and lift its median key into the parent `x` at slot `i`.
    ///
    /// The split is at `T - 1`: keys `[0, T-1)` stay in `y`, key `T-1` is the median that moves up,
    /// keys `[T, MAX_KEYS)` (that is `T-1` of them) go to `z`. An internal node also hands over
    /// `ptrs[T ..= MAX_KEYS]`, one more pointer than keys, which is the B-tree invariant
    /// `ptrs.len() == keys.len() + 1`.
    ///
    /// **Precondition: `x` must not itself be full**, which the callers guarantee by splitting
    /// top-down (see [`Self::put`]). That is why klib never needs to propagate a split upwards.
    ///
    /// Worked example with `T = 5`, a full leaf holding key slots 0..9: slots 0..4 stay in `y`, slot 4
    /// (`T - 1`) is the median lifted into the parent, slots 5..9 become `z`'s slots 0..4. Both
    /// children end with 4 keys, which is the `T - 1` minimum, so neither underflows.
    ///
    /// # Parameters
    /// - `x`: node id of the PARENT, which gains one key and one child pointer. Must not be full
    ///   (precondition above), and must be internal.
    /// - `i`: the child SLOT of `y` within `x.ptrs`, so `x.ptrs[i] == y`. The lifted median lands at
    ///   key slot `i` of `x` and the new sibling at child slot `i + 1`. Range `0 ..= x.keys.len()`.
    /// - `y`: node id of the full child being split. Must hold exactly `MAX_KEYS` keys.
    fn split(&mut self, x: usize, i: usize, y: usize) {
        let (z_internal, z_keys, z_ptrs, median) = {
            let full_node = &mut self.nodes[y];
            // Upper half of the keys, slots `T ..< MAX_KEYS` (that is `T - 1` of them). Slot `T - 1`
            // is deliberately excluded: it is the median that goes up, not into `z`.
            let z_keys: Vec<(i64, usize)> = full_node.keys[T..MAX_KEYS].to_vec();
            // The matching child node ids, `T ..= MAX_KEYS` inclusive, one more than `z_keys` so `z`
            // satisfies `ptrs.len() == keys.len() + 1`. A leaf hands over nothing.
            let z_ptrs: Vec<usize> = if full_node.is_internal {
                full_node.ptrs[T..=MAX_KEYS].to_vec()
            } else {
                Vec::new()
            };
            // Key slot `T - 1` of `y`: the separator that moves up into the parent.
            let median = full_node.keys[T - 1];
            full_node.keys.truncate(T - 1);
            if full_node.is_internal {
                full_node.ptrs.truncate(T);
            }
            (full_node.is_internal, z_keys, z_ptrs, median)
        };
        // Node id of the fresh right-hand sibling, allocated at the end of the arena.
        let z = self.nodes.len();
        self.nodes.push(Node { is_internal: z_internal, keys: z_keys, ptrs: z_ptrs });
        let parent = &mut self.nodes[x];
        parent.ptrs.insert(i + 1, z);
        parent.keys.insert(i, median);
    }

    /// `__kb_putp_aux` (`kbtree.h:200-217`), with klib's preemptive top-down splitting.
    ///
    /// **Duplicates are inserted, never replaced.** `chain_cmp` (`bwamem.cpp:40`) compares `pos`
    /// only, so two chains starting at the same reference position are equal keys as far as the tree
    /// is concerned, and both are stored. Reproducing the resulting array order is the whole reason
    /// this file exists.
    ///
    /// # Parameters
    /// - `x`: node id to insert into or descend from. Precondition, upheld by [`Self::put`] for the
    ///   root and by the preemptive split below for every other node: `x` is NOT full, so the leaf
    ///   this recursion terminates at always has room.
    /// - `k`: the `(pos, chain index)` pair to store. `pos` is a 2L-space REFERENCE position; the
    ///   index is opaque payload.
    fn putp_aux(&mut self, x: usize, k: (i64, usize)) {
        if !self.nodes[x].is_internal {
            // Key slot in `x` after which the new key belongs (`-1` = before all of them).
            let (i, _) = getp_aux(&self.nodes[x], k.0);
            // Insert *after* `i` (klib: `__KB_KEY(x)[i + 1] = *k` after a memmove). On an exact hit
            // `i` is the FIRST duplicate, so a newcomer lands directly behind it and *in front of*
            // every duplicate already stacked there. Inserting A, B, C at one pos therefore leaves
            // the array as [A, C, B], not [A, B, C]. In-order traversal reports that order, and
            // `mem_chain_flt`'s unstable equal-weight tie-break can see it.
            self.nodes[x].keys.insert((i + 1) as usize, k);
            return;
        }
        // Internal node: `i + 1` is the child to descend into (see `getp_aux`'s contract).
        // `i` is a CHILD slot in `x.ptrs` (0 ..= keys.len()), converted from the key slot `getp_aux`
        // returns. It is `mut` because a split below can move the target one slot right.
        let mut i = (getp_aux(&self.nodes[x], k.0).0 + 1) as usize;
        // Node id of the subtree the key would go into if no split happened.
        let child = self.nodes[x].ptrs[i];
        // Preemptive split: a full child is split on the way DOWN, before we know whether the insert
        // will actually land in it. This is what keeps the parent guaranteed non-full in `split`,
        // and it means the tree's shape depends on the insertion sequence, not just the final key
        // multiset. After splitting, the median has been lifted into `x` at slot `i`; if the new key
        // sorts above it, the insert belongs in the right half instead.
        if self.nodes[child].keys.len() == MAX_KEYS {
            self.split(x, i, child);
            // Strict `>`: a key EQUAL to the lifted median goes left, which is again what puts a new
            // duplicate ahead of existing ones.
            if k.0 > self.nodes[x].keys[i].0 {
                i += 1;
            }
        }
        // Node id to recurse into, re-read because `split` inserted into `x.ptrs` and may have shifted
        // things; guaranteed non-full, since it was either not full before or is a fresh split half.
        let next = self.nodes[x].ptrs[i];
        self.putp_aux(next, k);
    }

    /// `kb_putp` (`kbtree.h:218-232`): if the root is full, push a fresh empty root above it and
    /// split the old one into it (the only way the tree gains a level), then descend.
    ///
    /// `pos` is the chain's reference start; `idx` its index in the caller's `chains` vector, which
    /// the tree carries around as an opaque payload and never compares.
    ///
    /// # Parameters
    /// - `pos`: the chain's first seed's REFERENCE start in 2L space (bases, `>= l_pac` = reverse
    ///   strand). Duplicates are expected and preserved.
    /// - `idx`: the chain's index in the caller's `chains` vector. The caller must push the chain and
    ///   call this exactly once per chain, or `in_order` stops being a permutation of `0..n`.
    pub(crate) fn put(&mut self, pos: i64, idx: usize) {
        if self.nodes[self.root].keys.len() == MAX_KEYS {
            // Node ids: the full root that is about to become a child, and its replacement.
            let old_root = self.root;
            let new_root = self.nodes.len();
            self.nodes.push(Node {
                is_internal: true,
                keys: Vec::new(),
                ptrs: vec![old_root],
            });
            self.root = new_root;
            self.split(new_root, 0, old_root);
        }
        // Re-read after the possible root swap above; non-full either way, which is `putp_aux`'s
        // precondition.
        let root = self.root;
        self.putp_aux(root, (pos, idx));
    }

    /// In-order traversal, the order `mem_chain` emits chains in (and therefore the order
    /// `mem_chain_flt` sees, which drives its unstable equal-weight tie-break).
    ///
    /// # Returns
    /// Every stored payload (chain index) exactly once, ordered by `pos` ascending, and within one
    /// `pos` by klib's array order (see [`Self::putp_aux`]: A, B, C inserted at one `pos` read back as
    /// A, C, B). Length equals the number of `put` calls.
    pub(crate) fn in_order(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.walk(self.root, &mut out);
        out
    }

    /// Standard B-tree in-order walk: subtree `ptrs[i]`, then key `i`, ..., then the final subtree
    /// `ptrs[keys.len()]`. Yields payload indices, not positions.
    ///
    /// # Parameters
    /// - `x`: node id of the subtree root to emit. Recursion depth is the tree height, `O(log n)`.
    /// - `out`: accumulator, appended to in order; the caller supplies it empty.
    fn walk(&self, x: usize, out: &mut Vec<usize>) {
        let node = &self.nodes[x];
        if !node.is_internal {
            out.extend(node.keys.iter().map(|&(_, i)| i));
            return;
        }
        // `i` is a key slot, `ci` its payload; `ptrs[i]` is the child node id ordered before it.
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
