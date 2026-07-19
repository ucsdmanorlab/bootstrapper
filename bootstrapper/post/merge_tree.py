import numpy as np
from numba import njit


@njit(cache=True)
def _find_merges(us, vs, level, nxt, score, out):
    # For each (u, v) pair (dense indices, -1 = unknown), walk up the merge
    # tree from the lower-level node until the two meet; out = score of the
    # meeting node, or NaN if they never merge / a node is unknown.
    for k in range(us.shape[0]):
        u = us[k]
        v = vs[k]
        if u < 0 or v < 0:
            out[k] = np.nan
            continue
        while True:
            if u == v:
                out[k] = score[u]
                break
            if level[u] > level[v]:
                tmp = u
                u = v
                v = tmp
            if nxt[u] < 0:  # reached a root without meeting v
                out[k] = np.nan
                break
            u = nxt[u]


class MergeTree:
    """Merge tree over a waterz merge history (numba-accelerated).

    Vendored/adapted from lsd (lsd.post.merge_tree): build the tree by replaying
    a merge history with ``merge``, then ``find_merge(u, v)`` /
    ``find_merges(us, vs)`` return the score at which fragments ``u`` and ``v``
    first join the same segment (``None`` / NaN if they never merge).

    Nodes are stored in flat arrays indexed by a dense id (fragment ids are
    sparse/large, so they are remapped through ``_idx``); the hot query loop
    runs in compiled code, matching lsd's Cython implementation's behavior.
    """

    def __init__(self, leaf_nodes=None):
        self._idx = {}          # node id -> dense index
        self._level = []        # dense index -> tree level
        self._next = []         # dense index -> parent dense index (-1 = root)
        self._score = []        # dense index -> merge score (0.0 for leaves)
        self.id_to_node = {}    # node id -> id of its current (highest) merge node
        self.next_id = 0
        self.max_level = 0
        self._arrays = None

        if leaf_nodes is not None:
            leaves = [int(n) for n in leaf_nodes]
            for n in leaves:
                if n not in self._idx:
                    self._add(n, 0, 0.0)
                    self.id_to_node[n] = n
            self.next_id = max(leaves) + 1

    def _add(self, node_id, level, score):
        idx = len(self._level)
        self._idx[node_id] = idx
        self._level.append(level)
        self._next.append(-1)
        self._score.append(score)
        self._arrays = None
        return idx

    def merge(self, u, v, target, score):
        u, v, target = int(u), int(v), int(target)
        t = self.next_id
        self.next_id += 1

        iu = self._idx[self.id_to_node[u]]
        iv = self._idx[self.id_to_node[v]]
        level = max(self._level[iu], self._level[iv]) + 1
        self.max_level = max(self.max_level, level)

        it = self._add(t, level, float(score))
        self._next[iu] = it
        self._next[iv] = it
        self.id_to_node[target] = t

    def _finalize(self):
        if self._arrays is None:
            self._arrays = (
                np.asarray(self._level, dtype=np.int64),
                np.asarray(self._next, dtype=np.int64),
                np.asarray(self._score, dtype=np.float64),
            )
        return self._arrays

    def find_merges(self, us, vs):
        """Vectorized find_merge over id sequences. Returns a float64 array;
        NaN entries mean the pair never merges (equivalent to None)."""
        level, nxt, score = self._finalize()
        n = len(us)
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        uu = np.empty(n, dtype=np.int64)
        vv = np.empty(n, dtype=np.int64)
        idx = self._idx
        for k in range(n):
            uu[k] = idx.get(int(us[k]), -1)
            vv[k] = idx.get(int(vs[k]), -1)
        _find_merges(uu, vv, level, nxt, score, out)
        return out

    def find_merge(self, u, v):
        r = self.find_merges((u,), (v,))[0]
        return None if np.isnan(r) else r
