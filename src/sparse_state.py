"""Helpers for the sparse state: CSR adjacency, global id -> slot mapping and per-query top-k.

The reached entities of each query live in slots (B, S) instead of a dense (B, N) state, so no tensor of
size (B, N) is built except for the random-walk distance bins.
"""
import torch


class CSRGraph:
    """Adjacency in CSR format sorted by source node, to get the outgoing edges of a set of nodes."""

    _cache = {}

    def __init__(self, edge_index, num_nodes, device):
        # edge_index: (E, 3) = (src, rel, dst)
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        order = torch.argsort(src)
        # permutation from the original edge order to the CSR order (used to apply edge masks)
        self.order = order
        self.src = src[order].contiguous()
        self.rel = rel[order].contiguous()
        self.dst = dst[order].contiguous()
        self.num_nodes = num_nodes
        deg = torch.bincount(self.src, minlength=num_nodes)
        self.indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        torch.cumsum(deg, 0, out=self.indptr[1:])
        self.degree = deg

    @classmethod
    def get(cls, edge_index, num_nodes, device, key=None):
        """Cached CSR for a graph identified by `key`."""
        k = (key, num_nodes)
        if k not in cls._cache:
            if len(cls._cache) > 4:
                cls._cache.clear()
            cls._cache[k] = CSRGraph(edge_index, num_nodes, device)
        return cls._cache[k]

    def out_edges(self, nodes):
        """Outgoing edges of `nodes` (n,): edge indices (m,) and the position in `nodes` of their source (m,)."""
        deg = self.degree[nodes]                                  # (n,)
        total = int(deg.sum())
        if total == 0:
            z = nodes.new_zeros(0)
            return z, z
        owner = torch.repeat_interleave(torch.arange(len(nodes), device=nodes.device), deg)
        offs = torch.cumsum(deg, 0) - deg                         # start of each node's edges
        within = torch.arange(total, device=nodes.device) - offs[owner]
        eidx = self.indptr[nodes][owner] + within
        return eidx, owner


def global_to_slot(active_sorted, ids):
    """Map global ids to slots with a binary search on the sorted active set.

    active_sorted: (B, S) sorted global ids per row. ids: (B, K) ids to map.
    Returns slot (B, K) and found (B, K), whether each id is active.
    """
    slot = torch.searchsorted(active_sorted, ids)
    slot = slot.clamp(max=active_sorted.size(1) - 1)
    found = active_sorted.gather(1, slot) == ids
    return slot, found


def multikey_argsort(inputs, descending=False, break_tie=False):
    """Lexicographic sort by several keys (stable sorts from the last key to the first)."""
    if break_tie:
        order = torch.randperm(len(inputs[0]), device=inputs[0].device)
    else:
        order = torch.arange(len(inputs[0]), device=inputs[0].device)
    for key in inputs[::-1]:
        index = key[order].argsort(stable=True, descending=descending)
        order = order[index]
    return order


def variadic_topks(input, size, ks, largest=True, break_tie=False):
    """Top-k per sample on a flat tensor of variable-length segments.

    input: (total,) values of all samples, concatenated. size: (B,) segment lengths. ks: (B,) number to keep.
    Returns the selected values and their indices, grouped by sample.
    """
    index2sample = torch.repeat_interleave(size)
    if largest:
        index2sample = -index2sample
    order = multikey_argsort((index2sample, input), descending=largest, break_tie=break_tie)

    range = torch.arange(ks.sum(), device=input.device)
    offset = (size - ks).cumsum(0) - size + ks
    range = range + offset.repeat_interleave(ks)
    index = order[range]

    return input[index], index


def global_to_slot_flat(active, ids, qid, num_nodes):
    """Like `global_to_slot` for a flat tensor of ids with the query id of each element.

    The pair (query, node) is encoded as q * (N + 1) + node, which keeps the flattened active set sorted.
    """
    B, S = active.shape
    base = torch.arange(B, device=active.device).unsqueeze(1) * (num_nodes + 1)
    flat_sorted = (active + base).reshape(-1)
    key = ids + qid * (num_nodes + 1)
    pos = torch.searchsorted(flat_sorted, key).clamp(max=B * S - 1)
    found = flat_sorted[pos] == key
    slot = (pos - qid * S).clamp(0, S - 1)
    return slot, found


@torch.no_grad()
def personalized_pagerank(edge_index, num_nodes, src, alpha=0.8, num_iteration=20):
    """Personalized PageRank from each node in `src` (B,). Returns (B, N).

    Edge weight 1 / out-degree of the source; `alpha` weighs the propagation and 1 - alpha the restart.
    """
    dev = edge_index.device
    s, d = edge_index[:, 0], edge_index[:, 2]
    outdeg = torch.bincount(s, minlength=num_nodes).clamp(min=1).float()
    w = (1.0 / outdeg[s]).unsqueeze(1)                       # (E, 1)
    B = src.numel()
    init = torch.zeros(num_nodes, B, device=dev)
    init[src, torch.arange(B, device=dev)] = 1.0
    ppr = init
    for _ in range(num_iteration):
        nxt = torch.zeros_like(ppr)
        nxt.index_add_(0, d, ppr[s] * w)
        ppr = nxt * alpha + init * (1 - alpha)
    return ppr.t().contiguous()                              # (B, N)


def ppr_bins(ppr, num_nodes, num_bin=10):
    """Discretize PageRank values into `num_bin` logarithmic bins between 1/N and 1."""
    b = torch.logspace(-1, 0, num_bin, base=num_nodes, device=ppr.device)
    return torch.bucketize(ppr, b).clamp(max=num_bin - 1).to(torch.uint8)
