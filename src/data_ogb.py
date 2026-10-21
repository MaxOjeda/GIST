"""ogbl-wikikg2 with the official OGB evaluation protocol.

Each test triple gives two evaluation items, one with 500 negative tails and one with 500 negative heads.
The positive is placed first and the rank is 0.5 * (optimistic + pessimistic) + 1, as in the OGB evaluator.
Relations are interleaved: 2r is relation r and 2r + 1 its inverse.
"""
import torch
from src.data import Graph


class WikiKG2:
    """ogbl-wikikg2 loaded from a preprocessed file (see prep_wikikg2.py)."""

    def __init__(self, path='/home/jreutter/datasets/ogb/wikikg2.pt'):
        D = torch.load(path, map_location='cpu')
        self.num_entity = D['num_nodes']
        R = D['num_relation']
        self.num_relation = 2 * R
        tr = D['train']
        # edges (h, 2r, t) and their inverses (t, 2r+1, h)
        self.edge_index = torch.cat([
            torch.stack([tr[:, 0], 2 * tr[:, 1], tr[:, 2]], 1),
            torch.stack([tr[:, 2], 2 * tr[:, 1] + 1, tr[:, 0]], 1)])
        self.train_triplets = torch.stack([tr[:, 0], 2 * tr[:, 1], tr[:, 2]], 1)
        for sp in ('valid', 'test'):
            s = D[sp]
            setattr(self, sp + '_triplets', torch.stack([s[:, 0], 2 * s[:, 1], s[:, 2]], 1))
            setattr(self, sp + '_head_neg', D[sp + '_head_neg'])
            setattr(self, sp + '_tail_neg', D[sp + '_tail_neg'])
        g = Graph(self.edge_index, self.num_entity, self.num_relation)
        self.train_graph = self.valid_graph = self.test_graph = g

        # sorted int64 keys (h * R2 + r) * N + t, to test edge membership with a binary search
        N, R2 = self.num_entity, self.num_relation
        k = (self.edge_index[:, 0] * R2 + self.edge_index[:, 1]) * N + self.edge_index[:, 2]
        self.edge_key, self.edge_perm = k.sort()
        self._key_dev = {}

    def __getstate__(self):
        # do not pickle the GPU copies of the keys (data loader workers do not need them)
        st = self.__dict__.copy()
        st['_key_dev'] = {}
        return st

    def _key_on(self, dev):
        """Edge keys on `dev` (cached copy)."""
        dev = torch.device(dev)
        if dev.type == 'cpu':
            return self.edge_key
        if str(dev) not in self._key_dev:
            self._key_dev[str(dev)] = self.edge_key.to(dev)
        return self._key_dev[str(dev)]

    def query_edge_mask(self, h, r, t, device):
        """Edge mask (E,) that removes the query edges (h, r, t) and their inverses, with all duplicates."""
        N, R2 = self.num_entity, self.num_relation
        h, r, t = h.cpu(), r.cpu(), t.cpu()
        rev = torch.where(r % 2 == 0, r + 1, r - 1)
        q = torch.cat([(h * R2 + r) * N + t, (t * R2 + rev) * N + h])
        lo = torch.searchsorted(self.edge_key, q)
        hi = torch.searchsorted(self.edge_key, q, right=True)
        cnt = hi - lo
        pos = torch.repeat_interleave(lo, cnt) + (torch.arange(int(cnt.sum())) -
                                                 torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt))
        mask = torch.ones(self.edge_key.numel(), dtype=torch.bool, device=device)
        mask[self.edge_perm[pos].to(device)] = False
        return mask

    def _is_true(self, h, r, cand):
        """(B, k) mask: whether cand[i, j] is a known tail of (h_i, r_i)."""
        N, R2 = self.num_entity, self.num_relation
        key = self._key_on(cand.device)
        q = ((h.to(cand.device) * R2 + r.to(cand.device)) * N).unsqueeze(1) + cand
        pos = torch.searchsorted(key, q.reshape(-1)).clamp(max=key.numel() - 1)
        return (key[pos] == q.reshape(-1)).view(cand.shape)

    def sample_strict_negatives(self, h, r, k, max_tries=4):
        """Uniform negatives that are not known tails of (h, r), by rejection sampling."""
        B = h.size(0)
        # sample in chunks to limit memory with a large number of negatives
        chunk = max(1, 2_000_000 // max(B, 1))
        out = []
        for c0 in range(0, k, chunk):
            kc = min(chunk, k - c0)
            neg = torch.randint(self.num_entity, (B, kc), device=h.device)
            for _ in range(max_tries):
                bad = self._is_true(h, r, neg)
                if not bad.any():
                    break
                neg = torch.where(bad, torch.randint(self.num_entity, (B, kc), device=h.device), neg)
            out.append(neg)
        return torch.cat(out, 1) if len(out) > 1 else out[0]

    def train_collate_fn(self, batch):
        """First half of the batch predicts tails (h, 2r), second half heads (t, 2r+1)."""
        b = torch.stack(batch, 0)
        n = len(batch) // 2
        h = torch.cat([b[:n, 0], b[n:, 2]])
        r = torch.cat([b[:n, 1], b[n:, 1] + 1])       # inverse relation for head prediction
        t = torch.cat([b[:n, 2], b[n:, 0]])
        return {'h_index': h, 'r_index': r, 't_index': t,
                'graph': self.train_graph, 'ogb': True}

    def ogb_collate_fn(self, split):
        """Collate function for `OGBEvalSet` (a picklable object)."""
        return _OGBCollate(self, split)


class OGBEvalSet(torch.utils.data.Dataset):
    """Two items per triple (tail and head prediction).

    `subsample` > 0 evaluates a random subset of items, drawn with a fixed `seed`.
    """

    def __init__(self, n, subsample=0, seed=1024):
        self.n = n
        self.perm = None
        if subsample and subsample < 2 * n:
            g = torch.Generator().manual_seed(seed)
            self.perm = torch.randperm(2 * n, generator=g)[:subsample]

    def __len__(self):
        return self.perm.numel() if self.perm is not None else 2 * self.n

    def __getitem__(self, i):
        if self.perm is not None:
            i = int(self.perm[i])
        # interleaved: even items predict the tail, odd items the head (any prefix is balanced)
        return (i // 2, i % 2 == 0)


def ogb_ranks(score_cand):
    """OGB rank 0.5 * (optimistic + pessimistic) + 1, with the positive in column 0 of `score_cand` (B, 1 + k)."""
    pos = score_cand[:, :1]
    neg = score_cand[:, 1:]
    optimistic = (neg > pos).sum(dim=1)
    pessimistic = (neg >= pos).sum(dim=1)
    return 0.5 * (optimistic + pessimistic).float() + 1


class _OGBCollate:
    """Builds evaluation batches: query, positive and its 500 negatives."""

    def __init__(self, data, split):
        self.graph = data.train_graph
        self.trip = getattr(data, split + '_triplets')
        self.hneg = getattr(data, split + '_head_neg')
        self.tneg = getattr(data, split + '_tail_neg')

    def __call__(self, batch):
        idx = torch.tensor([i for i, _ in batch])
        is_tail = torch.tensor([b for _, b in batch])
        tr = self.trip[idx]
        h = torch.where(is_tail, tr[:, 0], tr[:, 2])
        r = torch.where(is_tail, tr[:, 1], tr[:, 1] + 1)   # head prediction uses the inverse relation
        gold = torch.where(is_tail, tr[:, 2], tr[:, 0])
        neg = torch.where(is_tail.unsqueeze(1), self.tneg[idx], self.hneg[idx])
        cand = torch.cat([gold.unsqueeze(1), neg], 1)      # positive in column 0
        return {'h_index': h, 'r_index': r, 't_index': gold, 'cand': cand,
                'graph': self.graph, 'ogb': True}
