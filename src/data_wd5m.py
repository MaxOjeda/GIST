"""Wikidata5M (transductive split) with filtered ranking against all entities.

Training reuses the machinery of `WikiKG2` (sorted edge keys, strict negatives, query edge masks).
Evaluation ranks each query against all entities, removing the known answers from train, valid and test.
Relations are interleaved: 2r is relation r and 2r + 1 its inverse.
"""
import os
import numpy as np
import torch
from src.data_ogb import WikiKG2
from src.data import Graph

RAW_DIR = '/home/jreutter/datasets/knowledge_graphs/wikidata5m'
CACHE = os.path.join(RAW_DIR, 'wikidata5m_transductive.pt')


def _read_ids(path):
    """Read 'Q123<TAB>P31<TAB>Q5' lines into an (n, 3) int64 array of Wikidata ids."""
    with open(path, 'r') as f:
        txt = f.read()
    a = np.array(txt.replace('Q', '').replace('P', '').split(), dtype=np.int64)
    assert a.size % 3 == 0, path
    return a.reshape(-1, 3)


def preprocess(raw_dir=RAW_DIR, out=CACHE):
    """Map Wikidata ids to contiguous integers and cache the three splits."""
    sp = {s: _read_ids(os.path.join(raw_dir, f'wikidata5m_transductive_{s}.txt'))
          for s in ('train', 'valid', 'test')}
    ent = np.unique(np.concatenate([np.concatenate([a[:, 0], a[:, 2]]) for a in sp.values()]))
    rel = np.unique(np.concatenate([a[:, 1] for a in sp.values()]))
    D = {'entity_qid': torch.from_numpy(ent), 'relation_pid': torch.from_numpy(rel),
         'num_nodes': int(ent.size), 'num_relation': int(rel.size)}
    train_ent = np.unique(np.concatenate([sp['train'][:, 0], sp['train'][:, 2]]))
    for s, a in sp.items():
        h = np.searchsorted(ent, a[:, 0]); r = np.searchsorted(rel, a[:, 1]); t = np.searchsorted(ent, a[:, 2])
        D[s] = torch.from_numpy(np.stack([h, r, t], 1))
        if s != 'train':
            # entities that do not appear in train (unreachable)
            miss = ~np.isin(np.concatenate([a[:, 0], a[:, 2]]), train_ent)
            D[s + '_entities_not_in_train'] = int(miss.sum())
    torch.save(D, out)
    return D


class Wikidata5M(WikiKG2):
    """Wikidata5M dataset; training utilities are inherited from `WikiKG2`."""

    def __init__(self, path=CACHE, eval_seed=1024):
        if not os.path.exists(path):
            preprocess(out=path)
        D = torch.load(path, map_location='cpu')
        self.num_entity = N = D['num_nodes']
        self.num_relation = R2 = 2 * D['num_relation']
        tr = D['train']
        self.edge_index = torch.cat([
            torch.stack([tr[:, 0], 2 * tr[:, 1], tr[:, 2]], 1),
            torch.stack([tr[:, 2], 2 * tr[:, 1] + 1, tr[:, 0]], 1)])
        self.train_triplets = torch.stack([tr[:, 0], 2 * tr[:, 1], tr[:, 2]], 1)
        # valid / test: both directions interleaved after a fixed shuffle (any prefix is a random sample)
        g = torch.Generator().manual_seed(eval_seed)
        for sp in ('valid', 'test'):
            s = D[sp][torch.randperm(D[sp].size(0), generator=g)]
            fwd = torch.stack([s[:, 0], 2 * s[:, 1], s[:, 2]], 1)
            inv = torch.stack([s[:, 2], 2 * s[:, 1] + 1, s[:, 0]], 1)
            setattr(self, sp + '_triplets', torch.stack([fwd, inv], 1).reshape(-1, 3))
        graph = Graph(self.edge_index, N, R2)
        self.train_graph = self.valid_graph = self.test_graph = graph

        # sorted int64 keys (h * R2 + r) * N + t
        k = (self.edge_index[:, 0] * R2 + self.edge_index[:, 1]) * N + self.edge_index[:, 2]
        self.edge_key, self.edge_perm = k.sort()
        self._key_dev = {}
        # evaluation filter: known answers from train, valid and test
        ev = torch.cat([self.valid_triplets, self.test_triplets])
        self.filt_key = torch.cat([self.edge_key, (ev[:, 0] * R2 + ev[:, 1]) * N + ev[:, 2]]).sort().values
        self.not_in_train = {sp: D.get(sp + '_entities_not_in_train', 0) for sp in ('valid', 'test')}

    def filter_mask(self, h, r):
        """(B, N) mask of the known answers of each query (h_i, r_i)."""
        N, R2 = self.num_entity, self.num_relation
        base = (h * R2 + r) * N
        lo = torch.searchsorted(self.filt_key, base)
        hi = torch.searchsorted(self.filt_key, base + N)
        cnt = hi - lo
        pos = torch.repeat_interleave(lo, cnt) + (torch.arange(int(cnt.sum())) -
                                                 torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt))
        row = torch.repeat_interleave(torch.arange(h.numel()), cnt)
        mask = torch.zeros(h.numel(), N, dtype=torch.bool)
        mask[row, self.filt_key[pos] - torch.repeat_interleave(base, cnt)] = True
        return mask

    def _eval_collate(self, batch):
        b = torch.stack(batch, 0)
        h, r, t = b.unbind(-1)
        return {'h_index': h, 'r_index': r, 't_index': t,
                'filter_mask': self.filter_mask(h, r), 'graph': self.train_graph}

    # same collate for valid and test
    valid_collate_fn = _eval_collate
    test_collate_fn = _eval_collate
