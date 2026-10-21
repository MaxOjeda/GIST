"""Paired comparison on Wikidata5M between a train.py rank dump and A*Net on the same queries.

Inputs (same protocol: filtered ranking against all entities, pessimistic ties):
  - train.py dump (--dump_ranks): rows (h, r2, t, rank); r2 = 2r predicts the tail, r2 = 2r + 1 the head.
  - A*Net dump (run_astarnet_wd5m.py): rows [rank_tail, rank_head, h, t, r], one per triple.
Usage: python analyze_wd5m_paired.py <split> [dump]   (split: valid | test)
"""
import sys
from collections import defaultdict
import torch

split = sys.argv[1] if len(sys.argv) > 1 else 'test'
ours_path = sys.argv[2] if len(sys.argv) > 2 else f'wd5m_ep5_{split}_ranks.pt'
ours = torch.load(ours_path, map_location='cpu')['rows'].double()
print('ours =', ours_path)
astar = torch.load(f'astarnet_wd5m_{split}_ranks_best_ep30.pt', map_location='cpu')['rows'].double()

# A*Net: (h, t, r) -> list of (rank_tail, rank_head)
amap = defaultdict(list)
for rt, rh, h, t, r in astar.tolist():
    amap[(int(h), int(t), int(r))].append((rt, rh))

o_r, a_r, is_tail = [], [], []
miss = 0
for h, r2, t, rk in ours.tolist():
    h, r2, t = int(h), int(r2), int(t)
    if r2 % 2 == 0:
        key, col, tail = (h, t, r2 // 2), 0, True        # tail query (h, r, ?)
    else:
        key, col, tail = (t, h, r2 // 2), 1, False       # head query (t, r^-1, ?) of the triple (h, r, t)
    if key not in amap:
        miss += 1
        continue
    a_r.append(amap[key][0][col]); o_r.append(rk); is_tail.append(tail)
o_r, a_r = torch.tensor(o_r), torch.tensor(a_r)
is_tail = torch.tensor(is_tail)
print(f'[{split}] queries nuestras {len(ours)} | pareadas {len(o_r)} | sin par {miss} | filas A*Net {len(astar)} '
      f'(= {2 * len(astar)} queries)')


def m(r):
    n = len(r); rr = 1 / r
    return (f'MRR {rr.mean():.4f}±{rr.std() / n ** 0.5:.4f} | MR {r.mean():9.0f} | H@1 {(r <= 1).double().mean():.4f} | '
            f'H@3 {(r <= 3).double().mean():.4f} | H@10 {(r <= 10).double().mean():.4f}')


for lab, mk in (('TODAS', torch.ones_like(is_tail)), ('cola', is_tail), ('cabeza', ~is_tail)):
    print(f'{lab:7s} nuestro  {m(o_r[mk])}')
    print(f'{lab:7s} A*Net    {m(a_r[mk])}')
    d = 1 / o_r[mk] - 1 / a_r[mk]
    print(f'{lab:7s} Δ MRR (nuestro − A*Net) = {d.mean():+.4f} ± {d.std() / len(d) ** 0.5:.4f}  '
          f'(t = {d.mean() / (d.std() / len(d) ** 0.5):.1f})')
better = (o_r < a_r).double().mean(); worse = (o_r > a_r).double().mean()
print(f'queries donde ganamos {better:.3f} | perdemos {worse:.3f} | empate {1 - better - worse:.3f}')
