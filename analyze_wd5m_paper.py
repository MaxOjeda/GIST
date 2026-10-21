"""Wikidata5M test analysis of GIST vs A*Net (same analysis as analyze_yago_paper.py).

  (1) per-query metrics,
  (2) MRR by the number of training answers of (h, r), with the rate of queries where both models fail (rank > 10),
      the median source degree, the fraction of head queries and the paired difference,
  (3) set metrics (R-Precision, MAP, Recall@K) for |T| >= 1, 2, 3,
  (4) the test queries with the most training answers.
Inputs: train.py rank dump (rows (h, r2, t, rank)) and A*Net dump from run_astarnet_wd5m.py.
Usage: python analyze_wd5m_paper.py [dump_gist] [dump_astarnet]
"""
import sys
from collections import defaultdict
import numpy as np
import torch
from analyze_set_metrics import positions_by_query, metrics

GIST = sys.argv[1] if len(sys.argv) > 1 else 'wd5m_neg1m_ep15_test_ranks.pt'
ASTAR = sys.argv[2] if len(sys.argv) > 2 else 'astarnet_wd5m_test_ranks_best_ep30.pt'
DATA = '/home/jreutter/datasets/knowledge_graphs/wikidata5m/wikidata5m_transductive.pt'
BINS = [(0, 0), (1, 3), (4, 10), (11, 30), (31, 100), (101, 1000), (1001, 10**9)]
KS = [1, 3, 10, 100]


def load():
    D = torch.load(DATA, map_location='cpu')
    R2 = 2 * D['num_relation']
    tr = D['train'].numpy()
    src = np.concatenate([tr[:, 0], tr[:, 2]])
    rel = np.concatenate([2 * tr[:, 1], 2 * tr[:, 1] + 1])
    uk, cnt = np.unique(src.astype(np.int64) * R2 + rel, return_counts=True)
    deg = np.bincount(src, minlength=D['num_nodes'])           # degree in the training graph (both directions)
    ours = torch.load(GIST, map_location='cpu')['rows'].double().numpy()
    astar = torch.load(ASTAR, map_location='cpu')['rows'].double().numpy()
    amap = {}
    for rt, rh, h, t, r in astar.tolist():
        amap.setdefault((int(h), int(t), int(r)), (rt, rh))
    rows, rk_g, rk_a = [], [], []
    for h, r2, t, rk in ours.tolist():
        h, r2, t = int(h), int(r2), int(t)
        key, col = ((h, t, r2 // 2), 0) if r2 % 2 == 0 else ((t, h, r2 // 2), 1)
        rows.append((h, r2, t)); rk_g.append(rk); rk_a.append(amap[key][col])
    rows = np.array(rows, dtype=np.int64)
    qk = rows[:, 0] * R2 + rows[:, 1]
    idx = np.clip(np.searchsorted(uk, qk), 0, len(uk) - 1)
    card = np.where(uk[idx] == qk, cnt[idx], 0)
    return rows, {'A*Net': np.array(rk_a), 'GIST': np.array(rk_g)}, card, deg, D


def main():
    rows, rk, card, deg, D = load()
    qid, pid = D['entity_qid'].numpy(), D['relation_pid'].numpy()
    names = list(rk)
    head = rows[:, 1] % 2 == 1

    print('=== (1) metricas por query ===')
    print(f"{'modelo':10s}{'MRR':>8}{'H@1':>8}{'H@3':>8}{'H@10':>8}{'MR':>10}")
    for n in names:
        r = rk[n]
        print(f'{n:10s}{(1/r).mean():8.4f}{(r<=1).mean():8.4f}{(r<=3).mean():8.4f}{(r<=10).mean():8.4f}{r.mean():10.0f}')

    both = (rk['A*Net'] > 10) & (rk['GIST'] > 10)
    d = 1 / rk['GIST'] - 1 / rk['A*Net']
    print('\n=== (2) por cardinalidad de (h, r) en train (fallan ambos = rank > 10 en A*Net y en GIST) ===')
    print(f"{'# resp. train':>14}{'n':>7}{'% test':>8}{'% cabeza':>9}{'fallan':>8}{'grado h':>9}"
          f"{'MRR A*Net':>11}{'MRR GIST':>10}{'Δ GIST − A*Net':>20}")
    for lo, hi in BINS + [(0, 10**9)]:
        m = (card >= lo) & (card <= hi)
        lab = 'todas' if (lo, hi) == (0, 10**9) else (f'{lo}' if lo == hi else (f'>{lo - 1}' if hi >= 10**9 else f'{lo}-{hi}'))
        dm = d[m]
        print(f'{lab:>14}{int(m.sum()):7d}{100*m.mean():7.1f}%{100*head[m].mean():8.1f}%{100*both[m].mean():7.1f}%'
              f'{np.median(deg[rows[m, 0]]):9.0f}{(1/rk["A*Net"][m]).mean():11.3f}{(1/rk["GIST"][m]).mean():10.3f}'
              f'{dm.mean():+12.4f} ± {dm.std()/len(dm)**.5:.4f}')

    print('\n=== (3) metricas de conjunto ===')
    POS = {n: positions_by_query(rows, rk[n]) for n in names}
    Tall = np.array([len(v) for v in POS['GIST'].values()])
    print('|T| en test: ' + ', '.join(f'{k}: {int((Tall == k).sum())}' for k in range(1, 6)) + f', >5: {int((Tall > 5).sum())}')
    for lo in (1, 2, 3):
        qs = [q for q, v in POS['GIST'].items() if len(v) >= lo]
        T = np.array([len(POS['GIST'][q]) for q in qs])
        if len(qs) < 30:
            print(f'\n--- |T| >= {lo}: solo {len(qs)} queries, se omite ---')
            continue
        print(f'\n--- |T| >= {lo}: {len(qs):,} queries ({T.sum():,} items = {100*T.sum()/Tall.sum():.1f}% del test) ---')
        cols = ['R-Prec', 'MAP'] + [f'R@{K}' for K in KS]
        print(f"{'modelo':10s}" + ''.join(f'{c:>9}' for c in cols))
        for n in names:
            res, _, _ = metrics({q: POS[n][q] for q in qs}, KS)
            print(f'{n:10s}' + ''.join(f'{x:9.4f}' for x in [res['R-Prec'], res['MAP']] + [res[f'R@{K}'] for K in KS]))

    print('\n=== (4) queries de test con mas respuestas en train ===')
    qs = sorted(POS['GIST'], key=lambda q: -card[(rows[:, 0] == q[0]) & (rows[:, 1] == q[1])][0])[:10]
    print(f"{'fuente':>11} {'relacion':>9} {'grado h':>8} {'resp.train':>10} {'resp.test':>9}   R-Prec (rank mediano) A*Net | GIST")
    for q in qs:
        m = (rows[:, 0] == q[0]) & (rows[:, 1] == q[1])
        n = int(m.sum())
        out = [f'{(POS[k][q] <= n).sum() / n:.3f} ({int(np.median(rk[k][m])):,})' for k in names]
        print(f"{'Q%d' % qid[q[0]]:>11} {'P%d%s' % (pid[q[1] // 2], '^-1' if q[1] % 2 else ''):>9} {deg[q[0]]:8d} "
              f"{int(card[m][0]):10d} {n:9d}   {out[0]} | {out[1]}")


if __name__ == '__main__':
    main()
