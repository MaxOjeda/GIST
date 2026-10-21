"""Set metrics for queries with several answers: Recall@K, Precision@K, R-Precision, MAP.

The standard protocol ranks each test triple separately, filtering the other answers. Since all answers of a query (h, r)
share the same filtered candidates, the real position of the m-th answer (0-based, sorted by filtered rank) is
filtered_rank + m. This rebuilds the set metrics exactly from the --dump_ranks files, without evaluating again.

R-Precision (K = |T| of each query) needs no choice of K. With |T| = 1, Recall@K equals Hits@K, so Precision@K is only
reported on queries with |T| >= K.

Usage: python analyze_set_metrics.py --ds fb237
"""
import argparse, csv, os, pickle, sys
from collections import defaultdict

import numpy as np
import torch

from analyze_overlap_v2 import DS


def positions_by_query(rows, rank):
    """{(h, r): sorted real positions of its test answers}."""
    by = defaultdict(list)
    for i in range(len(rows)):
        by[(int(rows[i, 0]), int(rows[i, 1]))].append(rank[i])
    out = {}
    for q, rs in by.items():
        r = np.sort(np.asarray(rs, dtype=float))
        out[q] = r + np.arange(len(r))      # the m-th answer is m positions lower
    return out


def metrics(pos, Ks):
    """Set metrics from {query: positions}."""
    T = np.array([len(v) for v in pos.values()])
    res = {'#queries': len(pos), '|T| medio': T.mean()}
    hits = {K: np.array([(v <= K).sum() for v in pos.values()], dtype=float) for K in Ks}
    for K in Ks:
        rec = hits[K] / T
        res[f'R@{K}'] = rec.mean()                                   # macro-recall
        res[f'microR@{K}'] = hits[K].sum() / np.minimum(K, T).sum()  # weighted by |T|
        res[f'nRec@{K}'] = hits[K].mean()                            # mean number of answers in the top K
    res['R-Prec'] = np.mean([(v <= len(v)).sum() / len(v) for v in pos.values()])
    res['MAP'] = np.mean([np.mean((np.arange(len(v)) + 1) / v) for v in pos.values()])
    return res, T, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, choices=list(DS))
    ap.add_argument('--Ks', type=int, nargs='+', default=[1, 3, 10, 100])
    args = ap.parse_args()
    C = DS[args.ds]

    inv_e, inv_r = {}, {}
    for f in C['raw']:
        for h, r, t in csv.reader(open(f), delimiter='\t'):
            inv_e.setdefault(h, len(inv_e)); inv_r.setdefault(r, len(inv_r)); inv_e.setdefault(t, len(inv_e))
    o_e = {int(i): e for e, i in (l.rstrip('\n').split('\t') for l in open(f"{C['data']}/entities.txt"))}
    o_r = {int(i): r for r, i in (l.rstrip('\n').split('\t') for l in open(f"{C['data']}/relations.txt"))}

    ours = [(n, torch.load(p, map_location='cpu')) for n, p in C['ours'] if os.path.exists(p)]
    rows = ours[0][1]['rows'].numpy()[:, :3]
    rk = {n: d['rows'].numpy()[:, 3].astype(float) for n, d in ours}
    ext = [(n, pickle.load(open(p, 'rb'))) for n, p in C['ext'] if os.path.exists(p)]
    if ext:
        htr = np.asarray(ext[0][1]['htr']); key = {}
        for i in range(len(htr)):
            h, t, r = map(int, htr[i]); key[(h, r, t, 0)] = (i, 0); key[(t, r, h, 1)] = (i, 1)
        idx, ok = [], []
        for i in range(len(rows)):
            h, r, t = map(int, rows[i]); rn = o_r[r]; rev = rn.startswith('-'); base = rn[1:] if rev else rn
            k = (inv_e.get(o_e[h], -1), inv_r.get(base, -1), inv_e.get(o_e[t], -1), int(rev))
            if k in key: idx.append(key[k]); ok.append(i)
        idx, ok = np.array(idx), np.array(ok)
        rows = rows[ok]; rk = {n: v[ok] for n, v in rk.items()}
        for n, d in ext: rk[n] = np.asarray(d['ranking'], float)[idx[:, 0], idx[:, 1]]
        print(f'alineadas {len(ok)} de {len(ours[0][1]["rows"])}')

    names = list(rk)
    POS = {n: positions_by_query(rows, rk[n]) for n in names}
    M = {n: metrics(POS[n], args.Ks) for n in names}
    T = M[names[0]][1]

    print(f"\n=== {args.ds}: {len(T):,} queries (h,r), |T| medio {T.mean():.2f}, "
          f"{100*np.mean(T == 1):.1f}% con |T|=1 ===")
    cols = ['R-Prec', 'MAP'] + [f'R@{K}' for K in args.Ks] + [f'microR@{K}' for K in args.Ks]
    w = max(len(n) for n in names) + 1
    print(f"{'modelo':>{w}}" + ''.join(f'{c:>11}' for c in cols))
    for n in names:
        print(f'{n:>{w}}' + ''.join(f'{M[n][0][c]:>11.4f}' for c in cols))

    print(f"\n--- metrica PONDERADA del usuario: (1/N) sum Recall@K*|T| = nro medio de respuestas "
          f"recuperadas en el top-K (maximo alcanzable = {np.minimum(args.Ks[-1], T).mean():.3f} a K={args.Ks[-1]}) ---")
    print(f"{'modelo':>{w}}" + ''.join(f'{f"nRec@{K}":>11}' for K in args.Ks))
    for n in names:
        print(f'{n:>{w}}' + ''.join(f'{M[n][0][f"nRec@{K}"]:>11.4f}' for K in args.Ks))

    # queries with several answers
    for lo in (2, 10):
        sel = {q: v for q, v in POS[names[0]].items() if len(v) >= lo}
        if len(sel) < 30: continue
        qs = set(sel)
        print(f"\n--- subconjunto |T| >= {lo}: {len(qs):,} queries "
              f"({100*len(qs)/len(T):.1f}% de las queries, "
              f"{100*sum(len(v) for v in sel.values())/T.sum():.1f}% de los items) ---")
        cols2 = ['R-Prec', 'MAP'] + [f'R@{K}' for K in args.Ks] + [f'P@{K}' for K in args.Ks if K <= lo or lo >= 10]
        print(f"{'modelo':>{w}}" + ''.join(f'{c:>11}' for c in cols2))
        for n in names:
            p = {q: POS[n][q] for q in qs}
            r, _, _ = metrics(p, args.Ks)
            row = [r['R-Prec'], r['MAP']] + [r[f'R@{K}'] for K in args.Ks]
            for K in args.Ks:
                if K <= lo or lo >= 10:
                    row.append(np.mean([(v <= K).sum() / K for v in p.values()]))
            print(f'{n:>{w}}' + ''.join(f'{x:>11.4f}' for x in row))


if __name__ == '__main__':
    main()
