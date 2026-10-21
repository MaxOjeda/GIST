"""YAGO3-10 test analysis of GIST, A*Net and their variants.

  (1) per-query metrics,
  (2) performance by the number of training answers of (h, r), with the rate of queries where A*Net and GIST both fail
      (rank > 10) and the median degrees of source and answer,
  (3) set metrics for queries with several answers (R-Precision, MAP, Recall@K),
  (4) examples of hub queries where A*Net and GIST both fail.
All ranks are filtered with pessimistic ties. Missing models are skipped.
Usage: python analyze_yago_paper.py
"""
import csv, os, pickle
from collections import defaultdict
import numpy as np
import torch
from analyze_overlap_v2 import DS
from analyze_set_metrics import positions_by_query, metrics

C = DS['yago']
OURS = [('GIST', 'yago_nvis_s42_ranks.pt'),
        ('GIST (presup. A*Net)', 'yago_nvis_budget037_s42_ranks.pt'),
        ('GIST - neg. duros', 'yago_global_s42_e20_ranks.pt'),
        ('GIST - neg. duros - tokens', 'gt_b200_s42_ranks.pt')]
EXT = [('A*Net', 'astarnet_yago_ranks_test.pkl'),
       ('A*Net + tokens', 'astarnet_git_yago_ranks_test.pkl'),
       ('A*Net + tokens + neg. duros', 'astarnet_git_hn_yago_ranks_test.pkl')]
KS = [1, 3, 10, 100]


def load():
    ours = [(n, torch.load(p, map_location='cpu')) for n, p in OURS if os.path.exists(p)]
    base = ours[0][1]
    rows = base['rows'].numpy()[:, :3].astype(np.int64)
    rk = {}
    for n, d in ours:
        r = d['rows'].numpy()
        assert np.array_equal(r[:, :3], rows), f'{n}: orden de queries distinto'
        rk[n] = r[:, 3].astype(float)
    # A*Net (torchdrug) uses other ids: match queries by entity and relation names
    inv_e, inv_r = {}, {}
    for f in C['raw']:
        for h, r, t in csv.reader(open(f), delimiter='\t'):
            inv_e.setdefault(h, len(inv_e)); inv_r.setdefault(r, len(inv_r)); inv_e.setdefault(t, len(inv_e))
    o_e = {int(i): e for e, i in (l.rstrip('\n').split('\t') for l in open(f"{C['data']}/entities.txt"))}
    o_r = {int(i): r for r, i in (l.rstrip('\n').split('\t') for l in open(f"{C['data']}/relations.txt"))}
    ext = [(n, pickle.load(open(p, 'rb'))) for n, p in EXT if os.path.exists(p)]
    keep = np.arange(len(rows))
    if ext:
        maps = []
        for n, d in ext:
            htr = np.asarray(d['htr']); key = {}
            for i in range(len(htr)):
                h, t, r = map(int, htr[i]); key[(h, r, t, 0)] = (i, 0); key[(t, r, h, 1)] = (i, 1)
            maps.append((n, d, key))
        idx = {n: [] for n, _, _ in maps}; ok = []
        for i in range(len(rows)):
            h, r, t = map(int, rows[i]); rn = o_r[r]; rev = rn.startswith('-'); b = rn[1:] if rev else rn
            k = (inv_e.get(o_e[h], -1), inv_r.get(b, -1), inv_e.get(o_e[t], -1), int(rev))
            if all(k in key for _, _, key in maps):
                ok.append(i)
                for n, _, key in maps:
                    idx[n].append(key[k])
        keep = np.array(ok)
        for n, d, _ in maps:
            ii = np.array(idx[n]); rk[n] = np.asarray(d['ranking'], float)[ii[:, 0], ii[:, 1]]
        for n in [m for m, _ in ours]:
            rk[n] = rk[n][keep]
        print(f'alineadas {len(keep)} de {len(rows)} queries con los modelos A*Net')
    return rows[keep], rk, base['edge_index'].numpy(), o_e, o_r


def main():
    rows, rk, ei, o_e, o_r = load()
    names = list(rk)
    N = len(o_e)
    deg = np.bincount(ei[:, 0], minlength=N)                 # the training graph already has both directions
    card = defaultdict(int)
    for h, r, _ in ei:
        card[(int(h), int(r))] += 1
    cq = np.array([card.get((int(h), int(r)), 0) for h, r, _ in rows])
    dh, dt = deg[rows[:, 0]], deg[rows[:, 2]]

    print('\n=== (1) metricas por query ===')
    print(f"{'modelo':30s}{'MRR':>8}{'H@1':>8}{'H@3':>8}{'H@10':>8}{'MR':>9}")
    for n in names:
        r = rk[n]
        print(f'{n:30s}{(1/r).mean():8.4f}{(r<=1).mean():8.4f}{(r<=3).mean():8.4f}{(r<=10).mean():8.4f}{r.mean():9.0f}')

    A, G = 'A*Net', 'GIST'
    both = (rk[A] > 10) & (rk[G] > 10) if A in rk and G in rk else np.zeros(len(rows), bool)
    print('\n=== (2) por cardinalidad de (h, r) en train (fallan ambos = rank > 10 en A*Net y en GIST) ===')
    hdr = f"{'# resp. train':>14}{'n':>7}{'% test':>8}{'fallan':>8}" + ''.join(f'{"MRR "+n[:14]:>20}' for n in names) \
        + ''.join(f'{"H@10 "+n[:13]:>20}' for n in names) + f"{'grado h':>9}{'grado t':>9}"
    print(hdr)
    for lo, hi in C['card']:
        m = (cq >= lo) & (cq <= hi)
        lab = f'{lo}' if lo == hi else (f'{lo}+' if hi > 10**8 else f'{lo}-{hi}')
        line = f'{lab:>14}{int(m.sum()):7d}{100*m.mean():7.1f}%{100*both[m].mean():7.1f}%'
        line += ''.join(f'{(1/rk[n][m]).mean():20.3f}' for n in names)
        line += ''.join(f'{(rk[n][m]<=10).mean():20.3f}' for n in names)
        line += f'{np.median(dh[m]):9.0f}{np.median(dt[m]):9.0f}'
        print(line)

    print('\n=== (3) metricas de conjunto ===')
    POS = {n: positions_by_query(rows, rk[n]) for n in names}
    Tall = np.array([len(v) for v in POS[names[0]].values()])
    print('|T| en test: ' + ', '.join(f'{k}: {int((Tall == k).sum())}' for k in range(1, 6)) + f', >5: {int((Tall > 5).sum())}')
    for lo in (1, 2, 3):
        qs = [q for q, v in POS[names[0]].items() if len(v) >= lo]
        T = np.array([len(POS[names[0]][q]) for q in qs])
        if len(qs) < 30:
            continue
        print(f'\n--- |T| >= {lo}: {len(qs):,} queries ({T.sum():,} items = {100*T.sum()/Tall.sum():.1f}% del test) ---')
        cols = ['R-Prec', 'MAP'] + [f'R@{K}' for K in KS]
        print(f"{'modelo':30s}" + ''.join(f'{c:>9}' for c in cols))
        for n in names:
            p = {q: POS[n][q] for q in qs}
            res, _, _ = metrics(p, KS)
            row = [res['R-Prec'], res['MAP']] + [res[f'R@{K}'] for K in KS]
            print(f'{n:30s}' + ''.join(f'{x:9.4f}' for x in row))

    print('\n=== (4) ejemplos: queries hub donde fallan A*Net y GIST (una respuesta de test por query) ===')
    seen, ex = set(), []
    order = np.argsort(-cq)
    for i in order:
        if not both[i]:
            continue
        h, r, t = map(int, rows[i])
        if (h, r) in seen:
            continue
        seen.add((h, r))
        n_test = int(((rows[:, 0] == h) & (rows[:, 1] == r)).sum())
        ex.append((o_e[h], o_r[r], int(deg[h]), int(cq[i]), n_test, o_e[t], [int(rk[n][i]) for n in names]))
        if len(ex) >= 8:
            break
    print(f"{'fuente':22s}{'relacion':24s}{'grado h':>9}{'resp.train':>11}{'resp.test':>10}  {'respuesta':28s} ranks ({' / '.join(names)})")
    for s, rl, d, ct, nt, ans, rr in ex:
        print(f"{s[:21]:22s}{rl[:23]:24s}{d:9d}{ct:11d}{nt:10d}  {ans[:27]:28s} {' / '.join(map(str, rr))}")


if __name__ == '__main__':
    main()
