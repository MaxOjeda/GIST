"""Overlap of failures (rank > 10) between models on the same test queries.

Prints the Jaccard index of the failure sets for every pair of models, and compares the newest model with an external
baseline by the number of training answers of (h, r) and, optionally, by the h-t distance.
Usage:
  python analyze_overlap_v2.py --ds fb237
  python analyze_overlap_v2.py --ds wn18rr --distance
"""
import argparse, csv, itertools, os, pickle, sys
from collections import Counter
from math import sqrt

import numpy as np
import torch

TD = os.path.expanduser('~/datasets/knowledge_graphs/torchdrug')
FB = os.path.expanduser('~/datasets/knowledge_graphs/FB15k-237/FB15k-237/raw')

DS = {
    'fb237': dict(
        raw=[f'{FB}/train.txt', f'{FB}/valid.txt', f'{FB}/test.txt'], data='data/fb15k-237',
        ours=[('GT viejo s42', 'fb_b8lr1_ranks.pt'), ('GT viejo s43', 'fb_b8lr1_s43_ranks.pt'),
              ('GT viejo s44', 'fb_b8lr1_s44_ranks.pt'),
              ('GT+global 20ep', 'fb_global_s42_ranks.pt'),
              ('GT+global 30ep', 'fb_global_s42_e30_ranks.pt')],
        ext=[('NBFNet', 'nbfnet_ranks_fb237_s1024_test.pkl'),
             ('A*Net', 'astarnet_ranks_bias_s1024_test.pkl'),
             ('A*Net s1025', 'astarnet_ranks_bias_s1025_test.pkl')],
        card=((0, 0), (1, 3), (4, 10), (11, 30), (31, 100), (101, 10**9))),
    'wn18rr': dict(
        raw=[f'{TD}/wn18rr_{s}.txt' for s in ('train', 'valid', 'test')], data='data/wn18rr',
        ours=[('GT viejo s42', 'wn_b8lr1_b100_ranks.pt'), ('GT viejo s43', 'wn_b8lr1_b100_s43_ranks.pt'),
              ('GT viejo s44', 'wn_b8lr1_b100_s44_ranks.pt'),
              ('GT+global s42', 'wn_global_s42_ranks.pt')],
        ext=[('NBFNet', 'nbfnet_wn_ranks_s1024_test.pkl'), ('A*Net', 'astarnet_wn_ranks_s1024_test.pkl')],
        card=((0, 0), (1, 1), (2, 3), (4, 10), (11, 30), (31, 10**9))),
    'yago': dict(
        raw=[f'{TD}/yago310_{s}.txt' for s in ('train', 'valid', 'test')], data='data/yago3-10',
        ours=[('GT viejo s42', 'gt_b200_s42_ranks.pt'), ('GT viejo s43', 'gt_b200_s43_ranks.pt'),
              ('GT viejo s44', 'gt_b200_s44_ranks.pt'),
              ('GT+global 10ep', 'yago_global_s42_ranks.pt'),
              ('GT+global 11ep', 'yago_global_s42_ep10_ranks.pt'),
              ('GT+global 20ep', 'yago_global_s42_e20_ranks.pt')],
        ext=[('A*Net', 'astarnet_yago_ranks_test.pkl')],
        card=((0, 0), (1, 3), (4, 10), (11, 30), (31, 100), (101, 10**9))),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, choices=list(DS))
    ap.add_argument('--distance', action='store_true', help='also group by h-t distance (slow)')
    args = ap.parse_args()
    C = DS[args.ds]

    inv_e, inv_r = {}, {}
    for f in C['raw']:
        for h, r, t in csv.reader(open(f), delimiter='\t'):
            inv_e.setdefault(h, len(inv_e)); inv_r.setdefault(r, len(inv_r)); inv_e.setdefault(t, len(inv_e))
    o_e = {int(i): e for e, i in (l.rstrip('\n').split('\t') for l in open(f"{C['data']}/entities.txt"))}
    o_r = {int(i): r for r, i in (l.rstrip('\n').split('\t') for l in open(f"{C['data']}/relations.txt"))}

    ours = [(n, torch.load(p, map_location='cpu')) for n, p in C['ours'] if os.path.exists(p)]
    if not ours:
        sys.exit('no hay volcados nuestros')
    rows = ours[0][1]['rows'].numpy()[:, :3]
    ei = ours[0][1]['edge_index'].numpy()
    for n, d in ours:
        assert (d['rows'].numpy()[:, :3] == rows).all(), f'{n}: otro orden de queries'

    ext = [(n, pickle.load(open(p, 'rb'))) for n, p in C['ext'] if os.path.exists(p)]
    rk = {n: d['rows'].numpy()[:, 3].astype(float) for n, d in ours}
    keep = np.arange(len(rows))
    if ext:
        htr = np.asarray(ext[0][1]['htr'])
        key = {}
        for i in range(len(htr)):
            h, t, r = map(int, htr[i]); key[(h, r, t, 0)] = (i, 0); key[(t, r, h, 1)] = (i, 1)
        idx, ok = [], []
        for i in range(len(rows)):
            h, r, t = map(int, rows[i]); rn = o_r[r]; rev = rn.startswith('-'); base = rn[1:] if rev else rn
            k = (inv_e.get(o_e[h], -1), inv_r.get(base, -1), inv_e.get(o_e[t], -1), int(rev))
            if k in key: idx.append(key[k]); ok.append(i)
        idx, keep = np.array(idx), np.array(ok)
        print(f'alineadas {len(keep)} de {len(rows)}')
        rk = {n: v[keep] for n, v in rk.items()}
        for n, d in ext:
            rk[n] = np.asarray(d['ranking'], float)[idx[:, 0], idx[:, 1]]
    rows = rows[keep]

    names = list(rk)
    print('\nMRR :', {k: round(float(np.mean(1 / v)), 4) for k, v in rk.items()})
    print('H@10:', {k: round(float(np.mean(v <= 10)), 3) for k, v in rk.items()})
    F = {k: v > 10 for k, v in rk.items()}
    J = lambda a, b: (F[a] & F[b]).sum() / (F[a] | F[b]).sum()
    w = max(len(n) for n in names) + 1
    print('\nJaccard de fallos (rank>10):')
    print(' ' * w + ''.join(f'{n[:9]:>10}' for n in names))
    for a in names:
        print(f'{a:>{w}}' + ''.join(f'{J(a, b):>10.3f}' for b in names))

    old = [n for n in names if n.startswith('GT viejo')]
    new = [n for n in names if n.startswith('GT+global')]
    exn = [n for _, n in [(0, n) for n, _ in ext]]
    g = lambda ps: np.mean([J(a, b) for a, b in ps]) if ps else float('nan')
    print('\n--- resumen: cuanto se PARECE cada familia a los baselines externos ---')
    if old and exn: print(f'  GT viejo   vs externos : {g(list(itertools.product(old, exn))):.3f}')
    if new and exn: print(f'  GT+global  vs externos : {g(list(itertools.product(new, exn))):.3f}')
    if len(exn) > 1: print(f'  externos entre si      : {g(list(itertools.combinations(exn, 2))):.3f}')
    if len(old) > 1: print(f'  GT viejo entre semillas: {g(list(itertools.combinations(old, 2))):.3f}')
    if old and new: print(f'  GT viejo vs GT+global  : {g(list(itertools.product(old, new))):.3f}')
    print('  ⇒ si "GT+global vs externos" NO baja respecto de "GT viejo vs externos", el bloque')
    print('    global afina la MISMA funcion; si baja, computa algo que la clase rawl2 no cubre.')

    # newest model vs the first external baseline
    if not (new and exn):
        return
    N = max(int(ei[:, [0, 2]].max()), int(rows[:, [0, 2]].max())) + 1
    deg = np.bincount(ei[:, 0], minlength=N) + np.bincount(ei[:, 2], minlength=N)
    cnt = Counter(zip(ei[:, 0].tolist(), ei[:, 1].tolist()))
    card = np.array([cnt.get((int(h), int(r)), 0) for h, r in zip(rows[:, 0], rows[:, 1])])
    a, b = new[-1], exn[0]
    both = F[a] & F[b]
    print(f'\n--- {a} vs {b}: fallo por CARDINALIDAD de (h,r) en train ---')
    print(f"{'#resp train':>12}{'n':>7}{'% test':>8}{'fallan ambos':>13}{'MRR ext':>9}{'MRR GT+g':>10}{'delta':>9}{'t':>7}")
    for lo, hi in C['card']:
        m = (card >= lo) & (card <= hi)
        if m.sum() < 20: continue
        d = (1 / rk[a][m]) - (1 / rk[b][m]); se = d.std(ddof=1) / sqrt(m.sum())
        lab = f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'
        print(f'{lab:>12}{m.sum():>7}{100*m.mean():>7.1f}%{100*both[m].mean():>12.1f}%'
              f'{np.mean(1/rk[b][m]):>9.3f}{np.mean(1/rk[a][m]):>10.3f}{d.mean():>+9.4f}{d.mean()/se if se else 0:>7.2f}')
    da = deg[rows[:, 2]]
    print(f'\n--- {a} vs {b}: fallos comunes vs aciertos comunes ---')
    for lab, m in (('fallan ambos', both), ('aciertan ambos', ~(F[a] | F[b]))):
        print(f'  {lab:15s} n={m.sum():6d}  grado resp mediana {np.median(da[m]):5.0f}  '
              f'grado fuente mediana {np.median(deg[rows[m, 0]]):5.0f}  #resp(h,r) mediana {np.median(card[m]):4.0f}')
    if args.distance:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        A = csr_matrix((np.ones(len(ei)), (ei[:, 0], ei[:, 2])), shape=(N, N))
        srcs = np.unique(rows[:, 0]); D = shortest_path(A, directed=False, unweighted=True, indices=srcs)
        pos = {s: i for i, s in enumerate(srcs)}
        dist = np.array([D[pos[h], t] for h, t in zip(rows[:, 0], rows[:, 2])])
        dist = np.where(np.isfinite(dist), dist, 99).astype(int)
        print(f'\n--- {a} vs {b}: por DISTANCIA h-t (99 = inalcanzable) ---')
        print(f"{'d':>6}{'n':>7}{'% test':>8}{'fallan ambos':>13}{'MRR ext':>9}{'MRR GT+g':>10}{'delta':>9}")
        for lo, hi in ((1, 2), (3, 3), (4, 4), (5, 6), (7, 98), (99, 99)):
            m = (dist >= lo) & (dist <= hi)
            if m.sum() < 20: continue
            d = (1 / rk[a][m]) - (1 / rk[b][m])
            print(f'{f"{lo}-{hi}":>6}{m.sum():>7}{100*m.mean():>7.1f}%{100*both[m].mean():>12.1f}%'
                  f'{np.mean(1/rk[b][m]):>9.3f}{np.mean(1/rk[a][m]):>10.3f}{d.mean():>+9.4f}')


if __name__ == '__main__':
    main()
