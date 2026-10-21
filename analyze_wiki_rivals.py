"""ogbl-wikikg2: which negatives score above the answer (reached by the search, unreached, ties).

Usage: python analyze_wiki_rivals.py dump.pt [astar_reach.pt]
"""
import sys, torch
R = torch.load(sys.argv[1])['rows']; rk = R[:, 3].double(); hit = R[:, 4].bool(); tail = R[:, 1] % 2 == 0
av, au, tie = R[:, 6].double(), R[:, 7].double(), R[:, 8].double()
A = torch.load(sys.argv[2])['rows'] if len(sys.argv) > 2 else None
for lab, m in (('todas', torch.ones_like(hit)), ('visitada', hit), ('visitada, cola', hit & tail), ('visitada, cabeza', hit & ~tail)):
    k = m & (rk > 1)
    print(f'{lab:17s} n={int(m.sum()):5d}  MRR {(1/rk[m]).mean():.4f}  rank>1: {100*k.double().mean()/m.double().mean():4.1f} %   '
          f'rivales por encima (media sobre rank>1): visitados {av[k].mean():6.2f}  NO visitados {au[k].mean():6.2f}  empates {tie[k].mean():5.2f}')
if A is not None:
    t = A[:, 3].bool(); both = t & hit; r = A[:, 2].double()
    print(f'\\nambos alcanzan (n={int(both.sum())}): A*Net MRR {(1/r[both]).mean():.4f}  GT {(1/rk[both]).mean():.4f}')
    for lab, m in (('A*Net top-1, GT no', both & (r <= 1) & (rk > 1)), ('GT top-1, A*Net no', both & (rk <= 1) & (r > 1))):
        print(f'  {lab:20s} n={int(m.sum()):5d}   GT: visitados encima {av[m].mean():.2f}  no visitados encima {au[m].mean():.2f}')
