"""Paired comparison on the same ogbl-wikikg2 test items between a train.py rank dump and A*Net."""
import sys, torch
O = torch.load(sys.argv[1])['rows']; A = torch.load('astarnet_wiki_reach_ranks.pt')['rows']
ref = torch.load('wiki_reach_vppr_ranks.pt')['rows']          # item order used for A*Net
assert torch.equal(O[:, :3], ref[:, :3]), 'los items no coinciden con los de A*Net'
o, a = O[:, 3].double(), A[:, 2].double(); tail = O[:, 1] % 2 == 0
def line(lab, m):
    d = 1 / o[m] - 1 / a[m]
    return (f'{lab:22s} n={int(m.sum()):5d}  GT {(1/o[m]).mean():.4f}  A*Net {(1/a[m]).mean():.4f}  '
            f'Δ {d.mean():+.4f} ± {d.std()/m.sum()**0.5:.4f}  (t={d.mean()/(d.std()/m.sum()**0.5):+.1f})')
print('MRR (protocolo OGB, 500 negativos, mismos items):')
for lab, m in (('TODOS', torch.ones_like(tail)), ('cola', tail), ('cabeza', ~tail)): print('  ' + line(lab, m))
for k in (1, 3, 10):
    print(f'  Hits@{k:<2d}  GT {(o <= k).double().mean():.4f}  A*Net {(a <= k).double().mean():.4f}')
print(f'  MR      GT {o.mean():.1f}  A*Net {a.mean():.1f}')
if O.shape[1] > 4:
    hit, at = O[:, 4].bool(), A[:, 3].bool()
    print(f'alcance: GT visita la respuesta {100*hit.double().mean():.1f} %  A*Net la toca {100*at.double().mean():.1f} %')
    for lab, m in (('ambos alcanzan', hit & at), ('solo A*Net', ~hit & at), ('solo GT', hit & ~at), ('ninguno', ~hit & ~at)):
        print('  ' + line(lab, m))
w = (1 / o > 1 / a).sum().item(); l = (1 / o < 1 / a).sum().item()
print(f'queries donde gana GT {w}, gana A*Net {l}, empate {len(o) - w - l}')
