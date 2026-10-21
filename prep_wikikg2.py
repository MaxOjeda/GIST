"""Convert ogbl-wikikg2 (OGB numpy arrays) into a .pt file read by src/data_ogb.py.

Requires the `ogb` package (installed in the astarnet environment).
"""
import numpy as np, torch
from ogb.linkproppred import LinkPropPredDataset

d = LinkPropPredDataset(name='ogbl-wikikg2', root='/home/jreutter/datasets/ogb')
s = d.get_edge_split()
N = int(d[0]['num_nodes']); R = int(s['train']['relation'].max()) + 1
def hrt(sp):
    return torch.from_numpy(np.stack([sp['head'], sp['relation'], sp['tail']], 1)).long()
out = {'num_nodes': N, 'num_relation': R, 'train': hrt(s['train'])}
for k in ('valid', 'test'):
    out[k] = hrt(s[k])
    out[k + '_head_neg'] = torch.from_numpy(s[k]['head_neg']).long()
    out[k + '_tail_neg'] = torch.from_numpy(s[k]['tail_neg']).long()
print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in out.items()})
torch.save(out, '/home/jreutter/datasets/ogb/wikikg2.pt')
print('guardado /home/jreutter/datasets/ogb/wikikg2.pt')
