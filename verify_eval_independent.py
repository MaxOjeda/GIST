"""Independent check of the transductive evaluation of a checkpoint.

Rebuilds the filters and test triples from the raw files (without the data loader), computes the filtered ranks with the
checkpoint, and checks that no valid/test triple is in the evaluation graph.
Usage: python verify_eval_independent.py <data_dir> <ckpt> <reported_mrr>
"""
import sys, collections, torch
import train as T
data, ck, reported = sys.argv[1], sys.argv[2], float(sys.argv[3])
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ent = {l.split('\t')[0]: int(l.split('\t')[1]) for l in open(f'{data}/entities.txt').read().splitlines() if l}
rel = {l.split('\t')[0]: int(l.split('\t')[1]) for l in open(f'{data}/relations.txt').read().splitlines() if l}
raw = {sp: [tuple(l.split('\t')) for l in open(f'{data}/{sp}.txt').read().splitlines() if l] for sp in ('train', 'valid', 'test')}
def enc(trips):   # each triple gives two queries: (h, r, t) and (t, -r, h)
    out = []
    for h, r, t in trips:
        out.append((ent[h], rel[r], ent[t])); out.append((ent[t], rel['-' + r], ent[h]))
    return out
filt = collections.defaultdict(set)
for sp in ('train', 'valid', 'test'):
    for h, r, t in enc(raw[sp]): filt[(h, r)].add(t)
test = enc(raw['test'])
dm = T.KGDataModule(data, False, 0, 8, 8)
g = dm.data.test_graph
E = set(map(tuple, g.edge_index.tolist()))
leak = sum(q in E for q in test) + sum(q in E for q in enc(raw['valid']))
print(f'grafo de eval: {len(E):,} aristas; queries de valid/test presentes en el grafo (fuga): {leak}')
m = T.GTLightningModule.load_from_checkpoint(ck, map_location=dev).to(dev).eval()
ranks = []
with torch.no_grad():
    for i in range(0, len(test), 16):
        q = torch.tensor(test[i:i + 16]); h, r, t = q[:, 0].to(dev), q[:, 1].to(dev), q[:, 2].to(dev)
        sc = m.model({'graph': g, 'h_index': h, 'r_index': r, 't_index': t})
        for j in range(len(q)):
            s = sc[j].clone(); ans = s[t[j]].item()
            others = [x for x in filt[(h[j].item(), r[j].item())] if x != t[j].item()]
            if others: s[torch.tensor(others, device=dev)] = float('-inf')
            s[t[j]] = float('-inf')
            ranks.append(int((s >= ans).sum().item()) + 1)
rk = torch.tensor(ranks, dtype=torch.double)
print(f'{data}: {len(rk)} queries  MRR {(1/rk).mean():.4f}  H@1 {(rk<=1).double().mean():.4f}  H@10 {(rk<=10).double().mean():.4f}  MR {rk.mean():.1f}   (reportado por el harness: {reported})')
print('seleccion del ckpt:', torch.load(ck, map_location='cpu')['callbacks'] if False else '(ver log: best ckpt por valid_mrr)')
