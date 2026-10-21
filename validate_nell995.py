"""Sanity checks of the transductive NELL-995 split in data/nell-995 (duplicates, leakage, filters, statistics)."""
import collections, torch
from src.data import TransductiveKnowledgeGraph
d = TransductiveKnowledgeGraph('./data/nell-995')
raw = lambda sp: [tuple(l.rstrip('\n').split('\t')) for l in open(f'./data/nell-995/{sp}.txt') if l.strip()]
tr, va, te = raw('train'), raw('valid'), raw('test')
print(f'entidades {d.num_entity:,}  relaciones {d.num_relation//2} (+inversas = {d.num_relation})  train {len(tr):,}  valid {len(va):,}  test {len(te):,}')
print(f'tensores del loader (con inversas): train {tuple(d.train_triplets.shape)}  valid {tuple(d.valid_triplets.shape)}  test {tuple(d.test_triplets.shape)}')
str_ = set(tr)
print(f'duplicados en train: {len(tr) - len(str_)}')
for name, sp in (('valid', va), ('test', te)):
    leak = sum(t in str_ for t in sp); inv = sum((t[2], t[1], t[0]) in str_ for t in sp)
    print(f'{name}: tripletas IDENTICAS en train (fuga) {leak}   su inversa (t,r,h) esta en train {inv}')
print(f'valid ∩ test: {len(set(va) & set(te))}')
ent_tr = collections.Counter([t[0] for t in tr] + [t[2] for t in tr])
for name, sp in (('valid', va), ('test', te)):
    iso = sum(1 for h, r, t in sp if ent_tr[h] == 0 or ent_tr[t] == 0)
    print(f'{name}: tripletas con alguna entidad SIN aristas en train (incontestables en una direccion) {iso} ({100*iso/len(sp):.1f} %)')
rel_tr = set(t[1] for t in tr); print(f'relaciones de valid/test ausentes en train: {len(set(t[1] for t in va + te) - rel_tr)}')
# training triples with another direct h-t edge (relevant for --remove_one_hop)
pairs = collections.Counter()
for h, r, t in tr: pairs[frozenset((h, t))] += 1
alt = sum(1 for h, r, t in tr if pairs[frozenset((h, t))] > 1)
print(f'train con OTRA arista directa h-t: {100*alt/len(tr):.1f} %   (FB15k-237 31.6 %, WN18RR ~1 %)')
alt_te = sum(1 for h, r, t in te if pairs[frozenset((h, t))] > 0)
print(f'test con alguna arista directa h-t en train: {100*alt_te/len(te):.1f} %')
deg = torch.tensor(sorted(ent_tr.values())).float()
print(f'grado en train: medio {deg.mean():.1f}  mediana {deg.median():.0f}  p99 {deg.quantile(.99):.0f}  max {deg.max():.0f}')
# filters: valid/test use train + valid + test, train uses train only
te_enc = d.test_triplets.tolist()
print(f'filtro de test contiene la respuesta de cada tripleta de test: {all(t in d.test_filters[(h, r)] for h, r, t in te_enc)}')
print(f'filtro de TRAIN contiene alguna respuesta de test (seria fuga): {sum(t in d.train_filters.get((h, r), ()) for h, r, t in te_enc)}')
