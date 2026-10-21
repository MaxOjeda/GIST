"""Extract test metrics, time per epoch, peak memory and settings from training logs.

Supports train.py logs (PyTorch Lightning) and torchdrug logs (A*Net / NBFNet, log.txt).
Usage: python collect_run_stats.py <log> [<log> ...]   (prints one JSON dict per log)
"""
import re, sys, statistics as st


def _read(p):
    with open(p, 'r', errors='replace') as f:
        return f.read().replace('\r', '\n')


def _hms(s):
    parts = [int(x) for x in s.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def ours(path):
    txt = _read(path)
    out = {'log': path, 'kind': 'ours'}
    cmd = re.findall(r'python train\.py[^\n]*', txt)
    if cmd:
        c = cmd[-1]
        for k in ('data_path', 'devices', 'batch_size', 'test_batch_size', 'max_epochs', 'limit_train_batches',
                  'neg_visited', 'num_negative', 'top_nodes', 'edge_budget', 'eval_ckpt', 'eval_split', 'accum'):
            m = re.search(r'--%s (\S+)' % k, c)
            if m:
                out[k] = m.group(1)
        out['grad_ckpt'] = '--grad_ckpt' in c
    # time per epoch: last progress bar at 100 % of each epoch
    ep_t = {}
    for m in re.finditer(r'Epoch (\d+): 100%\|[^\n]*?\| (\d+)/(\d+) \[([\d:]+)<', txt):
        ep_t[int(m.group(1))] = _hms(m.group(4))
    if ep_t:
        out['epochs_run'] = len(ep_t)
        out['epoch_time_min_median'] = round(st.median(ep_t.values()) / 60, 1)
    mem = [float(x) for x in re.findall(r'mem_gb=([\d.]+)', txt)]
    if mem:
        out['mem_gb_max'] = max(mem)
    b = re.findall(r'\[best ckpt\] (\S+)\s+valid_mrr=([\d.]+)', txt)
    if b:
        out['best_ckpt'] = b[-1][0]; out['best_valid_mrr'] = round(float(b[-1][1]), 4)
        e = re.search(r'epoch=(\d+)', b[-1][0])
        if e:
            out['best_epoch'] = int(e.group(1))
    tests = {}
    for k in ('test_mrr', 'test_mr', 'test_hits1', 'test_hits3', 'test_hits10'):
        v = re.findall(r'\s%s\s+([\d.]+)' % k, txt)
        if v:
            tests[k] = [float(x) for x in v]
    if tests:
        out['test_blocks'] = len(tests.get('test_mrr', []))
        out['test'] = {k: v[-1] for k, v in tests.items()}
        if out['test_blocks'] > 1:
            out['test_all'] = tests
    return out


def torchdrug(path):
    txt = _read(path)
    out = {'log': path, 'kind': 'torchdrug'}
    m = re.search(r"'dataset': \{'class': '([^']+)'", txt); out['dataset'] = m.group(1) if m else None
    m = re.search(r"'class': '(AStarNet|NBFNet|NeuralBellmanFordNetwork)'", txt); out['model'] = m.group(1) if m else None
    m = re.search(r"'batch_size': (\d+)", txt); out['batch_size'] = int(m.group(1)) if m else None
    m = re.search(r"'gpus': \[([^\]]*)\]", txt); out['gpus'] = len(m.group(1).split(',')) if m and m.group(1).strip() else 0
    m = re.search(r"'num_epoch': (\d+)", txt); out['num_epoch'] = int(m.group(1)) if m else None
    m = re.search(r"'batch_per_epoch': (\d+)", txt); out['batch_per_epoch'] = int(m.group(1)) if m else None
    m = re.search(r"'num_negative': (\d+)", txt); out['num_negative'] = int(m.group(1)) if m else None
    unit = {'secs': 1 / 3600, 'mins': 1 / 60, 'hours': 1.0}
    dur = [float(x) * unit[u] for x, u in re.findall(r'duration: ([\d.]+) (secs|mins|hours)', txt)]
    if dur:
        out['epochs_run'] = len(dur); out['epoch_time_min_median'] = round(st.median(dur) * 60, 1)
    mem = [float(x) for x in re.findall(r'max GPU memory: ([\d.]+) MiB', txt)]
    if mem:
        out['mem_gb_max'] = round(max(mem) / 1024, 1)
    # evaluation blocks in order
    blocks = re.split(r'Evaluate on (valid|test)', txt)
    evals = []
    for i in range(1, len(blocks), 2):
        sp, body = blocks[i], blocks[i + 1]
        d = {}
        for k in ('hits@1', 'hits@3', 'hits@10', 'mr', 'mrr'):
            mm = re.search(r'\s%s: ([\d.]+)' % re.escape(k), body)
            if mm:
                d[k] = float(mm.group(1))
        if d:
            evals.append((sp, d))
    valids = [d['mrr'] for sp, d in evals if sp == 'valid' and 'mrr' in d]
    tests = [d for sp, d in evals if sp == 'test']
    if valids:
        out['valid_curve'] = valids
    if tests:
        out['test'] = tests[-1]
        # final evaluation: valid (best epoch) and test
        out['final_valid'] = [d for sp, d in evals if sp == 'valid'][-1].get('mrr')
    return out


if __name__ == '__main__':
    import json
    for p in sys.argv[1:]:
        f = torchdrug if p.endswith('log.txt') or '/experiments_astarnet/' in p else ours
        print(json.dumps(f(p), default=str))
