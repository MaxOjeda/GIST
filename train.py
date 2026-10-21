"""Train and evaluate GIST on knowledge graph completion."""

import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model import SparseStateGraphTransformer
from src.data import TransductiveKnowledgeGraph, InductiveKnowledgeGraph
from src.data_ogb import WikiKG2, OGBEvalSet, ogb_ranks
from src.data_wd5m import Wikidata5M
from src.metric import MRMetric, MRRMetric, HitsMetric


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class KGDataModule(pl.LightningDataModule):
    def __init__(self, data_path, inductive, num_workers, batch_size, test_batch_size,
                 eval_subsample=0, eval_seed=1024):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        # random subset of the OGB evaluation set
        self.eval_subsample = eval_subsample
        self.eval_seed = eval_seed
        # ogbl-wikikg2 uses the OGB protocol (500 fixed negatives per query)
        self.is_ogb = 'wikikg2' in data_path
        if self.is_ogb:
            self.data = WikiKG2()
        elif 'wikidata5m' in data_path.lower():
            # Wikidata5M: filtered ranking against all entities
            self.data = Wikidata5M(eval_seed=eval_seed)
        else:
            cls = InductiveKnowledgeGraph if inductive else TransductiveKnowledgeGraph
            self.data = cls(data_path)
        self.num_relation = self.data.num_relation

    def train_dataloader(self):
        # forward relations only; the collate function adds the inverse queries
        triplets = self.data.train_triplets.clone()
        triplets = triplets[triplets[:, 1] % 2 == 0]
        return DataLoader(triplets, shuffle=True, collate_fn=self.data.train_collate_fn,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def _ogb_loader(self, split):
        n = getattr(self.data, split + '_triplets').size(0)
        return DataLoader(OGBEvalSet(n, self.eval_subsample, self.eval_seed), shuffle=False,
                          collate_fn=self.data.ogb_collate_fn(split),
                          batch_size=self.test_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.is_ogb:
            return self._ogb_loader('valid')
        return DataLoader(self.data.valid_triplets.clone(), shuffle=False,
                          collate_fn=self.data.valid_collate_fn,
                          batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.is_ogb:
            return self._ogb_loader('test')
        return DataLoader(self.data.test_triplets.clone(), shuffle=False,
                          collate_fn=self.data.test_collate_fn,
                          batch_size=self.test_batch_size, num_workers=self.num_workers)


# ----------------------------------------------------------------------------
# Lightning module
# ----------------------------------------------------------------------------
class GTLightningModule(pl.LightningModule):
    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 learning_rate, weight_decay, dependent=False,
                 node_ratio=0.1, degree_ratio=1.0, test_node_ratio=None,
                 global_channel='tokens', num_tokens=32, depth_encoding=False, rw_distance=False,
                 grad_ckpt=False, remove_one_hop=False,
                 num_negative=32, neg_visited=0, adversarial_temperature=0.5,
                 lr_milestones=(10, 15), no_decay_rel=False, dump_ranks=''):
        super().__init__()
        self.save_hyperparameters()
        self.remove_one_hop = remove_one_hop
        # optional dump of the per-query test ranks
        self.dump_ranks = dump_ranks
        self._rank_rows = []
        self.model = SparseStateGraphTransformer(
            num_relation, num_layer, hidden_dim, num_heads, drop,
            node_ratio=node_ratio, degree_ratio=degree_ratio, test_node_ratio=test_node_ratio,
            dependent=dependent, global_channel=global_channel, num_tokens=num_tokens,
            depth_encoding=depth_encoding, rw_distance=rw_distance, grad_ckpt=grad_ckpt)
        self.mr = MRMetric()
        self.mrr = MRRMetric()
        self.h1 = HitsMetric(topk=1)
        self.h3 = HitsMetric(topk=3)
        self.h10 = HitsMetric(topk=10)

    def remove_edge(self, batched_data):
        """Mask out the edges that make a training query trivial and store it as graph_mask (E,).

        By default the query edge (h, r, t) and its inverse are removed; with remove_one_hop, all edges
        between h and t.
        """
        h, r, t = batched_data['h_index'], batched_data['r_index'], batched_data['t_index']
        graph = batched_data['graph']
        if batched_data.get('ogb') and not self.remove_one_hop:
            # large graphs: binary search on the sorted edge keys
            batched_data['graph_mask'] = self.trainer.datamodule.data.query_edge_mask(h, r, t, h.device)
            return batched_data
        ei = graph.edge_index.to(h.device)
        h_rm = torch.cat([h, t], 0)
        t_rm = torch.cat([t, h], 0)
        if self.remove_one_hop:
            # match by (src, dst) pair: any relation between h and t
            encode2 = lambda a, c: c + a * graph.num_nodes
            src_hash = encode2(ei[:, 0], ei[:, 2])
            tgt_hash = encode2(h_rm, t_rm)
        else:
            rev_r = torch.where(r % 2 == 0, r + 1, r - 1)
            r_rm = torch.cat([r, rev_r], 0)
            encode = lambda a, b, c: c + (a + b * graph.num_nodes) * graph.num_nodes
            src_hash = encode(ei[:, 0], ei[:, 1], ei[:, 2])
            tgt_hash = encode(h_rm, r_rm, t_rm)
        mask = ~torch.isin(src_hash, tgt_hash)
        batched_data['graph_mask'] = mask
        return batched_data

    def _uniform_negatives(self, filter_mask, num_neg):
        """Uniform negatives that are not known answers (filter_mask), with replacement."""
        cand = 1.0 - filter_mask.float()          # 1 for valid candidates
        empty = cand.sum(-1) == 0                 # queries without candidates
        if empty.any():
            cand = cand.clone()
            cand[empty] = 1.0
        return torch.multinomial(cand, num_neg, replacement=True)        # (B, k)

    def _visited_negatives(self, score, batched_data, k):
        """Sample up to k negatives per query among the entities reached by the search (without replacement).

        Returns (idx, valid), each (B, k); valid is False for padding and for known answers.
        """
        with torch.no_grad():
            vis = self.model._last_visited                                     # (B, N) reached entities
            ogb = bool(batched_data.get('ogb'))
            if not ogb:
                vis = vis & ~batched_data['filter_mask'].bool()                # exclude known answers
            key = torch.rand(score.shape, device=score.device) * vis           # 0 for unreached entities
            keyv, idx = key.topk(min(k, score.size(1)), dim=1)                 # k random reached entities
            valid = keyv > 0
            if ogb:
                d = self.trainer.datamodule.data
                bad = d._is_true(batched_data['h_index'], batched_data['r_index'], idx)
                valid &= ~bad
        return idx, valid

    def _bce_loss(self, score, t_index, neg_index, neg_valid=None):
        """BCE of the answer against the given negatives, with self-adversarial weights.

        Negatives with neg_valid=False get weight 0.
        """
        pos = score.gather(1, t_index.unsqueeze(1))                      # (B, 1)
        neg = score.gather(1, neg_index)                                 # (B, k)
        pred = torch.cat([pos, neg], dim=1)                              # (B, 1+k)
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        neg_weight = torch.ones_like(pred)
        if self.hparams.adversarial_temperature > 0:
            with torch.no_grad():
                logit = pred[:, 1:] / self.hparams.adversarial_temperature
                if neg_valid is not None:
                    logit = logit.masked_fill(~neg_valid, float('-inf'))
                neg_weight[:, 1:] = F.softmax(logit, dim=-1)
        elif neg_valid is not None:
            neg_weight[:, 1:] = neg_valid.to(pred.dtype) / neg_valid.sum(1, keepdim=True).clamp(min=1)
        else:
            neg_weight[:, 1:] = 1 / neg.size(1)
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        return loss.mean()

    def training_step(self, batched_data, batch_idx):
        batched_data = self.remove_edge(batched_data)
        score = self.model(batched_data)                          # (B, N)
        t_index = batched_data['t_index']
        k = self.hparams.num_negative
        if batched_data.get('ogb'):
            # large graphs: strict negatives by rejection sampling
            neg = self.trainer.datamodule.data.sample_strict_negatives(
                batched_data['h_index'].to(score.device), batched_data['r_index'].to(score.device), k)
        else:
            neg = self._uniform_negatives(batched_data['filter_mask'], k)
        valid = None
        if self.hparams.neg_visited > 0:
            # uniform negatives plus negatives among the reached entities
            vidx, vok = self._visited_negatives(score, batched_data, self.hparams.neg_visited)
            self.log('neg_vis', vok.float().sum(1).mean(), prog_bar=True)
            valid = torch.cat([torch.ones_like(neg, dtype=torch.bool), vok], 1)
            neg = torch.cat([neg, vidx], 1)
        loss = self._bce_loss(score, t_index, neg, valid)
        self.log('train_loss', loss, prog_bar=True)
        self.log('mem_gb', torch.cuda.max_memory_allocated() / 1024**3, prog_bar=True)
        return loss

    def _eval_step(self, batched_data, collect=False):
        score = self.model(batched_data)
        # non-finite scores would give rank 1 to every query, so fail loudly
        if not torch.isfinite(score).all():
            raise FloatingPointError('non-finite scores: the model diverged.')
        if 'cand' in batched_data:
            # OGB protocol: rank against the given negatives
            sc = score.gather(1, batched_data['cand'])                   # (B, 1+k)
            ranks = ogb_ranks(sc)
            for m in (self.mr, self.mrr, self.h1, self.h3, self.h10):
                m.update(ranks)
            if collect and self.dump_ranks:
                # also store whether the answer and each negative were reached by the search,
                # and the negatives above the answer (reached / unreached) and ties
                hit = self.model._last_visited.gather(1, batched_data['cand'])
                above = sc[:, 1:] > sc[:, :1]
                self._rank_rows.append(torch.stack([
                    batched_data['h_index'], batched_data['r_index'],
                    batched_data['t_index'], ranks.long(),
                    hit[:, 0].long(), hit[:, 1:].sum(1),
                    (above & hit[:, 1:]).sum(1), (above & ~hit[:, 1:]).sum(1),
                    (sc[:, 1:] == sc[:, :1]).sum(1)], dim=1).detach().cpu())
            return
        answer = score.gather(1, batched_data['t_index'].unsqueeze(1))
        filt = batched_data['filter_mask'].bool()
        ranks = torch.sum((score >= answer) & (~filt), dim=1) + 1
        for m in (self.mr, self.mrr, self.h1, self.h3, self.h10):
            m.update(ranks)
        if collect and self.dump_ranks:
            self._rank_rows.append(torch.stack([
                batched_data['h_index'], batched_data['r_index'],
                batched_data['t_index'], ranks], dim=1).detach().cpu())

    def validation_step(self, batched_data, batch_idx):
        self._eval_step(batched_data)

    def test_step(self, batched_data, batch_idx):
        self._eval_step(batched_data, collect=True)

    def _save_ranks(self):
        if not (self.dump_ranks and self._rank_rows):
            return
        rows = torch.cat(self._rank_rows)          # (Q, 4) = (h, r, t, rank); more columns for OGB
        ei = self.trainer.datamodule.data.edge_index.cpu()
        torch.save({'rows': rows,
                    # the graph is not stored for large datasets
                    'edge_index': ei if ei.shape[0] < 5_000_000 else None,
                    'num_relation': self.trainer.datamodule.data.num_relation},
                   self.dump_ranks)
        print(f'[dump_ranks] {self.dump_ranks}  Q={len(rows)}', flush=True)
        self._rank_rows = []

    def _log_epoch(self, split):
        vals = {f'{split}_mr': self.mr.compute(), f'{split}_mrr': self.mrr.compute(),
                f'{split}_hits1': self.h1.compute(), f'{split}_hits3': self.h3.compute(),
                f'{split}_hits10': self.h10.compute()}
        for m in (self.mr, self.mrr, self.h1, self.h3, self.h10):
            m.reset()
        for k, v in vals.items():
            self.log(k, v, prog_bar=k.endswith(('mrr', 'hits10')), sync_dist=True)

    def validation_epoch_end(self, outputs):
        self._log_epoch('valid')

    def test_epoch_end(self, outputs):
        self._save_ranks()
        self._log_epoch('test')

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight', 'norm']
        if self.hparams.no_decay_rel:
            # rarely seen relations would be shrunk to zero by weight decay
            no_decay = no_decay + ['query_emb', 'relation_linear']
        groups = [
            {'params': [p for n, p in self.named_parameters()
                        if any(d in n for d in no_decay) and p.requires_grad],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.named_parameters()
                        if not any(d in n for d in no_decay) and p.requires_grad],
             'weight_decay': self.hparams.weight_decay},
        ]
        opt = torch.optim.Adam(groups, lr=self.hparams.learning_rate)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, list(self.hparams.lr_milestones), 0.1)
        return [opt], [{'scheduler': sched, 'interval': 'epoch'}]


INDUCTIVE_DATASETS = {'fb15k-237', 'wn18rr', 'nell-995'}


def is_inductive(data_path):
    name = os.path.basename(os.path.normpath(data_path))
    return '_v' in name and name.split('_v')[0] in INDUCTIVE_DATASETS


def main():
    p = ArgumentParser()
    p.add_argument('--data_path', type=str, required=True)

    g = p.add_argument_group('model')
    g.add_argument('--num_layer', type=int, default=6)
    g.add_argument('--hidden_dim', type=int, default=32)
    g.add_argument('--num_heads', type=int, default=8)
    g.add_argument('--drop', type=float, default=0.1)
    g.add_argument('--dependent', action='store_true',
                   help='Relation vectors conditioned on the query relation.')
    g.add_argument('--node_ratio', type=float, default=0.1,
                   help='Entities expanded per layer: K = node_ratio * |V|.')
    g.add_argument('--degree_ratio', type=float, default=1.0,
                   help='Edges propagated per layer: L = degree_ratio * K * |E| / |V|.')
    g.add_argument('--test_node_ratio', type=float, default=None,
                   help='node_ratio at evaluation time (default: same as training).')
    g.add_argument('--global_channel', choices=['tokens', 'sum', 'relation', 'none'], default='tokens',
                   help='Global channel: inducing tokens, sum readout, relation-specific readout, or none.')
    g.add_argument('--num_tokens', type=int, default=32, help='Number of inducing tokens.')
    g.add_argument('--depth_encoding', action='store_true',
                   help='Add an embedding of the layer at which each entity is reached.')
    g.add_argument('--rw_distance', action='store_true',
                   help='Use the random-walk distance to the query entity.')
    g.add_argument('--grad_ckpt', action='store_true',
                   help='Gradient checkpointing (lower memory, slower training).')

    g = p.add_argument_group('training')
    g.add_argument('--remove_one_hop', action='store_true',
                   help='During training, remove all edges between the query entities h and t.')
    g.add_argument('--num_negative', type=int, default=32,
                   help='Uniform negatives per positive.')
    g.add_argument('--neg_visited', type=int, default=0,
                   help='Additional negatives per query sampled among the entities reached by the search.')
    g.add_argument('--adversarial_temperature', type=float, default=0.5,
                   help='Temperature of the self-adversarial negative weights (<= 0: uniform weights).')
    g.add_argument('--learning_rate', type=float, default=5e-3)
    g.add_argument('--weight_decay', type=float, default=1e-4)
    g.add_argument('--no_decay_rel', action='store_true',
                   help='No weight decay on relation-specific parameters.')
    g.add_argument('--lr_milestones', type=int, nargs='+', default=[10, 15],
                   help='Epochs at which the learning rate is divided by 10.')
    g.add_argument('--batch_size', type=int, default=16)
    g.add_argument('--max_epochs', type=int, default=20)
    g.add_argument('--limit_train_batches', type=float, default=1.0,
                   help='Training batches per epoch (fraction or number).')
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--devices', type=int, default=1)
    g.add_argument('--num_workers', type=int, default=8)
    g.add_argument('--checkpoint_save_path', type=str, default='./experiments/gt')
    g.add_argument('--resume_from', type=str, default='',
                   help='Checkpoint to resume training from.')
    g.add_argument('--init_ckpt', type=str, default='',
                   help='Initialize the weights from this checkpoint (new optimizer).')

    g = p.add_argument_group('evaluation')
    g.add_argument('--test_batch_size', type=int, default=16)
    g.add_argument('--eval_only', action='store_true',
                   help='Only evaluate --eval_ckpt.')
    g.add_argument('--eval_ckpt', type=str, default='',
                   help='Checkpoint evaluated with --eval_only.')
    g.add_argument('--eval_split', choices=['test', 'valid'], default='test',
                   help='Split evaluated with --eval_only.')
    g.add_argument('--eval_subsample', type=int, default=0,
                   help='ogbl-wikikg2: evaluate a random subset of this many items.')
    g.add_argument('--eval_seed', type=int, default=1024,
                   help='Seed of the evaluation subset.')
    g.add_argument('--dump_ranks', type=str, default='',
                   help='File where the per-query test ranks are saved.')
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)

    dm = KGDataModule(args.data_path, is_inductive(args.data_path), args.num_workers,
                      args.batch_size, args.test_batch_size,
                      eval_subsample=args.eval_subsample, eval_seed=args.eval_seed)
    model = GTLightningModule(dm.num_relation, args.num_layer, args.hidden_dim,
                              args.num_heads, args.drop, args.learning_rate, args.weight_decay,
                              dependent=args.dependent, node_ratio=args.node_ratio,
                              degree_ratio=args.degree_ratio, test_node_ratio=args.test_node_ratio,
                              global_channel=args.global_channel, num_tokens=args.num_tokens,
                              depth_encoding=args.depth_encoding, rw_distance=args.rw_distance,
                              grad_ckpt=args.grad_ckpt, remove_one_hop=args.remove_one_hop,
                              num_negative=args.num_negative, neg_visited=args.neg_visited,
                              adversarial_temperature=args.adversarial_temperature,
                              lr_milestones=args.lr_milestones, no_decay_rel=args.no_decay_rel,
                              dump_ranks=args.dump_ranks)

    if args.init_ckpt:
        # weights only (new optimizer and schedule)
        sd = torch.load(args.init_ckpt, map_location='cpu')['state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[init_ckpt] {args.init_ckpt}  missing={len(missing)}  unexpected={len(unexpected)}', flush=True)
    ckpt = ModelCheckpoint(dirpath=args.checkpoint_save_path, monitor='valid_mrr',
                           mode='max', save_top_k=1, every_n_epochs=1, verbose=True,
                           save_last=True)
    trainer = pl.Trainer(accelerator='gpu', devices=args.devices, max_epochs=args.max_epochs,
                         callbacks=[ckpt], limit_train_batches=args.limit_train_batches,
                         num_sanity_val_steps=0, check_val_every_n_epoch=1, logger=False)
    if args.eval_only:
        if args.eval_split == 'valid':
            dm.test_dataloader = dm.val_dataloader      # evaluate the validation split
        print(f'[eval_only] {args.eval_ckpt}', flush=True)
        trainer.test(model, datamodule=dm, ckpt_path=args.eval_ckpt)
        return
    trainer.fit(model, datamodule=dm, ckpt_path=(args.resume_from or None))
    print(f"[best ckpt] {ckpt.best_model_path}  valid_mrr={ckpt.best_model_score}")
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path)


if __name__ == '__main__':
    main()
