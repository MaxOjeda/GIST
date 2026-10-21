"""GIST: a sparse graph transformer for knowledge graph completion.

At each layer a learned priority selects the entities to expand and the edges to propagate. Only the
reached entities keep a state. Messages flow along the selected edges with relational attention, and a
small set of tokens exchanges information across all reached entities.
"""

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

try:
    from src.sparse_state import (CSRGraph, global_to_slot, global_to_slot_flat,
                                  variadic_topks, ppr_bins)
except ImportError:
    from .sparse_state import (CSRGraph, global_to_slot, global_to_slot_flat,
                               variadic_topks, ppr_bins)


def _make_relation_linear(hidden_dim, num_relation):
    """Linear layer mapping the query to one vector per relation (weights 0 and bias 1 at start)."""
    lin = nn.Linear(hidden_dim, num_relation * hidden_dim)
    nn.init.zeros_(lin.weight)
    nn.init.ones_(lin.bias)
    return lin


class PrunedSparseAttentionLayer(nn.Module):
    """Relational attention restricted to the edges selected for each query.

    Each query has its own edge list (src, rel, dst) of shape (B, L). Weights use a sigmoid, so the
    aggregated message grows with the number of incoming edges.
    """

    def __init__(self, hidden_dim, num_heads, num_relation, drop, dependent=False):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.self_rel = num_relation
        R = num_relation + 1                     # real relations + self-loop
        self.num_rel_table = R
        self.dependent = dependent

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, R))          # attention bias per head and relation
        if dependent:
            self.rel_value = None
            self.relation_linear = _make_relation_linear(hidden_dim, R)   # relation vectors from the query
        else:
            self.rel_value = nn.Parameter(torch.ones(num_heads, R, self.head_dim))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, edges, q_emb=None, gate=None, boundary=None):
        """x: (B, S, D) states. edges: (src, rel, dst), each (B, L), as slot indices.
        gate: (B, L) priority of the sender of each edge, scales its message.
        boundary: (B, S, D) optional term added to the aggregated messages."""
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        src, rel, dst = edges
        L = src.size(1)

        h = self.norm1(x)
        q = self.to_q(h).view(B, N, H, hd)
        k = self.to_k(h).view(B, N, H, hd)
        v = self.to_v(h).view(B, N, H, hd)

        idx4 = lambda i: i.view(B, L, 1, 1).expand(B, L, H, hd)
        q_e = q.gather(1, idx4(dst))
        k_e = k.gather(1, idx4(src))
        v_e = v.gather(1, idx4(src))

        # index_select gives the same result as table[rel] with a faster backward
        rel_flat = rel.reshape(-1)
        if self.dependent:
            rel_emb = self.relation_linear(q_emb).view(B, self.num_rel_table, H, hd)
            g_e = rel_emb.gather(1, idx4(rel))                                   # (B, L, H, hd)
        else:
            g_e = self.rel_value.permute(1, 0, 2).contiguous() \
                      .index_select(0, rel_flat).view(B, L, H, hd)
        bias_e = self.rel_bias.t().contiguous().index_select(0, rel_flat).view(B, L, H)
        logit = (q_e * k_e).sum(-1) * self.scale + bias_e                       # (B, L, H)
        alpha = self.drop(torch.sigmoid(logit))

        msg = alpha.unsqueeze(-1) * (v_e * g_e)
        if gate is not None:
            msg = msg * gate.view(B, L, 1, 1)

        out = msg.new_zeros(B, N, H, hd)
        out.scatter_add_(1, idx4(dst), msg)
        out = out.reshape(B, N, D)
        if boundary is not None:
            out = out + boundary
        x = x + self.drop(self.to_out(out))
        x = x + self.ffn(self.norm2(x))
        return x


class GlobalInducedBlock(nn.Module):
    """Global inducing tokens.

    M tokens, conditioned on the query relation, attend to a pool of entities (pool); then every active
    entity attends to the tokens (broadcast). The output layer starts at zero, so the block initially
    leaves the states unchanged.
    """

    def __init__(self, hidden_dim, num_heads, num_tokens, drop=0.0):
        super().__init__()
        self.M = num_tokens
        self.tokens = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.tok_q = nn.Linear(hidden_dim, hidden_dim)
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.pool = nn.MultiheadAttention(hidden_dim, num_heads, dropout=drop, batch_first=True)
        self.bcast = nn.MultiheadAttention(hidden_dim, num_heads, dropout=drop, batch_first=True)
        self.ffn_t = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.out = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, valid, q_emb, pool_idx=None, pool_ok=None):
        """x: (B, S, D) states. valid: (B, S) mask. q_emb: (B, D).
        pool_idx / pool_ok: (B, P) pool slots and their mask; None pools all active slots."""
        B, S, D = x.shape
        xs = self.norm_x(x)
        if pool_idx is not None:
            src = xs.gather(1, pool_idx.unsqueeze(-1).expand(-1, -1, D))
            ok = pool_ok
        else:
            src, ok = xs, valid
        # keep at least one key per row (fully masked rows give NaN)
        kpm = ~ok
        kpm = kpm & ~(kpm.all(dim=1, keepdim=True) & (torch.arange(kpm.size(1), device=x.device) == 0))
        T = self.tokens.unsqueeze(0) + self.tok_q(q_emb).unsqueeze(1)          # (B, M, D)
        T = T + self.pool(T, src, src, key_padding_mask=kpm, need_weights=False)[0]
        T = T + self.ffn_t(self.norm_t(T))
        g = self.bcast(xs, T, T, need_weights=False)[0]                          # (B, S, D)
        return x + self.out(g) * valid.unsqueeze(-1).to(x.dtype)


class RelSpecificReadout(nn.Module):
    """Global readout added to every entity state.

    'global': W1 * (sum of all active states)
    'relq':   W1 * (sum over entities with an incoming edge of the query relation)
            + W2 * (sum over entities with an outgoing edge of the query relation)
    """

    def __init__(self, hidden_dim, mode='relq'):
        super().__init__()
        assert mode in ('relq', 'global')
        self.mode = mode
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False) if mode == 'relq' else None
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, valid, m_in=None, m_out=None):
        """x: (B, S, D). valid, m_in, m_out: (B, S) masks (m_in / m_out unused in 'global' mode)."""
        vf = valid.unsqueeze(-1).to(x.dtype)
        if self.mode == 'global':
            r = self.norm(self.w1((x * vf).sum(1)))                          # (B, D)
        else:
            s_in = (x * (m_in & valid).unsqueeze(-1).to(x.dtype)).sum(1)
            s_out = (x * (m_out & valid).unsqueeze(-1).to(x.dtype)).sum(1)
            r = self.norm(self.w1(s_in) + self.w2(s_out))
        return x + self.out(r).unsqueeze(1) * vf


NUM_DISTANCE_BINS = 10    # number of random-walk distance bins
GLOBAL_FROM_LAYER = 2     # first layer (0-based) with a global channel


class SparseStateGraphTransformer(nn.Module):
    """GIST model.

    Each query keeps the ids of its reached entities in `active` (B, S), sorted per row, and their states
    in `hidden` (B, S, d). Empty slots hold the id `num_nodes`, so the valid slots form a prefix.

    Per layer, K = node_ratio * |V| entities are expanded and L = degree_ratio * K * |E| / |V| edges are
    propagated; at most S = min(|V|, 6 L) entities are kept. `test_node_ratio` sets K at test time.

      dependent        relation vectors conditioned on the query relation
      global_channel   'tokens', 'sum', 'relation' (global readouts) or 'none'
      num_tokens       number of inducing tokens
      depth_encoding   add an embedding of the layer at which each entity is reached
      rw_distance      use the random-walk distance to the query entity (initial state, priority and score
                       of unreached entities)
      grad_ckpt        gradient checkpointing to save memory
    """

    def __init__(self, num_relation, num_layer=6, hidden_dim=32, num_heads=8, drop=0.0,
                 node_ratio=0.1, degree_ratio=1.0, test_node_ratio=None,
                 dependent=True, global_channel='tokens', num_tokens=32, depth_encoding=True,
                 rw_distance=False, grad_ckpt=False):
        super().__init__()
        assert global_channel in ('tokens', 'sum', 'relation', 'none')
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio
        # internal switches derived from rw_distance and global_channel
        self.indicator = 'ppr' if rw_distance else 'onehot'
        self.fallback = 'pprq' if rw_distance else 'scalar'
        self.prio_ppr = self.ppr_boundary = self.break_tie = rw_distance
        self.num_indicator_bin = NUM_DISTANCE_BINS
        self.read_global = global_channel == 'sum'
        # global readouts
        rmode = 'global' if self.read_global else 'relq'
        self.read_relq_blocks = nn.ModuleList([
            RelSpecificReadout(hidden_dim, mode=rmode)
            if (global_channel in ('sum', 'relation') and l >= GLOBAL_FROM_LAYER) else nn.Identity()
            for l in range(num_layer)])
        self._relq_cache = {}
        # inducing tokens
        self.global_blocks = nn.ModuleList([
            GlobalInducedBlock(hidden_dim, num_heads, num_tokens, drop)
            if (global_channel == 'tokens' and l >= GLOBAL_FROM_LAYER) else nn.Identity()
            for l in range(num_layer)])
        # depth encoding
        self.hop_pe = depth_encoding
        if depth_encoding:
            self.hop_emb = nn.Embedding(num_layer + 1, hidden_dim)
            nn.init.zeros_(self.hop_emb.weight)
        # score of unreached entities per distance bin and query relation
        if rw_distance:
            self.fallback_q = nn.Linear(hidden_dim, NUM_DISTANCE_BINS, bias=False)
            nn.init.zeros_(self.fallback_q.weight)
        self.self_rel = num_relation          # relation id of the self-loops
        self.grad_ckpt = grad_ckpt
        if rw_distance:
            # embedding of the random-walk distance to the query entity
            self.distance = nn.Embedding(NUM_DISTANCE_BINS, hidden_dim)
            nn.init.zeros_(self.distance.weight)
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        self.layers = nn.ModuleList([
            PrunedSparseAttentionLayer(hidden_dim, num_heads, num_relation, drop, dependent=dependent)
            for _ in range(num_layer)])
        # shared by the priority and the final score
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self.prio_g = nn.Linear(hidden_dim * 2, hidden_dim)
        self.unvisited_score = nn.Parameter(torch.zeros(1))

    def search_budget(self, num_nodes, num_edges):
        """Return (K, L, S) for a graph with the given number of nodes and edges."""
        ratio = self.test_node_ratio if (not self.training and self.test_node_ratio) else self.node_ratio
        K = max(1, round(ratio * num_nodes))
        L = int(self.degree_ratio * K * num_edges / num_nodes)
        return K, L, min(num_nodes, 6 * L)

    def _relq_masks(self, graph, dev):
        """(R, N) masks of the nodes with an outgoing / incoming edge of each relation, cached per graph."""
        key = (graph.num_nodes, int(graph.edge_index.shape[0]))
        if key not in self._relq_cache:
            if len(self._relq_cache) > 4:
                self._relq_cache.clear()
            ei = graph.edge_index.to(dev)
            h, r, t = ei[:, 0], ei[:, 1], ei[:, 2]
            R = int(r.max().item()) + 1
            N = graph.num_nodes
            out_m = torch.zeros(R, N, dtype=torch.bool, device=dev)
            in_m = torch.zeros(R, N, dtype=torch.bool, device=dev)
            out_m[r, h] = True
            in_m[r, t] = True
            self._relq_cache[key] = (out_m, in_m)
        return self._relq_cache[key]

    def _graph_dev(self, graph, dev):
        """Edges with self-loops and their CSR on `dev`, cached per graph."""
        ei = graph.edge_index
        sig = (graph.num_nodes, int(ei.shape[0]), tuple(ei[0].tolist()), tuple(ei[-1].tolist()), str(dev))
        cache = self.__dict__.setdefault('_gcache', {})
        if sig not in cache:
            if len(cache) >= 4:
                cache.clear()
            N = graph.num_nodes
            e = ei.to(dev)
            loop = torch.arange(N, device=dev)
            e_full = torch.cat([e, torch.stack([loop, torch.full_like(loop, self.self_rel), loop], 1)])
            cache[sig] = dict(ei=e, ei_full=e_full, csr=CSRGraph(e_full, N, dev))
        return cache[sig]

    def _ppr_clean_spmm(self, gc, N, src, gm, alpha=0.8, num_iteration=20):
        """Personalized PageRank from each query entity, without the edges of the training queries."""
        e = gc['ei']
        P = gc.get('ppr')
        if P is None:
            s, d = e[:, 0], e[:, 2]
            order = torch.argsort(d * N + s)                         # rows = destination
            crow = torch.zeros(N + 1, dtype=torch.long, device=e.device)
            crow[1:] = torch.cumsum(torch.bincount(d, minlength=N), 0)
            P = gc['ppr'] = dict(order=order, col=s[order].contiguous(), crow=crow,
                                 outdeg=torch.bincount(s, minlength=N).float())
        if gm is not None:
            keep = gm.to(e.device).bool()
            outdeg = P['outdeg'].index_add(0, e[:, 0][~keep], -torch.ones(int((~keep).sum()), device=e.device))
            w = (1.0 / outdeg.clamp(min=1))[P['col']] * keep[P['order']].float()
        else:
            w = (1.0 / P['outdeg'].clamp(min=1))[P['col']]
        A = torch.sparse_csr_tensor(P['crow'], P['col'], w, (N, N))
        B = src.numel()
        init = torch.zeros(N, B, device=e.device)
        init[src, torch.arange(B, device=e.device)] = 1.0
        ppr = init
        for _ in range(num_iteration):
            ppr = torch.sparse.mm(A, ppr) * alpha + init * (1 - alpha)
        return ppr.t().contiguous()                              # (B, N)

    def _boundary(self, nodes, bins, B, S, N, dev):
        """Initial state of new entities: zero, or the embedding of their distance bin."""
        if self.indicator != 'ppr':
            return torch.zeros(B, S, self.hidden_dim, device=dev)
        idx = bins.gather(1, nodes.clamp(max=N - 1)).long()
        w = self.distance.weight
        return F.one_hot(idx, w.size(0)).to(w.dtype) @ w

    def _priority(self, x, q_emb):
        """Priority in [0, 1] of each slot."""
        return torch.sigmoid(self._prio_logit(x, q_emb))

    def _ppr_logits(self, B, q_emb, dev):
        """Priority logits of an unreached entity per distance bin (B, nbins) and of the zero state (B, 1)."""
        dw = self.distance.weight
        tab = self._prio_logit(dw.unsqueeze(0).expand(B, -1, -1), q_emb)
        zero = self._prio_logit(torch.zeros(B, 1, self.hidden_dim, device=dev), q_emb)
        return tab, zero

    def _prio_logit(self, x, q_emb):
        """Priority before the sigmoid: MLP(LN(x * W[x; q]))."""
        W = self.prio_g.weight                       # (D, 2D)
        d = x.size(-1)
        g = torch.nn.functional.linear(x, W[:, :d], self.prio_g.bias)
        g = g + torch.nn.functional.linear(q_emb, W[:, d:]).unsqueeze(1)
        return self.readout(self.norm_out(x * g)).squeeze(-1)

    def forward(self, batched_data):
        graph = batched_data['graph']
        h_index, r_index = batched_data['h_index'], batched_data['r_index']
        N, B = graph.num_nodes, h_index.size(0)
        dev = h_index.device
        K, L, S = self.search_budget(N, int(graph.edge_index.shape[0]))

        # edges of the training queries are masked out with keep_csr
        gc = self._graph_dev(graph, dev)
        csr = gc['csr']
        gm = batched_data.get('graph_mask')
        if gm is not None:
            keep = torch.cat([gm.to(dev), torch.ones(N, dtype=torch.bool, device=dev)])   # self-loops always kept
            keep_csr = keep[csr.order]
        else:
            keep_csr = None

        q_emb = self.query_emb(r_index)
        ar = torch.arange(B, device=dev)
        bins = None
        if self.indicator == 'ppr' or self.fallback == 'pprq' or self.prio_ppr:
            # distance bins (B, N), computed once per batch
            ppr = self._ppr_clean_spmm(gc, N, h_index, batched_data.get('graph_mask'))
            bins = ppr_bins(ppr, N, self.num_indicator_bin)
            del ppr
        SENT = N                                     # id of empty slots
        # start from the query entity
        active = torch.full((B, 1), SENT, dtype=torch.long, device=dev)
        active[:, 0] = h_index
        hidden = self._boundary(active, bins, B, 1, N, dev)
        slot_h, _ = global_to_slot(active, h_index.unsqueeze(1))
        hidden[ar, slot_h.squeeze(1)] = q_emb + (self.hop_emb.weight[0] if self.hop_pe else 0.0)
        prio = torch.zeros(B, 1, device=dev)
        prio[ar, slot_h.squeeze(1)] = 1.0

        for li, layer in enumerate(self.layers):
            valid = active != SENT
            W = active.size(1)
            # (1) top-K active entities by priority (stable sort for deterministic ties)
            p = prio.masked_fill(~valid, float('-inf'))
            k_slot = p.sort(dim=1, descending=True, stable=True).indices[:, :min(K, W)]   # (B, K)
            k_nodes = active.gather(1, k_slot)
            k_ok = valid.gather(1, k_slot)

            base = self._priority(torch.zeros(B, 1, self.hidden_dim, device=dev), q_emb)
            base_tab = torch.sigmoid(self._ppr_logits(B, q_emb, dev)[0]) if self.prio_ppr else None

            # (2) outgoing edges of the selected entities
            flat = k_nodes.reshape(-1).clamp(max=N - 1)
            deg = csr.degree[flat] * k_ok.reshape(-1)
            total = int(deg.sum())
            Kc = k_nodes.size(1)
            owner = torch.repeat_interleave(torch.arange(B * Kc, device=dev), deg)
            offs = torch.cumsum(deg, 0) - deg
            eidx = csr.indptr[flat][owner] + (torch.arange(total, device=dev) - offs[owner])
            qid = owner.div(Kc, rounding_mode='floor')
            if keep_csr is not None:
                m = keep_csr[eidx]
                eidx, qid = eidx[m], qid[m]
            e_src, e_dst, e_rel = csr.src[eidx], csr.dst[eidx], csr.rel[eidx]

            # (3) top-L edges per query, ranked by the priority of their destination
            d_slot, d_found = global_to_slot_flat(active, e_dst, qid, N)
            d_prio = torch.where(d_found, prio.reshape(-1)[qid * W + d_slot],
                                 base_tab[qid, bins[qid, e_dst].long()] if base_tab is not None
                                 else base.reshape(-1)[qid])
            size = torch.bincount(qid, minlength=B)
            ks = size.clamp(max=L)
            _, sel = variadic_topks(d_prio, size, ks, break_tie=self.break_tie)

            nsel = int(ks.sum())
            sel_q = torch.repeat_interleave(torch.arange(B, device=dev), ks)
            pos_q = torch.arange(nsel, device=dev) - (ks.cumsum(0) - ks).repeat_interleave(ks)
            s_src = torch.zeros(B, L, dtype=torch.long, device=dev)
            s_dst = torch.zeros(B, L, dtype=torch.long, device=dev)
            s_rel = torch.full((B, L), self.self_rel, dtype=torch.long, device=dev)
            s_ok = torch.zeros(B, L, dtype=torch.bool, device=dev)
            s_src[sel_q, pos_q] = e_src[sel]
            s_dst[sel_q, pos_q] = e_dst[sel]
            s_rel[sel_q, pos_q] = e_rel[sel]
            s_ok[sel_q, pos_q] = True

            # (4) add the new destinations to the active set, keeping it sorted
            merged = torch.cat([active, torch.where(s_ok, s_dst, torch.full_like(s_dst, SENT))], 1)
            merged, _ = merged.sort(dim=1)
            dup = torch.zeros_like(merged, dtype=torch.bool)
            dup[:, 1:] = merged[:, 1:] == merged[:, :-1]
            merged = merged.masked_fill(dup, SENT)
            merged, _ = merged.sort(dim=1)
            new_active = merged[:, :S]
            w_new = int((new_active != SENT).sum(1).max().clamp(min=1))     # valid slots are a prefix
            new_active = new_active[:, :w_new]
            Wn = new_active.size(1)
            # move states to the new slots (dropped ones go to an extra slot)
            old_slot, old_found = global_to_slot(new_active, active)
            keep = old_found & (active != SENT)
            nb_base = self._boundary(new_active, bins, B, Wn, N, dev)
            nb = nb_base
            if self.hop_pe:
                # new entities get the encoding of the current depth
                nb = nb + self.hop_emb.weight[li + 1]
            nh = torch.cat([nb, hidden.new_zeros(B, 1, self.hidden_dim)], 1)
            np_ = prio.new_zeros(B, Wn + 1)
            idx = old_slot.masked_fill(~keep, Wn)
            nh.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
                        hidden * keep.unsqueeze(-1))
            np_.scatter_(1, idx, prio * keep)
            active, hidden, prio = new_active, nh[:, :Wn], np_[:, :Wn]

            # (5) relational attention along the selected edges
            a_src, _ = global_to_slot(active, s_src)
            a_dst, _ = global_to_slot(active, s_dst)
            gate = prio.gather(1, a_src) * s_ok
            bnd = None
            if self.ppr_boundary:
                # distance embedding added at every layer (query embedding at the source)
                s_h = global_to_slot(active, h_index.unsqueeze(1))[0]
                bnd = nb_base.scatter(1, s_h.view(B, 1, 1).expand(B, 1, self.hidden_dim), q_emb.unsqueeze(1))
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                hidden = torch.utils.checkpoint.checkpoint(
                    layer, hidden, (a_src, s_rel, a_dst), q_emb, gate, bnd, use_reentrant=False)
            else:
                hidden = layer(hidden, (a_src, s_rel, a_dst), q_emb=q_emb, gate=gate, boundary=bnd)

            # (6) inducing tokens, pooling the entities expanded at this layer
            gblock = self.global_blocks[li]
            if not isinstance(gblock, nn.Identity):
                valid_now = active != SENT
                p_idx, p_found = global_to_slot(active, k_nodes)
                p_ok = k_ok & p_found
                p_idx = p_idx.masked_fill(~p_ok, 0)
                if self.grad_ckpt and self.training and torch.is_grad_enabled():
                    hidden = torch.utils.checkpoint.checkpoint(
                        gblock, hidden, valid_now, q_emb, p_idx, p_ok, use_reentrant=False)
                else:
                    hidden = gblock(hidden, valid_now, q_emb, p_idx, p_ok)
            # (6') or a global readout
            rblock = self.read_relq_blocks[li]
            if not isinstance(rblock, nn.Identity):
                valid_now = active != SENT
                if self.read_global:
                    m_in = m_out = valid_now
                else:
                    out_m, in_m = self._relq_masks(graph, dev)
                    a_ids = active.clamp(max=N - 1)
                    m_in = in_m[r_index].gather(1, a_ids) & valid_now      # incoming edge of the query relation
                    m_out = out_m[r_index].gather(1, a_ids) & valid_now    # outgoing edge of the query relation
                if self.grad_ckpt and self.training and torch.is_grad_enabled():
                    hidden = torch.utils.checkpoint.checkpoint(
                        rblock, hidden, valid_now, m_in, m_out, use_reentrant=False)
                else:
                    hidden = rblock(hidden, valid_now, m_in, m_out)
            # (7) update priorities
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                prio = torch.utils.checkpoint.checkpoint(
                    self._priority, hidden, q_emb, use_reentrant=False)
            else:
                prio = self._priority(hidden, q_emb)

        # scores: reached entities from their state, others from the fallback
        sc = self.readout(self.norm_out(hidden)).squeeze(-1)
        v = active != SENT
        idx = active.masked_fill(~v, N)                      # invalid slots -> extra column
        z = sc.new_zeros(B, N + 1)
        scored = z.scatter(1, idx, torch.where(v, sc, torch.zeros_like(sc)))
        cover = z.scatter(1, idx, v.to(sc.dtype))
        fb = self.unvisited_score.expand(B, N)
        bins_l = bins.long() if bins is not None else None
        if self.fallback == 'pprq':
            fb = fb + self.fallback_q(q_emb).gather(1, bins_l)           # (B, nbins) -> (B, N)
        if self.prio_ppr:
            # unreached entities are also scored by the priority function (zero at start)
            tab, zero = self._ppr_logits(B, q_emb, dev)
            fb = fb + (tab - zero).gather(1, bins_l)
        # reached entities, used to sample hard negatives
        self._last_visited = (cover[:, :N] > 0).detach()
        return torch.where(cover[:, :N] > 0, scored[:, :N], fb)
