"""Ranking metrics accumulated over batches."""
import torch
from torchmetrics import Metric


class MRMetric(Metric):
    """Mean rank."""
    is_differentiable: None
    higher_is_better: False
    full_state_update: True

    def __init__(self):
        super().__init__()
        self.add_state('rank_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, ranks):
        self.total += ranks.size(0)
        self.rank_sum += ranks.sum()

    def compute(self):
        return self.rank_sum / self.total
    

class MRRMetric(Metric):
    """Mean reciprocal rank."""
    is_differentiable: None
    higher_is_better: True
    full_state_update: True

    def __init__(self):
        super().__init__()
        self.add_state('rank_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, ranks):
        self.total += ranks.size(0)
        self.rank_sum += (1./ ranks).sum()

    def compute(self):
        return self.rank_sum / self.total
    

class HitsMetric(Metric):
    """Fraction of ranks <= topk."""
    is_differentiable: None
    higher_is_better: True
    full_state_update: True

    def __init__(self, topk=1):
        super().__init__()
        self.add_state('rank_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.topk = topk

    def update(self, ranks):
        self.total += ranks.size(0)
        self.rank_sum += (ranks <= self.topk).sum()

    def compute(self):
        return self.rank_sum / self.total
    
