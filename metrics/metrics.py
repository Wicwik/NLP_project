import numpy as np
import torch

from .utils import normalize_squad, qa_metrics

from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix

# some are reused from https://github.com/AkariAsai/ATTEMPT/blob/main/attempt/metrics/metrics.py

# compute a f1 score from a list of strings
class F1ScoreWithInvalid(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def binary_reverse(targets):
        return ["0" if target == "1" else "1" for target in targets]

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        preds, targets = np.asarray(preds), np.asarray(targets)

        invalid_idx_mask = np.logical_and(preds != "0", preds != "1")
        preds[invalid_idx_mask] = self.binary_reverse(targets[invalid_idx_mask])

        preds, targets = torch.tensor(preds.astype(np.int32)), torch.tensor(targets.astype(np.int32))

        conf_mat = confusion_matrix(preds, targets, task="binary")
        self.tn += conf_mat[0,0]
        self.fp += conf_mat[0,1]
        self.fn += conf_mat[1,0]
        self.tp += conf_mat[1,1]

    def compute(self):
        if self.tp + self.fp == 0:
            return torch.tensor(0.0)

        if self.tp + self.fn == 0:
            return torch.tensor(0.0)

        precision = self.tp/(self.tp + self.fp)
        recall = self.tp/(self.tp + self.fn)

        if (precision * recall) == 0:
            return torch.tensor(0.0)
        
        if (precision + recall) == 0:
            return torch.tensor(0.0)

        return 100 * 2*(precision * recall) / (precision + recall)
    
class Accuraccy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        assert len(preds) == len(targets)
        
        preds = np.asarray(preds)
        targets = np.asarray(targets)

        self.correct += np.sum(preds == targets)
        self.total += preds.size

    def compute(self):
        return 100 * self.correct.float() / self.total
    

class SquadMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")


    def update(self, preds, targets):
        if type(targets[0]) is list:
            targets = [[normalize_squad(t) for t in u] for u in targets]
        else:
            targets = [[normalize_squad(u)] for u in targets]

        preds = [normalize_squad(p) for p in preds]

        self.targets += targets
        self.preds += preds

    def compute(self):
        return qa_metrics(self.targets, self.preds)