import numpy as np
import sklearn.metrics
import torch

from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix

# some are reused from https://github.com/AkariAsai/ATTEMPT/blob/main/attempt/metrics/metrics.py

def f1_score_with_invalid(predictions, targets) -> dict:
    """Computes F1 score,  with any prediction != 0 or 1 is counted as incorrect.
    Args:
      targets: list of targets, either 0 or 1
      predictions: list of predictions, any integer value
    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    def binary_reverse(labels):
        return ['0' if label == '1' else '1' for label in labels]
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions.
    invalid_idx_mask = np.logical_and(predictions != '0', predictions != '1')
    # For any prediction != 0 or 1, we set the prediction to the opposite of its corresponding target. NOT a mistake, they are using strings because the model returns strings of 1 and 0
    predictions[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask])
    targets = targets.astype(np.int32)
    predictions = predictions.astype(np.int32)
    return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}

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