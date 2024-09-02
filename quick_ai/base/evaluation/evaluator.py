from functools import wraps
from typing import Dict, List
from math import exp
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np


def normalization(func, technique='rev_exp', t_param: float = 0.1):
    """ Maps [0, inf) domain into [0, 1), so that highest value becomes the lowest value """
    @wraps(func)
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        if technique == 'rev_exp':
            return 1 - (value / (value + exp(-t_param * value)))
        elif technique == 'rev_rev_exp':
            value = -value + 1
            return 1 - (value / (value + exp(-t_param * value)))
        else:
            raise NotImplementedError(f"""This type of normalization{
                                      technique}, was not implemented yet""")
    return wrapper


_TWO_ARG = {'accuracy': accuracy_score, 'mse': normalization(mean_squared_error), 'mae': normalization(mean_absolute_error),
            'r2': normalization(r2_score, technique='rev_rev_exp')}
_WITH_AVG = {'precision': precision_score,
             'recall': recall_score, 'f1': f1_score}
_ROC_METRICS = {'roc_auc': roc_auc_score}


class BaseEvaluator:
    def __init__(self, metrics: List[str], weights: List[int] = None, average: str = 'weighted', multi_class: str = 'raise'):
        """
        Initialize the EvaluationScore object with customizable evaluation metrics and weights.

        Args:
        metrics (List[int]): A list of metrics to be computed. Default is None, which uses common classification metrics.
        weights (List[int]): A list of weights for the metrics. Default is None, which assumes equal weights.
        average (str): The averaging method for metrics like precision, recall, and F1 score (binary, micro, macro, weighted).
        multi_class (str): The method to handle multi-class classification ('raise', 'ovr', 'ovo'). Default is 'raise'.
        """
        self.metrics = metrics if metrics else [
            'accuracy', 'precision', 'recall', 'f1']

        if weights and len(weights) != len(self.metrics):
            raise ValueError(
                "Length of weights must match the number of metrics")
        self.weights = weights if weights else [
            1.0 / len(self.metrics)] * len(self.metrics)
        assert sum(self.weights) == 1.0, "Weights must sum to 1"

        self.average = average
        self.multi_class = multi_class

    def __call__(self, y_true, y_pred, y_prob=None) -> float:
        """
        Calculate and return a single float score, which is the weighted average of the selected metrics.

        Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_prob [optional] (array-like): Probability estimates for positive class (needed for metrics like roc_auc).

        Returns:
        float: A single float score representing the weighted average of the computed metrics.
        """
        total_score = 0.0

        for metric, weight in zip(self.metrics, self.weights):
            score = self._proper_metric(y_true, y_pred, y_prob, metric)
            assert score >= 0 and score <= 1, f'Invalid value encounter in {
                metric} metric'
            total_score += weight * score
        return total_score

    def _proper_metric(self, y_true, y_pred, y_prob, metric):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        if metric in _TWO_ARG:
            score = _TWO_ARG[metric](y_true, y_pred)
        elif metric in _WITH_AVG:
            score = _WITH_AVG[metric](y_true, y_pred, average=self.average)
        elif metric in _ROC_METRICS:
            if y_prob is not None:
                score = _ROC_METRICS[metric](
                    y_true, y_prob, average=self.average, multi_class=self.multi_class)
            else:
                raise ValueError(
                    "ROC AUC requires probability scores (y_prob)")
        elif callable(metric):
            score = metric(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        return score

    def score(self, y_true, y_pred, y_prob=None) -> Dict[str, float]:
        results = {}

        for metric in self.metrics:
            score = self._proper_metric(y_true, y_pred, y_prob, metric)
            results[metric] = score
        return results


def get_evaluator() -> BaseEvaluator:
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    weights = [0.5, 0.1, 0.2, 0.2]
    return BaseEvaluator(metrics, weights)


if __name__ == "__main__":
    y_true_bin = [0, 1, 1, 0, 1]
    y_pred_bin = [0, 1, 1, 1, 1]

    evaluator_bin = get_evaluator()
    overall_score_bin = evaluator_bin(y_true_bin, y_pred_bin)
    print(f"Weighted average score (binary): {overall_score_bin}")
    y_true_multi = [0, 1, 2, 2, 0, 2, 2]
    y_pred_multi = [0, 1, 2, 1, 0, 2, 2]
    y_prob_multi = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.5, 0.3],
                    [0.3, 0.5, 0.2], [0.6, 0.2, 0.2], [0.1, 0.2, 0.7],
                    [0.2, 0.1, 0.7]]

    evaluator_multi = get_evaluator()
    evaluator_multi.metrics[0] = 'r2'
    overall_score_multi = evaluator_multi(
        y_true_multi, y_pred_multi, y_prob=y_prob_multi)
    print(f"Weighted average score (multi-class): {overall_score_multi}")
