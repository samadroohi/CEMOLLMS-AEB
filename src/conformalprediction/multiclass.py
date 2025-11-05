import numpy as np
from typing import Iterable, List, Sequence, Union

from .base import BaseConformalPredictor


def _kth_higher_quantile(arr, alpha):
    arr = np.asarray(arr, dtype=float)
    m = arr.size
    if m == 0:
        return None
    k = int(np.ceil((1.0 - alpha) * (m + 1)))
    k = min(max(k, 1), m)  # clamp to [1, m]
    idx = k - 1
    part = np.partition(arr, idx)
    q = part[idx]
    return float(q)


class MulticlassConformalPredictor(BaseConformalPredictor):
    def __init__(
        self,
        classes: Sequence[str],
        mode: str = "hybrid",
        rare_shrink_tau: float = 10.0,
    ):
        """Conformal predictor for multilabel multiclass tasks.

        Args:
            classes: Ordered iterable representing the class vocabulary.
            mode: "global", "mondrian", or "hybrid" (per-class with shrink).
            rare_shrink_tau: τ parameter controlling shrinkage toward global.
        """
        super().__init__()
        self.task_type = "multiclass_classification"

        if not classes:
            raise ValueError("'classes' must be a non-empty sequence")
        mode = mode.lower()
        if mode not in {"global", "mondrian", "hybrid"}:
            raise ValueError("mode must be one of {'global','mondrian','hybrid'}")

        self.classes: List[str] = list(classes)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.mode = mode
        self.rare_shrink_tau = max(0.0, float(rare_shrink_tau))
        self.q_hat_global: float = 1.0
        self.q_hat_per_class: Union[None, np.ndarray] = None
        self.q_hat: Union[None, float, np.ndarray] = None
        self._empty_prob_calibration = 0
        self._empty_prob_test = 0

    @staticmethod
    def _aggregate_probs(P) -> np.ndarray:
        """Aggregate probs from generation steps to a single class distribution."""
        P = np.asarray(P, dtype=float)
        if P.ndim == 1:
            return P
        if P.ndim == 2 and P.shape[0] > 0:
            return P.mean(axis=0)
        if P.size == 0:
            raise ValueError("Empty probability tensor encountered during aggregation")
        raise ValueError(f"Unexpected probability tensor shape: {P.shape}")

    def _labels_to_indices(self, labels: Union[str, Iterable[str]]) -> List[int]:
        if isinstance(labels, (list, tuple, set)):
            idxs = []
            for label in labels:
                if label is None:
                    continue
                label_str = str(label)
                if label_str in self.class_to_idx:
                    idxs.append(self.class_to_idx[label_str])
            return sorted(set(idxs))
        label_str = str(labels)
        return [self.class_to_idx[label_str]] if label_str in self.class_to_idx else []

    def _to_q_array(self, q_hat) -> np.ndarray:
        if isinstance(q_hat, dict):
            arr = np.full(self.num_classes, self.q_hat_global, dtype=float)
            for key, value in q_hat.items():
                if isinstance(key, str) and key in self.class_to_idx:
                    arr[self.class_to_idx[key]] = float(value)
                elif isinstance(key, (int, np.integer)) and 0 <= key < self.num_classes:
                    arr[int(key)] = float(value)
            return arr
        arr = np.asarray(q_hat, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.num_classes:
            raise ValueError(
                f"Per-class q_hat must have length {self.num_classes}, got shape {arr.shape}"
            )
        return arr

    def fit(self, y_true_calib, pred_calibration, prob_pred_calib, alpha):
        scores_global: List[float] = []
        scores_per_class: List[List[float]] = [list() for _ in range(self.num_classes)]
        dropped_empty_probs = 0

        for labels, probs in zip(y_true_calib, prob_pred_calib):
            true_indices = self._labels_to_indices(labels)
            if not true_indices:
                continue  # skip examples without recognised labels

            if probs is None:
                dropped_empty_probs += 1
                continue

            p_vec = self._aggregate_probs(probs)
            if p_vec.size == 0:
                dropped_empty_probs += 1
                continue
            if p_vec.shape[0] != self.num_classes:
                raise ValueError(
                    f"Probability vector length {p_vec.shape[0]} does not match number of classes {self.num_classes}"
                )

            true_probs = p_vec[true_indices]
            score = 1.0 - float(np.min(true_probs))
            scores_global.append(score)
            for idx in true_indices:
                scores_per_class[idx].append(score)

        q_global = _kth_higher_quantile(scores_global, alpha)
        if q_global is None:
            q_global = 1.0
        self.q_hat_global = float(q_global)

        if self.mode == "global":
            self.q_hat_per_class = None
            self.q_hat = self.q_hat_global
            self._empty_prob_calibration = dropped_empty_probs
            return self.q_hat_global

        q_per_class = np.full(self.num_classes, self.q_hat_global, dtype=float)
        for cls_idx in range(self.num_classes):
            cls_scores = scores_per_class[cls_idx]
            q_cls = _kth_higher_quantile(cls_scores, alpha)
            if q_cls is None:
                q_cls = self.q_hat_global
            else:
                q_cls = float(q_cls)

            if self.mode == "hybrid":
                if self.rare_shrink_tau <= 0.0:
                    q_cls = self.q_hat_global if len(cls_scores) == 0 else q_cls
                else:
                    m_c = len(cls_scores)
                    lam = self.rare_shrink_tau / (m_c + self.rare_shrink_tau)
                    q_cls = float(lam * self.q_hat_global + (1.0 - lam) * q_cls)
            # Mondrian mode retains per-class quantiles without shrinkage; if a class has
            # no calibration support, the earlier fallback keeps the global estimate.

            q_per_class[cls_idx] = q_cls

        self.q_hat_per_class = q_per_class
        self.q_hat = q_per_class
        self._empty_prob_calibration = dropped_empty_probs
        return q_per_class

    def predict(self, prob_pred_test, q_hat):
        preds = []
        empty_prob_test = 0
        if isinstance(q_hat, (float, int, np.floating, np.integer)):
            thr_global = 1.0 - float(q_hat)
            thr_per_class = None
        else:
            q_array = self._to_q_array(q_hat)
            thr_global = None
            thr_per_class = 1.0 - q_array

        for probs in prob_pred_test:
            if probs is None:
                preds.append([])
                empty_prob_test += 1
                continue
            p_vec = self._aggregate_probs(probs)
            if p_vec.size == 0:
                empty_prob_test += 1
                p_vec = np.full(self.num_classes, 1.0 / self.num_classes, dtype=float)
            if p_vec.shape[0] != self.num_classes:
                raise ValueError(
                    f"Probability vector length {p_vec.shape[0]} does not match number of classes {self.num_classes}"
                )

            if thr_per_class is None:
                pred_set = np.where(p_vec >= thr_global)[0].tolist()
            else:
                pred_set = [idx for idx, prob in enumerate(p_vec) if prob >= thr_per_class[idx]]

            if not pred_set:
                pred_set = [int(np.argmax(p_vec))]
            preds.append(pred_set)

        self._empty_prob_test = empty_prob_test
        return preds

    def get_conformal_results(self, true_labels, pred_test, prob_pred_test, q_hat):
        pred_sets = self.predict(prob_pred_test, q_hat)

        covered = 0
        total = 0
        for labels, pred_set in zip(true_labels, pred_sets):
            idxs = self._labels_to_indices(labels)
            if not idxs:
                continue
            total += 1
            if set(idxs).issubset(set(pred_set)):
                covered += 1

        coverage = covered / max(1, total)
        avg_size = float(np.mean([len(S) for S in pred_sets])) if pred_sets else 0.0
        return pred_sets, coverage, avg_size, true_labels
