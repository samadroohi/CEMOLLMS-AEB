import numpy as np
from .base import BaseConformalPredictor

def _kth_higher_quantile(arr, alpha):
    arr = np.asarray(arr, dtype=float)
    m = arr.size
    if m == 0: return None
    k = int(np.ceil((1.0 - alpha) * (m + 1)))
    k = min(max(k, 1), m)  # clamp to [1, m]
    # kth (1-indexed) → index k-1 (0-indexed)
    idx = k - 1
    # linear-time selection
    part = np.partition(arr, idx)
    # tie-handling: 'higher' → take the smallest value ≥ part[idx]
    q = part[idx]
    return float(q)

class MulticlassConformalPredictor(BaseConformalPredictor):
    def __init__(self, mode: str = "mondrian", rare_shrink_tau: float = 0.0):
        """
        mode: "global" (one threshold) or "mondrian" (per-class threshold)
        rare_shrink_tau: >0 to shrink rare-class thresholds toward global
        """
        super().__init__()
        self.task_type = "multiclass_classification"
        assert mode in ("global", "mondrian")
        self.mode = mode
        self.rare_shrink_tau = float(rare_shrink_tau)
        self.q_hat = None  # float or dict[int]->float

    @staticmethod
    def _aggregate_probs(P):  # P: (K,) or (T,K)
        P = np.asarray(P, dtype=float)
        return P if P.ndim == 1 else P.mean(axis=0)

    def fit(self, y_true_calib, pred_calibration, prob_pred_calib, alpha):
        """
        y_true_calib: list/array of ints in [0..K-1] (or lists; first element used)
        prob_pred_calib: array/list length m; each (K,) or (T,K)
        """
        # Aggregate probs
        P_list = [self._aggregate_probs(p) for p in prob_pred_calib]  # each (K,)
        P = np.vstack(P_list)  # (m, K)

        # Flatten labels to ints
        ys = []
        for y in y_true_calib:
            ys.append(int(y[0] if isinstance(y, (list, tuple)) else y))
        ys = np.asarray(ys, dtype=int)

        # Scores: s_i = 1 - p_yi
        s = 1.0 - P[np.arange(P.shape[0]), ys]  # (m,)

        if self.mode == "global":
            q = _kth_higher_quantile(s, alpha)
            self.q_hat = 1.0 if q is None else q
            return self.q_hat

        # Mondrian per-class thresholds
        K = P.shape[1]
        qd = {}
        for c in range(K):
            mask = (ys == c)
            sc = s[mask]
            qc = _kth_higher_quantile(sc, alpha)
            qd[c] = 1.0 if qc is None else qc

        # Optional: shrink rare classes toward global (stabilizes when m_c small)
        if self.rare_shrink_tau > 0.0:
            q_global = _kth_higher_quantile(s, alpha)
            m_counts = np.bincount(ys, minlength=K)
            for c in range(K):
                mc = m_counts[c]
                lam = self.rare_shrink_tau / (mc + self.rare_shrink_tau)
                qd[c] = float(lam * q_global + (1 - lam) * qd[c])

        self.q_hat = qd
        return qd

    def predict(self, prob_pred_test, q_hat):
        """
        prob_pred_test: iterable of (K,) or (T,K)
        q_hat: float (global) or dict[int]->float (mondrian)
        Returns: list of lists of class indices
        """
        preds = []
        use_global = isinstance(q_hat, (float, int))
        for p in prob_pred_test:
            p = self._aggregate_probs(p)  # (K,)
            if use_global:
                thr = 1.0 - float(q_hat)
                pred_set = np.where(p >= thr)[0].tolist()
            else:
                pred_set = [c for c, pc in enumerate(p) if pc >= 1.0 - q_hat.get(c, 1.0)]
            preds.append(pred_set)
        return preds

    def get_conformal_results(self, true_labels, pred_test, prob_pred_test, q_hat):
        pred_sets = self.predict(prob_pred_test, q_hat)
        # coverage: all true labels must be in the set (handles multi-label)
        correct = 0
        for y, S in zip(true_labels, pred_sets):
            ys = y if isinstance(y, (list, tuple)) else [y]
            correct += set(int(t) for t in ys).issubset(set(S))
        coverage = correct / max(1, len(true_labels))
        avg_size = float(np.mean([len(S) for S in pred_sets])) if pred_sets else 0.0
        return pred_sets, coverage, avg_size, true_labels
