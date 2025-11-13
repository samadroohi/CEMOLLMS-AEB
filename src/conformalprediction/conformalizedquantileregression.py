import numpy as np
from .base import BaseConformalPredictor

class ConformalizedQuantileRegressionPredictor(BaseConformalPredictor):
    """
    Conformalized Quantile Regression (split version).
    Supports symmetric (single Q) or asymmetric (Q_lo, Q_hi) calibration.
    """
    def __init__(self, asymmetric: bool = False, lower_key: str = "lower", upper_key: str = "upper"):
        super().__init__()
        self.task_type = "regression"
        self.asymmetric = asymmetric
        self.lower_key = lower_key
        self.upper_key = upper_key
        self.Q = None
        self.Q_lo = None
        self.Q_hi = None

    @staticmethod
    def _order_statistic(scores, rank):
        """Return the 1-indexed order statistic with defensive bounds."""
        arr = np.sort(np.asarray(scores, dtype=float))
        if arr.size == 0:
            raise ValueError("No calibration scores provided.")
        idx = min(max(rank, 1), arr.size) - 1
        return float(arr[idx])

    def _extract_bounds(self, probs):
        """
        Accepts dict {'lower': arr, 'upper': arr} (or 'q_lo'/'q_hi') or array (n,2).
        Returns q_lo, q_hi arrays with non-crossing enforced.
        """
        if isinstance(probs, dict):
            lower = probs.get(self.lower_key, probs.get("q_lo", None))
            upper = probs.get(self.upper_key, probs.get("q_hi", None))
            if lower is None or upper is None:
                raise ValueError(
                    f"probs must contain '{self.lower_key}'/'{self.upper_key}' "
                    "or 'q_lo'/'q_hi'."
                )
            q_lo = np.asarray(lower, dtype=float)
            q_hi = np.asarray(upper, dtype=float)
        else:
            arr = np.asarray(probs, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("Array probs must have shape (n, 2) -> [lower, upper].")
            q_lo, q_hi = arr[:, 0], arr[:, 1]

        # enforce non-crossing
        swap_mask = q_lo > q_hi
        if np.any(swap_mask):
            a = np.minimum(q_lo, q_hi)
            b = np.maximum(q_lo, q_hi)
            q_lo, q_hi = a, b
        return q_lo, q_hi

    def fit(self, y_true, y_pred, probs_calibration, alpha):
        """
        Calibrate CQR on a holdout calibration set.
        Returns (Q, Q) if symmetric, else (Q_lo, Q_hi).
        """
        y_true = np.asarray(y_true, dtype=float)
        q_lo, q_hi = self._extract_bounds(probs_calibration)
        if y_true.shape[0] != q_lo.shape[0] or y_true.shape[0] != q_hi.shape[0]:
            raise ValueError("Length mismatch between y_true and quantile predictions.")

        # Positive parts of one-sided violations
        lower_scores = np.maximum(q_lo - y_true, 0.0)
        upper_scores = np.maximum(y_true - q_hi, 0.0)
        symmetric_scores = np.maximum(lower_scores, upper_scores)

        m = symmetric_scores.shape[0]
        if m == 0:
            raise ValueError("No calibration points provided.")

        # Symmetric split-conformal threshold (Romano et al., 2019)
        k_sym = int(np.ceil((1.0 - alpha) * (m + 1)))
        self.Q = self._order_statistic(symmetric_scores, k_sym)

        if self.asymmetric:
            # One-sided thresholds use alpha/2 to control each tail
            one_sided_alpha = alpha / 2.0
            k_one_side = int(np.ceil((1.0 - one_sided_alpha) * (m + 1)))
            self.Q_lo = self._order_statistic(lower_scores, k_one_side)
            self.Q_hi = self._order_statistic(upper_scores, k_one_side)
            return (self.Q_lo, self.Q_hi)

        return (self.Q, self.Q)

    def predict(self, y_pred, probs_test, quantiles):
        q_lo, q_hi = self._extract_bounds(probs_test)
        Q_lo, Q_hi = quantiles
        lower = q_lo - Q_lo
        upper = q_hi + Q_hi
        return lower, upper

    def get_conformal_results(self, y_true, y_pred, probs_test, quantiles):
        y_true = np.asarray(y_true, dtype=float)
        lower, upper = self.predict(y_pred, probs_test, quantiles)
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        interval_size = np.mean(upper - lower)
        print("\nResults:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Average interval size: {interval_size:.4f}")
        return (lower, upper), coverage, interval_size, y_true
