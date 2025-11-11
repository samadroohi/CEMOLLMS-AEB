import numpy as np
from typing import Iterable, List, Optional, Sequence, Union

from .base import BaseConformalPredictor


def _kth_higher_quantile(arr: Iterable[float], alpha: float) -> Optional[float]:
    values = np.asarray(list(arr), dtype=float)
    m = values.size
    if m == 0:
        return None
    k = int(np.ceil((1.0 - alpha) * (m + 1)))
    k = min(max(k, 1), m)
    idx = k - 1
    part = np.partition(values, idx)
    return float(part[idx])


class OrdinalClassificationConformalPredictor(BaseConformalPredictor):
    def __init__(
        self,
        classes: Sequence[str],
        mode: str = "global",
        rare_shrink_tau: Union[float, str] = 5.0,
        auto_tau_grid: Optional[Sequence[float]] = None,
        auto_holdout: float = 0.4,
        auto_seed: Optional[int] = None,
        auto_coverage_margin: float = 0.0,
        auto_size_tolerance: float = 1e-6,
    ) -> None:
        super().__init__()
        self.task_type = "ordinal_classification"

        if not classes:
            raise ValueError("'classes' must be a non-empty sequence for ordinal CP")
        self.classes: List[str] = list(classes)
        self.num_classes = len(self.classes)

        mode_norm = str(mode or "").strip().lower()
        if mode_norm not in {"global", "mondrian", "hybrid"}:
            raise ValueError("mode must be one of {'global','mondrian','hybrid'}")
        self.mode = mode_norm
        self._auto_shrink = isinstance(rare_shrink_tau, str) and rare_shrink_tau.strip().lower() == "auto"
        if self._auto_shrink:
            self.rare_shrink_tau = None
        else:
            self.rare_shrink_tau = max(0.0, float(rare_shrink_tau))
        self.tuned_tau: Optional[float] = None
        self.tuned_tau_diagnostics: Optional[dict] = None
        self.auto_tau_grid = None if auto_tau_grid is None else [float(x) for x in auto_tau_grid if float(x) > 0]
        if self.auto_tau_grid:
            self.auto_tau_grid = sorted(set(self.auto_tau_grid))
        self.auto_holdout = float(auto_holdout)
        self.auto_seed = auto_seed
        self.auto_coverage_margin = float(auto_coverage_margin)
        self.auto_size_tolerance = float(auto_size_tolerance)
        if self._auto_shrink and not self.auto_tau_grid:
            self.auto_tau_grid = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

        self.q_hat_global: float = 1.0
        self.q_hat_per_class: Optional[np.ndarray] = None
        self.q_hat: Union[None, float, np.ndarray] = None
        self._empty_prob_calibration = 0
        self._empty_prob_test = 0

    @staticmethod
    def _extract_index(label_entry) -> Optional[int]:
        if isinstance(label_entry, (list, tuple)) and len(label_entry) >= 2:
            return int(label_entry[1])
        if isinstance(label_entry, (int, np.integer)):
            return int(label_entry)
        return None

    @staticmethod
    def _make_contiguous(indices: List[int]) -> List[int]:
        if not indices:
            return []
        lo = min(indices)
        hi = max(indices)
        return list(range(lo, hi + 1))

    @staticmethod
    def _to_prob_vector(probs) -> Optional[np.ndarray]:
        if probs is None:
            return None
        arr = np.asarray(probs, dtype=float)
        if arr.ndim == 0:
            return None
        if arr.ndim > 1:
            arr = arr.reshape(arr.shape[-1])
        return arr

    def fit(self, y_true, y_pred, probs_calibration, alpha):
        scores_global: List[float] = []
        scores_per_class: List[List[float]] = [list() for _ in range(self.num_classes)]
        dropped_empty_probs = 0

        for label_entry, probs in zip(y_true, probs_calibration):
            target_idx = self._extract_index(label_entry)
            if target_idx is None or target_idx < 0 or target_idx >= self.num_classes:
                continue

            prob_vec = self._to_prob_vector(probs)
            if prob_vec is None:
                dropped_empty_probs += 1
                continue
            if prob_vec.shape[0] != self.num_classes:
                raise ValueError(
                    f"Probability vector length {prob_vec.shape[0]} does not match number of classes {self.num_classes}"
                )

            true_prob = float(prob_vec[target_idx])
            score = 1.0 - true_prob
            scores_global.append(score)
            scores_per_class[target_idx].append(score)

        q_global = _kth_higher_quantile(scores_global, alpha)
        if q_global is None:
            q_global = 1.0
        self.q_hat_global = float(q_global)

        if self.mode == "global":
            self.q_hat_per_class = None
            self.q_hat = self.q_hat_global
            self._empty_prob_calibration = dropped_empty_probs
            return self.q_hat_global

        tuned_tau = None
        if self.mode == "hybrid" and self._auto_shrink:
            tuned_tau = self._tune_shrink_tau(
                y_true,
                y_pred,
                probs_calibration,
                alpha,
            )
            self.rare_shrink_tau = tuned_tau
            self.tuned_tau = tuned_tau

        q_per_class = np.full(self.num_classes, self.q_hat_global, dtype=float)
        for cls_idx, cls_scores in enumerate(scores_per_class):
            q_cls = _kth_higher_quantile(cls_scores, alpha)
            if q_cls is None:
                q_cls = self.q_hat_global
            else:
                q_cls = float(q_cls)

            if self.mode == "hybrid":
                tau = float(self.rare_shrink_tau or 0.0)
                if tau <= 0.0:
                    q_cls = self.q_hat_global if len(cls_scores) == 0 else q_cls
                else:
                    m_c = len(cls_scores)
                    lam = tau / (m_c + tau)
                    q_cls = float(lam * self.q_hat_global + (1.0 - lam) * q_cls)

            q_per_class[cls_idx] = q_cls

        self.q_hat_per_class = q_per_class
        self.q_hat = q_per_class
        self._empty_prob_calibration = dropped_empty_probs
        return q_per_class

    def predict(self, y_pred, probs_test, q_hat):
        preds: List[List[int]] = []
        empty_prob_test = 0

        if isinstance(q_hat, (float, int, np.floating, np.integer)):
            thr_global = 1.0 - float(q_hat)
            thr_per_class = None
        else:
            q_array = np.asarray(q_hat, dtype=float)
            if q_array.shape != (self.num_classes,):
                raise ValueError(
                    f"Per-class q_hat must have shape ({self.num_classes},), got {q_array.shape}"
                )
            thr_global = None
            thr_per_class = 1.0 - q_array

        for probs in probs_test:
            prob_vec = self._to_prob_vector(probs)
            if prob_vec is None:
                preds.append([])
                empty_prob_test += 1
                continue
            if prob_vec.shape[0] != self.num_classes:
                raise ValueError(
                    f"Probability vector length {prob_vec.shape[0]} does not match number of classes {self.num_classes}"
                )

            if thr_per_class is None:
                pred_set = [idx for idx, prob in enumerate(prob_vec) if prob >= thr_global]
            else:
                pred_set = [idx for idx, prob in enumerate(prob_vec) if prob >= thr_per_class[idx]]

            if not pred_set:
                pred_set = [int(np.argmax(prob_vec))]

            preds.append(self._make_contiguous(pred_set))

        self._empty_prob_test = empty_prob_test
        return preds

    def get_conformal_results(self, y_true, y_pred, probs_test, q_hat):
        prediction_sets = self.predict(y_pred, probs_test, q_hat)

        true_indices = []
        for label_entry in y_true:
            idx = self._extract_index(label_entry)
            if idx is None:
                continue
            true_indices.append(idx)

        if not true_indices:
            coverage = 0.0
        else:
            covered = sum(
                int(true_idx in pred_set)
                for true_idx, pred_set in zip(true_indices, prediction_sets)
            )
            coverage = covered / len(true_indices)

        avg_set_size = float(np.mean([len(pred_set) for pred_set in prediction_sets])) if prediction_sets else 0.0
        return prediction_sets, coverage, avg_set_size, y_true

    def evaluate_set_size(self, prediction_sets: List[List[int]], precision: Optional[int] = 3) -> float:
        if len(prediction_sets) == 0:
            return 0.0

        avg_size = sum(len(pred_set) for pred_set in prediction_sets) / len(prediction_sets)
        return round(avg_size, precision) if precision is not None else avg_size

    # ------------------------------------------------------------------
    # Auto-tuning utilities
    # ------------------------------------------------------------------
    def _tune_shrink_tau(self, y_true, y_pred, probs_calibration, alpha) -> float:
        if not self.auto_tau_grid:
            return float(self.rare_shrink_tau or 0.0 or 5.0)

        n = len(y_true)
        if n == 0:
            return float(self.rare_shrink_tau or 5.0)

        holdout_frac = np.clip(self.auto_holdout, 0.05, 0.9)
        holdout_size = int(round(n * holdout_frac))
        if holdout_size <= 0 or holdout_size >= n:
            holdout_size = max(1, n // 4)
        if holdout_size <= 0 or holdout_size >= n:
            return float(self.auto_tau_grid[-1])

        indices = np.arange(n)
        rng = np.random.default_rng(self.auto_seed)
        rng.shuffle(indices)
        eval_idx = indices[:holdout_size]
        fit_idx = indices[holdout_size:]
        if len(fit_idx) == 0:
            return float(self.auto_tau_grid[-1])

        def subset(seq, idxs):
            return [seq[i] for i in idxs]

        y_fit = subset(y_true, fit_idx)
        y_fit_pred = subset(y_pred, fit_idx)
        probs_fit = subset(probs_calibration, fit_idx)
        y_eval = subset(y_true, eval_idx)
        y_eval_pred = subset(y_pred, eval_idx)
        probs_eval = subset(probs_calibration, eval_idx)

        global_cp = OrdinalClassificationConformalPredictor(
            self.classes,
            mode="global",
        )
        q_global = global_cp.fit(y_fit, y_fit_pred, probs_fit, alpha)
        _, global_cov, global_size, _ = global_cp.get_conformal_results(y_eval, y_eval_pred, probs_eval, q_global)

        mondrian_cp = OrdinalClassificationConformalPredictor(
            self.classes,
            mode="mondrian",
        )
        q_mond = mondrian_cp.fit(y_fit, y_fit_pred, probs_fit, alpha)
        _, mondrian_cov, mondrian_size, _ = mondrian_cp.get_conformal_results(y_eval, y_eval_pred, probs_eval, q_mond)

        margin = max(0.0, self.auto_coverage_margin)
        nominal_target = 1.0 - float(alpha)
        coverage_target = max(global_cov + margin, nominal_target + margin)
        coverage_target = min(coverage_target, 1.0)
        size_target = mondrian_size + abs(self.auto_size_tolerance)

        evaluations = []
        for tau in self.auto_tau_grid:
            candidate = OrdinalClassificationConformalPredictor(
                self.classes,
                mode="hybrid",
                rare_shrink_tau=float(tau),
            )
            q_candidate = candidate.fit(y_fit, y_fit_pred, probs_fit, alpha)
            _, cov, size, _ = candidate.get_conformal_results(y_eval, y_eval_pred, probs_eval, q_candidate)
            evaluations.append({
                "tau": float(tau),
                "coverage": float(cov),
                "set_size": float(size),
            })

        feasible = [
            e
            for e in evaluations
            if e["coverage"] >= coverage_target and e["set_size"] <= size_target
        ]
        selection_stage = "coverage+size"
        if feasible:
            feasible.sort(key=lambda e: (e["set_size"], -e["coverage"]))
            chosen = feasible[0]
        else:
            coverage_only = [e for e in evaluations if e["coverage"] >= coverage_target]
            if coverage_only:
                coverage_only.sort(key=lambda e: (-e["coverage"], e["set_size"]))
                chosen = coverage_only[0]
                selection_stage = "coverage-only"
            else:
                # fallback: prioritise coverage, then smallest set size
                evaluations.sort(key=lambda e: (-e["coverage"], e["set_size"]))
                chosen = evaluations[0]
                selection_stage = "best-effort"

        self.tuned_tau_diagnostics = {
            "evaluations": evaluations,
            "global_coverage_eval": float(global_cov),
            "global_set_size_eval": float(global_size),
            "mondrian_coverage_eval": float(mondrian_cov),
            "mondrian_set_size_eval": float(mondrian_size),
            "coverage_target": float(coverage_target),
            "size_target": float(size_target),
            "selection_stage": selection_stage,
            "chosen": chosen,
        }

        return float(chosen["tau"])