import json
import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold

class QuantileRegressor:
    """
    Train quantile regressors (pinball loss) and predict requested quantiles.
    Default backend: scikit-learn GradientBoostingRegressor (GBRT with loss='quantile').

    - Supports multiple quantiles in one object (trains 1 model per quantile).
    - Enforces non-crossing at prediction time.
    - Returns outputs in formats friendly to your CQR class.

    Parameters
    ----------
    quantiles : Iterable[float]
        Quantile levels in (0,1). Example: (0.05, 0.95).
    base_params : dict
        Hyperparameters for GradientBoostingRegressor *except* 'loss' and 'alpha'.
        Reasonable defaults are provided.
    random_state : Optional[int]
        RNG seed for reproducibility.

    Notes
    -----
    Pinball loss (for quantile τ) is minimized by the conditional τ-quantile:
        L_τ(y, f) = (τ - 1_{y < f}) * (y - f)
    """

    def __init__(
        self,
        quantiles: Iterable[float] = (0.05, 0.95),
        base_params: Optional[dict] = None,
        random_state: Optional[int] = 42,
        param_grid: Optional[Iterable[dict]] = None,
        cv_folds: int = 5,
        use_pca: bool = False,
        pca_variance: float = 0.98,
        tail_weight: float = 1.0,
        early_stopping: bool = False,
        early_stopping_rounds: int = 30,
        early_stopping_tol: float = 1e-4,
        validation_fraction: float = 0.1,
    ):
        self.quantiles = tuple(sorted(float(q) for q in quantiles))
        self.random_state = random_state

        default_params = dict(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.08,
            min_samples_leaf=5,
            subsample=0.8,
            max_features="sqrt",
        )
        if base_params:
            default_params.update(base_params)
        self._base_template = default_params
        self.param_grid = list(param_grid) if param_grid else []
        self.cv_folds = max(cv_folds, 1)
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.pca_: Optional[PCA] = None
        self.tail_weight = tail_weight
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tol = early_stopping_tol
        self.validation_fraction = validation_fraction

        # best params picked during fit
        self.base_params = None

        # one model per quantile
        self.models_: Dict[float, GradientBoostingRegressor] = {}

        self.is_fitted_: bool = False

    def _prepare_features(self, X, fit: bool = False):
        X = np.asarray(X, dtype=np.float32)
        if not self.use_pca:
            return X
        if fit:
            self.pca_ = PCA(n_components=self.pca_variance, svd_solver="full", random_state=self.random_state)
            return self.pca_.fit_transform(X)
        if self.pca_ is None:
            raise RuntimeError("PCA transformer not fitted; call fit before predict.")
        return self.pca_.transform(X)

    def _build_candidate_params(self):
        if not self.param_grid:
            return [self._base_template.copy()]
        candidates = []
        for override in self.param_grid:
            params = self._base_template.copy()
            params.update(override)
            candidates.append(params)
        return candidates

    def _mean_pinball(self, y_true, y_pred, alpha, weight=1.0):
        return weight * mean_pinball_loss(y_true, y_pred, alpha=alpha)

    def _evaluate_candidate(self, params, X, y):
        rng = np.random.RandomState(self.random_state)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=rng.randint(0, 10_000)) if self.cv_folds > 1 else None
        total_loss = 0.0
        denom = 0

        splits = kf.split(X) if kf else [(np.arange(X.shape[0]), np.arange(X.shape[0]))]

        for train_idx, val_idx in splits:
            if len(np.unique(train_idx)) < 2 or len(np.unique(val_idx)) < 2:
                continue
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            for q in self.quantiles:
                model_params = params.copy()
                model_params["random_state"] = self.random_state
                if self.early_stopping:
                    model_params.update(
                        {
                            "validation_fraction": self.validation_fraction,
                            "n_iter_no_change": self.early_stopping_rounds,
                            "tol": self.early_stopping_tol,
                        }
                    )

                model = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    verbose=0,
                    **model_params,
                )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                weight = self.tail_weight if (q == self.quantiles[0] or q == self.quantiles[-1]) else 1.0
                total_loss += self._mean_pinball(y_val, preds, alpha=q, weight=weight)
                denom += weight

        return total_loss / max(denom, 1)

    def _select_hyperparameters(self, X, y):
        candidates = self._build_candidate_params()
        if len(candidates) == 1 or self.cv_folds < 2:
            self.base_params = candidates[0]
            return

        best_score = float("inf")
        best_params = candidates[0]

        print(f"[DEBUG-QR] Hyperparameter search across {len(candidates)} configs × {self.cv_folds} folds")
        for idx, params in enumerate(candidates, 1):
            score = self._evaluate_candidate(params, X, y)
            print(f"[DEBUG-QR]   [{idx}/{len(candidates)}] params={params} → mean pinball={score:.6f}")
            if score < best_score:
                best_score = score
                best_params = params

        self.base_params = best_params
        print(f"[DEBUG-QR] Selected params: {best_params} (mean pinball={best_score:.6f})")

    def fit(self, X, y):
        import time
        X = self._prepare_features(X, fit=True)
        y = np.asarray(y).astype(float)
        if y.ndim != 1:
            raise ValueError("y must be 1D array of shape (n_samples,)")

        self._select_hyperparameters(X, y)

        self.models_.clear()
        print(
            f"[DEBUG-QR] Starting fit with {len(self.quantiles)} quantiles on {X.shape[0]} samples, {X.shape[1]} features"
        )
        print(f"[DEBUG-QR] GBRT params: {self.base_params}")
        if self.use_pca and self.pca_ is not None:
            print(
                f"[DEBUG-QR] PCA retained {self.pca_.n_components_} components (variance={self.pca_variance})"
            )
        
        # Initialize training log
        training_log = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_quantiles": len(self.quantiles),
            "gbrt_params": self.base_params.copy(),
            "quantile_training": {}
        }
        training_log["selected_params"] = self.base_params.copy()
        training_log["random_state"] = self.random_state
        training_log["pca"] = {
            "enabled": self.use_pca,
            "variance": self.pca_variance,
            "components": int(self.pca_.n_components_) if self.use_pca and self.pca_ is not None else None,
        }
        
        fit_start = time.time()
        for idx, q in enumerate(self.quantiles, 1):
            q_start = time.time()
            print(f"[DEBUG-QR] [{idx}/{len(self.quantiles)}] Training quantile τ={q:.4f}...", flush=True)
            
            # Each model is GBRT with quantile loss and its own alpha
            params = self.base_params.copy()
            params["random_state"] = self.random_state
            if self.early_stopping:
                params.update(
                    {
                        "validation_fraction": self.validation_fraction,
                        "n_iter_no_change": self.early_stopping_rounds,
                        "tol": self.early_stopping_tol,
                    }
                )
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                verbose=0,
                **params,
            )
            model.fit(X, y)
            self.models_[q] = model
            
            q_time = time.time() - q_start
            
            # Log training info for this quantile
            training_log["quantile_training"][str(q)] = {
                "quantile": q,
                "train_time_seconds": q_time,
                "final_train_loss": float(model.train_score_[-1]) if hasattr(model, 'train_score_') else None,
                "n_estimators_trained": model.n_estimators_,
                "completed_at": datetime.now().isoformat()
            }
            
            print(f"[DEBUG-QR] [{idx}/{len(self.quantiles)}] ✓ τ={q:.4f} completed in {q_time:.1f}s")

        self.is_fitted_ = True
        total_fit_time = time.time() - fit_start
        
        # Final summary
        training_log["total_fit_time_seconds"] = total_fit_time
        training_log["total_fit_time_minutes"] = total_fit_time / 60
        training_log["avg_time_per_quantile_seconds"] = total_fit_time / len(self.quantiles)
        
        # Save training log
        self._save_training_log(training_log)
        
        print(f"\n[DEBUG-QR] Fit completed in {total_fit_time:.1f}s ({total_fit_time/60:.2f}m)")
        print(f"[DEBUG-QR] Training log saved to: {self._get_log_path(training_log)}\n")
        return self

    def _get_log_path(self, training_log):
        """Generate log file path based on timestamp"""
        log_dir = "results/qr_training_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(log_dir, f"qr_training_{timestamp}.json")

    def _save_training_log(self, training_log):
        """Save training metrics to JSON file"""
        log_path = self._get_log_path(training_log)
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"[DEBUG-QR] Saved training metrics: {log_path}")

    def _check_is_fitted(self):
        if not self.is_fitted_ or not self.models_:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

    def predict_quantiles(self, X, quantiles: Optional[Iterable[float]] = None) -> np.ndarray:
        """
        Predict requested quantiles for each row in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        quantiles : optional Iterable of quantiles to return; if None, uses self.quantiles

        Returns
        -------
        preds : np.ndarray of shape (n_samples, n_q)
            Columns correspond to quantiles in ascending order.
        """
        import time

        self._check_is_fitted()
        X = self._prepare_features(X, fit=False)
        qs = tuple(sorted(self.quantiles if quantiles is None else (float(q) for q in quantiles)))

        # Ensure we have models for all requested quantiles
        missing = [q for q in qs if q not in self.models_]
        if missing:
            # train-on-demand for missing quantiles using same base params (cheap if needed)
            # You can remove this block if you prefer strict behavior.
            for q in missing:
                model = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    **self.base_params
                )
                # We need original training data to train missing qs; if unavailable, raise.
                raise ValueError(
                    f"Requested quantile {q} not trained. "
                    "Either refit with this quantile or provide a pool object that stores train data."
                )

        print(f"[DEBUG-QR] Predicting {len(qs)} quantiles on {X.shape[0]} samples...", end=" ", flush=True)
        pred_start = time.time()
        
        preds = []
        for q in qs:
            preds.append(self.models_[q].predict(X))
        preds = np.column_stack(preds)

        # Enforce non-crossing: monotonize across quantiles per row
        # A simple, safe choice is to sort the predicted quantiles per row.
        # (Monotone rearrangement; keeps coverage properties fine for CQR use.)
        preds.sort(axis=1)

        pred_time = time.time() - pred_start
        print(f"✓ ({pred_time:.2f}s, shape={preds.shape})")
        
        # Log prediction info
        self._log_prediction(len(qs), X.shape[0], pred_time, preds.shape)
        
        return preds

    def _log_prediction(self, n_quantiles, n_samples, pred_time, output_shape):
        """Log prediction metrics"""
        log_dir = "results/qr_prediction_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"qr_predictions_{timestamp}.json")
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "n_quantiles": n_quantiles,
            "n_samples": n_samples,
            "prediction_time_seconds": pred_time,
            "time_per_sample_ms": (pred_time / n_samples) * 1000,
            "output_shape": list(output_shape)
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def predict_interval(
        self,
        X,
        q_lo: float = 0.05,
        q_hi: float = 0.95,
        as_dict: bool = True
    ) -> Tuple[np.ndarray, np.ndarray] | Dict[str, np.ndarray]:
        """
        Convenience to get [q_lo, q_hi] interval.

        Returns
        -------
        (lower, upper) arrays OR {'lower': lower, 'upper': upper} dict
        """
        q_lo, q_hi = float(q_lo), float(q_hi)
        if q_lo >= q_hi:
            raise ValueError("Require q_lo < q_hi.")

        preds = self.predict_quantiles(X, quantiles=(q_lo, q_hi))
        lower, upper = preds[:, 0], preds[:, 1]
        if as_dict:
            return {"lower": lower, "upper": upper}
        return lower, upper

    # For completeness, a single-quantile helper:
    def predict_q(self, X, q: float) -> np.ndarray:
        q = float(q)
        return self.predict_quantiles(X, quantiles=(q,))[:, 0]
    
    def interpolate_quantile(preds_dict, q_target, quantile_grid, allow_extrapolate=False):
        """preds_dict: dict {q: vector}, same length for all vectors quantile_grid: sorted tuple/list of floats Returns vector at q_target, linearly interpolated if needed."""
        if q_target in preds_dict:
            return preds_dict[q_target]
        qs = np.array(sorted(quantile_grid))
        hi = int(np.searchsorted(qs, q_target, side="left"))
        if hi == 0:
            if not allow_extrapolate: raise ValueError("q_target below grid; extend grid.")
            return preds_dict[qs[0]]
        if hi == len(qs):
            if not allow_extrapolate: raise ValueError("q_target above grid; extend grid.")
            return preds_dict[qs[-1]]
        q0, q1 = float(qs[hi-1]), float(qs[hi])
        p0, p1 = preds_dict[q0], preds_dict[q1]
        w = (q_target - q0) / max(q1 - q0, 1e-12)
        return (1 - w) * p0 + w * p1

