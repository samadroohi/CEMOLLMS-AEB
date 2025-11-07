# Response to Reviewer Comment on Multiclass Conformal Prediction

> **Reviewer 1, Comment:** “Multiclass CP definition and construction of prediction sets. The current write-up appears to imply a score tied to max π(x)[c], which can degenerate the construction. Please define the nonconformity score precisely, and state whether thresholds are global or class-conditional. Provide pseudocode and note computational complexity.”

**Response:** We thank the reviewer for highlighting the ambiguity. We have revised the manuscript to clarify the multiclass (multilabel) conformal prediction procedure:

1. **Explicit nonconformity definition.** Section X.Y now states that the score for an example with probability vector \(p(x)\) and true label set \(Y\) is
   \[
     s(x, Y) = 1 - \min_{y \in Y} p_y(x),
   \]
   i.e., the complement of the smallest true-label probability. This is the monotone set-valued score recommended for multilabel conformal predictors and does not suffer the degeneracy described in the original draft.

2. **Threshold regimes.** The revision distinguishes three regimes:
   - *Global:* a single quantile \(q_{\text{global}}(\alpha)\) estimated from all calibration scores.
   - *Mondrian:* per-class quantiles \(q_c(\alpha)\) estimated from class-specific calibration buckets \(\mathcal{S}_c\).
   - *Hybrid:* a shrinkage blend \(\tilde{q}_c(\alpha) = \lambda_c q_{\text{global}}(\alpha) + (1-\lambda_c) q_c(\alpha)\) with \(\lambda_c = \tau / (|\mathcal{S}_c| + \tau)\).
   We emphasize that \(\tau\) is a configuration knob (default 5) that smoothly interpolates between global and Mondrian behavior, mitigating variance for rare classes.

3. **Pseudocode and complexity.** The manuscript now includes the following pseudocode (Algorithm 1) summarizing the calibration and prediction steps:

```
Algorithm 1: Multiclass Conformal Prediction (Hybrid form)
Inputs: calibration {(p_i, Y_i)}_{i=1}^m, test {p_j^*}_{j=1}^n, alpha, mode ∈ {global, mondrian, hybrid}, τ ≥ 0
1: scores_global ← [] ; scores_per_class ← {c: [] for c in classes}
2: for each (p_i, Y_i) in calibration do
3:     if Y_i is empty or p_i missing then continue
4:     score ← 1 − min_{y in Y_i} p_i[y]
5:     append score to scores_global
6:     for each y in Y_i do append score to scores_per_class[y]
7: q_global ← Quantile(scores_global, 1 − alpha)  // kth-higher rule
8: if mode == global then q_c ← q_global for all c
9: else
10:    for each class c do
11:        if scores_per_class[c] empty then q_hat ← q_global else q_hat ← Quantile(scores_per_class[c], 1 − alpha)
12:        if mode == hybrid or (mode == mondrian and τ > 0) then
13:            λ ← τ / (|scores_per_class[c]| + τ)
14:            q_c ← λ q_global + (1 − λ) q_hat
15:        else q_c ← q_hat
16: prediction_sets ← []
17: for each probability vector p_j^* in test do
18:    if mode == global then S ← {c : p_j^*[c] ≥ 1 − q_global}
19:    else S ← {c : p_j^*[c] ≥ 1 − q_c}
20:    if S empty then S ← {argmax_c p_j^*[c]}
21:    append S to prediction_sets
22: return prediction_sets
```

Computational complexity is dominated by the calibration pass and quantile computation. Both steps are linear in the number of calibration examples and classes: \(\mathcal{O}(mK)\). Prediction is \(\mathcal{O}(nK)\). All operations are vectorized in our implementation (see `src/conformalprediction/multiclass.py`).

4. **Empirical justification.** Section X.Z now discusses the trade-offs between modes using the GoEmotions benchmark. We show that the Mondrian and hybrid variants achieve closer-to-nominal coverage with only marginal increases in average set size, while the shrinkage parameter \(\tau\) lets practitioners interpolate between responsiveness and stability. This directly addresses the reviewer’s concern about the suitability of the construction.

We believe these clarifications resolve the ambiguity and convincingly justify our multiclass conformal prediction design. The revised manuscript and supplementary note `docs/multiclass_cp_methodology.md` incorporate the detailed definitions, pseudocode, and references cited in this response.
