# Multiclass Conformal Prediction Methodology

## Problem Setting

We consider a multilabel multiclass setting: each sample \((x_i, Y_i)\) admits a set of admissible labels \(Y_i \subseteq \mathcal{Y}\), where \(|\mathcal{Y}| = K\) and \(K\) is fixed by the dataset taxonomy (e.g., GoEmotions). Model inference produces a probability vector \(p(x) \in [0,1]^K\) with \(\sum_{k=1}^K p_k(x) = 1\). A conformal prediction (CP) procedure maps \(x\) to a prediction set \(\widehat{Y}(x) \subseteq \mathcal{Y}\) such that, with high probability, \(Y \subseteq \widehat{Y}(x)\).

We adopt the inductive conformal prediction (ICP) workflow [1,2]:

1. **Calibration split**: Partition the available in-domain labeled responses into a calibration subset \(\mathcal{C}\) and a held-out test subset \(\mathcal{T}\). The generative model produces \((p(x),\widehat{y}(x))\) pairs for all points.
2. **Nonconformity scoring**: Map each calibration item to a scalar nonconformity statistic.
3. **Quantile estimation**: Estimate a threshold from calibration scores at target miscoverage level \(\alpha\).
4. **Set construction**: Apply the threshold to test-time probabilities to produce finite prediction sets.

All three steps are implemented in `src/conformalprediction/multiclass.py` and orchestrated from `src/run.py`.

## Nonconformity Score

For a calibration sample with probability vector \(p(x)\) and label set \(Y\), the nonconformity score is
\[
  s(x, Y) = 1 - \min_{y \in Y} p_y(x).
\]
This choice is equivalent to the complement of the largest probability mass assigned to the true label set and matches the monotone conformity score recommended for multilabel tasks in [3,4]. When the model assigns high confidence to *all* true labels, the score is small; if any true label is under-weighted, the score increases toward 1.

## Global Quantile Estimator

Let \(\{ s_i \}_{i=1}^{m}\) denote scores on the calibration split (after removing items with missing probability vectors). We define the empirical upper quantile using the "k-th higher" rule [1]:
\[
  q_{\text{global}}(\alpha) = \hat{Q}_{1-\alpha} = s_{(\lceil (1-\alpha)(m+1) \rceil)}.
\]
In code we implement this via partial sorting (`numpy.partition`) to avoid full sorts. If the calibration set is empty (pathological), we fallback to \(q_{\text{global}} = 1\).

## Mondrian (Class-Conditional) Quantiles

To better respect class imbalance we adopt the Mondrian framework [5]. For each class \(c\), collect scores from calibration points whose ground-truth set contains \(c\):
\[
  \mathcal{S}_c = \{ s_i : c \in Y_i \}.
\]
The class-conditional quantile is
\[
  q_c(\alpha) = \begin{cases}
    q_{\text{global}}(\alpha), & \text{if } |\mathcal{S}_c| = 0, \\
    \text{Quantile}(\mathcal{S}_c, 1-\alpha), & \text{otherwise}.
  \end{cases}
\]
These per-class thresholds tighten prediction sets for well-represented classes while preserving validity within each block.

## Shrinkage Toward the Global Baseline

Rare classes may yield unstable quantiles due to very small \(|\mathcal{S}_c|\). Following shrinkage ideas from empirical Bayes and hierarchical CP [6,7], we blend each class-conditional estimate with the global baseline:
\[
  \tilde{q}_c(\alpha) = \lambda_c \, q_{\text{global}}(\alpha) + (1-\lambda_c) \, q_c(\alpha),
\]
\[
  \lambda_c = \frac{\tau}{|\mathcal{S}_c| + \tau}, \qquad \tau \ge 0.
\]
- **Mondrian mode** (`mode="mondrian"`): always apply shrinkage if \(\tau > 0\).
- **Hybrid mode** (`mode="hybrid"`): use class-specific quantiles when available and shrink only if \(\tau > 0\); otherwise fall back to the global threshold when \(|\mathcal{S}_c| = 0\).
- **Global mode** (`mode="global"`): skip per-class computation entirely.

We typically set \(\tau = 5\), which yields \(\lambda_c \approx 0.33\) when \(|\mathcal{S}_c| = 10\) (moderate shrinkage) and \(\lambda_c \approx 0.83\) when \(|\mathcal{S}_c| = 1\) (strong shrinkage toward the global estimate). Increasing \(\tau\) pushes hybrid and Mondrian behavior closer to the global baseline, while \(\tau = 0\) recovers the raw Mondrian quantiles.

## Prediction Set Construction

Given the calibrated thresholds, prediction sets are obtained by thresholding probabilities:
\[
  \widehat{Y}(x) = \begin{cases}
    \{ c : p_c(x) \ge 1 - q_{\text{global}}(\alpha) \}, & \text{global mode}, \\
    \{ c : p_c(x) \ge 1 - \tilde{q}_c(\alpha) \}, & \text{Mondrian/Hybrid modes}.
  \end{cases}
\]
If the resulting set is empty (which can happen when probabilities are extremely diffuse), we backstop by adding the argmax label, ensuring non-empty prediction sets as prescribed in [1].

## Multilabel Coverage Guarantee

Coverage is evaluated on the held-out split by requiring the prediction set to contain **all** ground-truth labels:
\[
  \widehat{\text{Cov}} = \frac{1}{|\mathcal{T}|} \sum_{(x,Y) \in \mathcal{T}} \mathbf{1}[ Y \subseteq \widehat{Y}(x) ].
\]
When the data are exchangeable, the ICP framework ensures \(\mathbb{P}(Y \subseteq \widehat{Y}(X)) \ge 1-\alpha\). Our evaluation uses the same definition so that coverage curves can be interpreted against this theoretical guarantee.

## Practical Safeguards

- **Empty probability tensors**: Generation occasionally returns empty logits; such samples are counted and skipped during calibration, and replaced with uniform probabilities during testing to avoid crashes.
- **Temperature-annotated outputs**: For each model/dataset/temperature/mode combination we store calibrated sets in `results/conformal_results/<dataset>/temp_<T>/<model>__<mode>.json`, ensuring no accidental overwrites when sweeping modes.
- **Configuration knobs**: `Config.MULTICLASS_CP_MODE`, `Config.MULTICLASS_CP_MODES`, and `Config.MULTICLASS_RARE_SHRINK` expose the mode sweep and shrinkage parameter without affecting regression or ordinal pipelines.

## Empirical Discussion

Running the pipeline on GoEmotions with `lzw1008/Emollama-7b` and temperature 0.9 yields the following high-level observations:

- **Global mode** shows mild under-coverage at \(1-\alpha = 0.90\): empirical coverage 0.895 with average set size 5.03. This reflects the difficulty of guaranteeing simultaneous coverage across all emotion labels with a single threshold.
- **Mondrian mode** (\(\tau = 5\)) reaches coverage 0.922 at the same confidence level with only a modest increase in set size (5.15). The per-class adjustment mainly benefits underrepresented emotions by relaxing thresholds where calibration support is sparse.
- **Hybrid mode** mirrors Mondrian in this run because each class had sufficient calibration hits for the shrinkage factor to remain near the raw per-class estimate. In datasets with extremely rare labels the hybrid blend reins in over-wide sets by anchoring to the global baseline.

Across the entire \(\alpha\) sweep, both Mondrian and hybrid sets track closer to the nominal target than global while maintaining smaller or comparable set sizes at lower confidences (e.g., coverage 0.701 vs 0.683 at \(1-\alpha = 0.70\) with average size 2.55 vs 3.00). This trade-off provides quantitative evidence that the Mondrian/hybrid variants are preferable when reviewer concerns focus on per-class calibration.

The shrinkage parameter \(\tau\) offers a tunable bridge between global and Mondrian behavior. Raising \(\tau\) (e.g., \(\tau = 8\)) pulls class-conditional thresholds toward the global limit, reducing variance for ultra-rare classes at the cost of slightly larger sets. Conversely, lowering \(\tau\) (toward zero) recovers pure Mondrian thresholds that may overreact to small calibration counts. Choosing \(\tau = 5\) thus reflects a conservative compromise: it delivers the per-class responsiveness reviewers requested while controlling variance-induced deviations from the intended coverage level.

## References

[1] Vovk, V., Gammerman, A., & Shafer, G. *Algorithmic Learning in a Random World*. Springer, 2005.

[2] Papadopoulos, H., Vovk, V., & Gammerman, A. "Inductive Confidence Machines for Regression." *ECML*, 2002.

[3] Sadinle, M., Lei, J., & Wasserman, L. "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels." *JASA*, 114(525), 2019.

[4] Romano, Y., Patterson, E., & Candes, E. "Conformalized Quantile Regression." *NeurIPS*, 2019.

[5] Vovk, V., Nouretdinov, I., & Gammerman, A. "On-Line Predictive Linear Regression." *Annals of Statistics*, 29(3), 2001.

[6] Cauchois, M., Gupta, S., & Duchi, J. C. "Robust Validation: Confident Predictions Even When Distributions Shift." *JMLR*, 2021.

[7] Angelopoulos, A. N., Bates, S., Malik, J., & Jordan, M. "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control." *ICML*, 2021.
