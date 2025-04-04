2. Comparison with Baseline Prediction Sets
Objective: Compare the efficiency of conformal prediction sets against sets derived from the baseline’s probabilities.

Description:

For the baseline (assumed to be an LLM providing softmax probabilities for classification), construct prediction sets by including all labels with probabilities above a threshold. Adjust the threshold to match the empirical coverage of conformal prediction (e.g., 90% for α = 0.1).
Compute the average set size for these baseline sets.
Compare these sizes with those from conformal prediction at the same coverage level.
Optionally, calibrate the baseline probabilities (e.g., using temperature scaling) to improve ECE, then repeat the set construction and comparison.

3. Analysis of Misclassified Examples
Objective: Highlight conformal prediction’s ability to mitigate baseline errors by including true labels in its sets.

Description:

Identify instances where the baseline’s top-1 prediction (highest probability label) is incorrect.
For these misclassified examples, check if the true label is included in the conformal prediction set (at a fixed α, e.g., 0.1).
Report the coverage rate specifically for these instances and compare it to the baseline’s 0% coverage (since the top-1 is wrong).


4. Set Size Distribution and Correlation with Difficulty
Objective: Show how conformal prediction adapts its uncertainty estimates to the difficulty of individual examples.

Description:

Compute the distribution of prediction set sizes across the test set (e.g., histogram showing frequency of sets with 1, 2, 3+ labels) at a fixed α (e.g., 0.1).
Optionally, define a difficulty metric (e.g., entropy of the baseline’s predicted probabilities or ground-truth label ambiguity) and correlate it with set size using a scatter plot or statistical test (e.g., Spearman correlation).
Why It’s Useful: A distribution skewed toward smaller sets (e.g., many singletons) indicates high confidence where warranted, while larger sets for harder examples show adaptive uncertainty. This adaptability is a strength over the baseline’s fixed-confidence point predictions, enhancing interpretability in affect recognition.