# Basic statistics and probability

## 1. Why Statistics Matters in AI Engineering

As an AI Engineer, you are working at the intersection of systems engineering, data pipelines, model deployment, and applied machine learning. You may not be deriving novel algorithms, but you will constantly be:

- **Evaluating model outputs** — deciding whether a model is performing well or degrading
- **Interpreting metrics** — understanding what accuracy, F1, perplexity, or BLEU scores actually mean
- **Designing experiments** — A/B testing prompts, comparing model versions, validating fine-tunes
- **Debugging data problems** — detecting distribution shift, data leakage, or sampling bias
- **Communicating uncertainty** — knowing when a model is confident vs. guessing
- **Prompting and RAG** — understanding why retrieval ranking or embedding similarity behaves as it does

Statistics is the language of all of the above. You do not need to be a statistician, but you do need fluency in the core concepts.

---

## 2. Descriptive Statistics

Descriptive statistics summarize datasets. They are your first tool when examining model outputs, token distributions, embedding spaces, or benchmark results.

### 2.1 Measures of Central Tendency

**Mean (arithmetic average)**

The mean is the sum of all values divided by the count. It is sensitive to outliers.

```python
import numpy as np

scores = [0.91, 0.85, 0.88, 0.02, 0.90]  # Note the outlier: 0.02
mean = np.mean(scores)
print(f"Mean: {mean:.3f}")  # 0.712 — pulled down by outlier
```

**Median**

The median is the middle value when sorted. Robust to outliers. Often more reliable than mean for skewed distributions (e.g., latency, cost).

```python
median = np.median(scores)
print(f"Median: {median:.3f}")  # 0.880 — unaffected by outlier
```

**Mode**

The most frequent value. Rarely used for continuous data but relevant for categorical outputs (e.g., most common predicted class).

> **AI Engineering use:** When reporting LLM evaluation scores, always check both mean and median. A single catastrophic failure (score = 0) can make mean scores misleading while median stays stable.

### 2.2 Measures of Spread

**Variance**

The average squared deviation from the mean. Measures how spread out values are. Higher variance means more inconsistency.

```
Variance = (1/n) * Σ(xᵢ - μ)²
```

```python
variance = np.var(scores)
print(f"Variance: {variance:.4f}")
```

**Standard Deviation (σ)**

The square root of variance. Expressed in the same units as the data — much more interpretable than variance.

```python
std = np.std(scores)
print(f"Std Dev: {std:.4f}")
```

**Percentiles and Quantiles**

Percentiles tell you what fraction of data falls below a given value. The 50th percentile is the median. The 95th percentile (P95) of latency tells you the worst-case experience for 95% of users.

```python
p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)
print(f"P50: {p50}ms | P95: {p95}ms | P99: {p99}ms")
```

> **AI Engineering use:** LLM inference latency is almost always right-skewed. Reporting only average latency hides the worst-case experience. Always report P95 and P99 for production systems.

**Interquartile Range (IQR)**

The range between the 25th and 75th percentiles. A robust measure of spread for skewed data.

```python
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
```

### 2.3 Skewness and Kurtosis

**Skewness** measures the asymmetry of a distribution.

- Positive skew: long right tail (e.g., token counts, API latencies)
- Negative skew: long left tail (e.g., model confidence near 1.0)
- Symmetric: skew ≈ 0

**Kurtosis** measures the "tailedness" of a distribution. High kurtosis means more extreme outliers (heavy tails). Relevant when reasoning about rare but catastrophic model failures.

```python
from scipy import stats

skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)  # excess kurtosis (normal = 0)
```

---

## 3. Probability Foundations

Probability is the mathematical framework for reasoning about uncertainty. Since LLMs are probabilistic models that output distributions over tokens, and since nearly every ML evaluation involves uncertainty, probability is non-negotiable knowledge.

### 3.1 Basic Probability Rules

**Sample space (Ω):** The set of all possible outcomes.

**Event (A):** A subset of outcomes. P(A) = probability that A occurs.

**Axioms:**

- P(A) ≥ 0 for all events A
- P(Ω) = 1
- If A and B are mutually exclusive: P(A ∪ B) = P(A) + P(B)

**Complement rule:** P(not A) = 1 - P(A)

**Addition rule (general):**

```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
```

### 3.2 Conditional Probability

The probability of A given that B has occurred:

```
P(A | B) = P(A ∩ B) / P(B)
```

This is fundamental to understanding how language models work. The next token prediction is literally `P(token | all previous tokens)`.

> **Example:** In a RAG pipeline, the conditional probability that the model gives a correct answer given that a relevant document was retrieved is much higher than its unconditional accuracy. That gap is what retrieval quality measures.

### 3.3 Independence

Events A and B are independent if:

```
P(A ∩ B) = P(A) * P(B)
```

Equivalently: P(A | B) = P(A). Knowing B provides no information about A.

> **AI Engineering use:** Naive Bayes classifiers assume features are conditionally independent — a simplification that surprisingly works well in text classification despite being obviously false in natural language.

### 3.4 The Chain Rule of Probability

Any joint probability can be decomposed:

```
P(A, B, C) = P(A) * P(B | A) * P(C | A, B)
```

This is the theoretical foundation of autoregressive language models. They learn to estimate:

```
P(w₁, w₂, ..., wₙ) = P(w₁) * P(w₂|w₁) * P(w₃|w₁,w₂) * ...
```

### 3.5 Bayes' Theorem

Bayes' theorem relates prior beliefs to updated beliefs after observing evidence:

```
P(H | E) = P(E | H) * P(H) / P(E)
```

Where:

- **P(H)** = prior probability of hypothesis H
- **P(E | H)** = likelihood: probability of evidence E given H is true
- **P(E)** = marginal probability of the evidence
- **P(H | E)** = posterior probability: updated belief after seeing E

> **Intuitive example:** A model flags a document as toxic with 90% confidence. If only 1% of documents are actually toxic, Bayes' theorem tells us the true positive rate is much lower than 90% suggests. This is the classic "base rate fallacy."

```python
# Bayes' Theorem example: spam detection
p_spam = 0.02                      # 2% of emails are spam (prior)
p_word_given_spam = 0.30           # "lottery" appears in 30% of spam
p_word_given_not_spam = 0.01       # "lottery" appears in 1% of normal email

# Marginal probability of the word
p_word = (p_word_given_spam * p_spam) + (p_word_given_not_spam * (1 - p_spam))

# Posterior: P(spam | word)
p_spam_given_word = (p_word_given_spam * p_spam) / p_word
print(f"P(spam | 'lottery'): {p_spam_given_word:.3f}")  # ~0.380
```

---

## 4. Probability Distributions

Distributions describe how probabilities are spread across possible outcomes. Understanding distributions helps you reason about model outputs, noise, and data generation processes.

### 4.1 Discrete Distributions

**Bernoulli Distribution**

A single binary outcome (success/failure) with probability p.

- Mean: p
- Variance: p(1-p)

Used for: Binary classification outputs, pass/fail evaluations.

**Binomial Distribution**

Number of successes in n independent Bernoulli trials.

```
P(X = k) = C(n,k) * pᵏ * (1-p)^(n-k)
```

```python
from scipy.stats import binom

# If a model has 85% accuracy and we run 100 evals:
n, p = 100, 0.85
mean = binom.mean(n, p)      # 85.0
std = binom.std(n, p)        # ~3.57
# P(scoring >= 90 correct)
prob_90_plus = 1 - binom.cdf(89, n, p)
print(f"P(≥90 correct): {prob_90_plus:.4f}")
```

**Categorical / Multinomial Distribution**

Generalization of Bernoulli to multiple categories. The raw output of a classifier softmax layer is a categorical distribution. Each LLM token generation is a sample from a categorical distribution over the vocabulary.

**Poisson Distribution**

Number of events in a fixed time interval, given average rate λ.

Used for: Modeling API request rates, error counts, rare events.

```python
from scipy.stats import poisson

lambda_rate = 3  # average 3 errors per hour
p_zero_errors = poisson.pmf(0, lambda_rate)   # 0.050
p_five_plus = 1 - poisson.cdf(4, lambda_rate)  # 0.185
```

### 4.2 Continuous Distributions

**Uniform Distribution**

All values in a range [a, b] are equally likely. Used in random initialization of model weights and for random sampling baselines.

**Normal (Gaussian) Distribution**

Defined by mean μ and standard deviation σ. Symmetric, bell-shaped. The most important distribution in statistics due to the Central Limit Theorem.

```
f(x) = (1 / σ√2π) * exp(-(x-μ)² / 2σ²)
```

68% of data falls within ±1σ, 95% within ±2σ, 99.7% within ±3σ.

```python
from scipy.stats import norm

mu, sigma = 0.85, 0.05
# Probability of a score below 0.75 (two sigma drop)
p_below_75 = norm.cdf(0.75, loc=mu, scale=sigma)
print(f"P(score < 0.75): {p_below_75:.4f}")  # ~0.0228
```

> **AI Engineering use:** Model evaluation scores across benchmark runs often approximate a normal distribution when N is large enough. Weight gradients during training are approximately normal. Neural network weight initializations (e.g., Xavier, He) are drawn from normal distributions.

**Log-Normal Distribution**

If log(X) is normally distributed, X is log-normally distributed. Right-skewed, always positive.

Common for: Inference latency, token generation time, cost per query — all values that can't be negative and have long right tails.

**Beta Distribution**

Defined on [0, 1], parametrized by α and β. Highly flexible shape. The natural distribution for probabilities and proportions.

Used for: Modeling click-through rates, conversion rates, evaluation accuracy estimates, Bayesian priors over probabilities.

```python
from scipy.stats import beta

# After observing 80 successes in 100 trials:
# Beta posterior with uniform prior Beta(1,1)
alpha_post = 80 + 1
beta_post  = 20 + 1
ci_low, ci_high = beta.ppf([0.025, 0.975], alpha_post, beta_post)
print(f"95% CI for true accuracy: [{ci_low:.3f}, {ci_high:.3f}]")
```

**Exponential Distribution**

Models time between events in a Poisson process. Memoryless property: past waiting time doesn't affect future.

Used for: Modeling API request inter-arrival times, time-to-failure in systems.

### 4.3 The Softmax Function and Categorical Distributions

The softmax function converts a vector of raw scores (logits) into a probability distribution:

```
softmax(zᵢ) = exp(zᵢ) / Σ exp(zⱼ)
```

This is the output layer of virtually every classifier and language model. The result is a valid probability distribution: all values ∈ (0,1) and they sum to 1.

**Temperature scaling** modifies softmax outputs. Temperature T < 1 sharpens the distribution (more confident), T > 1 flattens it (more random):

```python
import numpy as np

def softmax(logits, temperature=1.0):
    logits_scaled = np.array(logits) / temperature
    exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
    return exp_logits / exp_logits.sum()

logits = [2.0, 1.0, 0.5]
print("T=0.5 (sharp):", softmax(logits, 0.5))  # more confident
print("T=1.0 (normal):", softmax(logits, 1.0))
print("T=2.0 (flat):", softmax(logits, 2.0))   # more random
```

---

## 5. Bayesian Thinking

Bayesian thinking is a framework for updating beliefs based on evidence. It maps directly onto how you should reason about model performance, data quality, and system reliability.

### 5.1 Prior, Likelihood, and Posterior

- **Prior P(H):** What you believe before seeing data. Can be based on domain knowledge or past experiments.
- **Likelihood P(E|H):** How probable is the evidence if H were true?
- **Posterior P(H|E):** Your updated belief after seeing the evidence.

The cycle: observe evidence → update posterior → posterior becomes next prior.

### 5.2 Bayesian Confidence Intervals

Unlike frequentist confidence intervals, Bayesian credible intervals have the intuitive interpretation you might expect: "There is a 95% probability that the true parameter lies in this interval."

```python
from scipy.stats import beta
import numpy as np

# Estimating model accuracy with Bayesian approach
# Prior: Beta(1, 1) = uniform (no prior knowledge)
# After observing k=73 correct out of n=100:
k, n = 73, 100
alpha_prior, beta_prior = 1, 1

# Posterior is Beta(alpha_prior + k, beta_prior + n - k)
alpha_post = alpha_prior + k
beta_post = beta_prior + (n - k)

posterior = beta(alpha_post, beta_post)
mean_accuracy = posterior.mean()
ci = posterior.interval(0.95)

print(f"Posterior mean accuracy: {mean_accuracy:.3f}")
print(f"95% Credible Interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### 5.3 Bayesian A/B Testing

Traditional A/B testing asks: "Is there a statistically significant difference?" Bayesian A/B testing asks: "What is the probability that variant B is better than A?" — which is usually what you actually want to know.

```python
from scipy.stats import beta
import numpy as np

# Prompt A: 45 successes / 100 trials
# Prompt B: 58 successes / 100 trials
a_success, a_total = 45, 100
b_success, b_total = 58, 100

dist_a = beta(a_success + 1, a_total - a_success + 1)
dist_b = beta(b_success + 1, b_total - b_success + 1)

# Monte Carlo estimate: P(B > A)
samples = 100_000
a_samples = dist_a.rvs(samples)
b_samples = dist_b.rvs(samples)
p_b_better = (b_samples > a_samples).mean()
print(f"P(Prompt B > Prompt A): {p_b_better:.3f}")  # ~0.93
```

---

## 6. Statistical Inference & Hypothesis Testing

Hypothesis testing lets you make rigorous claims about whether an observed difference is real or just noise.

### 6.1 The Framework

1. **Null hypothesis H₀:** The default assumption (no effect, no difference)
2. **Alternative hypothesis H₁:** What you're trying to show
3. **p-value:** Probability of observing results at least this extreme if H₀ were true
4. **Significance level (α):** Your threshold for rejection, commonly 0.05
5. **Decision:** If p < α, reject H₀

> **Critical interpretation:** A p-value is NOT the probability that H₀ is true. It is the probability of observing your data (or more extreme) assuming H₀ is true. This distinction matters enormously.

### 6.2 Common Tests in AI Engineering

**Two-sample t-test:** Compare means of two groups (e.g., evaluation scores of model v1 vs v2)

```python
from scipy.stats import ttest_ind

model_v1_scores = [0.82, 0.79, 0.85, 0.81, 0.84, 0.80, 0.83]
model_v2_scores = [0.87, 0.89, 0.91, 0.86, 0.90, 0.88, 0.92]

stat, p_value = ttest_ind(model_v1_scores, model_v2_scores)
print(f"t-statistic: {stat:.3f}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Significant: model v2 is better (p < 0.05)")
```

**Paired t-test:** When the same examples are evaluated by both models (preferred for LLM benchmarks)

```python
from scipy.stats import ttest_rel

# Same 100 eval examples scored by both models
stat, p_value = ttest_rel(model_v1_scores, model_v2_scores)
```

**Chi-square test:** Test whether categorical distributions differ (e.g., error category distributions between models)

```python
from scipy.stats import chi2_contingency

# Observed errors by type: model_A vs model_B
observed = np.array([
    [45, 30],  # factual errors
    [12, 8],   # refusals
    [8, 22],   # formatting errors
])
chi2, p, dof, expected = chi2_contingency(observed)
print(f"Chi-square p-value: {p:.4f}")
```

### 6.3 Statistical Power and Sample Size

**Statistical power** is the probability of detecting a real effect when it exists (1 - Type II error rate). In AI Engineering, insufficient sample size is one of the most common evaluation mistakes.

- **Type I error (false positive):** Concluding models differ when they don't (α = 0.05)
- **Type II error (false negative):** Missing a real difference (β, typically 0.20)
- **Power = 1 - β** (typically want ≥ 0.80)

```python
from statsmodels.stats.power import TTestIndPower

# How many eval examples do I need?
analysis = TTestIndPower()
n = analysis.solve_power(
    effect_size=0.5,   # Cohen's d: expected difference / pooled std
    alpha=0.05,        # significance level
    power=0.80         # desired power
)
print(f"Required sample size per group: {int(np.ceil(n))}")  # ~64
```

> **AI Engineering use:** Running 10 eval examples and claiming one model is "better" is statistically meaningless. For most LLM benchmarks comparing two models with similar performance, you need hundreds of examples to have adequate statistical power.

### 6.4 Multiple Testing Correction

When running many hypothesis tests (e.g., evaluating across 20 different capabilities), you will get false positives by chance. At α=0.05, if you run 20 tests, you expect ~1 false positive even when there is no real effect.

**Bonferroni correction:** Divide α by the number of tests.

```python
n_tests = 20
alpha_corrected = 0.05 / n_tests  # 0.0025
```

**Benjamini-Hochberg (FDR correction):** Controls the expected proportion of false discoveries among all rejections. Less conservative than Bonferroni.

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.04, 0.03, 0.8, 0.06, 0.04, ...]
reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
```

---

## 7. Correlation, Covariance & Dependence

Understanding relationships between variables is critical for feature analysis, model debugging, and understanding embedding spaces.

### 7.1 Covariance

Covariance measures how two variables change together:

```
Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)]
```

Positive: X and Y increase together. Negative: one increases as the other decreases. Zero: no linear relationship (but may have non-linear dependence).

### 7.2 Pearson Correlation

Normalizes covariance to [-1, 1]:

```
ρ(X, Y) = Cov(X, Y) / (σₓ * σᵧ)
```

```python
from scipy.stats import pearsonr

# Does ROUGE score correlate with human preference ratings?
rouge_scores = [0.32, 0.45, 0.28, 0.61, 0.38, 0.55, 0.42]
human_ratings = [3.1, 4.2, 2.8, 4.8, 3.5, 4.5, 3.9]

corr, p_value = pearsonr(rouge_scores, human_ratings)
print(f"Pearson r: {corr:.3f}, p-value: {p_value:.4f}")
```

> **AI Engineering use:** Automated metrics (BLEU, ROUGE, BERTScore) often have weak correlation with human judgments. Computing this correlation is essential when validating that your automated eval pipeline is meaningful.

### 7.3 Spearman Rank Correlation

Non-parametric. Measures monotonic relationship (whether ranking is consistent), not linear relationship. More robust to outliers.

```python
from scipy.stats import spearmanr
corr, p_value = spearmanr(rouge_scores, human_ratings)
```

### 7.4 Cosine Similarity

The most common similarity measure in embedding spaces. Measures the angle between two vectors, not their magnitude:

```
cos(A, B) = (A · B) / (||A|| * ||B||)
```

Range: [-1, 1]. Values near 1 mean highly similar. Used in semantic search, RAG retrieval, deduplication, clustering.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Using sklearn for batch computation
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim

embeddings = np.random.randn(5, 1536)  # 5 documents, 1536-dim embeddings
sim_matrix = sk_cosine_sim(embeddings)  # 5x5 similarity matrix
```

---

## 8. Information Theory

Information theory provides the mathematical language of LLMs. Loss functions, perplexity, and tokenization all derive from information-theoretic concepts.

### 8.1 Entropy

Entropy measures the uncertainty (or information content) of a distribution:

```
H(X) = -Σ P(x) * log₂ P(x)
```

- High entropy = more uncertain = more bits needed to encode
- Zero entropy = perfectly deterministic (only one outcome possible)

```python
from scipy.stats import entropy
import numpy as np

# Uniform distribution over 4 tokens — max uncertainty
uniform = [0.25, 0.25, 0.25, 0.25]
print(f"H(uniform): {entropy(uniform, base=2):.3f} bits")  # 2.0 bits

# Peaked distribution — low uncertainty
peaked = [0.97, 0.01, 0.01, 0.01]
print(f"H(peaked): {entropy(peaked, base=2):.3f} bits")  # ~0.22 bits
```

> **AI Engineering use:** A model's output entropy tells you how confident it is. Monitoring entropy of model outputs can detect distribution shift or prompt injection attempts. High entropy across all outputs may signal a model is confused; suspiciously low entropy may indicate mode collapse.

### 8.2 Cross-Entropy

Cross-entropy measures how well distribution Q approximates true distribution P:

```
H(P, Q) = -Σ P(x) * log Q(x)
```

This is the loss function used to train virtually every neural network that outputs probabilities. The model (Q) is trained to match the true data distribution (P).

```python
# Cross-entropy loss (what your LLM optimizes during training)
def cross_entropy_loss(true_probs, predicted_probs):
    return -np.sum(true_probs * np.log(predicted_probs + 1e-10))

# True label: token index 2 out of 4 (one-hot)
true = [0, 0, 1, 0]
pred = [0.05, 0.10, 0.80, 0.05]  # model is fairly confident
print(f"Loss: {cross_entropy_loss(true, pred):.4f}")  # ~0.223
```

### 8.3 KL Divergence

KL divergence measures how different distribution Q is from distribution P:

```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
```

Properties: Always ≥ 0. Zero only when P = Q. Asymmetric: KL(P||Q) ≠ KL(Q||P).

> **AI Engineering use:** KL divergence appears in RLHF (fine-tuning with a KL penalty to prevent the model from drifting too far from the base model), VAEs (variational autoencoders), and distribution comparison in general.

```python
from scipy.stats import entropy  # entropy(pk, qk) = KL divergence

p = [0.4, 0.3, 0.2, 0.1]  # true distribution
q = [0.25, 0.25, 0.25, 0.25]  # approximate distribution

kl_pq = entropy(p, q)  # KL(p || q)
print(f"KL(P||Q): {kl_pq:.4f}")
```

### 8.4 Perplexity

Perplexity is the primary intrinsic metric for evaluating language models. It measures how surprised the model is by a text sample:

```
PPL(X) = exp(-(1/n) * Σ log P(xᵢ | x₁...xᵢ₋₁))
```

Lower perplexity = model assigns higher probability to the text = better model. A perplexity of k means the model is as uncertain as if choosing uniformly among k options.

```python
import numpy as np

def compute_perplexity(log_probs):
    """
    log_probs: list of log probabilities assigned to each token
    """
    avg_neg_log_prob = -np.mean(log_probs)
    return np.exp(avg_neg_log_prob)

# Model assigned these log-probs to each token in a sentence
token_log_probs = [-0.5, -0.8, -1.2, -0.3, -0.9, -0.4]
ppl = compute_perplexity(token_log_probs)
print(f"Perplexity: {ppl:.2f}")
```

---

## 9. Sampling & the Law of Large Numbers

### 9.1 The Law of Large Numbers

As the number of samples increases, the sample mean converges to the true population mean. This justifies why we trust large benchmark evaluations over small ones.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating: true model accuracy = 0.75
true_accuracy = 0.75
sample_sizes = [5, 10, 20, 50, 100, 500, 1000, 5000]

for n in sample_sizes:
    sample_accuracy = np.random.binomial(n, true_accuracy) / n
    error = abs(sample_accuracy - true_accuracy)
    print(f"n={n:5d}: estimated={sample_accuracy:.3f}, error={error:.3f}")
```

### 9.2 The Central Limit Theorem (CLT)

**The most important theorem in statistics for AI Engineering:**

No matter the underlying distribution of a random variable, the distribution of sample means approaches a normal distribution as sample size increases.

Formally: If X₁, ..., Xₙ are i.i.d. with mean μ and variance σ², then:

```
√n * (X̄ - μ) / σ → N(0, 1)  as n → ∞
```

This is why we can use normal-distribution-based confidence intervals for evaluation metrics even when individual samples aren't normally distributed.

```python
# CLT in action: sample means from a highly non-normal distribution
import numpy as np

# Underlying distribution: Poisson(λ=2) — very skewed
n_experiments = 10000
sample_size = 50
sample_means = [
    np.mean(np.random.poisson(lam=2, size=sample_size))
    for _ in range(n_experiments)
]
# sample_means will be approximately normally distributed
print(f"Mean of means: {np.mean(sample_means):.3f}")  # ≈ 2.0
print(f"Std of means: {np.std(sample_means):.3f}")    # ≈ σ/√n = √2/√50
```

### 9.3 Sampling Methods

**Simple random sampling:** Each item equally likely to be selected. Use for unbiased evaluation set construction.

**Stratified sampling:** Sample proportionally from subgroups. Use when you need evaluation coverage across categories (e.g., ensuring equal representation of short/long prompts, different domains).

```python
from sklearn.model_selection import train_test_split

# Stratified split: preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Bootstrap sampling:** Repeatedly resample with replacement to estimate the distribution of a statistic. Extremely useful for constructing confidence intervals on evaluation metrics without distributional assumptions.

```python
import numpy as np

def bootstrap_confidence_interval(data, statistic_fn, n_bootstrap=10000, ci=0.95):
    boot_stats = [
        statistic_fn(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    return np.percentile(boot_stats, [alpha * 100, (1 - alpha) * 100])

eval_scores = np.array([0.82, 0.79, 0.85, 0.81, 0.84, 0.80, 0.83, 0.88, 0.86, 0.82])
ci = bootstrap_confidence_interval(eval_scores, np.mean)
print(f"Bootstrap 95% CI for mean accuracy: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

---

## 10. Statistical Evaluation of Models

This section brings everything together for the most direct AI Engineering applications.

### 10.1 Confidence Intervals for Evaluation Metrics

Never report a point estimate (e.g., "accuracy = 0.84") without a confidence interval. The interval tells you how much sampling noise is in that estimate.

```python
import numpy as np
from scipy.stats import t

def accuracy_confidence_interval(n_correct, n_total, confidence=0.95):
    """Wilson score interval — better than normal approximation for proportions"""
    from statsmodels.stats.proportion import proportion_confint
    lower, upper = proportion_confint(n_correct, n_total, 
                                       alpha=1-confidence, method='wilson')
    return lower, upper

# 73 correct out of 100 evaluations
lower, upper = accuracy_confidence_interval(73, 100)
print(f"73/100 accuracy: 95% CI = [{lower:.3f}, {upper:.3f}]")

# 730 correct out of 1000 evaluations (same proportion, more data)
lower, upper = accuracy_confidence_interval(730, 1000)
print(f"730/1000 accuracy: 95% CI = [{lower:.3f}, {upper:.3f}]")
```

### 10.2 Detecting Distribution Shift

Distribution shift occurs when your production data differs from training/evaluation data. Statistical tests can detect this.

**Kolmogorov-Smirnov test:** Compare whether two samples come from the same distribution.

```python
from scipy.stats import ks_2samp

# Compare token length distributions: eval set vs recent production traffic
eval_lengths = np.random.normal(150, 40, 500)
prod_lengths = np.random.normal(200, 60, 500)  # shifted distribution

stat, p_value = ks_2samp(eval_lengths, prod_lengths)
print(f"KS stat: {stat:.3f}, p-value: {p_value:.6f}")
if p_value < 0.05:
    print("WARNING: Distribution shift detected!")
```

**Population Stability Index (PSI):** Measures how much a variable's distribution has shifted between two samples. Common in production ML monitoring.

```python
def compute_psi(expected, actual, n_bins=10):
    """Population Stability Index. PSI < 0.1 = stable, > 0.2 = significant shift."""
    eps = 1e-10
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] -= eps; bins[-1] += eps

    expected_percents = np.histogram(expected, bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins)[0] / len(actual)

    expected_percents = np.clip(expected_percents, eps, None)
    actual_percents = np.clip(actual_percents, eps, None)

    psi = np.sum((actual_percents - expected_percents) * 
                 np.log(actual_percents / expected_percents))
    return psi
```

### 10.3 Calibration

A well-calibrated model's confidence scores match actual accuracy. When a model says "90% confident," it should be correct 90% of the time.

```python
from sklearn.calibration import calibration_curve

# For a binary classifier
prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)

# A perfectly calibrated model: prob_true ≈ prob_pred
# ECE (Expected Calibration Error): weighted average absolute deviation
ece = np.mean(np.abs(prob_true - prob_pred))
print(f"Expected Calibration Error: {ece:.4f}")
```

### 10.4 Effect Size

Statistical significance tells you whether an effect exists. Effect size tells you how large it is. Both matter.

**Cohen's d** for comparing two means:

```
d = (μ₁ - μ₂) / σ_pooled
```

- d = 0.2: small effect
- d = 0.5: medium effect
- d = 0.8: large effect

```python
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

d = cohens_d(model_v2_scores, model_v1_scores)
print(f"Cohen's d: {d:.3f}")  # How large is the improvement?
```

---

## 11. Working with Distributions in Python

A practical reference for the most common operations.

```python
import numpy as np
from scipy import stats

# ── Normal distribution ──────────────────────────────────────────────────
dist = stats.norm(loc=0.85, scale=0.05)

dist.pdf(0.85)          # probability density at x=0.85
dist.cdf(0.80)          # P(X ≤ 0.80)
dist.ppf(0.025)         # value at 2.5th percentile
dist.rvs(size=1000)     # draw 1000 random samples
dist.interval(0.95)     # 95% interval: (mean - 1.96σ, mean + 1.96σ)

# ── Binomial distribution ────────────────────────────────────────────────
binom_dist = stats.binom(n=100, p=0.73)
binom_dist.pmf(73)      # P(exactly 73 successes)
binom_dist.cdf(70)      # P(70 or fewer successes)
binom_dist.mean()       # 73.0
binom_dist.std()        # ~4.43

# ── Beta distribution ─────────────────────────────────────────────────────
beta_dist = stats.beta(a=74, b=28)  # posterior after 73/100
beta_dist.mean()        # ≈ 0.725
beta_dist.interval(0.95)  # 95% credible interval

# ── Sampling and summary stats ───────────────────────────────────────────
sample = np.random.normal(loc=0.80, scale=0.08, size=500)

stats.describe(sample)  # returns: n, min/max, mean, variance, skewness, kurtosis
np.percentile(sample, [25, 50, 75, 95, 99])

# ── Testing normality ─────────────────────────────────────────────────────
stat, p = stats.shapiro(sample[:50])    # Shapiro-Wilk (small samples)
stat, p = stats.normaltest(sample)     # D'Agostino-Pearson (larger samples)

# ── Non-parametric tests ──────────────────────────────────────────────────
stat, p = stats.mannwhitneyu(group1, group2)     # non-parametric t-test alternative
stat, p = stats.wilcoxon(before_scores, after_scores)  # paired non-parametric
```

---

## 12. Quick Reference Formulas

|Concept|Formula|Python|
|---|---|---|
|Mean|μ = Σxᵢ/n|`np.mean(x)`|
|Variance|σ² = Σ(xᵢ-μ)²/n|`np.var(x)`|
|Std Dev|σ = √variance|`np.std(x)`|
|Conditional prob|P(A\|B) = P(A∩B)/P(B)|—|
|Bayes' theorem|P(H\|E) = P(E\|H)P(H)/P(E)|—|
|Entropy|H = -Σ p·log(p)|`scipy.stats.entropy(p)`|
|Cross-entropy|H(P,Q) = -Σ P·log(Q)|`F.cross_entropy(logits, targets)`|
|KL divergence|KL(P\|Q) = Σ P·log(P/Q)|`scipy.stats.entropy(p, q)`|
|Perplexity|PPL = exp(-Σlog P(wᵢ)/n)|`torch.exp(loss)`|
|Softmax|σ(z)ᵢ = exp(zᵢ)/Σexp(zⱼ)|`torch.softmax(logits, dim=-1)`|
|Cosine similarity|cos(A,B) = A·B/(‖A‖‖B‖)|`sklearn.metrics.pairwise.cosine_similarity`|
|Pearson r|ρ = Cov(X,Y)/(σₓσᵧ)|`scipy.stats.pearsonr(x, y)`|
|Cohen's d|d = (μ₁-μ₂)/σ_pooled|(see section 10.4)|

---

## Recommended Python Libraries

|Library|Use Case|
|---|---|
|`numpy`|Array math, descriptive stats, random sampling|
|`scipy.stats`|Distributions, hypothesis tests, correlation|
|`statsmodels`|Regression, power analysis, multiple testing|
|`sklearn.metrics`|Classification metrics, calibration|
|`sklearn.model_selection`|Cross-validation, stratified splits|
|`torch` / `torch.nn.functional`|Cross-entropy loss, softmax in models|
|`matplotlib` / `seaborn`|Distribution visualization|

---

_This document covers the statistical foundations most relevant to AI Engineering practice — evaluating models, debugging data, running experiments, and reasoning about uncertainty. For deeper study, the key bridge topics are: Basic linear algebra (for understanding embeddings and transformations), Optimization theory (for gradient descent), and Probabilistic graphical models (for structured prediction)._