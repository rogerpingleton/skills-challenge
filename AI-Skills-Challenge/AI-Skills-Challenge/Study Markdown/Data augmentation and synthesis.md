# Data augmentation and synthesis

## 1. Overview & Why This Matters {#overview}

In AI Engineering, **dataset engineering** is the discipline of building, curating, and managing the data that feeds model training. Within that discipline, two of the most powerful and frequently-used levers are:

- **Data Augmentation** — transforming _existing_ data samples to produce new, plausible variations
- **Synthetic Data Synthesis** — generating _net-new_ data samples from scratch (or from learned distributions), independent of specific source records

Both address the same core problems: data scarcity, class imbalance, privacy constraints, costly labeling, and insufficient coverage of edge cases. As of 2025, research shows that synthetic augmentation can yield **3–26% improvements in model performance** in low-data regimes, making these techniques indispensable in the modern AI engineer's toolkit.

The distinction matters in practice:

|Dimension|Augmentation|Synthesis|
|---|---|---|
|Source dependency|Requires existing data|Can work from scratch or from a distribution|
|Transformation type|Perturbations, flips, paraphrases|Net-new generation|
|Fidelity control|High (bounded by source)|Variable (model-dependent)|
|Privacy risk|Lower|Higher (risk of memorization)|
|Common tools|albumentations, nlpaug, SMOTE|GANs, VAEs, Diffusion, LLMs|

---

## 2. Data Augmentation {#data-augmentation}

### Core Concept

Data augmentation applies **label-preserving transformations** to existing data points to increase the effective size and diversity of a training set. The key constraint: a transformation must not change the ground-truth label. Rotating an image of a "cat" still produces a valid cat image. Paraphrasing a "positive review" should still produce a positive review.

The goals are:

- **Regularization**: Force the model to learn invariances (e.g., a cat is a cat regardless of orientation)
- **Dataset size expansion**: More training samples → better generalization
- **Class balance**: Oversample minority classes to counteract imbalance
- **Robustness**: Expose the model to realistic noise and variation

---

### 2a. Image Augmentation

Image augmentation is the most mature domain. Standard transformations include geometric transforms (flip, rotate, crop, scale), color jitter (brightness, contrast, saturation), noise injection (Gaussian, salt-and-pepper), and cutout/erasing.

**Modern advanced approaches:**

- **AutoAugment / RandAugment**: Learn optimal augmentation policies via RL or random search, rather than hand-tuning
- **MixUp**: Interpolate between two samples and their labels — `x̃ = λxᵢ + (1-λ)xⱼ`
- **CutMix**: Replace a rectangular patch of one image with a patch from another
- **AugMix**: Chains multiple augmentations with consistency loss
- **Diffusion-based augmentation**: Use a trained diffusion model to generate semantically meaningful variants

```python
# Example: albumentations pipeline (a standard industry choice)
import albumentations as A
import cv2

transform = A.Compose([
    A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.1),
    A.Rotate(limit=30, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

image = cv2.imread("cat.jpg")
augmented = transform(image=image)["image"]
```

---

### 2b. NLP / Text Augmentation

Text augmentation is more constrained than image augmentation because small changes can flip a label (e.g., adding "not" changes sentiment). Techniques span a spectrum from cheap-but-noisy to expensive-but-high-quality.

**Lexical-level techniques (Easy Data Augmentation, EDA):**

- **Synonym replacement**: Replace _n_ non-stopword tokens with WordNet synonyms
- **Random insertion**: Insert a synonym of a random word at a random position
- **Random swap**: Swap two random words
- **Random deletion**: Delete each word with probability _p_

**Semantic-level techniques (higher quality):**

- **Back translation**: Translate to an intermediate language (e.g., English → French → English) to get paraphrases. Preserves meaning while varying surface form.
- **Contextual word embedding substitution**: Use BERT/RoBERTa masked-prediction to substitute words in their contextual neighborhood
- **T5/Pegasus paraphrase generation**: Prompt a seq2seq model to rephrase the entire sentence

**LLM-based augmentation (state-of-the-art):**

- Prompt an LLM to generate _n_ paraphrases, rewrite in a different style, or produce an equivalent question
- Most powerful for instruction-following and chat datasets

```python
# Example: back-translation using Helsinki-NLP models (HuggingFace)
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text: str, src: str = "en", pivot: str = "fr") -> str:
    def translate(texts, model_name):
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    forward = f"Helsinki-NLP/opus-mt-{src}-{pivot}"
    backward = f"Helsinki-NLP/opus-mt-{pivot}-{src}"
    pivot_text = translate([text], forward)
    return translate(pivot_text, backward)[0]

original = "The product quality exceeded my expectations."
augmented = back_translate(original)
print(f"Original:  {original}")
print(f"Augmented: {augmented}")
```

```python
# Example: EDA-style synonym replacement with nlpaug
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src="wordnet", aug_p=0.3)

texts = [
    "The customer service was terrible and unhelpful.",
    "I loved every moment of this experience.",
]
augmented = aug.augment(texts, n=2)  # 2 augmentations per sample
for original, variants in zip(texts, augmented):
    print(f"Original: {original}")
    for v in variants:
        print(f"  -> {v}")
```

---

### 2c. Tabular Data Augmentation

Tabular augmentation is the hardest modality because:

1. Mixed types (categorical + numerical) break many transformations
2. Inter-column correlations must be preserved (e.g., age and retirement_status are correlated)
3. Domain constraints must hold (negative ages are invalid)

**Numerical techniques:**

- **Gaussian noise injection**: Add `N(0, σ)` noise to numerical features
- **Feature scaling perturbation**: Multiply by `U(1-ε, 1+ε)`
- **Interpolation**: Linear interpolation between two samples (SMOTE-style)

**Categorical techniques:**

- **Label smoothing** (soft targets instead of hard 0/1)
- **Controlled category swap**: Swap categories within a logical constraint group

**Class-imbalance techniques (critical for fraud, medical, etc.):**

- **SMOTE** (Synthetic Minority Over-Sampling Technique): For each minority sample, find its k-nearest neighbors and interpolate along connecting line segments
- **Borderline-SMOTE**: Only augment minority samples near the decision boundary
- **ADASYN** (Adaptive Synthetic): Generate more samples in sparser regions of minority class space

```python
# Example: SMOTE for class imbalance with imbalanced-learn
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from collections import Counter

# Create imbalanced dataset (5% minority class)
X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=5,
    weights=[0.95, 0.05], random_state=42
)
print(f"Before SMOTE: {Counter(y)}")

# Apply SMOTE
smote = SMOTE(sampling_strategy="minority", k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"After SMOTE:  {Counter(y_resampled)}")

# Borderline-SMOTE — better for high-dimensional or noisy data
bl_smote = BorderlineSMOTE(kind="borderline-1", random_state=42)
X_bl, y_bl = bl_smote.fit_resample(X, y)
print(f"After Borderline-SMOTE: {Counter(y_bl)}")

# ADASYN — focuses on harder-to-learn regions
adasyn = ADASYN(sampling_strategy="minority", random_state=42)
X_ada, y_ada = adasyn.fit_resample(X, y)
print(f"After ADASYN: {Counter(y_ada)}")
```

```python
# Example: Gaussian noise injection for numerical augmentation
import pandas as pd
import numpy as np

def augment_numerical(df: pd.DataFrame, num_cols: list, noise_std: float = 0.05, n_copies: int = 3) -> pd.DataFrame:
    """
    Augment numerical features by adding Gaussian noise.
    noise_std: fraction of each column's std dev to use as noise scale
    """
    augmented_frames = [df]
    for _ in range(n_copies):
        noisy = df.copy()
        for col in num_cols:
            col_std = df[col].std()
            noise = np.random.normal(0, noise_std * col_std, size=len(df))
            noisy[col] = noisy[col] + noise
        augmented_frames.append(noisy)
    return pd.concat(augmented_frames, ignore_index=True)

df = pd.DataFrame({
    "age": [25, 34, 45, 55, 62],
    "income": [45000, 72000, 89000, 102000, 135000],
    "label": [0, 1, 1, 0, 1]
})
df_augmented = augment_numerical(df, num_cols=["age", "income"], noise_std=0.03)
print(f"Original size: {len(df)}, Augmented size: {len(df_augmented)}")
```

---

### 2d. Time-Series Augmentation

- **Time warping**: Non-linearly scale the time axis
- **Window slicing / sliding window**: Extract overlapping subsequences
- **Magnitude warping**: Scale amplitude with a smooth random curve
- **Jittering**: Add random noise to each timestep
- **Permutation**: Slice series into segments and shuffle them

```python
# Example: jittering and time warping for time series
import numpy as np

def jitter(series: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    return series + np.random.normal(0, sigma, series.shape)

def time_warp(series: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(len(series))
    knots = np.linspace(0, len(series) - 1, num=knot + 2)
    warp_steps = knots + np.random.normal(0, sigma * len(series), size=knot + 2)
    warp_steps[0], warp_steps[-1] = 0, len(series) - 1
    warper = CubicSpline(knots, np.sort(warp_steps))
    warped = warper(orig_steps)
    return np.interp(warped, orig_steps, series)

ts = np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.normal(0, 0.1, 200)
ts_jittered = jitter(ts, sigma=0.05)
ts_warped = time_warp(ts, sigma=0.1)
```

---

## 3. Synthetic Data Synthesis {#synthetic-data-synthesis}

### Core Concept

Synthetic data synthesis creates **entirely new data points** by learning and sampling from an underlying data distribution. Unlike augmentation, synthesis is not tied to specific source records — it can produce samples in areas of feature space that are absent or sparse in the original dataset.

The applications are broad:

- **Pre-training data generation**: Create instruction-following datasets, QA pairs, code examples at scale
- **Privacy-safe data sharing**: Replace sensitive records with statistically equivalent synthetic ones (healthcare, finance)
- **Rare/edge-case coverage**: Generate examples of rare events (fraud, equipment failure, rare diseases)
- **Domain-specific fine-tuning**: Produce specialized text/records for niche domains where real data is scarce

---

### 3a. Generative Model Architectures

#### GANs (Generative Adversarial Networks)

A generator G and discriminator D play a minimax game. G learns to produce samples indistinguishable from real data. Strong for images; **CTGAN** and **TableGAN** extend this to tabular data.

```python
# CTGAN for tabular synthesis
from ctgan import CTGAN
import pandas as pd

df = pd.read_csv("transactions.csv")
discrete_cols = ["category", "merchant_type", "is_fraud"]

ctgan = CTGAN(epochs=300, batch_size=500, verbose=True)
ctgan.fit(df, discrete_cols)

synthetic_df = ctgan.sample(n=10000)
print(synthetic_df.head())
```

#### VAEs (Variational Autoencoders)

Encode data into a continuous latent space and sample from it. More stable than GANs. Used in **TVAE** (tabular VAE, part of SDV) and for structured data.

#### Diffusion Models

Add noise to data iteratively and learn to reverse it. State-of-the-art for images (Stable Diffusion), and increasingly used for complex tabular/multimodal data generation.

#### SDV (Synthetic Data Vault) — The Swiss Army Knife

SDV is the go-to Python library for tabular, relational, and time-series synthetic data. It wraps multiple backends (Gaussian Copula, CTGAN, TVAE) with a unified API.

```python
# SDV for relational synthetic data
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

df = pd.read_csv("patient_records.csv")

# Infer metadata automatically
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Inspect and fix metadata if needed
metadata.update_column("patient_id", sdtype="id")
metadata.update_column("diagnosis_date", sdtype="datetime", datetime_format="%Y-%m-%d")
metadata.update_column("is_diabetic", sdtype="boolean")

# Train synthesizer
synthesizer = CTGANSynthesizer(metadata, epochs=500)
synthesizer.fit(df)

# Generate synthetic records
synthetic = synthesizer.sample(num_rows=5000)

# Evaluate quality
from sdv.evaluation.single_table import evaluate_quality
quality_report = evaluate_quality(df, synthetic, metadata)
quality_report.get_score()  # Score between 0.0 and 1.0
```

---

### 3b. LLM-Driven Data Synthesis

LLMs are now the dominant approach for text and instruction dataset synthesis. The standard patterns are:

**Pattern 1 — Self-Instruct / Bootstrapping** Prompt an LLM with a handful of seed examples and ask it to generate many more. This is how datasets like Alpaca (52K instructions from 175 seeds) and WizardCoder were produced.

```python
# Example: LLM-based instruction synthesis using the Anthropic API
import anthropic
import json

client = anthropic.Anthropic()

SEED_EXAMPLES = [
    {"instruction": "Explain gradient descent", "response": "Gradient descent is..."},
    {"instruction": "What is overfitting?", "response": "Overfitting occurs when..."},
]

SYSTEM_PROMPT = """You are a data synthesis expert. Generate diverse, high-quality
instruction-response pairs for training a machine learning assistant.
Each pair must be distinct from the seeds provided.
Return a JSON array of objects with keys: instruction, response."""

def synthesize_instructions(seed_examples: list, n: int = 10) -> list:
    seed_text = json.dumps(seed_examples, indent=2)
    user_prompt = f"""Here are seed examples:
{seed_text}

Generate {n} new diverse instruction-response pairs on machine learning topics.
Vary the difficulty, style, and topic. Return only a valid JSON array."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return json.loads(response.content[0].text)

new_pairs = synthesize_instructions(SEED_EXAMPLES, n=20)
print(f"Generated {len(new_pairs)} new instruction pairs")
```

**Pattern 2 — Persona-driven Synthesis** Generate diverse examples by prompting the LLM to adopt different personas, skill levels, or writing styles. This is the key to diversity without repetition.

```python
# Persona-driven synthesis for diverse question generation
import anthropic

client = anthropic.Anthropic()

PERSONAS = [
    "a curious 10-year-old asking their first question about computers",
    "a senior software engineer debugging a production issue",
    "a non-technical product manager trying to understand AI limitations",
    "a PhD researcher looking for nuanced technical depth",
]

def generate_with_persona(topic: str, persona: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Write a question and ideal answer about "{topic}" from the perspective of: {persona}.
Format as JSON: {{"question": "...", "answer": "...", "persona": "..."}}"""
        }]
    )
    import json
    return json.loads(response.content[0].text)

topic = "how neural networks learn"
dataset = [generate_with_persona(topic, p) for p in PERSONAS]
```

**Pattern 3 — Document-Grounded Synthesis (RAG-style)** Feed real documents to an LLM and ask it to generate QA pairs, summaries, or annotations grounded in those documents. This is powerful for domain-specific fine-tuning datasets.

```python
# RAG-style dataset generation from real documents
def generate_qa_from_document(document_text: str, n_questions: int = 5) -> list:
    import anthropic, json
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system="You are an expert at generating high-quality training data. Always respond with valid JSON only.",
        messages=[{
            "role": "user",
            "content": f"""Given this document:

---
{document_text[:3000]}
---

Generate {n_questions} diverse question-answer pairs that test comprehension of this document.
Include easy, medium, and hard questions.
Return a JSON array: [{{"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}}]"""
        }]
    )
    return json.loads(response.content[0].text)
```

**Pattern 4 — LLM-as-Judge + Filtering** Generate a large pool of synthetic samples, then use another LLM call (or rule-based filters) to score and filter for quality. This is the "generate then curate" paradigm used by modern RLHF-style pipelines.

```python
# Quality filtering with LLM-as-judge
def score_sample(sample: dict, rubric: str) -> dict:
    import anthropic, json
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Score this training sample on a 1-5 scale based on: {rubric}

Sample:
Instruction: {sample['instruction']}
Response: {sample['response']}

Respond with JSON only: {{"score": <1-5>, "reason": "..."}}"""
        }]
    )
    result = json.loads(response.content[0].text)
    return {**sample, "quality_score": result["score"], "quality_reason": result["reason"]}

def filter_dataset(dataset: list, min_score: int = 4) -> list:
    rubric = "clarity, factual accuracy, instructional value, and appropriate length"
    scored = [score_sample(s, rubric) for s in dataset]
    return [s for s in scored if s["quality_score"] >= min_score]
```

---

### 3c. Scenario Simulation & Edge Case Generation

A specific and highly valuable use of synthesis is generating **rare-but-critical scenarios** that real data doesn't cover adequately.

```python
# Edge case generation: adversarial / rare scenarios for a classifier
EDGE_CASE_PROMPT = """Generate {n} edge-case financial transaction records that are:
- Borderline fraud cases (not obviously fraudulent, but suspicious)
- Include realistic merchant names, amounts, and patterns
- Include the ground truth label (is_fraud: true/false)
Return as a JSON array with fields: amount, merchant, category, time_of_day, is_fraud, reasoning"""

def generate_edge_cases(n: int = 50) -> list:
    import anthropic, json
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        messages=[{"role": "user", "content": EDGE_CASE_PROMPT.format(n=n)}]
    )
    return json.loads(response.content[0].text)
```

---

## 4. Use Cases & When to Apply Each Strategy {#use-cases}

|Problem|Recommended Approach|Why|
|---|---|---|
|Image classifier with 500 training samples|Heavy image augmentation (albumentations + RandAugment)|Cheap, high-quality, well-validated|
|NLP classifier with 200 examples|Back-translation + EDA + LLM paraphrase|Mix cheap and quality|
|Fraud detection (0.1% positive rate)|SMOTE/ADASYN + CTGAN for minority class|Imbalance is the core problem|
|LLM instruction fine-tuning|Self-instruct + persona-driven synthesis + LLM-as-judge filter|Pure synthesis at scale|
|Healthcare ML (HIPAA data, can't share real records)|SDV/CTGAN to generate synthetic patient records|Privacy-preserving utility|
|Rare equipment failure prediction|CTGAN + scenario simulation|Real failure data is too sparse|
|Domain-specific RAG improvement|Document-grounded QA synthesis|Ground truth from documents|
|NLP robustness testing|Adversarial text augmentation|Find failure modes|
|Code generation fine-tuning|LLM self-synthesis + execution-validated filtering|WizardCoder pattern|

---

## 5. Risks, Pitfalls & Mitigations {#risks}

### 5a. Bias Amplification

**Risk**: If real data encodes biases (e.g., underrepresentation of minorities in hiring data), synthetic data trained on that real data will propagate and potentially amplify those biases.

**Mitigation**:

- Audit original data for bias before synthesizing
- Use controlled generation to enforce demographic parity
- Apply post-hoc bias evaluation on synthetic data before use
- Use tools like `aif360` (IBM AI Fairness 360) or `fairlearn` to measure distributional fairness

---

### 5b. Model Collapse ⚠️ (Critical for 2024–2025)

**Risk**: When models are iteratively trained on outputs of previous model generations (recursive training), the output distribution degrades. Outputs become less diverse, creative, and eventually degrade toward high-probability "safe" patterns. This has been confirmed in peer-reviewed Nature research.

**Mechanics**:

- Each generation amplifies statistical modes and forgets tails
- Rare patterns (rare words, unusual reasoning chains) are progressively lost
- Affects LLMs, diffusion models, and VAEs

**Key research findings (2025)**:

- Replacing real data with synthetic data each generation guarantees collapse
- **Accumulating** synthetic data alongside a non-shrinking real-data anchor avoids collapse
- Higher diversity of synthetic data sources mitigates collapse significantly
- Using synthetic data from a single source model worsens self-preference bias

**Mitigation checklist**:

```
✅ Never replace your real data corpus — only augment/supplement it
✅ Use multiple diverse synthesis sources (not just one LLM)
✅ Implement provenance tagging on all synthetic samples
✅ Watermark synthetic outputs so future pipelines can identify and filter them
✅ Retain a held-out "clean real data" anchor that is never contaminated
✅ Monitor output diversity metrics (vocabulary entropy, perplexity, n-gram diversity)
✅ Use LLM-as-judge filtering to reject low-quality synthetic samples early
✅ Apply Maximum Mean Discrepancy (MMD) or entropy checks to ensure
   synthetic data adds new information vs. duplicating existing patterns
```

```python
# Monitoring output diversity — a simple diversity metric
from collections import Counter
import numpy as np

def distinct_n(texts: list, n: int = 2) -> float:
    """Compute Distinct-N: fraction of unique n-grams across all texts."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        all_ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

# Use this to monitor synthetic output diversity over time
generations = [gen_0_outputs, gen_1_outputs, gen_2_outputs]
for i, gen in enumerate(generations):
    print(f"Gen {i} Distinct-2: {distinct_n(gen, 2):.3f}")
    # Declining Distinct-N is an early warning of collapse
```

---

### 5c. Data Leakage & Privacy

**Risk**: Generative models (especially LLMs) can memorize training data and reproduce it in synthetic outputs, creating privacy violations even when the intent was to anonymize data.

**Mitigation**:

- Use differential privacy during generative model training (`opacus` for PyTorch)
- Run membership inference attacks on synthetic outputs to test for leakage
- Apply regex/NER-based PII scrubbing as a post-processing step
- Prefer statistical/distributional synthesis methods (Gaussian Copula) for highly sensitive domains

```python
# PII scrubbing from synthetic text outputs
import re

PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "phone": r'\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "credit_card": r'\b(?:\d[ -]?){13,16}\b',
}

def scrub_pii(text: str) -> str:
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
    return text
```

---

### 5d. Distribution Mismatch

**Risk**: Synthetic data that doesn't match the real deployment distribution will hurt model performance — the model overfits to synthetic artifacts rather than learning the true task.

**Evaluation approach: Train-on-Synthetic, Test-on-Real (TSTR)**

```python
# TSTR evaluation framework
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def tstr_evaluation(real_df, synthetic_df, feature_cols, target_col):
    """
    Train on Synthetic, Test on Real.
    If TSTR score ≈ TRTR score, synthetic data is high quality.
    """
    X_real = real_df[feature_cols]
    y_real = real_df[target_col]
    X_syn = synthetic_df[feature_cols]
    y_syn = synthetic_df[target_col]

    # Split real data for testing only
    _, X_test, _, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

    # Train on real (TRTR baseline)
    X_train_real, _, y_train_real, _ = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
    clf_real = RandomForestClassifier(random_state=42).fit(X_train_real, y_train_real)
    trtr_report = classification_report(y_test, clf_real.predict(X_test))

    # Train on synthetic (TSTR)
    clf_syn = RandomForestClassifier(random_state=42).fit(X_syn, y_syn)
    tstr_report = classification_report(y_test, clf_syn.predict(X_test))

    print("=== TRTR (Train-Real Test-Real) ===")
    print(trtr_report)
    print("=== TSTR (Train-Synthetic Test-Real) ===")
    print(tstr_report)
```

---

## 6. Tooling & Ecosystem {#tooling}

### Image

|Library|Use Case|
|---|---|
|`albumentations`|Fast, composable image transforms|
|`torchvision.transforms`|PyTorch-native image augmentation|
|`imgaug`|Rich library with bounding-box-aware augmentation|
|`Augmentor`|Pipeline-based augmentation|

### NLP / Text

|Library|Use Case|
|---|---|
|`nlpaug`|Comprehensive NLP augmentation (EDA, back-translation, BERT, etc.)|
|`textaugment`|Lightweight EDA implementation|
|`transformers` (HuggingFace)|Back-translation via MarianMT, T5 paraphrase|
|`checklist`|Behavioral testing + augmentation|

### Tabular

|Library|Use Case|
|---|---|
|`imbalanced-learn`|SMOTE, Borderline-SMOTE, ADASYN|
|`sdv` (Synthetic Data Vault)|CTGAN, TVAE, Gaussian Copula for tabular + relational + time-series|
|`ctgan`|Standalone CTGAN for tabular synthesis|
|`ydata-synthetic`|Multiple GAN architectures for tabular data|
|`gretel-synthetics`|Cloud-hosted synthetic data platform|

### LLM-Driven Synthesis

|Tool|Use Case|
|---|---|
|Anthropic API (Claude)|High-quality instruction/QA synthesis, LLM-as-judge|
|`distilabel` (Argilla)|Scalable synthetic data pipelines with LLM backends|
|`DataDreamer`|Research-grade LLM-powered dataset synthesis|
|HuggingFace Synthetic Data Generator|No-code synthetic dataset generation|

### Time-Series

|Library|Use Case|
|---|---|
|`tsaug`|Time-series augmentation transforms|
|`torchaudio`|Audio/waveform augmentation|
|`audiomentations`|Deep-learning-friendly audio augmentation|

### Evaluation

|Tool|Purpose|
|---|---|
|`sdmetrics`|Fidelity, utility, privacy metrics for synthetic data|
|`aif360`|Bias/fairness evaluation|
|`deepeval`|LLM output evaluation for synthetic instruction data|

---

## 7. Decision Framework {#decision-framework}

```
START: I need more / better training data
│
├── What is my data modality?
│   ├── Image → albumentations + consider diffusion-based synthesis for low-data
│   ├── Text/NLP → EDA for fast baseline; back-translation for quality; LLM for scale
│   ├── Tabular → check imbalance first (SMOTE), then CTGAN/SDV for volume
│   └── Time-Series → tsaug, jittering, window slicing
│
├── What is my primary problem?
│   ├── Not enough data total → Synthesis (LLM / GAN / VAE / Diffusion)
│   ├── Class imbalance → SMOTE/ADASYN first; CTGAN if complex distributions
│   ├── Missing edge cases → Targeted synthesis / scenario simulation
│   ├── Privacy prevents sharing real data → SDV/CTGAN synthetic replacement
│   └── Improving model robustness → Augmentation (noise, perturbation, adversarial)
│
├── Do I have existing real data?
│   ├── Yes → Augmentation is always the first choice (cheaper, safer)
│   └── No/Very little → Synthesis required; use LLMs for text, GANs for structured
│
└── How will I validate quality?
    ├── Tabular → TSTR evaluation + sdmetrics fidelity score
    ├── Text → Distinct-N diversity + LLM-as-judge scoring
    └── ALL → Always test: does model trained with augmented/synthetic data
              actually perform better on held-out REAL data?
```

---

## 8. Production Checklist {#checklist}

Before deploying augmented or synthetic data in a training pipeline:

```
DATA QUALITY
□ Validated that augmentations preserve ground-truth labels
□ Checked inter-column correlations are maintained (tabular)
□ Ran TSTR evaluation — synthetic data score within acceptable range of TRTR
□ Measured Distinct-N or equivalent diversity metric on generated text
□ Spot-checked 100+ synthetic samples manually

BIAS & FAIRNESS
□ Audited source data for representation bias before synthesis
□ Verified synthetic data doesn't amplify demographic imbalance
□ Run fairness evaluation (aif360 or fairlearn) on the augmented training set

PRIVACY
□ PII scrubbing applied to all synthetic text outputs
□ Membership inference attack run on sample of synthetic data
□ Sensitive columns (SSN, DOB, name) are not being memorized/reproduced

MODEL COLLAPSE PREVENTION
□ All synthetic data is tagged with source and generation timestamp
□ Real data corpus is preserved separately and not replaced
□ Diversity metrics logged and alarmed for degradation
□ Training pipeline will alert if synthetic-to-real ratio exceeds threshold

PIPELINE HYGIENE
□ Augmentation applied only to training split (never val/test)
□ Augmentation is deterministic-or-seeded for reproducibility
□ Synthetic data stored separately from real data with clear provenance
□ Quality filter (LLM-as-judge or rule-based) applied before inclusion
```

---

_Last updated: April 2026. Key references: arXiv:2503.14023 (LLM synthetic data survey), Nature (Shumailov et al., model collapse), arXiv:2511.01490 (synthetic data diversity), SDV documentation, imbalanced-learn documentation._