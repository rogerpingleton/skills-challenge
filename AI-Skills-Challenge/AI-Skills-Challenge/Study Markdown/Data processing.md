# Data processing

## Introduction

Data processing is the operational core of dataset engineering. Before any model is trained, fine-tuned, or queried, raw data must be transformed from a messy, heterogeneous state into a clean, structured, and semantically meaningful form that a model can learn from effectively. As an AI Engineer, you will spend a significant portion of your time in this domain — research consistently shows that **data engineers spend ~57% of their time building and maintaining datasets and ETL/ELT pipelines**.

The central truth of the field is simple: **model quality is bounded by data quality**. No architecture, no optimizer, and no prompt engineering can compensate for noisy, biased, or poorly structured training data.

In 2025–2026, the data processing landscape has shifted significantly. Open table formats (Apache Iceberg, Delta Lake) have standardized data lake architectures, streaming-first ingestion has become the default, and AI is now being embedded directly into transformation pipelines through tools like Databricks Lakeflow's AI Functions. As a practitioner, you operate at the intersection of classical data engineering and AI system design.

---

## The Data Processing Lifecycle

Data processing for AI follows a structured lifecycle. Each stage has well-defined inputs, outputs, and failure modes that you must control:

```
Raw Sources
    │
    ▼
[1] Ingestion         ← APIs, databases, files, streams, web scraping
    │
    ▼
[2] Cleaning          ← Remove nulls, duplicates, noise, PII, malformed records
    │
    ▼
[3] Transformation    ← Normalize, encode, parse, tokenize, format standardization
    │
    ▼
[4] Feature Eng.      ← Derive new features, embeddings, aggregations
    │
    ▼
[5] Validation        ← Schema checks, statistical assertions, drift detection
    │
    ▼
[6] Storage/Version   ← Parquet/Delta/Iceberg, DVC, lakeFS
    │
    ▼
[7] Orchestration     ← Airflow, Dagster, Prefect — scheduling, monitoring, alerting
    │
    ▼
  Model-Ready Dataset
```

Each of these stages is covered below with techniques, examples, and tool recommendations.

---

## Stage 1: Data Ingestion

Ingestion is the process of pulling raw data from its source into your pipeline. In 2025, streaming-first ingestion is the default, with transformation, schema evolution, and observability increasingly embedded at the ingestion layer.

### Source Types

**Structured:** SQL databases (Postgres, MySQL), data warehouses (Snowflake, BigQuery, Redshift), CSV, Parquet, Excel.

**Semi-structured:** JSON, XML, AVRO, log files, API responses.

**Unstructured:** PDFs, HTML/web pages, images, audio/video, email archives, Confluence pages, Slack archives.

### Batch vs. Streaming

**Batch ingestion** processes data in large chunks on a schedule (e.g., nightly). It is simpler, easier to debug, and appropriate for training dataset construction.

**Streaming ingestion** processes data continuously as events arrive (e.g., Kafka topics). It is appropriate for RAG pipelines, real-time feature stores, and systems where data freshness matters.

### Python Example: Multi-Source Batch Ingestion

```python
import polars as pl
import duckdb
import requests
from pathlib import Path

# --- Ingest from CSV ---
df_csv = pl.read_csv("data/customer_records.csv", infer_schema_length=10000)

# --- Ingest from Parquet (efficient columnar format) ---
df_parquet = pl.read_parquet("data/transactions/*.parquet")

# --- Ingest from SQL database ---
conn = duckdb.connect("warehouse.duckdb")
df_sql = conn.execute("SELECT * FROM events WHERE event_date >= '2024-01-01'").pl()

# --- Ingest from REST API ---
def ingest_api(endpoint: str, params: dict) -> pl.DataFrame:
    response = requests.get(endpoint, params=params, timeout=30)
    response.raise_for_status()
    return pl.from_dicts(response.json()["data"])

df_api = ingest_api("https://api.example.com/records", {"limit": 5000})

# --- Ingest unstructured: PDF text extraction ---
import pymupdf  # fitz

def extract_pdf_text(pdf_path: str) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    records = []
    for i, page in enumerate(doc):
        records.append({"source": pdf_path, "page": i + 1, "text": page.get_text()})
    return records

pdf_records = extract_pdf_text("data/manual.pdf")
df_pdf = pl.from_dicts(pdf_records)
```

### Best Practices: Ingestion

- **Validate schema immediately at ingestion.** Catching schema drift at the source is far cheaper than discovering it downstream.
- **Store raw data in its original form** before any transformation. Use a `raw/` layer in your data lake. You will need to reprocess it when your pipeline logic changes.
- **Use Parquet as your default intermediate format.** It is columnar, compressed, and natively supported by every major processing engine.
- **Implement idempotent ingestion.** Re-running an ingestion job should not produce duplicate records. Use watermarks, source checksums, or deduplication keys.
- **Log provenance.** Record where each record came from, when it was ingested, and with what version of the pipeline. This is essential for reproducibility and debugging.

---

## Stage 2: Data Cleaning

Cleaning is the most labor-intensive and high-impact stage of data processing. Its goal is to produce a dataset that is complete, consistent, accurate, and free of artifacts that would corrupt model learning.

### 2.1 Handling Missing Values

Missing values distort distributions and can cause silent failures in downstream transformations. Your strategy depends on the nature of the missingness:

**Missing Completely at Random (MCAR):** Drop rows if the percentage is small (<5%). Otherwise, impute.

**Missing at Random (MAR):** Impute using statistical methods (median for numerical, mode for categorical) or model-based imputation (e.g., k-NN imputation).

**Missing Not at Random (MNAR):** The fact of missingness is itself informative. Add a binary indicator column and impute.

```python
import polars as pl

df = pl.read_parquet("data/records.parquet")

# Inspect missingness
missing_report = df.null_count()
print(missing_report)

# Strategy 1: Drop rows with any null in critical columns
df_clean = df.drop_nulls(subset=["user_id", "timestamp", "label"])

# Strategy 2: Fill numerical nulls with median
median_age = df["age"].median()
df_clean = df_clean.with_columns(pl.col("age").fill_null(median_age))

# Strategy 3: Fill categorical nulls with a sentinel value
df_clean = df_clean.with_columns(pl.col("category").fill_null("UNKNOWN"))

# Strategy 4: Add missingness indicator before imputing
df_clean = df_clean.with_columns(
    pl.col("income").is_null().cast(pl.Int8).alias("income_was_null"),
    pl.col("income").fill_null(df["income"].median())
)
```

### 2.2 Deduplication

Duplicate records in training data cause a model to memorize and overfit. For LLM training, near-duplicate documents are particularly harmful, causing the model to regurgitate memorized text rather than generalize.

```python
# Exact deduplication
df_dedup = df.unique(subset=["record_id"])

# Content-level deduplication (for text data)
df_dedup = df.unique(subset=["text_content"])

# Near-duplicate detection using MinHash / SimHash (for LLM datasets)
# Use the `datasketch` library for scalable near-dedup
from datasketch import MinHash, MinHashLSH

def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode("utf8"))
    return m

lsh = MinHashLSH(threshold=0.8, num_perm=128)
texts = df["text_content"].to_list()
hashes = [(i, get_minhash(t)) for i, t in enumerate(texts)]

# Insert into LSH index
for i, mh in hashes:
    lsh.insert(str(i), mh)

# Find duplicates
duplicates = set()
for i, mh in hashes:
    result = lsh.query(mh)
    if len(result) > 1:
        # Mark all but the first occurrence as duplicates
        for dup in result[1:]:
            duplicates.add(int(dup))

df_dedup = df.filter(~pl.Series(range(len(df))).is_in(list(duplicates)))
```

### 2.3 Outlier Detection and Removal

Outliers can represent genuine rare events (valuable) or data errors (harmful). You must distinguish between them.

```python
# Z-score based outlier detection
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df["value"].to_numpy()))
df_no_outliers = df.filter(pl.Series(z_scores) < 3.0)

# IQR-based outlier detection (more robust)
Q1 = df["value"].quantile(0.25)
Q3 = df["value"].quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df.filter(
    (pl.col("value") >= Q1 - 1.5 * IQR) &
    (pl.col("value") <= Q3 + 1.5 * IQR)
)

# Isolation Forest for multivariate outlier detection
from sklearn.ensemble import IsolationForest

features = df.select(["feature_a", "feature_b", "feature_c"]).to_numpy()
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(features)  # -1 = outlier, 1 = inlier

df_clean = df.filter(pl.Series(outlier_labels) == 1)
```

### 2.4 Text-Specific Cleaning (for NLP/LLM Pipelines)

```python
import re
import unicodedata
import ftfy  # fixes text encoding issues

def clean_text(text: str) -> str:
    # Fix broken unicode encoding
    text = ftfy.fix_text(text)

    # Unicode normalization (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)

    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove non-printable characters
    text = "".join(c for c in text if c.isprintable() or c == "\n")

    return text

# Apply to a Polars DataFrame
df_clean = df.with_columns(
    pl.col("text").map_elements(clean_text, return_dtype=pl.Utf8)
)
```

### 2.5 PII Removal (Privacy Compliance)

For GDPR, CCPA, and responsible AI development, PII must be identified and removed or anonymized before a dataset is used for training.

```python
import spacy
import re

nlp = spacy.load("en_core_web_lg")

PII_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PHONE": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

def redact_pii(text: str) -> str:
    # Regex-based PII removal
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{label}]", text)

    # NER-based person/location name removal
    doc = nlp(text)
    for ent in reversed(doc.ents):
        if ent.label_ in ("PERSON", "GPE", "LOC", "ORG"):
            text = text[:ent.start_char] + f"[{ent.label_}]" + text[ent.end_char:]

    return text
```

---

## Stage 3: Data Transformation & Normalization

Transformation converts cleaned raw data into a format suitable for model consumption. This includes type coercion, encoding, scaling, and format standardization.

### 3.1 Numerical Feature Scaling

Most ML algorithms are sensitive to the scale of input features. Two standard approaches:

**Min-Max Normalization:** Scales features to [0, 1]. Best when the distribution is not Gaussian and you need bounded values.

**Z-score Standardization:** Centers features to mean 0, standard deviation 1. Best for Gaussian-distributed features and algorithms that assume normality (SVMs, linear regression).

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

data = df.select(["age", "income", "transaction_amount"]).to_numpy()

# Min-Max normalization
scaler_minmax = MinMaxScaler()
data_normalized = scaler_minmax.fit_transform(data)

# Z-score standardization
scaler_std = StandardScaler()
data_standardized = scaler_std.fit_transform(data)

# IMPORTANT: Fit on TRAINING data only, then transform train and test
# Never fit on the full dataset — this constitutes data leakage
X_train, X_test = data[:8000], data[8000:]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit here
X_test_scaled = scaler.transform(X_test)         # only transform here
```

### 3.2 Categorical Encoding

```python
import polars as pl

# One-hot encoding (for nominal categories with low cardinality)
df_encoded = df.to_dummies(columns=["product_category", "region"])

# Label encoding (for ordinal categories or tree-based models)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = df.with_columns(
    pl.Series(le.fit_transform(df["priority"].to_list())).alias("priority_encoded")
)

# Target encoding (for high-cardinality categoricals with supervised tasks)
# Replace category with mean of target variable — reduces dimensionality
target_means = df.group_by("city").agg(pl.col("churn").mean().alias("city_target_enc"))
df = df.join(target_means, on="city", how="left")

# Frequency encoding (unsupervised — useful for LLM pre-training data curation)
freq = df.group_by("domain").agg(pl.len().alias("domain_freq"))
df = df.join(freq, on="domain", how="left")
```

### 3.3 Text Normalization for NLP

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download(["punkt", "wordnet", "stopwords"])

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def normalize_text(text: str, lemmatize: bool = True, remove_stopwords: bool = False) -> str:
    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords (use with care — can hurt LLM training)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    # Lemmatize: "running" → "run", "better" → "good"
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

# Note: For LLM fine-tuning, aggressive normalization (stopword removal,
# lemmatization) is usually NOT recommended. Preserve natural language structure.
# These techniques are more appropriate for classical ML text features.
```

### 3.4 Data Type Optimization

Storing data with the most memory-efficient type reduces RAM consumption and speeds up processing:

```python
import polars as pl

def optimize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Automatically downcast numeric columns to their smallest valid type."""
    exprs = []
    for col_name in df.columns:
        dtype = df[col_name].dtype
        if dtype == pl.Int64:
            col_min = df[col_name].min()
            col_max = df[col_name].max()
            if col_min >= 0 and col_max <= 255:
                exprs.append(pl.col(col_name).cast(pl.UInt8))
            elif col_min >= -128 and col_max <= 127:
                exprs.append(pl.col(col_name).cast(pl.Int8))
            elif col_min >= -32768 and col_max <= 32767:
                exprs.append(pl.col(col_name).cast(pl.Int16))
            elif col_min >= -2**31 and col_max <= 2**31 - 1:
                exprs.append(pl.col(col_name).cast(pl.Int32))
            else:
                exprs.append(pl.col(col_name))
        elif dtype == pl.Float64:
            exprs.append(pl.col(col_name).cast(pl.Float32))
        else:
            exprs.append(pl.col(col_name))
    return df.with_columns(exprs)

df_optimized = optimize_dtypes(df)
```

---

## Stage 4: Feature Engineering

Feature engineering is the process of creating new input representations from raw data that make the model's job easier. It sits at the boundary between domain knowledge and machine learning intuition.

### 4.1 Temporal Features

```python
import polars as pl

df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

df = df.with_columns([
    pl.col("timestamp").dt.year().alias("year"),
    pl.col("timestamp").dt.month().alias("month"),
    pl.col("timestamp").dt.weekday().alias("day_of_week"),
    pl.col("timestamp").dt.hour().alias("hour_of_day"),
    pl.col("timestamp").dt.week().alias("week_of_year"),
    # Cyclical encoding (preserves periodicity for ML models)
    (2 * 3.14159 * pl.col("timestamp").dt.month() / 12).sin().alias("month_sin"),
    (2 * 3.14159 * pl.col("timestamp").dt.month() / 12).cos().alias("month_cos"),
])
```

### 4.2 Aggregation / Window Features

```python
# Rolling statistics (e.g., 7-day average transaction value per user)
df = df.sort("user_id", "timestamp")

df = df.with_columns([
    pl.col("amount")
      .rolling_mean(window_size=7)
      .over("user_id")
      .alias("rolling_7d_avg_amount"),
    pl.col("amount")
      .rolling_std(window_size=7)
      .over("user_id")
      .alias("rolling_7d_std_amount"),
])

# Lag features (value N steps ago)
df = df.with_columns([
    pl.col("amount").shift(1).over("user_id").alias("prev_amount"),
    pl.col("amount").shift(7).over("user_id").alias("amount_7d_ago"),
])
```

### 4.3 Text Embeddings (for Semantic ML Tasks)

For modern AI pipelines, raw text is converted into dense vector embeddings for downstream use in models, vector databases, or RAG systems.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import polars as pl

model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, fast

texts = df["description"].to_list()

# Batch encode for efficiency
embeddings = model.encode(
    texts,
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True  # L2 normalize for cosine similarity
)

# embeddings.shape → (n_records, 384)
np.save("embeddings/description_embeddings.npy", embeddings)

# For larger-scale embedding generation, use NVIDIA NeMo Curator
# or Databricks' ai_query() function for distributed batch inference
```

### 4.4 Interaction Features

```python
# Ratio features
df = df.with_columns(
    (pl.col("clicks") / (pl.col("impressions") + 1)).alias("ctr"),
    (pl.col("revenue") / (pl.col("cost") + 1e-6)).alias("roas"),
)

# Cross features (manual polynomial interaction)
df = df.with_columns(
    (pl.col("age") * pl.col("tenure_months")).alias("age_x_tenure"),
)
```

---

## Stage 5: Data Validation & Quality Assurance

Validation is the safety net of your pipeline. It catches data quality issues before they silently corrupt your model. This is one of the most underinvested areas in AI engineering, and one of the highest-leverage.

### Philosophy: Data Contracts

Treat your datasets like software interfaces. Define explicit contracts (schemas, statistical properties, value constraints) and test them automatically on every pipeline run.

### 5.1 Schema Validation with Pandera

```python
import pandera.polars as pa
import polars as pl

# Define a schema (the "contract" for your dataset)
schema = pa.DataFrameSchema({
    "user_id": pa.Column(pl.Utf8, nullable=False, unique=True),
    "age": pa.Column(pl.Int32, pa.Check.between(0, 120), nullable=True),
    "email": pa.Column(pl.Utf8, pa.Check.str_matches(r"[^@]+@[^@]+\.[^@]+"), nullable=True),
    "label": pa.Column(pl.Int8, pa.Check.isin([0, 1]), nullable=False),
    "score": pa.Column(pl.Float32, pa.Check.between(0.0, 1.0), nullable=False),
    "timestamp": pa.Column(pl.Datetime, nullable=False),
})

# Validate — raises SchemaError with detailed report on failure
try:
    validated_df = schema.validate(df, lazy=True)  # lazy=True collects all errors
    print("✅ Schema validation passed")
except pa.errors.SchemaErrors as e:
    print(f"❌ Schema validation failed:\n{e.failure_cases}")
```

### 5.2 Statistical Assertions with Great Expectations

Great Expectations is the industry-standard tool for asserting statistical properties of data at scale.

```python
import great_expectations as gx

context = gx.get_context()

# Define expectations on your dataset
expectation_suite = context.add_or_update_expectation_suite("my_dataset_suite")

# Column-level expectations
validator = context.get_validator(...)
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_unique("user_id")
validator.expect_column_values_to_be_between("age", min_value=0, max_value=120)
validator.expect_column_proportion_of_unique_values_to_be_between(
    "label", min_value=0.3, max_value=0.7  # Enforce class balance
)
validator.expect_column_mean_to_be_between("score", min_value=0.4, max_value=0.6)
validator.expect_table_row_count_to_be_between(min_value=10000)

# Save and run
validator.save_expectation_suite()
results = validator.validate()
print(f"Validation success: {results.success}")
```

### 5.3 Data Distribution Monitoring (Drift Detection)

Data drift — where the statistical distribution of production data diverges from training data — is a primary cause of model degradation in production.

```python
from scipy.stats import ks_2samp, chi2_contingency
import numpy as np

def detect_numerical_drift(reference: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> dict:
    """Kolmogorov-Smirnov test for continuous feature drift."""
    statistic, p_value = ks_2samp(reference, current)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < alpha
    }

def detect_categorical_drift(reference: list, current: list) -> dict:
    """Chi-squared test for categorical feature drift."""
    # Build contingency table
    all_categories = set(reference + current)
    ref_counts = [reference.count(c) for c in all_categories]
    cur_counts = [current.count(c) for c in all_categories]
    chi2, p_value, _, _ = chi2_contingency([ref_counts, cur_counts])
    return {"chi2": chi2, "p_value": p_value, "drift_detected": p_value < 0.05}

# Usage
ref_scores = train_df["score"].to_numpy()
prod_scores = production_df["score"].to_numpy()
result = detect_numerical_drift(ref_scores, prod_scores)
if result["drift_detected"]:
    print(f"⚠️  Drift detected in 'score': KS={result['statistic']:.3f}, p={result['p_value']:.4f}")
```

### 5.4 Class Balance Validation

Imbalanced datasets cause classifiers to develop systematic biases toward the majority class.

```python
# Check class distribution
label_dist = df.group_by("label").agg(pl.len().alias("count"))
label_dist = label_dist.with_columns(
    (pl.col("count") / pl.col("count").sum()).alias("proportion")
)
print(label_dist)

# Remedy: Oversample minority class (SMOTE)
from imblearn.over_sampling import SMOTE

X = df.select(feature_cols).to_numpy()
y = df["label"].to_numpy()
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Remedy: Undersample majority class
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

---

## Stage 6: Data Storage & Versioning

### 6.1 Storage Formats

|Format|Best Use Case|Advantages|
|---|---|---|
|**Parquet**|Default for structured/tabular|Columnar, compressed, universally supported|
|**Delta Lake**|Mutable tables, ACID transactions|Versioned, supports upserts, time travel|
|**Apache Iceberg**|Multi-engine lakehouse tables|Schema evolution, partition pruning|
|**JSONL**|Text/LLM datasets|Human-readable, line-delimited, streamable|
|**HDF5**|Large numerical arrays (embeddings)|Efficient for multidimensional arrays|
|**Arrow/Feather**|In-memory exchange, fast I/O|Zero-copy, fastest read/write|

```python
import polars as pl

# Write / read Parquet (recommended default)
df.write_parquet("data/processed/records.parquet", compression="zstd")
df = pl.read_parquet("data/processed/records.parquet")

# Partition by date for large datasets
df.write_parquet(
    "data/processed/records/",
    use_pyarrow=True,
    pyarrow_options={"partition_cols": ["year", "month"]}
)

# JSONL for LLM training datasets
with open("data/llm_training.jsonl", "w") as f:
    for record in df.iter_rows(named=True):
        f.write(json.dumps(record) + "\n")
```

### 6.2 Dataset Versioning with DVC

Data Version Control (DVC) extends Git to track datasets and model artifacts, enabling reproducible experiments.

```bash
# Initialize DVC in your project
git init && dvc init

# Track a dataset file
dvc add data/processed/records.parquet

# Push to remote storage (S3, GCS, Azure Blob, etc.)
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push

# On another machine or after a git checkout:
dvc pull

# Tag a dataset version
git tag dataset-v1.0 && git push --tags
```

```python
# In Python: load a specific versioned dataset
import subprocess
subprocess.run(["git", "checkout", "dataset-v1.0"])
subprocess.run(["dvc", "pull"])
df = pl.read_parquet("data/processed/records.parquet")
```

---

## Stage 7: Pipeline Orchestration

Orchestration turns your ad-hoc scripts into a reliable, monitored, production-grade system. It handles scheduling, dependency management, retries, alerting, and observability.

### Tools

**Dagster** — Asset-centric orchestration. Models your pipeline as a graph of data assets. Excellent for AI/ML workloads with built-in data catalog. Recommended for new pipelines in 2025.

**Apache Airflow** — The most widely deployed orchestrator. DAG-based. Mature ecosystem and extensive integrations.

**Prefect** — Pythonic, low-boilerplate. Great for teams that want fast iteration without heavy infrastructure.

### Dagster Example: Asset-Based Pipeline

```python
from dagster import asset, AssetIn, define_asset_job
import polars as pl

@asset
def raw_records() -> pl.DataFrame:
    """Ingest raw records from source."""
    return pl.read_parquet("s3://data-lake/raw/records/*.parquet")

@asset(ins={"raw": AssetIn("raw_records")})
def cleaned_records(raw: pl.DataFrame) -> pl.DataFrame:
    """Clean and deduplicate raw records."""
    return (
        raw
        .drop_nulls(subset=["user_id", "label"])
        .unique(subset=["user_id"])
        .filter(pl.col("age").is_between(0, 120))
    )

@asset(ins={"cleaned": AssetIn("cleaned_records")})
def feature_engineered_records(cleaned: pl.DataFrame) -> pl.DataFrame:
    """Add derived features."""
    return cleaned.with_columns([
        (pl.col("revenue") / pl.col("cost")).alias("roas"),
        pl.col("timestamp").dt.month().alias("month"),
    ])

# Define and materialize the pipeline
pipeline_job = define_asset_job(
    name="dataset_pipeline",
    selection=["raw_records", "cleaned_records", "feature_engineered_records"]
)
```

---

## Special Topic: Processing Data for LLMs & RAG

LLM-specific data processing has unique requirements beyond classical tabular ML. The quality bar is extremely high — data that would be "good enough" for a business analytics dashboard can actively harm an LLM's behavior.

### Processing Pipeline for LLM Pre-training / Fine-tuning

```
Raw Text Sources (web, books, code, papers)
    │
    ▼
Text Extraction (OCR, HTML parsing, PDF parsing)
    │
    ▼
Language Identification (filter to target languages)
    │
    ▼
Heuristic Quality Filtering
  - Minimum word count (e.g., ≥ 100 words)
  - Maximum repetition ratio
  - Perplexity score (low-perplexity = high quality)
  - Symbol/word ratio
    │
    ▼
Deduplication (exact + near-duplicate via MinHash)
    │
    ▼
PII Removal
    │
    ▼
Toxicity / Harm Filtering (safety classifiers)
    │
    ▼
Tokenization
    │
    ▼
Final Dataset (JSONL format)
```

### Heuristic Quality Filtering

```python
import polars as pl
import re

def quality_score(text: str) -> dict:
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # Repetition ratio (ratio of unique words to total)
    unique_ratio = len(set(words)) / max(word_count, 1)

    # Symbol density (high symbol ratio = spam/boilerplate)
    symbol_count = len(re.findall(r"[^a-zA-Z0-9\s]", text))
    symbol_ratio = symbol_count / max(char_count, 1)

    # Line structure (many short lines = table/list spam)
    lines = text.split("\n")
    avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)

    return {
        "word_count": word_count,
        "unique_ratio": unique_ratio,
        "symbol_ratio": symbol_ratio,
        "avg_line_len": avg_line_len,
        "passes": (
            word_count >= 100 and
            unique_ratio >= 0.3 and
            symbol_ratio <= 0.2 and
            avg_line_len >= 40
        )
    }

# Apply filter to a corpus
scores = [quality_score(t) for t in df["text"].to_list()]
df = df.with_columns(pl.Series([s["passes"] for s in scores]).alias("passes_quality"))
df_filtered = df.filter(pl.col("passes_quality"))
print(f"Retained {len(df_filtered)} / {len(df)} documents ({len(df_filtered)/len(df):.1%})")
```

### RAG Data Processing

For Retrieval-Augmented Generation (RAG), data processing focuses on chunking, embedding, and indexing rather than model training preparation.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- Step 1: Chunking ---
# Chunk size and overlap are critical hyperparameters for RAG quality
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,           # tokens per chunk
    chunk_overlap=64,         # overlap to preserve context across chunks
    separators=["\n\n", "\n", ". ", " ", ""]
)

documents = []
for _, row in df.iter_rows(named=True):
    chunks = splitter.split_text(row["text"])
    for i, chunk in enumerate(chunks):
        documents.append({
            "id": f"{row['doc_id']}_chunk_{i}",
            "text": chunk,
            "source": row["source"],
            "page": row.get("page", 0),
        })

# --- Step 2: Embedding ---
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [d["text"] for d in documents]
embeddings = model.encode(texts, batch_size=256, normalize_embeddings=True)

# --- Step 3: Indexing into vector store ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("knowledge_base")

collection.add(
    ids=[d["id"] for d in documents],
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[{"source": d["source"], "page": d["page"]} for d in documents]
)

# --- Step 4: Query ---
query_embedding = model.encode(["How do I handle missing values?"], normalize_embeddings=True)
results = collection.query(query_embeddings=query_embedding.tolist(), n_results=5)
```

### Synthetic Data Generation

When real data is scarce, synthetic data generation using an LLM can augment your training set.

```python
import anthropic
import json

client = anthropic.Anthropic()

def generate_synthetic_examples(
    task_description: str,
    few_shot_examples: list[dict],
    n: int = 20
) -> list[dict]:
    """Generate synthetic training examples for fine-tuning."""

    prompt = f"""You are generating training data for an AI model.
Task: {task_description}

Here are example input-output pairs:
{json.dumps(few_shot_examples, indent=2)}

Generate {n} diverse new examples in the same JSON format.
Return ONLY a JSON array of objects, no other text.
"""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(message.content[0].text)

# Example usage: generate synthetic Q&A pairs
few_shots = [
    {"question": "What is gradient descent?", "answer": "An optimization algorithm..."},
    {"question": "What is overfitting?", "answer": "When a model performs well on..."},
]
synthetic_data = generate_synthetic_examples(
    task_description="Answer machine learning questions concisely and accurately.",
    few_shot_examples=few_shots,
    n=50
)
```

---

## Tools & Frameworks Reference

### DataFrame Processing Libraries

|Library|Best For|When to Use|
|---|---|---|
|**Polars**|Production ETL, large datasets|> 1M rows; need speed; parallel processing|
|**Pandas**|Prototyping, small datasets, ecosystem|< 1M rows; Jupyter exploration; sklearn integration|
|**DuckDB**|SQL-based analytics, Parquet queries|SQL-first workflows; large Parquet files; no ETL needed|
|**PySpark**|Distributed, multi-node clusters|Petabyte-scale; existing Spark/Databricks infrastructure|
|**Dask**|Distributed Pandas-compatible|Existing Pandas code; need to scale to cluster|
|**cuDF**|GPU-accelerated DataFrame processing|NVIDIA GPUs available; maximum speed|

### Data Validation

|Tool|Description|
|---|---|
|**Pandera**|Schema validation with statistical checks; Polars + Pandas support|
|**Great Expectations**|Enterprise data quality; expectations, data docs, CI/CD integration|
|**Pointblank**|Polars-native validation with rich HTML reports|
|**Validoopsie**|Lightweight, composable; impact-level thresholds; Polars-first|

### Pipeline Orchestration

|Tool|Description|
|---|---|
|**Dagster**|Asset-centric; built-in lineage; recommended for AI/ML|
|**Apache Airflow**|Mature, widely deployed; DAG-based|
|**Prefect**|Pythonic, low boilerplate; fast iteration|
|**Kubeflow Pipelines**|Kubernetes-native; ML-focused|

### Data Versioning

|Tool|Description|
|---|---|
|**DVC**|Git-based dataset and model versioning; S3/GCS/Azure storage backends|
|**lakeFS**|Git-like branching for data lakes; Delta Lake / Iceberg integration|
|**MLflow**|Experiment tracking + artifact versioning|

### LLM-Specific Data Processing

|Tool|Description|
|---|---|
|**NVIDIA NeMo Curator**|Scalable LLM data curation; GPU-accelerated; heuristic + model filtering|
|**Hugging Face Datasets**|Dataset loading, streaming, processing; transformers integration|
|**IBM Data Prep Kit**|PII redaction, deduplication, quality filtering for LLM training|
|**LangChain / LlamaIndex**|Chunking, embedding, RAG pipeline construction|
|**datasketch**|MinHash LSH for near-duplicate detection at scale|

### Storage & Infrastructure

|Tool|Description|
|---|---|
|**Apache Iceberg**|Open table format; multi-engine; schema evolution|
|**Delta Lake**|ACID transactions; time travel; Databricks-native|
|**ChromaDB / Weaviate / Qdrant**|Vector databases for RAG|
|**Apache Arrow**|In-memory columnar format; cross-engine data exchange|

---

## Best Practices Summary

**Data Quality Above All Else** The single most impactful investment you can make is in data quality. A smaller, cleaner dataset consistently outperforms a larger, noisier one for model training.

**Preserve Raw Data** Always store your raw, unmodified data. Pipeline logic will change. You must be able to reprocess from the original source. Never overwrite raw data in place.

**Treat Your Pipeline as Code** Version control your processing scripts. Use CI/CD to run data validation checks automatically. Data pipelines should be as rigorously tested as application code.

**Avoid Data Leakage** Fit all scalers, encoders, and imputers on training data only. Never allow information from the test or validation split to influence transformations. This is one of the most common and consequential mistakes in ML engineering.

**Make Processing Reproducible** Set random seeds. Pin library versions. Log every parameter of every transformation. Use DVC or lakeFS to version your datasets. Every experiment must be reproducible from scratch.

**Validate at Every Stage** Do not wait until the end to check data quality. Validate immediately after ingestion, after cleaning, and after transformation. Each validation checkpoint narrows the blast radius of bugs.

**Profile Before You Process** Before writing any transformation code, profile your data. Understand distributions, cardinalities, null rates, and value ranges. Surprises in production almost always originate in data you didn't look at closely enough.

**Choose the Right Processing Engine** Use Polars for production pipelines over 1M rows. Use DuckDB for SQL-based analytics on Parquet files. Keep Pandas for quick exploration and sklearn-compatible code under 1M rows. Reserve PySpark for true distributed workloads.

**For LLM Datasets: Quality > Quantity** Research consistently shows that a smaller, carefully curated dataset produces better LLMs than a large, noisy one. Deduplication and quality filtering are among the highest-leverage preprocessing steps for language model training.

---

## Performance Benchmarks: Choosing the Right Engine

The following benchmarks reflect 2025 results on a 12.7 million row dataset (NYC Yellow Taxi, 2.1 GB):

|Operation|Pandas|Polars|DuckDB|
|---|---|---|---|
|CSV Load|~45s|~6s|~4s|
|Group By + Agg|~18s|~1.2s|~1.5s|
|Filter + Sort|~12s|~0.8s|~1.0s|
|Join (large)|~35s|~3.5s|~2.8s|
|Memory Usage|~8 GB|~2.5 GB|~1.8 GB|

**Rule of thumb:** Polars is the right default for Python DataFrame operations in production pipelines. DuckDB is the right default when your workflow is SQL-centric or reads directly from Parquet/CSV files. Use Pandas where ecosystem compatibility (sklearn, matplotlib) is required.

For datasets exceeding available RAM, use Polars' streaming (lazy) API or DuckDB's out-of-core execution. Both handle multi-GB datasets gracefully on a single machine before you need to graduate to distributed compute.

---

_Report compiled from research through April 2026. Tools and benchmarks reflect the current state of the AI engineering ecosystem._