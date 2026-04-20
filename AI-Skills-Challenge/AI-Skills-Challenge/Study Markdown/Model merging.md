
# Model merging

## Model Merging in AI Engineering

### What Is It?

Model merging is a technique that combines two or more LLMs into a single model. It's a relatively new and experimental method to create new models cheaply — no GPU required during the merge itself. Rather than training a new model from scratch or running joint multi-task training, you combine the already-learned weights of multiple finetuned models into one.

The core insight that makes this work: model merging combines the weights of multiple customized LLMs to increase resource utilization and add value to successful models, reducing experimentation waste and offering a cost-effective alternative to joint training.

### Why It Matters

When you finetune a model on Task A and a separate model on Task B, merging lets you get a single model with both capabilities — without retraining. This is powerful because:

- You avoid **catastrophic forgetting** (a common finetuning problem where learning new skills degrades old ones)
- Merging also mitigates catastrophic forgetting during finetuning and continual learning, helping models retain base-model knowledge.
- Model merging is an efficient technique that does not require the collection of raw training data and does not require expensive computation.

---

### Core Concept: Task Vectors

Before diving into specific methods, you need to understand **task vectors**. A task vector is simply the _difference_ between a finetuned model's weights and the base model's weights:

```
task_vector = finetuned_weights - base_weights
```

You can then add, scale, or combine these vectors arithmetically to create new merged models. This is called **Task Arithmetic** and is the foundation for most advanced merging techniques.

---

### The Major Merging Techniques

#### 1. Model Soup (Linear Averaging)

The simplest approach: just average the weights of multiple finetuned models element-wise. Model Soup refers to the simple idea of averaging model weights across multiple finetuned models. It works surprisingly well as a baseline and is computationally trivial.

```python
# Conceptual example with PyTorch
merged_weights = {}
for key in model_a.state_dict():
    merged_weights[key] = (model_a.state_dict()[key] + model_b.state_dict()[key]) / 2
```

**Limitation:** Naive averaging can cause destructive interference when models have diverged significantly.

#### 2. SLERP (Spherical Linear Interpolation)

Instead of linear interpolation, SLERP moves along a _spherical path_ between two weight vectors, preserving the geometric structure of the weight space.

SLERP interpolates between points by moving along a spherical path and takes into account the angle between the original points. This is important because the magnitude of weight vectors often encodes meaningful information (e.g., feature learning strength).

**Key constraint:** Only two models can be merged with SLERP at a time. One model serves as the "base."

A `mergekit` YAML config for SLERP looks like:

```yaml
slices:
  - sources:
      - model: mistral-7b-instruct-v0.2
        layer_range: [0, 32]
      - model: openchat-3.5
        layer_range: [0, 32]
merge_method: slerp
base_model: mistral-7b-v0.1
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
```

#### 3. TIES-Merging (Trim, Elect Sign & Merge)

TIES addresses **task interference** — the problem where merging causes the models to compete and cancel each other out on certain weights.

It works in three steps:

1. **Trim** — discard small weight changes (they're likely noise)
2. **Elect Sign** — for each weight, determine which direction (+ or −) had the most total "votes" across models
3. **Merge** — only average parameters that agree with the elected sign

TIES-Merging seeks to resolve interference by enabling the models that had the most significant weight updates for any given weight take precedence during the merging process — the models that "cared" more about that weight would be prioritized over those that did not.

#### 4. DARE (Drop And REscale)

DARE is a pre-processing augment you apply _before_ merging. The key observation: most delta parameters (task vector entries) are near-zero and essentially redundant.

DARE drops delta parameters with a ratio `p`, and rescales the remaining ones by `1/(1 – p)` to approximate the original embeddings. DARE has been shown to be effective even when dropping upwards of 90%, or even 99% of the task vector weights.

In practice, **DARE + TIES** is often the strongest combination:

```yaml
models:
  - model: mistral-7b-v0.1   # base
  - model: mistral-7b-code-v0.1
    parameters:
      density: 0.5   # keep 50% of delta params
      weight: 0.4
  - model: mistral-7b-instruct-v0.2
    parameters:
      density: 0.5
      weight: 0.6
merge_method: dare_ties
base_model: mistral-7b-v0.1
parameters:
  int8_mask: true
dtype: bfloat16
```

#### 5. Passthrough / Frankenmerging

A completely different approach: instead of blending weights from the same layers, you **concatenate layers** from different models.

By concatenating layers from different LLMs, passthrough can produce models with an exotic number of parameters (e.g., 9B with two 7B parameter models). These models are often referred to as "frankenmerges" or "Frankenstein models" by the community. This technique is very experimental, but it managed to create impressive models, like goliath-120b using two Llama 2 70B models.

---

### The Primary Tool: `mergekit`

The go-to library is **`mergekit`** (by Arcee AI / Charles Goddard). It's YAML-configured, supports all the major methods above, and runs on CPU (though GPU is faster for large models).

```bash
pip install mergekit
```

```bash
mergekit-yaml merge_config.yaml ./output-model --copy-tokenizer
```

You can also use it programmatically in Python if you need it in a pipeline.

---

### Real-World Examples

- **NeuralBeagle-7B / Marcoro14-7B-slerp**: Merges of Mistral-7B-based models that became the best-performing 7B models on the Open LLM Leaderboard at the time of release.
- **goliath-120B**: A Frankenmerge of two Llama-2-70B models, creating a 120B model by layer concatenation — no additional training.
- **Daredevil-7B**: A merge of three different models based on Mistral-7B using `dare_ties`.

---

### Important Limitations to Know

All model merging approaches remain noticeably behind the individually fine-tuned models. Simple model merging strategies typically deliver only moderate performance, whereas more advanced approaches can further alleviate conflicts but still fall short of the upper bound given by individual models.

Other key constraints:

- **Architecture must match** — you can only merge models with identical architectures (same layer count, hidden dimensions, attention heads). You can't merge a Llama-3 8B with a Mistral 7B, for instance.
- **Same base model strongly preferred** — models finetuned from the same base checkpoint merge far more cleanly than models with different training lineages.
- **Benchmark contamination risk** — the problem with public leaderboards is that people can train LLMs on the test data to get better results. By merging the best leaderboard models, you also contaminate your own results. Be cautious interpreting leaderboard-topping merged models.

---

### When Should You Use It as an AI Engineer?

|Scenario|Use Merging?|
|---|---|
|You have a coding-finetuned model and a reasoning-finetuned model on the same base, and want both|✅ Yes|
|You ran multiple finetuning experiments and want to combine the best checkpoints|✅ Yes (Model Soup)|
|You want to reduce catastrophic forgetting after SFT|✅ Yes (merge with base)|
|You need maximum single-task performance|❌ Use a dedicated finetune|
|Models have different architectures|❌ Not feasible (without Passthrough)|

Model merging sits in the toolbox between "prompt engineering" and "full retraining" — it's a cheap, creative, and often surprisingly effective way to compose capabilities from models you already have.