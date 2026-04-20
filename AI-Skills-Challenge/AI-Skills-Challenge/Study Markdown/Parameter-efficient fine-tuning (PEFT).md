# Parameter-efficient fine-tuning (PEFT)

### The Core Problem

Training state-of-the-art LLMs involves billions (sometimes trillions) of parameters. Traditional full fine-tuning involves slight adjustments to _all_ parameters in a pretrained LLM to adapt it for a specific task. But as models have grown larger and more complex, this process has become too demanding on computational resources and energy. Each fully fine-tuned model is also the same size as the original, consuming enormous storage.

### What is PEFT?

PEFT modifies only a _subset_ of parameters in pre-trained neural networks, rather than updating all model parameters. It aims to mitigate computational challenges by adjusting a limited number of parameters to achieve similar or better performance with reduced overhead. By targeting only specific layers or components, PEFT can significantly reduce the training time and resources needed.

PEFT works by freezing most of the pretrained model parameters and layers while adding a few trainable parameters — known as **adapters** — to specific layers for downstream tasks. The fine-tuned models retain all the learning from pretraining while specializing in their respective tasks.

### Key Benefits

PEFT addresses several critical problems in LLM engineering: it dramatically reduces **time-to-value** (since fewer parameters need updating), guards against **catastrophic forgetting** (since most original parameters stay frozen), and reduces **overfitting** risk (since most parameters remain static).

Practically, PEFT produces tiny checkpoints worth only a few MBs compared to full fine-tuning. For example, a model like `bigscience/mt0-xxl` takes 40GB of storage, and full fine-tuning produces a 40GB checkpoint per downstream dataset — whereas PEFT methods produce just a few MBs per task, all while achieving comparable performance.

---

### PEFT Techniques with Examples

PEFT methods fall into three broad categories: **additive**, **reparameterized**, and **selective**.

#### 1. LoRA — Low-Rank Adaptation _(Reparameterized)_

The most widely used PEFT method today. In LoRA, a model's original weights remain frozen, and new, small, trainable parameters are injected using low-dimensional matrices. This allows efficient fine-tuning by adding a limited number of parameters trained with minimal computational cost while maintaining the integrity and performance of the original model.

The creators of LoRA demonstrated adapting GPT-3 (175B parameters) to new tasks by training as few as ~37.7 million parameters (~0.02% of the model) — a 10,000× reduction in trainable parameters and about a 3× reduction in GPU memory usage during training.

**QLoRA** is an extension combining LoRA with 4-bit quantization. Using QLoRA, one can fine-tune a 65B parameter model on a single GPU using 4-bit precision together with LoRA.

A minimal Python example using Hugging Face's `peft` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,              # rank of the low-rank matrices
    lora_alpha=32,     # scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # which layers to adapt
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622%
```

---

#### 2. Adapter Layers _(Additive)_

Adapters are small, trainable modules inserted into the transformer layers of a frozen pretrained model. They typically consist of a down-projection, a non-linearity, and an up-projection, and are placed after attention and feed-forward sublayers.

```python
from peft import get_peft_model, AdaptionPromptConfig, TaskType

# Adapters via PEFT (LLaMA-Adapter style)
config = AdaptionPromptConfig(
    task_type=TaskType.CAUSAL_LM,
    adapter_len=10,       # number of adapter tokens
    adapter_layers=30     # number of layers to apply adapters
)
peft_model = get_peft_model(model, config)
```

Adapter-based methods and LoRA achieve nearly identical accuracy to full fine-tuning while reducing trainable parameters by over 95% on NLP benchmarks like GLUE and SuperGLUE.

---

#### 3. Prefix Tuning & Prompt Tuning _(Additive — Soft Prompts)_

P-tuning automatically optimizes input embeddings during training rather than requiring manual prompt design. It modifies input embeddings and adjusts them based on downstream task requirements, and has shown significant performance gains on benchmarks like SuperGLUE.

```python
from peft import get_peft_model, PromptTuningConfig, TaskType

config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,     # number of learnable "soft" tokens prepended to input
    tokenizer_name_or_path="gpt2"
)
peft_model = get_peft_model(model, config)
```

In sequence-to-sequence tasks like machine translation and text summarization, LoRA and prefix tuning have demonstrated competitive performance, though prefix tuning often requires careful prompt engineering to match the effectiveness of adapter-based approaches.

---

#### 4. BitFit _(Selective)_

One of the simplest PEFT approaches — only the **bias terms** of the pretrained model are updated during fine-tuning, leaving all weights frozen. BitFit, while computationally efficient, exhibits slight performance degradation, particularly on tasks requiring deeper model modifications. It's best suited for low-resource or quick-iteration scenarios.

```python
# BitFit: manually freeze everything except bias terms
for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False
```

---

#### 5. IA³ — Infused Adapter by Inhibiting and Amplifying Inner Activations _(Additive)_

IA³ is a PEFT technique that optimizes the fine-tuning process by focusing on rescaling the inner activations of a pre-trained model, injecting learned vectors that scale attention keys, values, and feed-forward activations. It introduces even fewer parameters than LoRA.

---

### Summary Comparison

|Method|Strategy|Trainable Params|Best For|
|---|---|---|---|
|**LoRA / QLoRA**|Reparameterized|Very low (~0.1%)|LLMs, diffusion models|
|**Adapters**|Additive modules|Low|Multi-task learning|
|**Prefix Tuning**|Soft prompts|Low|Seq2seq, translation|
|**Prompt Tuning**|Soft prompts|Very low|Few-shot tasks|
|**BitFit**|Selective (bias only)|Minimal|Quick experiments|
|**IA³**|Additive (rescaling)|Extremely low|Resource-constrained settings|

In practice, LoRA-based methods are the dominant choice for autoregressive LLMs today due to their efficiency and popularity, with other PEFT methods rarely chosen in production settings.