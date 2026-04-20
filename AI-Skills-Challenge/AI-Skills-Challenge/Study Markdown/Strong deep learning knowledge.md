# Strong deep learning knowledge

## 1. What Is Deep Learning?

Deep learning is a subfield of machine learning that uses **artificial neural networks with many layers** to learn hierarchical representations of data. The "deep" refers to the depth of these layers — each layer progressively transforms the input into increasingly abstract representations.

Unlike classical ML (e.g., SVMs, decision trees), which relies heavily on hand-crafted features, deep learning **learns features automatically** from raw data given enough data and compute.

**Key distinction from classical ML:**

|Classical ML|Deep Learning|
|---|---|
|Hand-engineered features|Learned features|
|Works well on small data|Excels at scale|
|Interpretable|Often opaque (black box)|
|Faster to train|Compute-intensive|
|e.g., XGBoost, SVM|e.g., transformers, CNNs|

Deep learning powers modern speech recognition, large language models, image generation, autonomous vehicles, drug discovery, and much more.

---

## 2. Mathematical Foundations

As an AI engineer, you need solid grounding in these mathematical areas:

### Linear Algebra

- **Vectors and matrices**: Inputs, weights, and activations are all tensors (multi-dimensional arrays).
- **Matrix multiplication**: The core operation of a neural network layer: `Z = W · X + b`.
- **Dot products**: Used in attention mechanisms.
- **Eigenvalues/eigenvectors**: Relevant to PCA and understanding optimization landscapes.
- **Norms**: L1 and L2 norms are used in regularization and distance metrics.

### Calculus (Multivariable)

- **Partial derivatives**: The gradient of a loss function with respect to each weight.
- **Chain rule**: The backbone of backpropagation — how gradients flow backward through layers.
- **Jacobians and Hessians**: Advanced training and optimization theory.

### Probability and Statistics

- **Probability distributions**: Gaussian, Bernoulli, Categorical — used in modeling outputs and noise.
- **Maximum likelihood estimation (MLE)**: Cross-entropy loss is directly derived from MLE.
- **Bayes' theorem**: Bayesian deep learning and uncertainty estimation.
- **Information theory**: Entropy, KL divergence (used in VAEs and knowledge distillation).

### Numerical Computation

- Floating-point precision (float32, float16, bfloat16)
- Numerical stability (log-sum-exp trick, gradient clipping)

---

## 3. The Neuron and the Neural Network

### The Artificial Neuron

A single neuron computes:

```
output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
       = activation(Wᵀx + b)
```

where `W` are the learned weights, `b` is the bias, and `activation` is a non-linear function.

### Layers in a Neural Network

- **Input layer**: Raw features (e.g., pixel values, token embeddings).
- **Hidden layers**: Intermediate representations. The number and size of these define the network's capacity.
- **Output layer**: Final prediction (e.g., class probabilities, regression values).

### The Universal Approximation Theorem

A neural network with a **single hidden layer** and sufficient neurons can approximate any continuous function to arbitrary precision. In practice, **depth** (more layers) is more efficient than width (more neurons per layer) for learning complex functions. This is a core theoretical justification for deep architectures.

### Python example: simple feedforward net in PyTorch

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleMLP(784, 256, 10)  # e.g. MNIST
```

---

## 4. Backpropagation and Gradient Descent

### The Training Loop

Training a neural network consists of four steps repeated over mini-batches:

1. **Forward pass**: Compute predictions from inputs.
2. **Loss computation**: Measure prediction error with a loss function.
3. **Backward pass (backprop)**: Compute gradients of the loss w.r.t. every parameter.
4. **Parameter update**: Adjust weights in the direction that reduces loss.

### Backpropagation

Backpropagation is the application of the **chain rule** from calculus to compute gradients efficiently across all layers. For a composition of functions `L = f(g(h(x)))`:

```
dL/dx = (dL/df) · (df/dg) · (dg/dh) · (dh/dx)
```

Frameworks like PyTorch and TensorFlow implement **automatic differentiation (autograd)**, which builds a computational graph during the forward pass and differentiates it automatically during `.backward()`.

```python
# PyTorch autograd example
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x  # y = x³ + 2x
y.backward()
print(x.grad)  # dy/dx = 3x² + 2 = 14
```

### Vanishing and Exploding Gradients

- **Vanishing gradients**: Gradients shrink exponentially in deep networks, making early layers learn very slowly. Common with sigmoid/tanh activations.
- **Exploding gradients**: Gradients grow uncontrollably. Mitigated with gradient clipping.

Solutions: ReLU activations, residual connections (ResNets), batch normalization, careful weight initialization (Xavier/He init).

---

## 5. Activation Functions

Activation functions introduce **non-linearity**, allowing networks to learn complex patterns. Without them, stacking layers is equivalent to a single linear transformation.

|Function|Formula|Use case|Notes|
|---|---|---|---|
|**ReLU**|`max(0, x)`|Hidden layers (default)|Fast, sparse; dead neuron problem|
|**Leaky ReLU**|`max(αx, x)`|Alternative to ReLU|Fixes dead neurons|
|**GELU**|`x · Φ(x)`|Transformers (BERT, GPT)|Smooth approximation, state of the art|
|**Sigmoid**|`1 / (1 + e⁻ˣ)`|Binary output|Saturates; causes vanishing gradients|
|**Tanh**|`(eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)`|RNNs, some hidden layers|Zero-centered|
|**Softmax**|`eˣⁱ / Σeˣʲ`|Multi-class output|Outputs a probability distribution|
|**Swish**|`x · sigmoid(x)`|Efficient nets|Self-gated; often outperforms ReLU|

---

## 6. Loss Functions

The loss function measures how wrong the model's predictions are. Choosing the right one depends on the task.

### Regression

- **Mean Squared Error (MSE)**: `L = (1/n) Σ(yᵢ - ŷᵢ)²`  
    Penalizes large errors heavily. Sensitive to outliers.
- **Mean Absolute Error (MAE)**: `L = (1/n) Σ|yᵢ - ŷᵢ|`  
    More robust to outliers.
- **Huber Loss**: Combines MSE and MAE; less sensitive to outliers than MSE.

### Classification

- **Binary Cross-Entropy**: `L = -[y log(ŷ) + (1-y) log(1-ŷ)]`  
    Used for binary classification.
- **Categorical Cross-Entropy**: `L = -Σ yᵢ log(ŷᵢ)`  
    Standard for multi-class classification. Equivalent to negative log-likelihood.

### Other

- **KL Divergence**: Measures difference between two distributions. Used in VAEs.
- **Contrastive / Triplet loss**: Used in metric learning, embeddings, and face recognition.
- **RLHF reward models**: Custom loss functions designed to align models with human preferences.

---

## 7. Optimization Algorithms

### Stochastic Gradient Descent (SGD)

```
W ← W - η · ∇L(W)
```

Uses mini-batches rather than the full dataset. The learning rate `η` is the most critical hyperparameter.

### Momentum

Adds a velocity term that accumulates gradients in directions of consistent update, helping escape local minima and speed convergence:

```python
v = β * v + ∇L
W = W - η * v
```

### Adam (Adaptive Moment Estimation)

The default optimizer for most deep learning tasks. Combines momentum (first moment) with RMSProp (second moment, per-parameter adaptive learning rates):

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
```

### Learning Rate Scheduling

The learning rate should typically decay during training. Common strategies:

- **Cosine annealing**: Smooth decay following a cosine curve.
- **OneCycleLR**: Warms up then decays; very effective in practice.
- **ReduceLROnPlateau**: Reduces LR when validation loss stops improving.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

### Warmup

Many modern training regimes (especially transformers) use a **linear warmup** period where the LR starts near zero and increases to the target LR. This stabilizes early training.

---

## 8. Regularization Techniques

Regularization prevents **overfitting** — when a model performs well on training data but poorly on unseen data.

### Dropout

Randomly "drops" (sets to zero) a fraction of neurons during training, forcing the network to learn redundant representations:

```python
self.dropout = nn.Dropout(p=0.3)
x = self.dropout(x)  # only during training; disabled at inference
```

### Batch Normalization

Normalizes the activations of each layer within a mini-batch. Significantly accelerates training, allows higher learning rates, and acts as a mild regularizer:

```python
self.bn = nn.BatchNorm1d(hidden_dim)
x = self.bn(x)
```

### Layer Normalization

Used in transformers instead of batch norm. Normalizes across feature dimensions rather than across the batch, making it independent of batch size:

```python
self.ln = nn.LayerNorm(d_model)
```

### L1 and L2 Weight Decay

L2 regularization (weight decay) penalizes large weights, pushing them toward zero. L1 promotes sparsity. In PyTorch, L2 is applied via the optimizer's `weight_decay` parameter:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

### Early Stopping

Monitor validation loss and stop training when it starts to increase. A practical and highly effective technique.

### Data Augmentation

Artificially increases dataset size by applying transformations (flipping, cropping, noise) to training samples. Essential in computer vision.

---

## 9. Core Architectures

### 9.1 Feedforward Networks (MLPs)

Multi-layer perceptrons are the building block of deep learning. They process fixed-size input vectors through a series of linear transformations and non-linearities. Used for tabular data, embeddings, and as components within larger architectures.

### 9.2 Convolutional Neural Networks (CNNs)

CNNs are designed for grid-structured data (images, audio spectrograms). They exploit **spatial locality** and **translation invariance** through shared convolutional filters.

**Key components:**

- **Convolutional layer**: Slides a learned filter over the input, computing dot products. Learns local features (edges, textures, objects).
- **Pooling layer**: Downsamples the spatial dimensions (max pooling, average pooling).
- **Receptive field**: The region of input that influences a given neuron. Grows deeper in the network.

**Landmark architectures**: LeNet, AlexNet, VGG, ResNet, EfficientNet.

**ResNet residual connection** — the key innovation that enabled very deep networks (100+ layers):

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return nn.functional.relu(x + self.block(x))  # skip connection
```

### 9.3 Recurrent Neural Networks and LSTMs

RNNs process sequential data by maintaining a hidden state that carries information across time steps. The vanishing gradient problem made vanilla RNNs hard to train on long sequences.

**LSTM (Long Short-Term Memory)** solves this with gating mechanisms:

- **Forget gate**: What to discard from memory.
- **Input gate**: What new information to store.
- **Output gate**: What to use from memory for the current output.
- **Cell state**: Long-term memory highway.

While largely replaced by transformers for NLP, RNNs and LSTMs remain useful for time-series, streaming inference, and on-device models.

### 9.4 Transformers and Attention

The transformer is the dominant architecture in modern deep learning, forming the foundation of GPT, BERT, ViT, Whisper, and virtually all state-of-the-art models.

**Self-attention mechanism**: Each token in a sequence attends to every other token, computing a weighted sum of values based on query-key similarity:

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

Where `Q`, `K`, `V` are linear projections of the input.

**Multi-head attention**: Run self-attention multiple times in parallel with different projections, allowing the model to attend to different aspects of the sequence simultaneously.

**Transformer architecture:**

```
Input → Embedding + Positional Encoding
     → [Multi-Head Attention → Layer Norm → FFN → Layer Norm] × N
     → Output
```

**Types:**

- **Encoder-only** (BERT): Bidirectional, good for classification and understanding tasks.
- **Decoder-only** (GPT): Autoregressive, good for generation.
- **Encoder-decoder** (T5, BART): Good for sequence-to-sequence (translation, summarization).

**Vision Transformer (ViT)**: Applies transformers to images by splitting them into patches.

### 9.5 Generative Models (GANs & Diffusion)

**GANs (Generative Adversarial Networks)**: A generator and discriminator trained adversarially. The generator tries to produce realistic outputs; the discriminator tries to distinguish real from fake. Powerful but notoriously unstable to train.

**Diffusion models**: State-of-the-art generative models (Stable Diffusion, DALL-E). They learn to reverse a gradual noising process — starting from pure noise and iteratively denoising to generate high-quality outputs. Slower at inference than GANs but more stable and controllable.

**VAEs (Variational Autoencoders)**: Learn a latent space where similar inputs cluster together. Used for generation, compression, and representation learning.

---

## 10. Training Deep Networks in Practice

### The Training Loop in PyTorch

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()           # clear accumulated gradients
        outputs = model(batch_x)        # forward pass
        loss = criterion(outputs, batch_y)  # compute loss
        loss.backward()                 # backward pass (compute gradients)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()                # update weights

        total_loss += loss.item()
    return total_loss / len(dataloader)
```

### Mixed Precision Training

Using float16 (or bfloat16) for forward/backward passes reduces memory by ~2x and speeds up training on modern GPUs:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Monitoring Training

- **Training loss vs. validation loss**: The gap indicates overfitting.
- **Learning curves**: Loss and accuracy over steps/epochs.
- **Gradient norms**: Watch for exploding or vanishing gradients.
- Tools: **TensorBoard**, **Weights & Biases (wandb)**, **MLflow**.

### Hyperparameter Tuning

Key hyperparameters to tune: learning rate, batch size, number of layers, hidden dimensions, dropout rate, weight decay. Methods include grid search, random search, and Bayesian optimization (e.g., with Optuna).

---

## 11. Transfer Learning and Fine-Tuning

Rather than training from scratch, it is almost always better to start from a pretrained model.

### Transfer Learning

A model pretrained on a large dataset (ImageNet, Common Crawl) has learned rich, general representations. You can reuse these by:

1. **Feature extraction**: Freeze all pretrained layers; only train a new task-specific head.
2. **Full fine-tuning**: Continue training all layers on your task data.

### Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning of large models is expensive. PEFT methods adapt large models by only training a small subset of parameters:

- **LoRA (Low-Rank Adaptation)**: Injects trainable low-rank matrices into attention layers. The original weights are frozen; only the small adapter matrices are trained.
- **Prefix tuning / Prompt tuning**: Prepend learnable tokens to the input sequence.
- **Adapter layers**: Insert small trainable modules between layers.

```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,                    # rank of the adapter matrices
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # typically < 1% of total params
```

---

## 12. Applied Domains

### Natural Language Processing (NLP) and LLMs

- **Tokenization**: Converting text into subword tokens (BPE, SentencePiece).
- **Embeddings**: Dense vector representations of tokens and sentences (Word2Vec, GloVe, contextual embeddings from transformers).
- **Tasks**: Classification, NER, question answering, summarization, generation.
- **LLMs**: GPT-style autoregressive models trained on web-scale data. Capabilities include in-context learning, instruction following, and chain-of-thought reasoning.

### Computer Vision

- **Image classification**: Assign a class label to an image (ResNet, EfficientNet, ViT).
- **Object detection**: Locate and classify objects (YOLO, Faster R-CNN, DETR).
- **Semantic segmentation**: Classify every pixel (U-Net, SegFormer).
- **Instance segmentation**: Detect and segment individual object instances (Mask R-CNN, SAM).

### Reinforcement Learning (RL)

An agent learns to take actions in an environment to maximize cumulative reward. Deep RL combines neural networks with RL.

- **Policy gradient methods** (PPO, REINFORCE): Directly optimize a policy function.
- **Q-learning / DQN**: Learn a value function estimating future rewards.
- **RLHF (Reinforcement Learning from Human Feedback)**: Used to fine-tune LLMs for alignment. Humans rate model outputs; a reward model is trained; PPO optimizes the LLM against it.

### Multimodal Learning

Models that process and reason across multiple modalities simultaneously.

- **CLIP**: Jointly trained image and text encoder for zero-shot classification.
- **Flamingo / LLaVA**: Visual language models for image understanding.
- **Whisper**: Encoder-decoder transformer for robust speech recognition.
- **GPT-4V / Gemini**: Large multimodal models handling text, images, audio.

---

## 13. Model Deployment and Production

Training a model is only half the job. Production ML involves:

### Inference Optimization

- **Batching**: Process multiple inputs simultaneously to maximize GPU utilization.
- **TorchScript / ONNX export**: Convert models to optimized intermediate representations for production serving.
- **TensorRT**: NVIDIA's optimizer for high-throughput GPU inference.
- **vLLM / TGI**: Specialized serving frameworks for LLMs with paged attention and continuous batching.

### Serving Frameworks

- **FastAPI + PyTorch**: For custom inference APIs.
- **Triton Inference Server**: NVIDIA's high-performance model server.
- **BentoML**: Simplifies packaging and deploying ML models.
- **SageMaker / Vertex AI**: Managed ML platforms on AWS/GCP.

### Monitoring in Production

- **Data drift**: Input distribution shifts over time.
- **Model degradation**: Performance drops as the world changes.
- **Latency and throughput**: P50/P95/P99 latencies, requests-per-second.

### KV Cache (for LLM inference)

Transformer inference computes key-value pairs for each token. Caching these for previously processed tokens dramatically reduces inference cost for long contexts.

---

## 14. Model Compression

### Quantization

Reduce numerical precision to save memory and speed up inference.

- **Post-training quantization (PTQ)**: Quantize after training; quick but may reduce accuracy.
- **Quantization-aware training (QAT)**: Simulate quantization during training; higher accuracy.
- Common targets: float16, bfloat16, int8, int4 (GGUF format for LLMs).

```python
# Post-training dynamic quantization
import torch.quantization
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### Pruning

Remove weights or entire neurons/heads that contribute little to the output. Can achieve 50–90% sparsity with minimal accuracy loss.

### Knowledge Distillation

Train a small "student" model to mimic the outputs (or intermediate representations) of a larger "teacher" model. The student learns a richer signal than hard labels alone.

```python
# Distillation loss: combine task loss with soft-label KL divergence
distill_loss = F.kl_div(
    F.log_softmax(student_logits / temperature, dim=-1),
    F.softmax(teacher_logits / temperature, dim=-1),
    reduction='batchmean'
) * (temperature ** 2)
```

---

## 15. Python Tools and Ecosystem

An AI engineer working in deep learning should be fluent with:

|Category|Tools|
|---|---|
|**Core frameworks**|PyTorch (primary), JAX, TensorFlow/Keras|
|**Data loading**|`torch.utils.data`, HuggingFace `datasets`|
|**Models & pretrained weights**|HuggingFace `transformers`, `timm`|
|**Fine-tuning**|HuggingFace `peft`, `trl`, `accelerate`|
|**Experiment tracking**|Weights & Biases, MLflow, TensorBoard|
|**Hyperparameter tuning**|Optuna, Ray Tune|
|**Distributed training**|PyTorch DDP, DeepSpeed, FSDP|
|**Inference serving**|vLLM, TGI, ONNX Runtime, TorchServe|
|**Compute**|CUDA, `nvidia-smi`, GPU profiling|
|**Visualization**|matplotlib, seaborn, `torchviz`|

---

## 16. What AI Engineers Are Expected to Know

Below is a tiered summary of deep learning knowledge by depth:

### Foundational (Everyone)

- How a neural network learns (forward pass → loss → backprop → update)
- The purpose of activation functions and why non-linearity matters
- What overfitting is and the main ways to prevent it
- The difference between training, validation, and test sets
- Common loss functions and when to use each
- The basics of CNNs, RNNs, and transformers — what problems they solve and why
- How to load and train a model in PyTorch
- Transfer learning and why you almost never train from scratch

### Intermediate (Working AI Engineer)

- Deep intuition for the transformer architecture, especially self-attention and positional encoding
- Optimization algorithms (SGD, Adam, AdamW) and their tradeoffs
- Learning rate scheduling and warmup
- Batch norm vs. layer norm and when each applies
- Fine-tuning strategies (full fine-tuning vs. LoRA/PEFT)
- Mixed precision training and gradient scaling
- How to read training curves and diagnose training instability
- How to instrument experiments and compare runs systematically
- Model deployment basics: ONNX export, inference batching, latency vs. throughput
- Familiarity with HuggingFace ecosystem

### Advanced (Senior / Specialist)

- Architectural design choices: attention variants (MQA, GQA, flash attention), positional encodings (RoPE, ALiBi)
- Distributed training: data parallelism, tensor parallelism, pipeline parallelism, FSDP
- RLHF pipeline: reward modeling, PPO, DPO
- Scaling laws and their implications for model and data size decisions
- Quantization and model compression in depth (GPTQ, AWQ, GGUF)
- Custom CUDA kernels and GPU memory management
- Evaluation methodology: benchmarks, human evaluation, safety testing
- LLM inference optimization: KV cache, speculative decoding, continuous batching

---

_This document covers the deep learning knowledge expected of AI engineers across foundational theory, practical training, modern architectures, and production deployment. The field evolves rapidly — follow arXiv, Papers With Code, and the HuggingFace blog to stay current._