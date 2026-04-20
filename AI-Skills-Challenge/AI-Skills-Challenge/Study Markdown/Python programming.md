# Python programming

## 1. List Comprehensions & Generator Expressions

### Why You Need This as an AI Engineer
Data preprocessing is 80% of AI work. List comprehensions let you clean, filter, and transform data in a single readable line — far faster than writing manual `for` loops. Generator expressions do the same thing but without loading everything into memory, which is critical when your dataset has millions of rows.

### Working Examples

```python
# ---- LIST COMPREHENSIONS ----

# Basic: Square all numbers
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition: Filter only even squares
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# Nested: Flatten a matrix (common in data preprocessing)
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Real AI use case: Tokenize and lowercase text data
sentences = ["Hello World", "AI is Amazing", "Python Rocks"]
tokens = [word.lower() for sentence in sentences for word in sentence.split()]
print(tokens)  # ['hello', 'world', 'ai', 'is', 'amazing', 'python', 'rocks']

# Dictionary comprehension: Create a word-to-index vocabulary
vocab = {word: idx for idx, word in enumerate(sorted(set(tokens)))}
print(vocab)
# {'ai': 0, 'amazing': 1, 'hello': 2, 'is': 3, 'python': 4, 'rocks': 5, 'world': 6}


# ---- GENERATOR EXPRESSIONS ----

# Memory-efficient: Process 10 million items without loading all into RAM
total = sum(x ** 2 for x in range(10_000_000))  # Uses almost no memory
print(f"Sum of squares: {total}")

# Lazy evaluation: Only computes values as needed
big_data = (x * 0.01 for x in range(1_000_000))
first_five = [next(big_data) for _ in range(5)]
print(first_five)  # [0.0, 0.01, 0.02, 0.03, 0.04]
```

### Tips & Tricks
- **Use list comps for small-to-medium data**, generator expressions for large/streaming data.
- **Set comprehensions** `{x for x in items}` are great for deduplication.
- **Avoid nesting more than 2 levels** — beyond that, use a regular loop for readability.
- **Conditional expressions work inline**: `[x if x > 0 else 0 for x in data]` (ReLU activation!).

---

## 2. Generators & the `yield` Keyword

### Why You Need This as an AI Engineer
Training AI models means working with massive datasets that won't fit in memory. Generators let you stream data batch-by-batch to your model — this is exactly how PyTorch's `DataLoader` and TensorFlow's `tf.data` work under the hood. Understanding `yield` is non-negotiable.

### Working Examples

```python
# Basic generator function
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num, end=" ")  # 5 4 3 2 1
print()


# AI Use Case: Batch data generator for model training
import random

def batch_generator(data, batch_size):
    """Yields batches of data — just like a DataLoader."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Simulate a dataset of 100 samples
dataset = [random.random() for _ in range(100)]

for batch_num, batch in enumerate(batch_generator(dataset, batch_size=16)):
    print(f"Batch {batch_num}: {len(batch)} samples, mean={sum(batch)/len(batch):.3f}")


# Infinite generator: Useful for data augmentation or cycling
def infinite_cycle(data):
    """Cycles through data forever — useful for training loops."""
    while True:
        for item in data:
            yield item

cycler = infinite_cycle(["augment_flip", "augment_rotate", "augment_crop"])
augmentations = [next(cycler) for _ in range(7)]
print(augmentations)
# ['augment_flip', 'augment_rotate', 'augment_crop', 'augment_flip', ...]


# Generator pipeline: Chain generators for data processing
def read_lines(filename_sim):
    """Simulates reading lines from a file."""
    lines = ["  Hello World  \n", "  AI Engineering  \n", "  Python 3.12  \n"]
    for line in lines:
        yield line

def strip_lines(lines):
    for line in lines:
        yield line.strip()

def to_lower(lines):
    for line in lines:
        yield line.lower()

# Compose the pipeline
pipeline = to_lower(strip_lines(read_lines("data.txt")))
for processed in pipeline:
    print(repr(processed))
# 'hello world'
# 'ai engineering'
# 'python 3.12'
```

### Tips & Tricks
- **`yield from`** delegates to a sub-generator: `yield from another_generator()`.
- **Use `send()`** to push values INTO a generator (used in coroutines).
- **Generators are one-shot** — once exhausted, you need to create a new one.
- **`itertools`** is your best friend: `chain`, `islice`, `tee`, `cycle` all work with generators.
- **Memory rule of thumb**: If your data fits in RAM, use a list. If it doesn't, use a generator.

---

## 3. Decorators

### Why You Need This as an AI Engineer
Decorators add cross-cutting functionality — timing, logging, caching, retrying failed API calls — without touching your core logic. In AI engineering, you'll use them to time training loops, cache expensive model predictions, add authentication to API endpoints, and implement retry logic for flaky LLM API calls.

### Working Examples

```python
import functools
import time

# ---- BASIC DECORATOR: Timer ----
def timer(func):
    """Measures execution time of any function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"⏱  {func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def train_model(epochs):
    """Simulate model training."""
    total_loss = 0
    for epoch in range(epochs):
        time.sleep(0.01)  # Simulate computation
        total_loss += 1.0 / (epoch + 1)
    return total_loss

loss = train_model(10)
print(f"Final loss: {loss:.4f}")


# ---- DECORATOR WITH ARGUMENTS: Retry ----
def retry(max_attempts=3, delay=1.0):
    """Retries a function on failure — essential for LLM API calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise Exception(f"{func.__name__} failed after {max_attempts} attempts")
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.1)
def call_llm_api(prompt):
    """Simulates a flaky API call."""
    import random
    if random.random() < 0.6:
        raise ConnectionError("API timeout")
    return f"Response to: {prompt}"

# This will retry automatically on failure
try:
    result = call_llm_api("Explain transformers")
    print(result)
except Exception as e:
    print(e)


# ---- CACHING DECORATOR (Built-in) ----
@functools.lru_cache(maxsize=128)
def expensive_embedding(text):
    """Simulates computing an embedding — cached after first call."""
    print(f"  Computing embedding for: '{text}'")
    time.sleep(0.1)  # Simulate expensive computation
    return [hash(text) % 100 / 100.0 for _ in range(4)]  # Fake embedding

# First call computes, second call is instant (cached)
print(expensive_embedding("hello world"))
print(expensive_embedding("hello world"))  # Cache hit — no recomputation!
print(f"Cache info: {expensive_embedding.cache_info()}")
```

### Tips & Tricks
- **Always use `@functools.wraps(func)`** — it preserves the original function's name and docstring.
- **`@functools.lru_cache`** is a free performance boost for pure functions (same input → same output).
- **Stack decorators** by placing them on consecutive lines above a function; they apply bottom-up.
- **`@property`** is a decorator that turns a method into an attribute — great for computed values.
- **In AI**: Use decorators for experiment tracking, GPU memory management, and API rate limiting.

---

## 4. Context Managers (`with` statement)

### Why You Need This as an AI Engineer
AI engineering means managing resources — files, database connections, GPU memory, API sessions, model weights. Context managers guarantee cleanup even when exceptions occur. Every time you open a file, connect to a database, or manage a PyTorch training context, you're using context managers.

### Working Examples

```python
from contextlib import contextmanager
import time

# ---- BASIC: File handling ----
with open("example.txt", "w") as f:
    f.write("Training data goes here")
# File is automatically closed, even if an error occurs

with open("example.txt", "r") as f:
    content = f.read()
    print(content)


# ---- CUSTOM CONTEXT MANAGER: Timer ----
@contextmanager
def timer_context(label="Block"):
    """Time any block of code with a `with` statement."""
    start = time.perf_counter()
    yield  # Code inside the `with` block runs here
    elapsed = time.perf_counter() - start
    print(f"⏱  {label}: {elapsed:.4f}s")

with timer_context("Data preprocessing"):
    data = [x ** 2 for x in range(1_000_000)]

with timer_context("Model inference simulation"):
    result = sum(data) / len(data)
    print(f"Mean: {result:.2f}")


# ---- CUSTOM CONTEXT MANAGER: Temporary directory ----
import tempfile
import os

@contextmanager
def temp_model_dir():
    """Creates a temp directory for model checkpoints, cleans up after."""
    dirpath = tempfile.mkdtemp()
    print(f"📂 Created temp dir: {dirpath}")
    try:
        yield dirpath
    finally:
        # Cleanup
        for f in os.listdir(dirpath):
            os.remove(os.path.join(dirpath, f))
        os.rmdir(dirpath)
        print(f"🗑  Cleaned up: {dirpath}")

with temp_model_dir() as model_dir:
    # Save a "model checkpoint"
    checkpoint_path = os.path.join(model_dir, "model_v1.bin")
    with open(checkpoint_path, "w") as f:
        f.write("model weights here")
    print(f"Saved checkpoint to {checkpoint_path}")
# Directory and files are automatically cleaned up


# ---- CLASS-BASED CONTEXT MANAGER ----
class DatabaseConnection:
    """Simulates a database connection lifecycle."""
    def __init__(self, db_name):
        self.db_name = db_name

    def __enter__(self):
        print(f"🔌 Connecting to {self.db_name}...")
        self.connection = {"status": "connected", "db": self.db_name}
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"🔌 Disconnecting from {self.db_name}...")
        self.connection["status"] = "disconnected"
        return False  # Don't suppress exceptions

with DatabaseConnection("vector_db") as conn:
    print(f"Status: {conn['status']}")
    print("Inserting embeddings...")
```

### Tips & Tricks
- **`contextlib.suppress(ExceptionType)`** — silently ignores specific exceptions.
- **Use `contextlib.ExitStack`** to manage a dynamic number of context managers.
- **In PyTorch**: `torch.no_grad()` is a context manager that disables gradient computation during inference.
- **In AI workflows**: Wrap GPU memory allocation, file handles, and API sessions in context managers.

---

## 5. Type Hints & Static Typing

### Why You Need This as an AI Engineer
Modern AI codebases are large and collaborative. Type hints make your code self-documenting, catch bugs before runtime, and enable IDE autocompletion. Frameworks like Pydantic (used in LangChain, FastAPI) are built entirely around type hints. Proficiency in Python 3.10+ type hints is now a standard hiring requirement.

### Working Examples

```python
from typing import Optional, Union
from dataclasses import dataclass

# ---- BASIC TYPE HINTS ----
def compute_loss(predictions: list[float], targets: list[float]) -> float:
    """Compute mean squared error."""
    assert len(predictions) == len(targets)
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

loss = compute_loss([1.0, 2.0, 3.0], [1.1, 2.2, 2.8])
print(f"MSE Loss: {loss:.4f}")


# ---- OPTIONAL & UNION ----
def load_model(path: str, device: Optional[str] = None) -> dict:
    """Load a model, optionally specifying a device."""
    model = {"weights": [0.1, 0.2, 0.3], "path": path}
    model["device"] = device or "cpu"
    return model

model = load_model("model.bin")
print(model)

model_gpu = load_model("model.bin", device="cuda:0")
print(model_gpu)


# ---- TYPED DICTIONARIES (Python 3.12+) ----
# Using TypedDict for structured configs
from typing import TypedDict

class TrainingConfig(TypedDict):
    learning_rate: float
    batch_size: int
    epochs: int
    model_name: str
    use_gpu: bool

def train(config: TrainingConfig) -> dict[str, float]:
    """Train with a typed configuration."""
    print(f"Training {config['model_name']} for {config['epochs']} epochs")
    print(f"  LR: {config['learning_rate']}, Batch: {config['batch_size']}")
    return {"final_loss": 0.05, "accuracy": 0.95}

config: TrainingConfig = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "model_name": "transformer-v2",
    "use_gpu": True,
}
results = train(config)
print(results)


# ---- CALLABLE TYPE HINTS ----
from typing import Callable

def apply_activation(x: list[float], func: Callable[[float], float]) -> list[float]:
    """Apply any activation function to a list of values."""
    return [func(val) for val in x]

import math

relu: Callable[[float], float] = lambda x: max(0.0, x)
sigmoid: Callable[[float], float] = lambda x: 1.0 / (1.0 + math.exp(-x))

data = [-2.0, -1.0, 0.0, 1.0, 2.0]
print(f"ReLU:    {apply_activation(data, relu)}")
print(f"Sigmoid: {[f'{v:.3f}' for v in apply_activation(data, sigmoid)]}")
```

### Tips & Tricks
- **Python 3.10+**: Use `X | Y` instead of `Union[X, Y]` and `list[int]` instead of `List[int]`.
- **Use `mypy` or `pyright`** to run static type checking on your codebase.
- **`TypeVar` and `Generic`** let you write type-safe reusable classes (like custom containers).
- **Pydantic models** auto-validate data using type hints — essential for API inputs/outputs.
- **Type hints are NOT enforced at runtime** by default — they're for tooling and documentation.

---

## 6. Object-Oriented Programming

### Why You Need This as an AI Engineer
Every AI framework is built on OOP. PyTorch models inherit from `nn.Module`. Custom datasets extend `Dataset`. LangChain tools are classes. Understanding classes, inheritance, and `dataclasses` is essential for writing models, custom layers, training pipelines, and production-grade AI systems.

### Working Examples

```python
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# ---- BASIC CLASS: Neural Network Layer ----
class LinearLayer:
    """Simulates a single dense layer: y = Wx + b"""
    def __init__(self, input_size: int, output_size: int):
        import random
        self.weights = [[random.gauss(0, 0.1) for _ in range(input_size)]
                        for _ in range(output_size)]
        self.biases = [0.0] * output_size

    def forward(self, x: list[float]) -> list[float]:
        output = []
        for neuron_weights, bias in zip(self.weights, self.biases):
            activation = sum(w * xi for w, xi in zip(neuron_weights, x)) + bias
            output.append(max(0, activation))  # ReLU
        return output

    def __repr__(self):
        return f"LinearLayer({len(self.weights[0])} → {len(self.weights)})"

layer = LinearLayer(3, 2)
print(layer)
result = layer.forward([1.0, 0.5, -0.3])
print(f"Output: {result}")


# ---- INHERITANCE & ABSTRACT CLASSES ----
class BaseModel(ABC):
    """Abstract base class for all models."""
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False

    @abstractmethod
    def predict(self, x: list[float]) -> list[float]:
        pass

    def summary(self):
        status = "trained" if self.is_trained else "untrained"
        print(f"Model '{self.name}' [{status}]")


class SimpleClassifier(BaseModel):
    def __init__(self, name: str, threshold: float = 0.5):
        super().__init__(name)
        self.threshold = threshold

    def predict(self, x: list[float]) -> list[float]:
        return [1.0 if val > self.threshold else 0.0 for val in x]

    def train(self, data: list):
        print(f"Training {self.name} on {len(data)} samples...")
        self.is_trained = True

clf = SimpleClassifier("BinaryClassifier", threshold=0.5)
clf.summary()
clf.train([1, 2, 3, 4, 5])
clf.summary()
predictions = clf.predict([0.2, 0.7, 0.4, 0.9, 0.1])
print(f"Predictions: {predictions}")


# ---- DATACLASSES: Clean configuration objects ----
@dataclass
class ExperimentConfig:
    """Typed, clean configuration with defaults and auto-generated methods."""
    model_name: str
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    dropout: float = 0.1
    tags: list[str] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return self.epochs * (1000 // self.batch_size)  # Assume 1000 samples

config = ExperimentConfig(
    model_name="GPT-mini",
    learning_rate=5e-4,
    tags=["experiment", "baseline"]
)
print(config)
print(f"Total steps: {config.total_steps}")

# Dataclasses are comparable and hashable (if frozen=True)
config2 = ExperimentConfig(model_name="GPT-mini", learning_rate=5e-4, tags=["experiment", "baseline"])
print(f"Configs equal? {config == config2}")
```

### Tips & Tricks
- **`@dataclass(frozen=True)`** makes instances immutable and hashable — use for configs.
- **`__slots__`** saves memory by replacing `__dict__` with a fixed attribute layout.
- **`super().__init__()`** is essential when extending framework classes (PyTorch, LangChain).
- **Composition over inheritance** — prefer having objects as attributes rather than deep class hierarchies.
- **`__repr__` and `__str__`** are your debugging lifelines — always implement them.

---

## 7. Async/Await & `asyncio`

### Why You Need This as an AI Engineer
AI systems in production make many concurrent I/O calls — hitting LLM APIs, fetching embeddings, querying vector databases, calling microservices. Synchronous code waits for each call to finish before starting the next. Async code runs them all concurrently, often making your pipeline 5-10x faster.

### Working Examples

```python
import asyncio
import time

# ---- BASIC ASYNC ----
async def fetch_embedding(text: str, delay: float = 0.5) -> dict:
    """Simulates an async API call to get embeddings."""
    print(f"  ⏳ Fetching embedding for '{text}'...")
    await asyncio.sleep(delay)  # Non-blocking wait
    fake_embedding = [hash(text + str(i)) % 100 / 100 for i in range(4)]
    return {"text": text, "embedding": fake_embedding}

async def main_sequential():
    """Sequential — each call waits for the previous one."""
    texts = ["hello", "world", "python", "AI"]
    start = time.perf_counter()
    results = []
    for text in texts:
        result = await fetch_embedding(text)
        results.append(result)
    elapsed = time.perf_counter() - start
    print(f"Sequential: {elapsed:.2f}s for {len(results)} embeddings\n")

async def main_concurrent():
    """Concurrent — all calls run at the same time!"""
    texts = ["hello", "world", "python", "AI"]
    start = time.perf_counter()
    results = await asyncio.gather(
        *[fetch_embedding(text) for text in texts]
    )
    elapsed = time.perf_counter() - start
    print(f"Concurrent: {elapsed:.2f}s for {len(results)} embeddings\n")

# Run both to see the difference
asyncio.run(main_sequential())   # ~2.0 seconds (0.5s × 4)
asyncio.run(main_concurrent())   # ~0.5 seconds (all at once!)


# ---- REAL-WORLD PATTERN: Rate-limited API calls ----
async def rate_limited_calls(prompts: list[str], max_concurrent: int = 3):
    """Use a semaphore to limit concurrent API calls."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_call(prompt):
        async with semaphore:
            return await fetch_embedding(prompt, delay=0.3)

    results = await asyncio.gather(*[limited_call(p) for p in prompts])
    return results

prompts = [f"prompt_{i}" for i in range(8)]
start = time.perf_counter()
results = asyncio.run(rate_limited_calls(prompts, max_concurrent=3))
elapsed = time.perf_counter() - start
print(f"Rate-limited ({len(results)} calls, max 3 concurrent): {elapsed:.2f}s")
```

### Tips & Tricks
- **`asyncio.gather()`** runs multiple coroutines concurrently — your go-to for batch API calls.
- **`asyncio.Semaphore`** limits concurrency — essential for API rate limits.
- **`asyncio.wait_for(coro, timeout=10)`** adds a timeout to any async call.
- **Don't mix sync and async** — use `asyncio.to_thread()` to run blocking code in async contexts.
- **`httpx` and `aiohttp`** are the async HTTP libraries — use them instead of `requests` for async work.

---

## 8. Lambda Functions, `map`, `filter`, `reduce`

### Why You Need This as an AI Engineer
Functional programming tools let you write concise data transformations. You'll use lambdas as activation functions, loss function selectors, and quick transformations in data pipelines. They're also common in sorting, Pandas operations, and callback-based frameworks.

### Working Examples

```python
from functools import reduce
import math

# ---- LAMBDA FUNCTIONS ----
# Quick one-liner functions
relu = lambda x: max(0, x)
sigmoid = lambda x: 1 / (1 + math.exp(-x))
normalize = lambda x, mean, std: (x - mean) / std

print(f"ReLU(-3): {relu(-3)}")        # 0
print(f"ReLU(5): {relu(5)}")          # 5
print(f"Sigmoid(0): {sigmoid(0):.3f}")  # 0.500

# ---- MAP: Apply function to every element ----
raw_scores = [-2.0, -0.5, 0.0, 1.5, 3.0]

# Apply sigmoid to all scores
probabilities = list(map(sigmoid, raw_scores))
print(f"Probabilities: {[f'{p:.3f}' for p in probabilities]}")

# Convert strings to floats (common in CSV parsing)
string_values = ["1.5", "2.7", "3.14", "0.001"]
float_values = list(map(float, string_values))
print(f"Parsed floats: {float_values}")

# ---- FILTER: Keep elements that pass a test ----
predictions = [0.1, 0.8, 0.3, 0.95, 0.05, 0.7, 0.6]

# Keep only high-confidence predictions
high_confidence = list(filter(lambda p: p > 0.7, predictions))
print(f"High confidence: {high_confidence}")  # [0.8, 0.95]

# ---- REDUCE: Accumulate a single result ----
# Multiply all values (useful for probability chains)
probs = [0.9, 0.8, 0.95, 0.7]
joint_probability = reduce(lambda a, b: a * b, probs)
print(f"Joint probability: {joint_probability:.4f}")  # 0.4788

# ---- SORTING WITH LAMBDAS ----
models = [
    {"name": "GPT-4", "accuracy": 0.95, "cost": 0.03},
    {"name": "Claude", "accuracy": 0.93, "cost": 0.015},
    {"name": "Llama-3", "accuracy": 0.89, "cost": 0.001},
]

# Sort by accuracy (descending)
by_accuracy = sorted(models, key=lambda m: m["accuracy"], reverse=True)
print("By accuracy:", [m["name"] for m in by_accuracy])

# Sort by cost-effectiveness (accuracy per dollar)
by_value = sorted(models, key=lambda m: m["accuracy"] / m["cost"], reverse=True)
print("By value:", [m["name"] for m in by_value])
```

### Tips & Tricks
- **Lambdas are best for short, throwaway functions** — if it's complex, use `def`.
- **`map()` and `filter()` return iterators** — wrap in `list()` to see results.
- **List comprehensions are usually preferred** over `map`/`filter` in modern Python.
- **`operator` module** has pre-built functions: `operator.add`, `operator.itemgetter("key")`.

---

## 9. Exception Handling & Custom Exceptions

### Why You Need This as an AI Engineer
AI systems fail in creative ways — corrupted data files, API rate limits, out-of-memory GPUs, invalid model inputs, NaN losses during training. Proper exception handling means your 12-hour training run doesn't crash at hour 11 because of one bad data point.

### Working Examples

```python
# ---- BASIC TRY/EXCEPT/FINALLY ----
def safe_divide(a: float, b: float) -> float:
    try:
        result = a / b
    except ZeroDivisionError:
        print("⚠  Warning: Division by zero, returning 0.0")
        return 0.0
    except TypeError as e:
        print(f"⚠  Type error: {e}")
        return 0.0
    else:
        # Runs only if NO exception occurred
        print(f"✅ {a} / {b} = {result}")
        return result
    finally:
        # ALWAYS runs — cleanup goes here
        pass

safe_divide(10, 3)
safe_divide(10, 0)
safe_divide(10, "abc")


# ---- CUSTOM EXCEPTIONS ----
class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class InvalidInputError(ModelError):
    """Raised when model receives invalid input."""
    def __init__(self, expected_shape, actual_shape):
        self.expected = expected_shape
        self.actual = actual_shape
        super().__init__(
            f"Expected input shape {expected_shape}, got {actual_shape}"
        )

class TrainingDivergedError(ModelError):
    """Raised when training loss becomes NaN or infinite."""
    def __init__(self, epoch: int, loss: float):
        self.epoch = epoch
        self.loss = loss
        super().__init__(f"Training diverged at epoch {epoch}: loss={loss}")


# ---- USING CUSTOM EXCEPTIONS ----
def validate_input(data: list, expected_dim: int):
    if len(data) != expected_dim:
        raise InvalidInputError(
            expected_shape=(expected_dim,),
            actual_shape=(len(data),)
        )
    return True

def train_step(epoch: int) -> float:
    import random
    loss = random.random()
    if loss > 0.95:  # Simulate divergence
        raise TrainingDivergedError(epoch, float('nan'))
    return loss

# Robust training loop with exception handling
print("\n--- Robust Training Loop ---")
losses = []
for epoch in range(20):
    try:
        loss = train_step(epoch)
        losses.append(loss)
    except TrainingDivergedError as e:
        print(f"⚠  {e}")
        print(f"  Recovering: reducing learning rate and retrying...")
        losses.append(losses[-1] if losses else 1.0)  # Use last good loss
    except KeyboardInterrupt:
        print(f"\n⏹  Training interrupted at epoch {epoch}")
        break

print(f"Completed {len(losses)} epochs. Final loss: {losses[-1]:.4f}")
```

### Tips & Tricks
- **Catch specific exceptions**, not bare `except:` — you'll hide real bugs.
- **Create exception hierarchies** — catch `ModelError` to handle all model-related errors.
- **Use `else` for success-only code** and `finally` for guaranteed cleanup.
- **`raise ... from e`** chains exceptions to preserve the original traceback.
- **In training loops**: Catch `NaN` losses early and save checkpoints frequently.

---

## 10. NumPy — Vectorized Computation

### Why You Need This as an AI Engineer
NumPy is the foundation of all numerical computing in Python. Every AI framework (PyTorch, TensorFlow, scikit-learn, JAX) is built on top of NumPy arrays or their derivatives. Vectorized operations run 10-100x faster than Python loops because they execute in optimized C code under the hood.

### Working Examples

```python
import numpy as np

# ---- ARRAY CREATION ----
# From list
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Array: {arr}, dtype: {arr.dtype}, shape: {arr.shape}")

# Common initializations
zeros = np.zeros((3, 4))         # 3x4 matrix of zeros
ones = np.ones((2, 3))           # 2x3 matrix of ones
random_arr = np.random.randn(3, 3)  # 3x3 standard normal
linspace = np.linspace(0, 1, 5)  # 5 evenly spaced values from 0 to 1
print(f"Linspace: {linspace}")

# ---- VECTORIZED OPERATIONS (No loops needed!) ----
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

print(f"Add:      {a + b}")
print(f"Multiply: {a * b}")
print(f"Power:    {a ** 2}")
print(f"Dot prod: {np.dot(a, b)}")

# ---- BROADCASTING ----
# Automatically expands dimensions to match shapes
matrix = np.random.randn(3, 4)
row_mean = matrix.mean(axis=1, keepdims=True)  # Shape: (3, 1)
normalized = matrix - row_mean  # Broadcasting: (3,4) - (3,1) → (3,4)
print(f"Matrix shape: {matrix.shape}")
print(f"Normalized means: {normalized.mean(axis=1)}")  # Should be ~0

# ---- AI-SPECIFIC OPERATIONS ----

# Softmax function (used in every classifier)
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Softmax: {probs} (sum={probs.sum():.4f})")

# Cosine similarity (used in embeddings/search)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

emb1 = np.random.randn(768)  # Simulated BERT embedding
emb2 = emb1 + np.random.randn(768) * 0.1  # Similar embedding
emb3 = np.random.randn(768)  # Random embedding

print(f"Similar: {cosine_similarity(emb1, emb2):.4f}")  # High
print(f"Random:  {cosine_similarity(emb1, emb3):.4f}")  # Low

# ---- INDEXING & SLICING (Essential for data manipulation) ----
data = np.random.randn(100, 5)  # 100 samples, 5 features

# Boolean indexing (filtering)
positive_rows = data[data[:, 0] > 0]  # Rows where first feature > 0
print(f"Rows with positive feature 0: {positive_rows.shape[0]}")

# Fancy indexing
indices = np.array([0, 5, 10, 50, 99])
selected = data[indices]
print(f"Selected {selected.shape[0]} specific rows")

# Reshape (critical for model inputs)
flat = np.arange(12)
reshaped = flat.reshape(3, 4)    # 3 rows, 4 columns
batched = flat.reshape(2, 2, 3)  # 2 batches of 2x3
print(f"Flat: {flat.shape} → Reshaped: {reshaped.shape} → Batched: {batched.shape}")
```

### Tips & Tricks
- **Avoid Python loops over NumPy arrays** — always use vectorized operations.
- **`axis=0` means "across rows" (down), `axis=1` means "across columns" (right)**.
- **`np.einsum()`** is incredibly powerful for complex tensor operations — learn it.
- **Use `np.random.seed(42)`** for reproducible experiments (or `np.random.default_rng(42)`).
- **Memory layout matters**: Use `np.ascontiguousarray()` before passing to C/CUDA code.
- **`@` operator**: `A @ B` is matrix multiplication, same as `np.matmul(A, B)`.

---

## 11. Pandas — Data Manipulation

### Why You Need This as an AI Engineer
Before any model sees your data, it passes through Pandas. Loading CSVs, cleaning missing values, feature engineering, merging datasets, computing statistics — Pandas handles it all. It's the bridge between raw data and your model's input pipeline.

### Working Examples

```python
import pandas as pd
import numpy as np

# ---- CREATING DATAFRAMES ----
df = pd.DataFrame({
    "model": ["GPT-4", "Claude-3", "Llama-3", "Gemini", "Mistral"],
    "accuracy": [0.95, 0.93, 0.89, 0.92, 0.88],
    "latency_ms": [120, 85, 45, 110, 35],
    "cost_per_1k": [0.03, 0.015, 0.001, 0.025, 0.002],
    "open_source": [False, False, True, False, True],
})
print(df)
print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")

# ---- FILTERING & SELECTION ----
# Fast models
fast_models = df[df["latency_ms"] < 100]
print(f"\nFast models:\n{fast_models[['model', 'latency_ms']]}")

# Multiple conditions
affordable_accurate = df[(df["accuracy"] > 0.90) & (df["cost_per_1k"] < 0.02)]
print(f"\nAffordable & accurate:\n{affordable_accurate['model'].tolist()}")

# ---- FEATURE ENGINEERING ----
df["value_score"] = df["accuracy"] / df["cost_per_1k"]
df["speed_tier"] = pd.cut(df["latency_ms"], bins=[0, 50, 100, 200],
                           labels=["fast", "medium", "slow"])
print(f"\nWith engineered features:\n{df[['model', 'value_score', 'speed_tier']]}")

# ---- HANDLING MISSING DATA ----
messy_df = pd.DataFrame({
    "feature1": [1.0, np.nan, 3.0, np.nan, 5.0],
    "feature2": [10, 20, np.nan, 40, 50],
    "label": ["pos", "neg", "pos", None, "neg"],
})
print(f"\nMissing values:\n{messy_df.isnull().sum()}")

# Fill numeric with mean, categorical with mode
messy_df["feature1"] = messy_df["feature1"].fillna(messy_df["feature1"].mean())
messy_df["feature2"] = messy_df["feature2"].fillna(messy_df["feature2"].median())
messy_df["label"] = messy_df["label"].fillna(messy_df["label"].mode()[0])
print(f"\nCleaned:\n{messy_df}")

# ---- GROUPBY & AGGREGATION ----
experiments = pd.DataFrame({
    "run": list(range(1, 13)),
    "optimizer": ["adam", "sgd", "adamw"] * 4,
    "loss": np.random.exponential(0.5, 12),
    "accuracy": np.random.uniform(0.7, 0.99, 12),
})

summary = experiments.groupby("optimizer").agg(
    mean_loss=("loss", "mean"),
    std_loss=("loss", "std"),
    best_accuracy=("accuracy", "max"),
    num_runs=("run", "count"),
).round(4)
print(f"\nExperiment summary:\n{summary}")

# ---- APPLY: Custom transformations ----
df["model_upper"] = df["model"].apply(str.upper)
df["log_cost"] = df["cost_per_1k"].apply(np.log10)
print(f"\nTransformed:\n{df[['model_upper', 'log_cost']]}")
```

### Tips & Tricks
- **`.query("column > value")`** is often cleaner than boolean indexing.
- **`pd.read_csv("file.csv", nrows=100)`** — peek at data without loading the full file.
- **Use `.loc[]` for label-based and `.iloc[]` for integer-based indexing**.
- **Method chaining** keeps code clean: `df.dropna().reset_index().sort_values("col")`.
- **For large datasets**: Use `dtype` specification when reading CSVs to reduce memory by 50-80%.
- **`df.memory_usage(deep=True)`** tells you exactly how much RAM your DataFrame consumes.

---

## 12. Dictionary & Unpacking Mastery

### Why You Need This as an AI Engineer
Dictionaries are everywhere in AI — model configs, hyperparameters, API responses, JSON data, tokenizer vocabularies. Mastering dict operations, merging, unpacking, and `defaultdict`/`Counter` will save you hours of data wrangling.

### Working Examples

```python
from collections import defaultdict, Counter

# ---- DICTIONARY MERGING (Python 3.9+) ----
default_config = {"lr": 0.001, "batch_size": 32, "epochs": 10, "optimizer": "adam"}
override = {"lr": 0.0005, "epochs": 20, "scheduler": "cosine"}

# Merge with override taking priority
final_config = default_config | override  # Python 3.9+
print(f"Final config: {final_config}")

# ---- UNPACKING ----
def create_model(name, lr, epochs, **kwargs):
    print(f"Creating '{name}' with lr={lr}, epochs={epochs}")
    if kwargs:
        print(f"  Extra params: {kwargs}")

# Unpack a dict as keyword arguments
create_model(**final_config, name="transformer")

# ---- DEFAULTDICT: Auto-initialize missing keys ----
# Build an inverted index (word → list of documents)
documents = {
    "doc1": "the cat sat on the mat",
    "doc2": "the dog chased the cat",
    "doc3": "the mat was on the floor",
}

inverted_index = defaultdict(list)
for doc_id, text in documents.items():
    for word in text.split():
        inverted_index[word].append(doc_id)

print(f"\n'cat' appears in: {inverted_index['cat']}")
print(f"'mat' appears in: {inverted_index['mat']}")

# ---- COUNTER: Frequency analysis ----
all_words = " ".join(documents.values()).split()
word_counts = Counter(all_words)
print(f"\nTop 5 words: {word_counts.most_common(5)}")

# Build a simple vocabulary
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
print(f"Vocabulary: {vocab}")

# ---- DICTIONARY COMPREHENSIONS ----
# Invert a mapping (useful for decoding model outputs)
idx_to_word = {idx: word for word, idx in vocab.items()}
print(f"Reverse vocab: {idx_to_word}")

# Filter a dict
expensive_models = {k: v for k, v in {"gpt4": 0.03, "claude": 0.015, "llama": 0.001}.items()
                    if v > 0.01}
print(f"Expensive models: {expensive_models}")

# ---- NESTED DICT ACCESS (Safe) ----
api_response = {
    "choices": [
        {"message": {"content": "Hello, world!", "role": "assistant"}}
    ],
    "usage": {"total_tokens": 42}
}

# Safe nested access with .get()
content = api_response.get("choices", [{}])[0].get("message", {}).get("content", "N/A")
tokens = api_response.get("usage", {}).get("total_tokens", 0)
print(f"\nAPI content: {content}")
print(f"Tokens used: {tokens}")
```

### Tips & Tricks
- **`dict.get(key, default)`** prevents `KeyError` — use it for API responses.
- **`dict.setdefault(key, default)`** sets AND returns if missing — useful in loops.
- **`Counter` supports arithmetic**: `counter1 + counter2`, `counter1 - counter2`.
- **`ChainMap`** from `collections` layers multiple dicts without copying — great for configs.
- **JSON keys are always strings** — remember this when loading API responses.

---

## 13. File I/O & JSON/YAML Handling

### Why You Need This as an AI Engineer
AI engineering means constantly reading/writing data — model configs (JSON/YAML), datasets (CSV/Parquet), model weights (binary), logs, and API request/response payloads. Efficient file I/O is a daily task.

### Working Examples

```python
import json
import csv
import os

# ---- JSON: The universal data format ----
# Writing JSON (model config)
config = {
    "model_name": "transformer-v2",
    "hyperparameters": {
        "learning_rate": 0.001,
        "hidden_size": 768,
        "num_heads": 12,
        "layers": 6,
    },
    "training": {
        "epochs": 50,
        "batch_size": 32,
        "early_stopping": True,
    },
    "tags": ["production", "v2", "optimized"],
}

with open("model_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("✅ Saved config to model_config.json")

# Reading JSON
with open("model_config.json", "r") as f:
    loaded_config = json.load(f)
print(f"Loaded model: {loaded_config['model_name']}")
print(f"Hidden size: {loaded_config['hyperparameters']['hidden_size']}")

# ---- JSONL: One JSON object per line (common for datasets) ----
training_examples = [
    {"prompt": "What is AI?", "completion": "AI is artificial intelligence."},
    {"prompt": "What is ML?", "completion": "ML is machine learning."},
    {"prompt": "What is NLP?", "completion": "NLP is natural language processing."},
]

# Write JSONL
with open("training_data.jsonl", "w") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")

# Read JSONL (streaming — memory efficient for large files)
loaded_examples = []
with open("training_data.jsonl", "r") as f:
    for line in f:
        loaded_examples.append(json.loads(line.strip()))
print(f"\nLoaded {len(loaded_examples)} training examples")
print(f"Example: {loaded_examples[0]}")

# ---- CSV: Tabular data ----
results = [
    {"epoch": 1, "train_loss": 2.5, "val_loss": 2.8, "accuracy": 0.45},
    {"epoch": 2, "train_loss": 1.8, "val_loss": 2.1, "accuracy": 0.62},
    {"epoch": 3, "train_loss": 1.2, "val_loss": 1.5, "accuracy": 0.78},
]

# Write CSV
with open("training_log.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Read CSV
with open("training_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Epoch {row['epoch']}: loss={row['train_loss']}, acc={row['accuracy']}")

# ---- PATH HANDLING (os.path and pathlib) ----
from pathlib import Path

model_dir = Path("models") / "transformer" / "v2"
model_dir.mkdir(parents=True, exist_ok=True)

checkpoint = model_dir / "checkpoint_best.pt"
checkpoint.write_text("fake model weights")

print(f"\nCheckpoint path: {checkpoint}")
print(f"Exists: {checkpoint.exists()}")
print(f"Parent: {checkpoint.parent}")
print(f"Stem: {checkpoint.stem}")
print(f"Suffix: {checkpoint.suffix}")

# List all files in a directory
for path in Path(".").glob("*.json"):
    print(f"  Found: {path}")

# Cleanup
import shutil
shutil.rmtree("models")
for f in ["example.txt", "model_config.json", "training_data.jsonl", "training_log.csv"]:
    if os.path.exists(f):
        os.remove(f)
```

### Tips & Tricks
- **Use `pathlib.Path`** instead of `os.path` — it's more readable and Pythonic.
- **JSONL is the standard** for LLM training data (OpenAI, Anthropic, Hugging Face all use it).
- **`json.dumps(obj, default=str)`** handles datetime and other non-serializable types.
- **For large files**: Read line-by-line instead of `json.load()` to avoid memory issues.
- **Parquet** (`pd.read_parquet()`) is 5-10x faster than CSV for large datasets — use it.

---

## 14. `*args`, `**kwargs`, and Function Design

### Why You Need This as an AI Engineer
AI codebases heavily use `*args` and `**kwargs` for flexible function signatures — model constructors accept variable configs, wrapper functions pass arguments through, and API clients need to handle arbitrary parameters. Understanding these is essential for reading and writing framework code.

### Working Examples

```python
# ---- *args: Variable positional arguments ----
def compute_metrics(*predictions_targets_pairs):
    """Accept any number of (prediction, target) tuples."""
    for i, (pred, target) in enumerate(predictions_targets_pairs):
        error = abs(pred - target)
        print(f"  Sample {i}: pred={pred:.2f}, target={target:.2f}, error={error:.2f}")

compute_metrics((0.9, 1.0), (0.3, 0.0), (0.7, 0.8))


# ---- **kwargs: Variable keyword arguments ----
def create_experiment(name: str, **hyperparameters):
    """Create an experiment with any hyperparameters."""
    experiment = {"name": name, "params": hyperparameters}
    print(f"\nExperiment: {name}")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    return experiment

exp = create_experiment(
    "baseline",
    learning_rate=0.001,
    batch_size=64,
    dropout=0.1,
    warmup_steps=1000,
    weight_decay=0.01
)


# ---- COMBINING: Passthrough wrapper ----
def log_function_call(func):
    """Decorator that logs any function call with its arguments."""
    def wrapper(*args, **kwargs):
        args_str = ", ".join(map(repr, args))
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        print(f"📞 Calling {func.__name__}({all_args})")
        return func(*args, **kwargs)
    return wrapper

@log_function_call
def train(model_name, epochs=10, lr=0.001):
    return f"Trained {model_name}"

train("bert", epochs=5, lr=0.0005)


# ---- KEYWORD-ONLY ARGUMENTS (after *) ----
def safe_predict(data: list, *, model: str, temperature: float = 0.7):
    """Force callers to name 'model' and 'temperature' explicitly."""
    print(f"Predicting with {model} (temp={temperature}) on {len(data)} samples")

safe_predict([1, 2, 3], model="gpt-4", temperature=0.3)
# safe_predict([1, 2, 3], "gpt-4")  # TypeError! 'model' must be keyword


# ---- POSITIONAL-ONLY ARGUMENTS (before /) ----
def loss(predictions, targets, /, *, reduction="mean"):
    """predictions and targets must be positional, reduction must be keyword."""
    errors = [abs(p - t) for p, t in zip(predictions, targets)]
    if reduction == "mean":
        return sum(errors) / len(errors)
    return errors

print(f"\nLoss: {loss([1, 2, 3], [1.1, 1.9, 3.2], reduction='mean'):.4f}")
```

### Tips & Tricks
- **`*` alone in a signature** forces all following args to be keyword-only.
- **`/` in a signature** forces all preceding args to be positional-only (Python 3.8+).
- **Use `**kwargs` sparingly** — it hides the function's interface; prefer explicit parameters.
- **Unpack dicts into function calls**: `func(**config_dict)` is a common AI pattern.
- **`*args` in class `__init__`** with `super().__init__(*args)` is essential for framework inheritance.

---

## 15. Virtual Environments & Dependency Management

### Why You Need This as an AI Engineer
Different AI projects need different library versions. Your NLP project might need `transformers==4.40` while your CV project uses `transformers==4.35`. Without virtual environments, you'll face dependency hell — broken imports, version conflicts, and unreproducible experiments.

### Key Commands

```bash
# ---- VENV (Built-in, no extra install needed) ----
python -m venv myproject_env          # Create
source myproject_env/bin/activate     # Activate (Linux/Mac)
# myproject_env\Scripts\activate      # Activate (Windows)
pip install numpy pandas torch        # Install packages
pip freeze > requirements.txt         # Save dependencies
deactivate                            # Deactivate

# ---- RECREATE AN ENVIRONMENT ----
python -m venv new_env
source new_env/bin/activate
pip install -r requirements.txt       # Install exact versions

# ---- UV (Modern, ultra-fast alternative — recommended in 2026) ----
# pip install uv
# uv venv                             # Create venv
# uv pip install numpy pandas torch   # 10-100x faster than pip
# uv pip compile requirements.in      # Lock dependencies
```

### Example `requirements.txt`

```text
# Core AI libraries
numpy>=1.26,<2.0
pandas>=2.2
scikit-learn>=1.4
torch>=2.3
transformers>=4.40
pydantic>=2.0

# API & serving
fastapi>=0.111
httpx>=0.27
uvicorn>=0.29

# Utilities
python-dotenv>=1.0
tqdm>=4.66
```

### Tips & Tricks
- **One venv per project** — never install AI packages globally.
- **Pin exact versions** for production: `numpy==1.26.4`, not `numpy>=1.26`.
- **`uv`** is the modern replacement for pip — it's 10-100x faster.
- **Docker** is the gold standard for reproducible AI environments.
- **`.python-version`** file + `pyenv` lets you switch Python versions per project.
- **`pip install -e .`** (editable install) lets you develop packages while testing them.

---

## 16. Logging

### Why You Need This as an AI Engineer
`print()` is for debugging. `logging` is for production. When your model is running in production serving 10,000 requests/minute, you need structured, leveled logs that can be filtered, searched, and monitored — not random print statements.

### Working Examples

```python
import logging

# ---- BASIC SETUP ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("ai_pipeline")

# ---- LOG LEVELS ----
logger.debug("Detailed debug info — hidden in production")
logger.info("Training started with batch_size=32")
logger.warning("GPU memory usage above 80%")
logger.error("Failed to load checkpoint: file not found")
logger.critical("Out of memory — training aborted")


# ---- STRUCTURED LOGGING FOR ML ----
def train_epoch(epoch: int, num_batches: int = 10):
    import random
    total_loss = 0
    for batch in range(num_batches):
        loss = random.expovariate(1.0)
        total_loss += loss

    avg_loss = total_loss / num_batches
    logger.info(
        "Epoch %d complete | avg_loss=%.4f | batches=%d",
        epoch, avg_loss, num_batches
    )
    if avg_loss > 1.5:
        logger.warning("High loss detected at epoch %d: %.4f", epoch, avg_loss)
    return avg_loss

# Run a training loop with proper logging
logger.info("=" * 50)
logger.info("Starting training run")
for epoch in range(1, 6):
    loss = train_epoch(epoch)
logger.info("Training complete")


# ---- LOGGING TO FILE ----
file_handler = logging.FileHandler("training.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
logger.addHandler(file_handler)
logger.info("This goes to both console AND file")

# Cleanup
import os
if os.path.exists("training.log"):
    os.remove("training.log")
```

### Tips & Tricks
- **Use `logger = logging.getLogger(__name__)`** in every module.
- **Never use f-strings in log calls** — use `%s` formatting (lazy evaluation).
- **`logging.exception("msg")`** automatically includes the traceback.
- **In production**: Use structured JSON logging (e.g., `python-json-logger`).
- **Log hyperparameters at the start** of every training run for reproducibility.

---

## 17. Multiprocessing & Threading

### Why You Need This as an AI Engineer
Data preprocessing is CPU-bound — tokenizing millions of documents, augmenting images, computing features. Multiprocessing runs these tasks across all CPU cores. Threading is useful for I/O-bound tasks like loading data from disk or downloading files.

### Working Examples

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

# ---- CPU-BOUND: Multiprocessing ----
def process_document(doc_id: int) -> dict:
    """Simulate CPU-heavy document processing."""
    # Simulate tokenization + feature extraction
    result = sum(i * i for i in range(100_000))
    return {"doc_id": doc_id, "features": result % 1000}

# Sequential
start = time.perf_counter()
sequential_results = [process_document(i) for i in range(8)]
print(f"Sequential: {time.perf_counter() - start:.2f}s")

# Parallel with ProcessPoolExecutor
start = time.perf_counter()
with ProcessPoolExecutor(max_workers=4) as executor:
    parallel_results = list(executor.map(process_document, range(8)))
print(f"Parallel (4 workers): {time.perf_counter() - start:.2f}s")


# ---- I/O-BOUND: Threading ----
def download_model_shard(shard_id: int) -> str:
    """Simulate downloading a model shard."""
    time.sleep(0.3)  # Simulate network I/O
    return f"shard_{shard_id}_downloaded"

start = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as executor:
    shards = list(executor.map(download_model_shard, range(8)))
elapsed = time.perf_counter() - start
print(f"\nDownloaded {len(shards)} shards in {elapsed:.2f}s (threaded)")


# ---- PRACTICAL: Parallel data preprocessing ----
def preprocess_text(text: str) -> dict:
    """Simulate text preprocessing pipeline."""
    words = text.lower().split()
    return {
        "original": text,
        "num_words": len(words),
        "unique_words": len(set(words)),
        "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
    }

texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming every industry",
    "Python is the preferred language for AI development",
    "Natural language processing enables human computer interaction",
] * 3  # 12 texts total

with ProcessPoolExecutor() as executor:
    results = list(executor.map(preprocess_text, texts))

for r in results[:3]:
    print(f"  '{r['original'][:30]}...' → {r['num_words']} words, "
          f"avg length: {r['avg_word_length']:.1f}")
```

### Tips & Tricks
- **CPU-bound → `multiprocessing`**, **I/O-bound → `threading` or `asyncio`**.
- **`concurrent.futures`** is the cleanest API — use `ProcessPoolExecutor` and `ThreadPoolExecutor`.
- **Python's GIL** prevents true parallel threading for CPU work — that's why `multiprocessing` exists.
- **Shared data between processes** needs `mp.Queue`, `mp.Value`, or `mp.Manager`.
- **`Pool.imap_unordered()`** gives results as they complete — great for progress bars.
- **Be careful with memory**: Each process copies the full memory space.

---

## 18. Pydantic — Data Validation

### Why You Need This as an AI Engineer
Pydantic validates and parses data using Python type hints. It's the backbone of FastAPI (for serving models), LangChain (for agent configs), and practically every modern AI API. When your model receives bad input, Pydantic catches it before it causes a cryptic error 10 layers deep.

### Working Examples

```python
from dataclasses import dataclass

# ---- SIMULATING PYDANTIC-STYLE VALIDATION ----
# (Full Pydantic requires `pip install pydantic`)
# Here's the pattern you'll use:

class ModelConfig:
    """Validated model configuration — mimics Pydantic BaseModel."""
    def __init__(self, model_name: str, temperature: float = 0.7,
                 max_tokens: int = 1000, top_p: float = 1.0):
        # Validation
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        if not 0 <= temperature <= 2:
            raise ValueError(f"temperature must be 0-2, got {temperature}")
        if max_tokens < 1 or max_tokens > 100000:
            raise ValueError(f"max_tokens must be 1-100000, got {max_tokens}")
        if not 0 <= top_p <= 1:
            raise ValueError(f"top_p must be 0-1, got {top_p}")

        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

    def __repr__(self):
        return f"ModelConfig({self.to_dict()})"

# Valid config
config = ModelConfig(model_name="claude-3", temperature=0.3, max_tokens=500)
print(config)

# Invalid config — caught immediately
try:
    bad_config = ModelConfig(model_name="gpt-4", temperature=5.0)
except ValueError as e:
    print(f"Validation error: {e}")


# ---- WHAT REAL PYDANTIC CODE LOOKS LIKE ----
# (Install with: pip install pydantic)
"""
from pydantic import BaseModel, Field, validator

class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1000, ge=1, le=100000)
    stream: bool = False

    @validator("messages")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content'")
        return v

# Auto-validates and coerces types
request = ChatRequest(
    model="claude-3",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.5,
)
print(request.model_dump_json(indent=2))
"""
print("\n(Pydantic example shown as reference — install pydantic to run)")
```

### Tips & Tricks
- **Pydantic v2** is dramatically faster than v1 — always use `pydantic>=2.0`.
- **`Field(default=..., ge=0, le=1)`** adds validation constraints declaratively.
- **`.model_dump()`** converts to dict, **`.model_dump_json()`** to JSON string.
- **`@field_validator`** and `@model_validator`** add custom validation logic.
- **Use Pydantic for all API request/response schemas** — it's the standard.

---

## 19. API Interaction with `requests` and `httpx`

### Why You Need This as an AI Engineer
AI engineering in 2026 is API-first. You call LLM APIs (OpenAI, Anthropic, Google), embedding APIs, vector database APIs, and deploy your own models behind APIs. Knowing how to make robust HTTP requests with proper error handling, retries, and timeouts is essential.

### Working Examples

```python
import json
import time

# ---- SIMULATED API CLIENT ----
# (Shows the pattern — replace with real `requests` or `httpx` calls)

class LLMClient:
    """Simulates an LLM API client with best practices."""

    def __init__(self, api_key: str, base_url: str = "https://api.example.com",
                 timeout: float = 30.0, max_retries: int = 3):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._request_count = 0

    def _make_request(self, endpoint: str, payload: dict) -> dict:
        """Simulate an API request with retry logic."""
        import random
        for attempt in range(1, self.max_retries + 1):
            try:
                self._request_count += 1

                # Simulate network issues 30% of the time
                if random.random() < 0.3 and attempt < self.max_retries:
                    raise ConnectionError("Simulated timeout")

                # Simulate successful response
                response = {
                    "id": f"resp_{self._request_count}",
                    "content": f"Response to: {payload.get('prompt', 'N/A')}",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 25},
                }
                return response

            except ConnectionError as e:
                wait = 2 ** attempt * 0.1  # Exponential backoff
                print(f"  ⚠  Attempt {attempt} failed: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

        raise Exception(f"Request failed after {self.max_retries} attempts")

    def chat(self, prompt: str, temperature: float = 0.7) -> str:
        """Send a chat request."""
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 1000,
        }
        response = self._make_request("/v1/chat", payload)
        return response["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        payload = {"texts": texts}
        response = self._make_request("/v1/embed", payload)
        # Simulate embeddings
        import random
        return [[random.random() for _ in range(4)] for _ in texts]

# ---- USAGE ----
client = LLMClient(api_key="sk-fake-key")

# Single request
result = client.chat("Explain transformers in one sentence")
print(f"Response: {result}")

# Batch embeddings
embeddings = client.embed(["hello world", "python AI", "machine learning"])
print(f"\nEmbeddings for 3 texts: {len(embeddings)} vectors of dim {len(embeddings[0])}")


# ---- WHAT REAL API CALLS LOOK LIKE ----
"""
import httpx

# Synchronous
response = httpx.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": "sk-...",
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    },
    json={
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "Hello!"}],
    },
    timeout=30.0,
)
result = response.json()

# Async
async with httpx.AsyncClient() as client:
    response = await client.post(url, headers=headers, json=payload)
"""
print("\n(Real API examples shown as reference)")
```

### Tips & Tricks
- **Always set timeouts** — `timeout=30.0` prevents hanging forever.
- **Implement exponential backoff** for retries: `wait = 2 ** attempt` seconds.
- **Use `httpx` over `requests`** — it supports both sync and async.
- **Stream responses** for LLM APIs: `for chunk in response.iter_lines()`.
- **Store API keys in environment variables**: `os.environ["API_KEY"]`, never in code.
- **Use `python-dotenv`** to load `.env` files automatically.

---

## 20. String Formatting & Regular Expressions

### Why You Need This as an AI Engineer
Prompt engineering IS string engineering. You'll build complex prompts with variables, parse LLM outputs, extract structured data from text, clean training data, and validate formats. Regex is essential for text extraction and cleaning in NLP pipelines.

### Working Examples

```python
import re

# ---- F-STRINGS: Modern prompt building ----
model = "claude-3"
temperature = 0.7
system_prompt = "You are a helpful AI assistant."

# Basic f-string
prompt = f"Using {model} at temperature {temperature}"
print(prompt)

# Multi-line f-string (prompt template)
user_query = "What is machine learning?"
context = "Machine learning is a subset of AI that learns from data."

full_prompt = f"""System: {system_prompt}

Context: {context}

User: {user_query}

Please answer the user's question based on the provided context.
Keep your response under {100} words."""

print(full_prompt)

# F-string formatting tricks
accuracy = 0.9567
loss = 0.00342
print(f"Accuracy: {accuracy:.2%}")     # 95.67%
print(f"Loss: {loss:.2e}")             # 3.42e-03
print(f"{'Model':<15} {'Score':>8}")   # Left/right alignment
print(f"{'BERT':<15} {0.92:>8.4f}")
print(f"{'GPT-4':<15} {0.95:>8.4f}")


# ---- REGEX: Text extraction and cleaning ----
# Extract code blocks from LLM output
llm_response = """Here's the code:

```python
def hello():
    print("Hello, World!")
```

And here's another:

```javascript
console.log("Hi!");
```
"""

code_blocks = re.findall(r"```(\w+)\n(.*?)```", llm_response, re.DOTALL)
for lang, code in code_blocks:
    print(f"Language: {lang}")
    print(f"Code: {code.strip()}\n")


# Extract JSON from LLM output (common task!)
messy_output = """
Sure! Here's the analysis:

{"sentiment": "positive", "confidence": 0.95, "keywords": ["great", "amazing"]}

Let me know if you need anything else.
"""

json_match = re.search(r'\{[^{}]*\}', messy_output)
if json_match:
    import json
    extracted = json.loads(json_match.group())
    print(f"Extracted JSON: {extracted}")


# Clean text for NLP
def clean_text(text: str) -> str:
    """Standard NLP text cleaning pipeline."""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)     # Remove URLs
    text = re.sub(r'<[^>]+>', '', text)                # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)                # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()           # Normalize whitespace
    return text

dirty = "  Check out <b>https://example.com</b>!!! It's AMAZING!!! 🔥🔥  "
clean = clean_text(dirty)
print(f"Dirty: {dirty!r}")
print(f"Clean: {clean!r}")


# Validate email format
def is_valid_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

print(f"\nvalid@email.com: {is_valid_email('valid@email.com')}")
print(f"not-an-email: {is_valid_email('not-an-email')}")
```

### Tips & Tricks
- **Use raw strings** for regex: `r'\d+'` (avoids escaping backslashes).
- **`re.DOTALL`** makes `.` match newlines — essential for multi-line LLM outputs.
- **Named groups** `(?P<name>...)` make extraction readable: `match.group("name")`.
- **Compile frequently-used patterns**: `pattern = re.compile(r'...')` for performance.
- **For prompt templates**: Consider `string.Template` or Jinja2 for complex templates.
- **Test regex** at regex101.com before putting it in code.

---

## 21. Bonus: Tips, Tricks, and AI-Specific Wisdom

### General Python Power Tips

```python
# ---- ENUMERATE: Always use instead of range(len()) ----
models = ["BERT", "GPT", "T5"]
for idx, model in enumerate(models, start=1):
    print(f"{idx}. {model}")

# ---- ZIP: Iterate multiple sequences together ----
names = ["model_a", "model_b", "model_c"]
scores = [0.92, 0.88, 0.95]
latencies = [100, 50, 200]

for name, score, latency in zip(names, scores, latencies):
    print(f"{name}: score={score}, latency={latency}ms")

# ---- WALRUS OPERATOR (:=): Assign and use in one step ----
import random
data = [random.gauss(0, 1) for _ in range(10)]

# Without walrus
n = len(data)
if n > 5:
    print(f"Large dataset: {n} items")

# With walrus
if (n := len(data)) > 5:
    print(f"Large dataset: {n} items")

# ---- TERNARY EXPRESSIONS ----
score = 0.85
label = "pass" if score >= 0.5 else "fail"
print(f"Score {score} → {label}")

# ---- ANY() and ALL() for conditions ----
predictions = [0.8, 0.9, 0.7, 0.95, 0.6]
all_confident = all(p > 0.5 for p in predictions)
any_uncertain = any(p < 0.3 for p in predictions)
print(f"All confident: {all_confident}, Any uncertain: {any_uncertain}")

# ---- UNPACKING ----
first, *middle, last = [1, 2, 3, 4, 5]
print(f"First: {first}, Middle: {middle}, Last: {last}")

# Swap without temp variable
a, b = 1, 2
a, b = b, a
print(f"Swapped: a={a}, b={b}")

# ---- COLLECTIONS TRICKS ----
from collections import OrderedDict

# LRU-style cache with OrderedDict
class SimpleCache:
    def __init__(self, maxsize=100):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

cache = SimpleCache(maxsize=3)
cache.put("emb_hello", [0.1, 0.2])
cache.put("emb_world", [0.3, 0.4])
cache.put("emb_python", [0.5, 0.6])
print(f"Cache hit: {cache.get('emb_hello')}")
cache.put("emb_new", [0.7, 0.8])  # Evicts oldest unused
print(f"Evicted: {cache.get('emb_world')}")  # None — was evicted
```

### AI-Specific Pro Tips

1. **Set random seeds everywhere** for reproducibility:
   ```python
   import random, numpy as np
   random.seed(42)
   np.random.seed(42)
   # torch.manual_seed(42)
   # torch.cuda.manual_seed_all(42)
   ```

2. **Use `tqdm` for progress bars** — never leave a long loop without feedback:
   ```python
   # from tqdm import tqdm
   # for batch in tqdm(dataloader, desc="Training"):
   #     ...
   ```

3. **Environment variables for secrets**:
   ```python
   import os
   api_key = os.environ.get("ANTHROPIC_API_KEY", "")
   assert api_key, "Set ANTHROPIC_API_KEY environment variable"
   ```

4. **Profile before optimizing**:
   ```python
   # python -m cProfile my_script.py
   # python -m memory_profiler my_script.py
   ```

5. **Use `__all__` in modules** to control public API:
   ```python
   __all__ = ["ModelConfig", "train", "predict"]
   ```

6. **Structured project layout**:
   ```
   my_ai_project/
   ├── src/
   │   ├── models/
   │   ├── data/
   │   ├── training/
   │   └── serving/
   ├── tests/
   ├── configs/
   ├── notebooks/
   ├── requirements.txt
   ├── pyproject.toml
   └── README.md
   ```

7. **Learn these keyboard shortcuts for Jupyter**:
   - `Shift+Enter` — Run cell
   - `Esc+A` — Insert cell above
   - `Esc+B` — Insert cell below
   - `Esc+DD` — Delete cell
   - `Esc+M` — Change to Markdown

8. **Git commit messages for experiments**:
   ```
   exp: lr=5e-4, batch=64, dropout=0.2 → val_acc=0.934
   ```

---

## Quick Reference: What to Learn When

| Priority | Concept | Use Case |
|----------|---------|----------|
| 🔴 Critical | List comprehensions, dicts, f-strings | Daily data manipulation |
| 🔴 Critical | NumPy, Pandas | Every data pipeline |
| 🔴 Critical | Classes & inheritance | PyTorch models, LangChain tools |
| 🔴 Critical | File I/O & JSON | Config management, data loading |
| 🟡 Important | Generators | Memory-efficient data loading |
| 🟡 Important | Decorators | Logging, caching, retries |
| 🟡 Important | Type hints & Pydantic | API schemas, code quality |
| 🟡 Important | Async/await | Concurrent API calls |
| 🟡 Important | Exception handling | Robust production systems |
| 🟢 Advanced | Context managers | Resource management |
| 🟢 Advanced | Multiprocessing | Parallel data preprocessing |
| 🟢 Advanced | Regex | NLP text cleaning, output parsing |
| 🟢 Advanced | Logging | Production monitoring |

---

## Final Words

The best AI engineers aren't necessarily the ones who know the most about neural architectures — they're the ones who can **ship reliable systems**. That means writing clean, typed, well-structured Python code that handles errors gracefully, scales efficiently, and can be understood by your teammates six months later.

Master these 20 concepts, and you'll have the Python foundation to tackle any AI engineering challenge — from fine-tuning LLMs to building production RAG systems to deploying real-time inference APIs.

**Start building. Start breaking things. Start fixing them. That's how you become excellent.**
