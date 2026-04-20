# Basic understanding of Docker and containers

## 1. Why Docker Matters for AI Engineering

A 2025 Gartner report found that **85% of ML models never reach production** — the primary culprit being inconsistent environments and deployment friction. Docker directly addresses this.

### The Core Problem Docker Solves

AI/ML projects are notorious for the "it works on my machine" problem. A model may depend on Python 3.10, PyTorch 2.1, CUDA 12.1, and a specific version of `transformers`. Reproducing that exact environment on a colleague's laptop, a CI server, or a cloud VM is error-prone without containerization.

Docker packages your model, all its dependencies, and its runtime environment into a single, portable, immutable artifact called an **image**. When that image runs, it becomes a **container** — an isolated process that behaves identically everywhere.

### Key Benefits for AI Engineers

- **Reproducibility** — Pin every dependency at build time; the same image runs identically in dev, staging, and production.
- **Portability** — Run locally, push to a cloud registry, deploy to Kubernetes, or hand off to a colleague.
- **Isolation** — Multiple models with conflicting Python/CUDA requirements can coexist on the same host.
- **Version control for environments** — Image tags give you auditable, rollback-able snapshots of your entire runtime.
- **Compliance** — Docker is the standard compliance layer for regulated industries (healthcare, finance) needing audit trails of what ran and when.

---

## 2. Core Concepts: Containers, Images, and Registries

### Images vs. Containers

|Concept|Analogy|Description|
|---|---|---|
|**Dockerfile**|Recipe|Instructions for building an image|
|**Image**|Frozen snapshot|Read-only template; the artifact you build and push|
|**Container**|Running process|A live instance of an image|
|**Registry**|App store|Central store for images (Docker Hub, ECR, GCR, GHCR)|
|**Volume**|External hard drive|Persistent storage that survives container restarts|
|**Network**|LAN|Virtual network connecting containers to each other|

### The Docker Layered Filesystem

Docker images are built in **layers**. Each instruction in a Dockerfile (`RUN`, `COPY`, `ADD`) creates a new layer. Layers are cached — if a layer hasn't changed, Docker reuses it from cache. This is critical for AI projects where `pip install` can take minutes.

**Implication:** Put instructions that change least often (installing system packages, pinning library versions) near the top, and instructions that change most often (copying your application code) near the bottom.

### Container Lifecycle

```
docker pull  →  [Image exists locally]
docker run   →  Container starts (Created → Running)
docker stop  →  Container gracefully stops (Running → Stopped)
docker start →  Restart a stopped container
docker rm    →  Remove a stopped container
docker rmi   →  Remove an image from local storage
```

---

## 3. Dockerfile Fundamentals

A `Dockerfile` is a text file with sequential build instructions. For AI workloads, the craft is in balancing image size, build speed, and runtime performance.

### Anatomy of a Dockerfile

```dockerfile
# Base image — choose wisely for AI workloads
FROM python:3.11-slim

# Metadata
LABEL maintainer="you@company.com"
LABEL version="1.0"

# Set working directory inside the container
WORKDIR /app

# Copy dependency file first (layer cache optimization!)
COPY requirements.txt .

# Install dependencies as a separate layer
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (changes more often — put it later)
COPY . .

# Expose the port your app listens on
EXPOSE 8000

# Default command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-Stage Builds: Critical for AI Production Images

AI images can balloon to 10+ GB if you're not careful (CUDA base images are large, dev tools add weight). Multi-stage builds keep production images lean:

```dockerfile
# ─── Stage 1: Builder ───────────────────────────────────────────────────────
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

WORKDIR /build
COPY requirements.txt .

# Install build tools and compile any C extensions
RUN apt-get update && apt-get install -y build-essential \
    && pip install --user --no-cache-dir -r requirements.txt

# ─── Stage 2: Production ────────────────────────────────────────────────────
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Only copy the installed packages from the builder — NOT the build tools
COPY --from=builder /root/.local /root/.local

WORKDIR /app
COPY . .

# Run as non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH=/root/.local/bin:$PATH
CMD ["python", "inference_server.py"]
```

**Result:** The `devel` CUDA image (for building) might be 8 GB; the `runtime` image (for serving) is ~2 GB.

### Real-World Example: FastAPI LLM Inference Server

```dockerfile
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app/ ./app/
COPY models/ ./models/

# Non-root user
RUN useradd -m -u 1001 aiuser && chown -R aiuser /app
USER aiuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### Base Image Selection Guide

|Use Case|Recommended Base Image|
|---|---|
|General Python API|`python:3.11-slim`|
|PyTorch inference|`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`|
|TensorFlow inference|`tensorflow/tensorflow:2.15.0-gpu`|
|Building with CUDA|`nvidia/cuda:12.1-devel-ubuntu22.04`|
|Lightweight ML serving|`python:3.11-alpine` (no CUDA)|
|Hugging Face models|`huggingface/transformers-pytorch-gpu`|

---

## 4. Essential Docker CLI Commands

### Building & Running

```bash
# Build an image from the current directory
docker build -t my-ai-app:1.0 .

# Build with build arguments (e.g., inject model version)
docker build --build-arg MODEL_VERSION=llama3 -t my-ai-app:1.0 .

# Run a container interactively (great for debugging)
docker run -it --rm my-ai-app:1.0 /bin/bash

# Run in detached mode (background)
docker run -d --name ai-server -p 8000:8000 my-ai-app:1.0

# Run with GPU access (requires NVIDIA Container Toolkit)
docker run --gpus all my-ai-app:1.0

# Run with specific GPU(s)
docker run --gpus '"device=0,1"' my-ai-app:1.0

# Run with resource limits
docker run --memory=8g --cpus=4 my-ai-app:1.0

# Mount a local directory as a volume
docker run -v /local/models:/app/models:ro my-ai-app:1.0

# Pass environment variables
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY my-ai-app:1.0
```

### Inspecting & Debugging

```bash
# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# View logs (follow mode)
docker logs -f ai-server

# View last 100 lines of logs
docker logs --tail 100 ai-server

# Execute a command inside a running container
docker exec -it ai-server /bin/bash

# Inspect container configuration and state
docker inspect ai-server

# View resource usage (CPU, memory, GPU)
docker stats

# Copy a file from a container to local
docker cp ai-server:/app/logs/output.log ./output.log
```

### Image Management

```bash
# List local images
docker images

# Pull from a registry
docker pull pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Tag an image for a registry
docker tag my-ai-app:1.0 registry.example.com/team/my-ai-app:1.0

# Push to a registry
docker push registry.example.com/team/my-ai-app:1.0

# Remove dangling (untagged) images — keeps disk clean
docker image prune

# Remove ALL unused images (be careful)
docker image prune -a

# Check image layers and sizes
docker history my-ai-app:1.0
```

---

## 5. Docker Compose: Multi-Service AI Stacks

Docker Compose is the tool for defining and running **multi-container applications** from a single YAML file. For AI engineering, this means you can stand up your inference API, a vector database, a message queue, and a monitoring stack with one command.

### Core Compose Concepts

```
docker compose up       # Start all services (add -d for detached)
docker compose down     # Stop and remove containers, networks
docker compose down -v  # Also remove volumes (caution: deletes data)
docker compose ps       # List running services
docker compose logs -f  # Follow logs for all services
docker compose logs -f api  # Follow logs for a specific service
docker compose build    # Rebuild images
docker compose pull     # Pull updated images from registries
docker compose exec api bash  # Shell into the 'api' service
docker compose restart api    # Restart one service without restarting others
```

### Real-World Example: RAG (Retrieval-Augmented Generation) Stack

This is a common production AI pattern — a Python API that uses a vector database and an LLM:

```yaml
# compose.yaml
name: rag-application

services:

  # ── Your AI API ──────────────────────────────────────────────────────────
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - REDIS_URL=redis://redis:6379
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./models:/app/models:ro   # Read-only model mount
    deploy:
      resources:
        limits:
          memory: 8g
          cpus: '4'
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # ── Vector Database ───────────────────────────────────────────────────────
  qdrant:
    image: qdrant/qdrant:v1.8.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 20s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # ── Cache / Session Store ─────────────────────────────────────────────────
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # ── Background Worker (async embeddings, jobs) ────────────────────────────
  worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: ["python", "-m", "celery", "-A", "app.tasks", "worker", "--loglevel=info"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_HOST=qdrant
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - qdrant
    restart: unless-stopped

  # ── Monitoring ────────────────────────────────────────────────────────────
  prometheus:
    image: prom/prometheus:v2.50.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  qdrant_data:
  redis_data:
  grafana_data:

networks:
  default:
    name: rag-network
```

### Using `.env` Files

Always keep secrets out of `compose.yaml`:

```bash
# .env (add to .gitignore!)
OPENAI_API_KEY=sk-...
GRAFANA_PASSWORD=supersecret
POSTGRES_PASSWORD=dbpassword
```

Compose automatically loads `.env` from the same directory. Reference values with `${VARIABLE_NAME}`.

### Compose Profiles (Dev vs. Prod)

Use profiles to conditionally include services:

```yaml
services:
  api:
    build: .
    # No profile = always runs

  pgadmin:          # Dev-only database UI
    image: dpage/pgadmin4
    profiles: ["dev"]

  nginx:            # Only in production
    image: nginx:alpine
    profiles: ["prod"]
```

```bash
# Start with dev tools
docker compose --profile dev up

# Start production configuration
docker compose --profile prod up
```

### The New `models` Top-Level Element (2025)

Docker Compose now supports AI models natively as first-class citizens:

```yaml
# compose.yaml — with native LLM support
models:
  llama3:
    model: ai/meta-llama3.2

services:
  api:
    build: .
    environment:
      - LLM_ENDPOINT=http://model-runner.docker.internal/engines/llama3/v1
```

This integrates with Docker Model Runner to pull and serve open-weight models locally, identical to how you'd configure a cloud LLM endpoint — enabling true local-to-production parity.

---

## 6. GPU Support for AI/ML Workloads

### Prerequisites

1. Install the **NVIDIA Container Toolkit** on the host:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### GPU in `docker run`

```bash
# All GPUs
docker run --gpus all my-ai-image

# Specific GPU by index
docker run --gpus '"device=0"' my-ai-image

# Multiple specific GPUs
docker run --gpus '"device=0,1"' --memory=32g my-ai-image

# Limit GPU memory (per-GPU fraction — less common, usually set in code)
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0 my-ai-image
```

### GPU in Docker Compose

```yaml
services:
  trainer:
    image: my-training-image
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all          # or count: 1 for a single GPU
              capabilities: [gpu]
```

### Choosing the Right CUDA Base Image

```
nvidia/cuda:{version}-base        # Minimal — just CUDA runtime
nvidia/cuda:{version}-runtime     # + cuBLAS, cuDNN runtime libs (use for inference)
nvidia/cuda:{version}-devel       # + headers, compilers (use for building)
```

For most inference workloads: `nvidia/cuda:12.1-runtime-ubuntu22.04` For training workloads: `nvidia/cuda:12.1-devel-ubuntu22.04` (or use PyTorch's own images)

---

## 7. Deploying AI Applications to Production

### The Production Deployment Spectrum

```
Local Dev          →  Docker Compose     →  Single VM         →  Kubernetes / Cloud
(docker compose up)   (same compose.yaml)   (docker run / swarm)  (EKS, GKE, AKS)
```

The remarkable 2025 development: **the same `compose.yaml` used in development now deploys directly to production** on Google Cloud Run and Azure Container Apps.

### Path 1: Single-Server Deployment (Small-Scale)

Suitable for: Internal tools, low-traffic APIs, MVP deployments.

```bash
# On your production server:

# 1. Install Docker
curl -fsSL https://get.docker.com | sh

# 2. Clone your repo or copy your compose.yaml
git clone https://github.com/your-org/your-ai-app.git
cd your-ai-app

# 3. Set environment variables
cp .env.example .env
nano .env  # fill in your secrets

# 4. Start the stack
docker compose -f compose.yaml -f compose.prod.yaml up -d

# 5. Verify
docker compose ps
docker compose logs -f api
```

### Path 2: Direct Cloud Deploy (New in 2025)

Docker's new cloud integrations allow the same compose experience in production:

```bash
# Deploy to Google Cloud Run (no Kubernetes needed)
gcloud run compose up

# Deploy to Azure Container Apps
az containerapp compose create --compose-file compose.yaml
```

### Path 3: Docker Swarm (Medium-Scale)

Swarm turns multiple Docker hosts into a cluster with zero new tools:

```bash
# Initialize the swarm on your manager node
docker swarm init --advertise-addr <MANAGER-IP>

# Join worker nodes (command output from init)
docker swarm join --token <TOKEN> <MANAGER-IP>:2377

# Deploy a stack (uses the same compose.yaml format)
docker stack deploy -c compose.yaml my-ai-app

# Scale a service
docker service scale my-ai-app_api=5

# View services
docker service ls
docker service logs my-ai-app_api
```

### Path 4: Kubernetes (Large-Scale / Enterprise)

For production AI at scale, Kubernetes is the industry standard. The typical workflow:

```bash
# 1. Build and push your image to a registry
docker build -t your-registry/ai-inference:v1.2 .
docker push your-registry/ai-inference:v1.2

# 2. Write a Kubernetes Deployment
# (see deployment.yaml below)

# 3. Apply to your cluster
kubectl create namespace ai-workloads
kubectl apply -f k8s/ -n ai-workloads

# 4. Set up autoscaling
kubectl apply -f hpa.yaml -n ai-workloads
```

**`k8s/deployment.yaml`** — Example Kubernetes deployment for an inference service:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-inference
  namespace: ai-workloads
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-inference
  template:
    metadata:
      labels:
        app: ai-inference
    spec:
      containers:
        - name: inference
          image: your-registry/ai-inference:v1.2
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "4Gi"
              cpu: "2"
              nvidia.com/gpu: "1"
            limits:
              memory: "8Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
          env:
            - name: MODEL_PATH
              value: "/models/llm"
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: ai-secrets
                  key: openai-api-key
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          volumeMounts:
            - name: model-storage
              mountPath: /models
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ai-inference-service
  namespace: ai-workloads
spec:
  selector:
    app: ai-inference
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-inference-hpa
  namespace: ai-workloads
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Blue-Green Deployments for AI Models

Zero-downtime model updates are critical in production. The blue-green pattern maintains two identical environments — only one receives live traffic at a time:

```bash
# Current version (blue) is live
# Deploy new version (green) without touching blue

kubectl set image deployment/ai-inference-green inference=your-registry/ai-inference:v1.3
kubectl rollout status deployment/ai-inference-green

# Validate green is healthy
curl http://green-endpoint/health

# Switch traffic to green (update service selector)
kubectl patch service ai-inference-service -p '{"spec":{"selector":{"version":"green"}}}'

# Keep blue running for quick rollback; tear down after validation
```

---

## 8. Kubernetes: Orchestration at Scale

### Why Kubernetes for AI?

|Need|Kubernetes Feature|
|---|---|
|Auto-scale on high traffic|Horizontal Pod Autoscaler (HPA)|
|Restart crashed model servers|Self-healing (liveness probes)|
|Efficient GPU allocation|Resource requests/limits + NVIDIA device plugin|
|Roll back a bad model|`kubectl rollout undo`|
|Multi-tenant AI platform|Namespaces + RBAC|
|Batch training jobs|Kubernetes Jobs / CronJobs|
|Secret management|Kubernetes Secrets + Vault integration|

### Managed Kubernetes Services (Recommended for Most Teams)

|Provider|Service|Notes|
|---|---|---|
|AWS|EKS|Best ecosystem; use with ECR|
|Google Cloud|GKE|Best Kubernetes UX; Autopilot for hands-off ops|
|Azure|AKS|Strong enterprise/hybrid support|
|DigitalOcean|DOKS|Simple, cost-effective for smaller workloads|

### Compose Bridge: Converting Compose to Kubernetes

Docker's Compose Bridge tool converts your `compose.yaml` directly to Kubernetes manifests or Helm charts — no manual YAML writing required:

```bash
# Install Compose Bridge
docker compose bridge

# Generate Kubernetes manifests
docker compose bridge --output kubernetes --destination ./k8s

# Generate a Helm chart
docker compose bridge --output helm --destination ./helm-chart
```

---

## 9. CI/CD Pipelines for AI Models

A production AI pipeline should automatically build, test, and deploy whenever you push new code or a new model version.

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy AI Model

on:
  push:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ai-inference

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Log in to registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha   # GitHub Actions cache
          cache-to: type=gha,mode=max

  run-tests:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run integration tests in container
        run: |
          docker compose -f compose.test.yml up --build --abort-on-container-exit
          docker compose -f compose.test.yml down -v

  deploy-staging:
    needs: run-tests
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/ai-inference \
            inference=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n staging
          kubectl rollout status deployment/ai-inference -n staging

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production  # Requires manual approval
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/ai-inference \
            inference=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n production
          kubectl rollout status deployment/ai-inference -n production
```

### Test Compose File

```yaml
# compose.test.yml
services:
  sut:  # System Under Test
    build: .
    command: ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
    environment:
      - TEST_MODE=true
    depends_on:
      - qdrant-test
    
  qdrant-test:
    image: qdrant/qdrant:v1.8.0
```

---

## 10. Security Best Practices

### Container Security Checklist

**1. Never run as root**

```dockerfile
# Create and switch to a non-root user
RUN useradd -m -u 1001 appuser
USER appuser
```

**2. Use read-only filesystems where possible**

```yaml
# In compose.yaml
services:
  api:
    read_only: true
    tmpfs:
      - /tmp          # Allow writes only to /tmp
      - /app/cache
```

**3. Manage secrets properly — never hardcode**

```yaml
# compose.yaml — using Docker secrets
services:
  api:
    secrets:
      - openai_api_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key

secrets:
  openai_api_key:
    environment: OPENAI_API_KEY   # Read from host env, not stored in compose file
```

For Kubernetes, use external secret managers:

```bash
# Store in Kubernetes secret (encrypted at rest)
kubectl create secret generic ai-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  -n ai-workloads

# Or better: use HashiCorp Vault / AWS Secrets Manager / GCP Secret Manager
# and inject at runtime via the Vault agent sidecar
```

**4. Scan images for vulnerabilities**

```bash
# Using Docker Scout (built into Docker CLI)
docker scout cves my-ai-app:1.0

# Using Trivy (open source, excellent for CI)
trivy image my-ai-app:1.0

# In CI — fail build if HIGH or CRITICAL CVEs found
trivy image --exit-code 1 --severity HIGH,CRITICAL my-ai-app:1.0
```

**5. Pin image versions — never use `latest` in production**

```dockerfile
# BAD — non-deterministic
FROM python:latest
FROM pytorch/pytorch:latest

# GOOD — pinned
FROM python:3.11.8-slim
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
```

**6. Set resource limits — always**

```yaml
# Without limits, one container can starve the entire host
services:
  api:
    deploy:
      resources:
        limits:
          memory: 8g
          cpus: '4'
        reservations:   # Minimum guaranteed
          memory: 2g
          cpus: '1'
```

---

## 11. Monitoring & Observability

### The Observability Stack for AI

A minimal production monitoring setup:

```yaml
# Add to your compose.yaml
services:
  prometheus:
    image: prom/prometheus:v2.50.0
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana

  # For LLM-specific metrics (latency, token counts, cost)
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3001:3000"
    environment:
      - DATABASE_URL=postgresql://...
```

### Instrumenting Your Python AI App

```python
# In your FastAPI app
from prometheus_client import Counter, Histogram, generate_latest
import time

# Define metrics
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total inference requests', ['model', 'status'])
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference latency', ['model'],
                               buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
TOKEN_USAGE = Counter('llm_tokens_total', 'Total tokens used', ['model', 'type'])

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict")
async def predict(request: InferenceRequest):
    start = time.time()
    try:
        result = model.predict(request.input)
        INFERENCE_REQUESTS.labels(model="my-model", status="success").inc()
        TOKEN_USAGE.labels(model="my-model", type="completion").inc(result.tokens_used)
        return result
    except Exception as e:
        INFERENCE_REQUESTS.labels(model="my-model", status="error").inc()
        raise
    finally:
        INFERENCE_LATENCY.labels(model="my-model").observe(time.time() - start)
```

### Key AI-Specific Metrics to Track

|Metric|What It Tells You|
|---|---|
|Inference latency (p50, p95, p99)|User experience; detect model regressions|
|Token usage|Cost management for LLM APIs|
|GPU utilization|Whether you're under/over-provisioned|
|GPU memory usage|Risk of OOM crashes|
|Request queue depth|Whether you need more replicas|
|Error rate by model version|Catch regressions after updates|
|Model accuracy / drift|Production data vs. training data divergence|

---

## 12. Common Pitfalls & Troubleshooting

### Pitfall 1: Image Size Bloat

**Problem:** Your AI Docker image is 15+ GB, making deployments slow and expensive.

**Causes:**

- Using `devel` CUDA images instead of `runtime`
- Not using multi-stage builds
- Installing dev tools (`gcc`, `build-essential`) in the final stage
- Caching pip downloads inside the image

**Fix:**

```dockerfile
# Multi-stage: build in devel, run in runtime
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder
RUN pip install --user --no-cache-dir -r requirements.txt

FROM nvidia/cuda:12.1-runtime-ubuntu22.04
COPY --from=builder /root/.local /root/.local
# Dev tools NOT copied to final image
```

Also: check your `.dockerignore` file — it's as important as `.gitignore`:

```
# .dockerignore
.git
__pycache__
*.pyc
.pytest_cache
*.ipynb
data/
checkpoints/
.env
*.egg-info
```

---

### Pitfall 2: "It Works Locally But Not in Production"

**Problem:** CUDA version mismatch between dev machine and production host.

**Diagnosis:**

```bash
# On the host, check driver version
nvidia-smi

# Inside the container, check CUDA toolkit version
nvcc --version

# The host driver must be >= the CUDA version your container uses
# CUDA 12.1 requires driver >= 525.85.12
```

**Fix:** Pin your CUDA image to match the minimum driver version available in production. Use the CUDA compatibility matrix: https://docs.nvidia.com/deploy/cuda-compatibility/

---

### Pitfall 3: Container Starts But Model Fails to Load

**Problem:** Container health check passes but the model returns errors.

**Causes:** The model download happens at startup and takes time; health check fires too early.

**Fix:** Implement a two-phase health check:

```python
# In your FastAPI app
import asyncio

model_ready = False

@app.on_event("startup")
async def load_model():
    global model_ready
    # Load model (this can take minutes for large LLMs)
    await asyncio.to_thread(load_heavy_model)
    model_ready = True

@app.get("/health")     # Liveness: is the process alive?
async def health():
    return {"status": "alive"}

@app.get("/ready")      # Readiness: is the model loaded and ready for traffic?
async def ready():
    if not model_ready:
        raise HTTPException(503, "Model not loaded yet")
    return {"status": "ready"}
```

In Kubernetes/Compose, use `/health` for liveness and `/ready` for readiness probes. Set a generous `start_period` on the health check.

---

### Pitfall 4: OOM (Out of Memory) Kills

**Problem:** Container is killed with exit code 137 (OOM).

**Diagnosis:**

```bash
docker inspect <container_id> | grep -i oom
# or
docker stats <container_id>
# or in Kubernetes
kubectl describe pod <pod-name> | grep -A5 OOMKilled
```

**Fix:**

```yaml
# Set memory limits based on actual profiling, not guessing
services:
  api:
    deploy:
      resources:
        limits:
          memory: 16g    # For a 7B parameter model
```

For LLMs: as a rule of thumb, a 7B model in fp16 needs ~14 GB VRAM; in 4-bit quantization, ~4 GB.

---

### Pitfall 5: Slow Container Builds (Especially in CI)

**Problem:** Every CI run rebuilds everything from scratch; `pip install` takes 5+ minutes.

**Fix — Use BuildKit cache mounts:**

```dockerfile
# Enable BuildKit: DOCKER_BUILDKIT=1 docker build .
# Or in Docker 23+, BuildKit is on by default

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

```yaml
# In GitHub Actions, enable GHA cache
- name: Build
  uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

---

### Pitfall 6: Container Networking Confusion

**Problem:** Services can't talk to each other; connection refused errors.

**Key Rule:** Containers on the same Compose network communicate by **service name**, not `localhost`.

```python
# WRONG — trying to reach another container
response = requests.get("http://localhost:6333/collections")

# CORRECT — use the service name from compose.yaml
response = requests.get("http://qdrant:6333/collections")
```

**Diagnosis:**

```bash
# Test connectivity from inside a container
docker compose exec api curl -v http://qdrant:6333/healthz

# Inspect networks
docker network ls
docker network inspect rag-application_default
```

---

### Pitfall 7: Kubernetes — Not Setting Resource Requests/Limits

**Problem:** A single noisy workload consumes all GPU/CPU on the node, crashing other pods.

**Fix:** Always set both `requests` (guaranteed) and `limits` (maximum) in every Kubernetes manifest. Never deploy to production without them.

---

### Pitfall 8: Storing Secrets in Images or Compose Files

**Problem:** API keys committed to `compose.yaml` or baked into Docker images; visible in `docker history`.

**Fix:** Use environment variable references, Docker secrets, or external secret managers (HashiCorp Vault, AWS Secrets Manager). Check with:

```bash
# Scan for secrets accidentally baked into your image
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    trufflesecurity/trufflehog:latest docker --image my-ai-app:1.0
```

---

### Pitfall 9: Not Persisting Model Weights

**Problem:** Container restarts and has to re-download a 10 GB model.

**Fix:** Mount a persistent volume for model caches:

```yaml
services:
  api:
    volumes:
      - model_cache:/app/.cache/huggingface
    environment:
      - HF_HOME=/app/.cache/huggingface

volumes:
  model_cache:   # Persists across container restarts
```

---

### Pitfall 10: Using `latest` Tag in Production

**Problem:** A `docker pull` silently updates your production image to a broken version.

**Fix:** Always use immutable tags — either a semantic version (`v1.2.3`) or a Git commit SHA (`abc1234`). In CI, tag with the commit SHA:

```bash
docker build -t registry.example.com/ai-app:${GITHUB_SHA} .
```

---

### Quick Diagnostic Commands

```bash
# Why did my container exit?
docker inspect <container> --format='{{.State.ExitCode}} {{.State.Error}}'

# What's consuming disk?
docker system df

# Clean up everything unused (safe to run periodically)
docker system prune --volumes

# What's in my image layers? (Find unexpected large files)
docker history --no-trunc my-image:tag

# Real-time resource usage
docker stats --no-stream

# Follow logs for a crashing container with restart loop
docker logs --tail 50 -f <container_name>

# Kubernetes: why is my pod not starting?
kubectl describe pod <pod-name> -n ai-workloads
kubectl logs <pod-name> -n ai-workloads --previous   # logs from crashed instance
```

---

## 13. Quick Reference Cheat Sheet

### Dockerfile Best Practices Summary

|Rule|Why|
|---|---|
|Use multi-stage builds|Smaller production images|
|Pin base image versions|Reproducible builds|
|Copy `requirements.txt` before code|Layer cache for faster rebuilds|
|Use `--no-cache-dir` with pip|Smaller image|
|Add `.dockerignore`|Exclude unnecessary files from context|
|Run as non-root user|Security|
|Add `HEALTHCHECK`|Visibility into container health|
|Use `EXPOSE` and document ports|Team clarity|

### Compose Best Practices Summary

|Rule|Why|
|---|---|
|Use `depends_on` with health conditions|Correct startup ordering|
|Set `restart: unless-stopped`|Auto-recovery from crashes|
|Always set memory/CPU limits|Prevent resource starvation|
|Use named volumes for persistence|Data survives container removal|
|Store secrets in `.env` (not compose file)|Security|
|Use `healthcheck` on every service|Enables `depends_on: condition: healthy`|

### Production Deployment Decision Tree

```
How many users / requests?
├── Low (internal tool, MVP)
│   └── Single VM + Docker Compose
│       └── Add Nginx reverse proxy + Let's Encrypt SSL
│
├── Medium (growing startup)
│   └── Docker Swarm OR
│       Cloud-managed containers (Cloud Run, Azure Container Apps, Fargate)
│
└── High (enterprise, >100 RPS, multi-region)
    └── Kubernetes (EKS / GKE / AKS)
        └── Add: Istio service mesh, Argo CD for GitOps,
                  Prometheus + Grafana, Cert-manager
```

### AI-Specific Infrastructure Reference

|Component|Common Choice|Docker Image|
|---|---|---|
|Vector DB|Qdrant|`qdrant/qdrant`|
|Vector DB|Weaviate|`semitechnologies/weaviate`|
|Vector DB|Chroma|`chromadb/chroma`|
|LLM serving (local)|Ollama|`ollama/ollama`|
|LLM serving (production)|vLLM|`vllm/vllm-openai`|
|Message queue|Redis|`redis:7-alpine`|
|Task queue|Celery (w/ Redis)|your image|
|Metrics|Prometheus|`prom/prometheus`|
|Dashboards|Grafana|`grafana/grafana`|
|LLM observability|Langfuse|`langfuse/langfuse`|
|Model registry|MLflow|`ghcr.io/mlflow/mlflow`|

---

_Report compiled April 2026. Docker ecosystem information reflects Docker Desktop 5.x, Docker Compose 2.x, and Kubernetes 1.32+._