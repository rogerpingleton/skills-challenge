# Cloud platforms

Here's a comprehensive overview of the major cloud platforms used in AI Engineering, with what you need to know as a practitioner working in Python.

---

# Cloud Platforms for AI Engineering

The cloud market is dominated by three hyperscalers. AWS, Azure, and GCP currently account for a combined 62% share of the cloud infrastructure market. Each has carved out a distinct identity for AI work: AWS is ideal for scalable and extensive deployment environments; Azure is optimal for regulated enterprise environments with hybrid cloud needs; and GCP is best for cutting-edge AI workloads and innovative research.

---

## 1. Amazon Web Services (AWS)

AWS remains the global market leader with a 28% market share. AWS provides the most mature and extensive ecosystem of more than 200 services.

### Amazon Bedrock

Amazon Bedrock is a fully managed serverless service that enables developers to integrate foundation models (from AI21, Anthropic, Cohere, and Meta) into their apps via an API. It's the primary entry point for generative AI on AWS. Bedrock integrates seamlessly with AWS services like Lambda and SageMaker, and it allows you to customize and fine-tune models with domain information to improve accuracy for specific use cases.

As a Python engineer, you interact with Bedrock through `boto3`. A minimal example using the Converse API:

```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.converse(
    modelId="anthropic.claude-sonnet-4-5",
    messages=[{"role": "user", "content": [{"text": "Summarize the key risks of RAG pipelines."}]}]
)

print(response["output"]["message"]["content"][0]["text"])
```

The Bedrock ecosystem covers prompt engineering, agents and their components, custom model import, multimodal data, RAG implementation, and responsible AI tooling.

**Key things to know:**

- IAM permissions are the most common source of friction. Your execution role must have `bedrock:InvokeModel` and related permissions explicitly granted.
- Bedrock allows prompt engineering but not direct fine-tuning in the traditional sense — for that, SageMaker is the right tool.
- The `bedrock-agentcore-sdk-python` is a newer Python SDK for transforming AI agents into production-ready applications, offering framework-agnostic primitives for runtime, memory, authentication, and tools with AWS-managed infrastructure.

### Amazon SageMaker

SageMaker is the full-lifecycle ML platform. SageMaker provides a platform to host, train, and fine-tune models to improve their speed and accuracy and to adapt them to required use cases.

SageMaker Pipeline is a sequence of connected steps organized in a directed acyclic graph (DAG), which can be created using a drag-and-drop interface or the Pipelines SDK. It's the right choice when you need to own the training loop — custom datasets, custom architectures, or reproducible experimentation workflows.

```python
from sagemaker.sklearn import SKLearn

estimator = SKLearn(
    entry_point="train.py",
    role="arn:aws:iam::123456789:role/SageMakerRole",
    instance_type="ml.m5.xlarge",
    framework_version="1.2-1"
)
estimator.fit({"train": "s3://my-bucket/train"})
predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m5.large")
```

**Key things to know:**

- SageMaker Studio is the recommended IDE for the full ML lifecycle.
- SageMaker's value comes when you have data that no one else does — and predictions so specific to your situation that a generic model wouldn't suffice.
- Notebook execution roles are _separate_ from your AWS Console user, which causes IAM confusion for newcomers.

### AWS Compute for AI — EC2 & Specialized Silicon

Beyond standard VMs, AWS offers specialized instances powered by their own custom silicon — Trainium and Inferentia — designed specifically to lower the cost of training and running AI models. For Python engineers, this matters when optimizing inference cost at scale; the `torch-neuronx` library provides Inferentia support.

---

## 2. Microsoft Azure

Azure is the second-largest provider. Azure's growth has been accelerated by its exclusive partnership with OpenAI, making it the primary home for GPT-4o and other cutting-edge models via Azure AI Foundry.

### Azure AI Foundry

In 2025, Azure AI Foundry has emerged as a centralized hub for building, fine-tuning, and deploying AI models at scale. It's the successor to Azure AI Studio and consolidates model access, fine-tuning, evaluation, and deployment pipelines under one roof.

Azure AI Foundry provides deep enterprise integration with direct access to OpenAI's models (GPT-4, GPT-5), plus a strong MLOps studio.

Python access uses the `azure-ai-inference` SDK:

```python
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

client = ChatCompletionsClient(
    endpoint="https://<your-resource>.openai.azure.com/",
    credential=AzureKeyCredential("<your-api-key>")
)

response = client.complete(
    messages=[
        SystemMessage("You are a helpful AI engineer assistant."),
        UserMessage("What is the difference between RAG and fine-tuning?")
    ],
    model="gpt-4o"
)
print(response.choices[0].message.content)
```

**Key things to know:**

- For teams whose highest priority is access to OpenAI's flagship GPT models within an enterprise-grade Microsoft environment, Azure AI Foundry is the best fit — especially when seamless integration with Microsoft 365, Cognitive Search, and Active Directory is needed.
- Azure integrates GitHub Copilot with Azure DevOps, and Azure Active Directory simplifies identity management across on-premises and cloud.
- Azure's documentation and management console can be inconsistent across services. Budget extra time for navigating the portal when setting up new services.

### Azure Machine Learning

Azure ML is Azure's counterpart to SageMaker — a full platform for training, tracking experiments, building pipelines, and deploying models. It integrates tightly with MLflow for experiment tracking, which is a major plus for Python workflows.

```python
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<subscription-id>",
    resource_group_name="<rg>",
    workspace_name="<workspace>"
)

job = command(
    code="./src",
    command="python train.py --epochs ${{inputs.epochs}}",
    inputs={"epochs": 10},
    environment="AzureML-sklearn-1.0-ubuntu20.04:1",
    compute="cpu-cluster"
)
ml_client.jobs.create_or_update(job)
```

**Key things to know:**

- `DefaultAzureCredential` is the recommended auth pattern — it chains through environment variables, managed identity, and interactive login automatically.
- Azure ML pipelines use a YAML-based component system that's well-suited to CI/CD integration.

### Azure Kubernetes Service (AKS) for AI Inference

Azure's ND H100 v5-series VMs are specifically built for massive-scale AI workloads, offering integrated NVIDIA H100 Tensor Core GPUs and high-speed networking. AKS is the deployment target of choice for serving custom models at scale in the Azure ecosystem.

---

## 3. Google Cloud Platform (GCP)

Google Cloud Platform is the AI-native cloud. Google invented the Transformer architecture that underpins modern LLMs, and GCP's Vertex AI platform offers Gemini 1.5 Pro, Imagen 3, and the most advanced MLOps tooling available.

### Vertex AI

Vertex AI is a machine learning platform that lets you train and deploy ML models and AI applications. Vertex AI combines data engineering, data science, and ML engineering workflows, which lets teams collaborate using a common toolset.

Google Vertex AI is a unified Google-native ML platform, featuring the Gemini family, PaLM, and Model Garden with third-party and open-source models like Llama, Gemma, and BERT — along with advanced, data-driven MLOps.

**Important SDK note for 2025/2026:** Several generative AI modules in the Vertex AI SDK are deprecated as of June 2025 and will be removed June 2026: `vertexai.generative_models`, `vertexai.language_models`, `vertexai.vision_models`. You should use the Google Gen AI SDK (`google-genai`) to access these features.

The current recommended Python pattern for Gemini on Vertex:

```python
from google import genai

# For Vertex AI (vs. direct Gemini Developer API)
client = genai.Client(
    vertexai=True,
    project="your-project-id",
    location="us-central1"
)

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Explain the tradeoffs between vector databases and BM25 search for RAG."
)
print(response.text)
```

For deploying agents on Vertex using Google's Agent Development Kit (ADK):

```python
from google.adk.agents import Agent
from vertexai.agent_engines import AdkApp

app = AdkApp(agent=Agent(
    model="gemini-2.0-flash",
    name="my_agent",
    tools=[my_tool_function],
))

# Deploy to Vertex
remote_app = client.agent_engines.create(
    agent=app,
    config={"requirements": ["google-cloud-aiplatform[agent_engines,adk]"]},
)
```

**Key things to know:**

- For organizations training or fine-tuning large models, Google's Tensor Processing Units (TPUs) remain the gold standard for compute efficiency at scale.
- GCP's Cloud Run is a fully managed serverless environment that runs containers and automatically scales them, providing more flexibility than traditional functions — it's the preferred deployment target for containerized inference APIs.
- Google Kubernetes Engine (GKE) is widely considered the most mature and feature-rich managed Kubernetes service available.

### BigQuery for AI Data Engineering

BigQuery, Google's serverless data warehouse, continues to lead the market for real-time analytics at petabyte scale. For AI engineers, BigQuery ML lets you train and run inference directly in SQL, and BigQuery is a first-class data source for Vertex AI training pipelines and feature stores.

---

## Cross-Cutting Concerns for AI Engineers

**Infrastructure as Code.** Regardless of which cloud you use, learn Terraform. Terraform for IaC helps overcome vendor lock-in challenges. All three platforms have Terraform providers, and reproducible infrastructure is essential for ML experiments.

**IAM and Security.** All three providers offer robust IAM, but Azure integrates best with enterprise Active Directory. The common best practice across all three is least-privilege access — scope your roles as narrowly as possible, especially for anything touching training data or model endpoints.

**Kubernetes.** Kubernetes has become the de facto standard for container orchestration. GKE is widely regarded as the most mature and developer-friendly managed Kubernetes offering. AWS EKS, while the most flexible, requires more manual configuration and has a steeper learning curve.

**Cost Management.** Cloud AI costs can escalate rapidly with GPU/TPU usage, large context windows, and high-throughput inference. Tag all resources, set billing alerts, and use spot/preemptible instances for non-critical training jobs. Each platform has a cost optimization advisor (AWS Compute Optimizer, Azure Advisor, GCP Recommender).

**Multi-cloud reality.** Most large organizations use two or all three clouds in a multi-cloud strategy, leveraging each provider's best-in-class services for specific workloads. Tools like Terraform, Kubernetes, and open standards make this increasingly practical without vendor lock-in. In practice, you'll often find model inference on one cloud, data warehousing on another, and orchestration in a cloud-agnostic tool like Airflow or Prefect.

---

## Quick Reference: Which Platform for What

| Task                                     | Best Fit                             |
| ---------------------------------------- | ------------------------------------ |
| Access GPT-4o / OpenAI models            | Azure AI Foundry                     |
| Access Claude models                     | AWS Bedrock or Vertex AI             |
| Train custom models at scale             | SageMaker (AWS) or Vertex AI (GCP)   |
| LLM fine-tuning, enterprise governance   | Azure ML                             |
| TPU training / large-scale LLM training  | GCP Vertex AI                        |
| Real-time analytics feeding AI pipelines | BigQuery (GCP)                       |
| Container-based inference serving        | Cloud Run (GCP) or ECS/Fargate (AWS) |
| Broadest third-party model selection     | AWS Bedrock                          |
| Kubernetes-based serving                 | GKE (GCP) is most mature             |
| Microsoft ecosystem integration          | Azure across the board               |