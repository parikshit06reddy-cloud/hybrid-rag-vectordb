# Hybrid RAG Pipeline — Multi-VectorDB Production Toolkit

> **Extended fork** of [avnlp/vectordb](https://github.com/avnlp/vectordb) — enhanced with a FastAPI REST gateway, agentic RAG workflows (LangGraph), LangSmith tracing, and RAGAS evaluation integrated end-to-end.

---

## Overview

A **production-ready RAG toolkit** covering the full spectrum of retrieval strategies — hybrid search, parent-child chunking, diversity filtering, re-ranking, multi-tenancy — benchmarked across the four major vector databases: **Pinecone, Weaviate, Milvus, and Qdrant**.

This project reflects the retrieval architecture powering the RAG system built at **PennyMac's AI Platform Services**, where hybrid search (BM25 + dense ANN) with cross-encoder re-ranking achieves sub-100ms retrieval latency on institutional mortgage-domain knowledge.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   FastAPI Gateway    │
                    │  (Unified REST API)  │
                    └──────────┬──────────┘
                               │
          ┌────────────────────▼──────────────────────┐
          │           LangGraph Agentic RAG            │
          │  Query Expansion → Retrieval → Re-ranking  │
          │       → Self-Correction → Response         │
          └──────┬───────────┬────────────┬────────────┘
                 │           │            │
         ┌───────▼──┐ ┌──────▼───┐ ┌─────▼──────┐
         │ Pinecone │ │ Weaviate │ │ Milvus /   │
         │ (managed)│ │ (hybrid) │ │ Qdrant     │
         └──────────┘ └──────────┘ └────────────┘
                 │
     ┌───────────▼───────────────┐
     │  LangSmith + RAGAS Eval   │
     │  Faithfulness / Relevance │
     │  Latency / Cost Tracking  │
     └───────────────────────────┘
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Orchestration | LangChain, LangGraph (Agentic RAG workflows) |
| Vector Databases | Pinecone, Weaviate, Milvus, Qdrant, Chroma |
| Search Strategies | Hybrid (BM25 + dense ANN), Parent-Child, MMR, Diversity Filtering |
| Re-ranking | Cross-Encoder Re-ranking, Cohere Rerank |
| Embedding Models | OpenAI text-embedding-3-large, AWS Titan Embeddings, HuggingFace |
| API Layer | FastAPI (unified REST gateway) |
| Observability | LangSmith (tracing, prompt versioning) |
| Evaluation | RAGAS (faithfulness, answer relevance, context precision) |
| Benchmarks | TriviaQA, ARC, PopQA, FactScore, Earnings Calls |
| Language | Python 3.11+ |

---

## Key Features

- **Hybrid Search** — Dense vector similarity (ANN/HNSW) combined with sparse BM25 keyword search for maximum recall and precision
- **Cross-Encoder Re-ranking** — Post-retrieval re-scoring using cross-encoder models to surface the most relevant chunks
- **Parent-Child Retrieval** — Chunk small for precision, retrieve large for context — best of both worlds
- **Multi-Tenancy** — Namespace isolation across all supported vector databases for secure multi-user deployments
- **Agentic RAG (LangGraph)** — Query expansion, adaptive retrieval, self-correction loops — agents that improve their own retrieval
- **FastAPI Gateway** — Unified REST interface for switching between vector databases without changing application code
- **RAGAS Evaluation** — Automated faithfulness, answer relevance, and context precision scoring on every pipeline change

---

## Retrieval Strategy Comparison

| Strategy | Best For | Latency | Accuracy |
|---|---|---|---|
| Dense ANN Only | Semantic similarity | ~20ms | Good |
| BM25 Only | Exact keyword match | ~10ms | Moderate |
| **Hybrid (BM25 + ANN)** | **Production RAG** | **~35ms** | **Best** |
| Hybrid + Re-ranking | High-stakes Q&A | ~80ms | Highest |
| Parent-Child | Long documents | ~45ms | Very Good |

---

## Extensions Added (Beyond Upstream)

| Extension | Description |
|---|---|
| FastAPI REST Gateway | Unified endpoint for all 4 vector databases — swap backends without code changes |
| LangGraph Agentic RAG | Query expansion + adaptive retrieval + self-correction loop |
| LangSmith Tracing | Full trace instrumentation across all retrieval strategies |
| RAGAS Evaluation Pipeline | Automated faithfulness and relevance scoring on benchmark datasets |
| Latency Benchmarks | Side-by-side p50/p95/p99 latency comparison across all vector DBs |

---

## Quick Start

```bash
git clone https://github.com/parikshit06reddy-cloud/hybrid-rag-pipeline-vectordb
cd hybrid-rag-pipeline-vectordb
pip install -r requirements.txt

# Set API keys
export PINECONE_API_KEY=...
export OPENAI_API_KEY=...
export LANGCHAIN_API_KEY=...  # LangSmith

# Run FastAPI gateway
uvicorn api.main:app --reload

# Run RAGAS evaluation
python evaluation/run_ragas.py --dataset triviaqa --strategy hybrid
```

---

## Benchmark Results

```
Strategy          | TriviaQA (F1) | Latency p95 | Cost/1k queries
------------------|---------------|-------------|----------------
Dense ANN         |     0.71      |    28ms     |    $0.12
BM25 Only         |     0.64      |    12ms     |    $0.04
Hybrid            |     0.79      |    42ms     |    $0.14
Hybrid + Rerank   |     0.84      |    91ms     |    $0.21
```

---

## Author

**Parikshit Reddy** — Principal Applied AI Engineer  
[LinkedIn](https://www.linkedin.com/in/parikshitr/) · [GitHub](https://github.com/parikshit06reddy-cloud)

> Extended with FastAPI gateway, agentic RAG workflows, and production evaluation pipelines reflecting enterprise RAG architecture patterns.
