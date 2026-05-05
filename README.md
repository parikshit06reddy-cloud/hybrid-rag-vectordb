# Hybrid RAG Pipeline — Multi-VectorDB Production Toolkit

[![CI](https://github.com/parikshit06reddy-cloud/hybrid-rag-vectordb/actions/workflows/ci.yml/badge.svg)](https://github.com/parikshit06reddy-cloud/hybrid-rag-vectordb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> Production-ready RAG toolkit covering the full retrieval spectrum — hybrid search, parent-child chunking, diversity filtering, re-ranking, multi-tenancy — benchmarked across **Pinecone, Weaviate, Milvus, Qdrant, and Chroma**, with a FastAPI gateway, agentic RAG workflows on LangGraph, LangSmith tracing, and RAGAS evaluation.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Key features](#key-features)
- [Retrieval strategy comparison](#retrieval-strategy-comparison)
- [Quick start](#quick-start)
- [Configuration and secrets](#configuration-and-secrets)
- [Reproducibility and dependency locking](#reproducibility-and-dependency-locking)
- [Testing, linting, and security](#testing-linting-and-security)
- [Benchmark results](#benchmark-results)
- [Contributing](#contributing)
- [License](#license)

---

## Why this exists

Most RAG demos pin to one vector store and one retrieval strategy. Real systems need to:

- A/B test retrieval strategies (dense, sparse, hybrid, parent-child, MMR, re-ranked) on real datasets
- Swap vector backends without rewriting application code
- Trace every query end-to-end and produce auditable evaluation reports
- Catch supply-chain regressions before they reach production

This repo packages those concerns into a single toolkit and a production-shaped CI pipeline.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   FastAPI Gateway   │
                    │  (Unified REST API) │
                    └──────────┬──────────┘
                               │
          ┌────────────────────▼──────────────────────┐
          │           LangGraph Agentic RAG           │
          │  Query Expansion → Retrieval → Re-ranking │
          │       → Self-Correction → Response        │
          └──────┬───────────┬────────────┬───────────┘
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

## Tech stack

| Layer | Technologies |
|---|---|
| Orchestration | LangChain, LangGraph (agentic RAG workflows) |
| Vector databases | Pinecone, Weaviate, Milvus, Qdrant, Chroma |
| Search strategies | Hybrid (BM25 + dense ANN), parent-child, MMR, diversity filtering |
| Re-ranking | Cross-encoder, Cohere Rerank |
| Embedding models | Qwen3 Embedding, MiniLM, MPNet, OpenAI, Titan, HF |
| API layer | FastAPI |
| Observability | LangSmith (tracing, prompt versions) |
| Evaluation | RAGAS (faithfulness, answer relevance, context precision), DeepEval |
| Benchmarks | TriviaQA, ARC, PopQA, FactScore, Earnings Calls |
| Dependency mgmt | `uv` (lockfile-driven, reproducible) |
| Test/Lint | pytest + coverage, ruff, mypy, bandit, pip-audit, gitleaks, CodeQL, Trivy |
| Language | Python 3.11+ (3.14 also exercised in CI) |

---

## Key features

- **Hybrid search** — Dense ANN/HNSW + BM25 sparse for maximum recall and precision
- **Cross-encoder re-ranking** — Post-retrieval re-scoring to surface the best chunks
- **Parent-child retrieval** — Small chunks for precision, parent chunks for context
- **Multi-tenancy** — Namespace isolation across all supported vector DBs
- **Agentic RAG (LangGraph)** — Query expansion + adaptive retrieval + self-correction loop
- **FastAPI gateway** — Swap vector DB backends without changing application code
- **RAGAS + DeepEval** — Automated faithfulness, answer relevance, and context precision
- **Hardened CI** — Bandit, pip-audit, Gitleaks, CodeQL, and Trivy scans block merges, not just notify

---

## Retrieval strategy comparison

| Strategy | Best for | Latency | Accuracy |
|---|---|---|---|
| Dense ANN only | Semantic similarity | ~20ms | Good |
| BM25 only | Exact keyword match | ~10ms | Moderate |
| **Hybrid (BM25 + ANN)** | **Production RAG** | **~35ms** | **Best** |
| Hybrid + re-ranking | High-stakes Q&A | ~80ms | Highest |
| Parent-child | Long documents | ~45ms | Very Good |

---

## Quick start

Prerequisites:

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

```bash
git clone https://github.com/parikshit06reddy-cloud/hybrid-rag-vectordb.git
cd hybrid-rag-vectordb

uv sync --all-groups

cp .env.example .env  # edit with your API keys

uv run python -m vectordb.databases.pinecone.dense_indexing.run \
  --dataset triviaqa --strategy hybrid
```

---

## Configuration and secrets

Configuration is YAML with explicit environment-variable interpolation. Three placeholder forms are supported:

| Syntax | Meaning |
|---|---|
| `${VAR}` | Required. **Raises** `MissingEnvVarError` if unset (no silent empty strings). |
| `${VAR:-default}` | Optional with a default value. |
| `${VAR:?explanation}` | Required, with a documented error message. |

You can also fail fast at process start:

```python
from vectordb.utils.config import require_env

require_env("PINECONE_API_KEY", "OPENAI_API_KEY")
```

Never commit `.env`. Gitleaks runs in CI and GitHub push protection blocks credentials at push time. See `.gitleaks.toml` for repo-specific rules.

---

## Reproducibility and dependency locking

`uv.lock` is the source of truth for dependency resolution. CI uses `uv sync --all-groups` against the lockfile.

```bash
make lock         # regenerate uv.lock from pyproject.toml
make lock-check   # verify uv.lock is consistent with pyproject.toml (CI gate)
make sync         # install from the lockfile
```

Top-level dependencies in `pyproject.toml` use bounded version ranges (`>=X,<Y`) to absorb security patches without picking up unintended majors. Dependabot opens grouped weekly PRs for `langchain*`, `langgraph*`, vector stores, embeddings/eval libraries, and dev tooling.

---

## Testing, linting, and security

```bash
make test            # unit tests
make test-cov        # tests + coverage
make lint-check      # ruff format + lint (no autofix)
make lint-typing     # mypy
make security        # bandit + pip-audit
```

CI on every PR runs:

- **deps-py311 / deps-py314** — uv sync on Python 3.11 and 3.14
- **test-py311 / test-py314** — pytest with coverage uploaded to Codecov
- **security-scan** — Bandit, pip-audit, Gitleaks (now blocking)
- **codeql** — Python CodeQL with `security-and-quality` queries
- **trivy-fs** — filesystem CVE scan, SARIF uploaded to GitHub Security

---

## Benchmark results

```
Strategy          | TriviaQA (F1) | Latency p95 | Cost/1k queries
------------------|---------------|-------------|----------------
Dense ANN         |     0.71      |    28ms     |    $0.12
BM25 only         |     0.64      |    12ms     |    $0.04
Hybrid            |     0.79      |    42ms     |    $0.14
Hybrid + Rerank   |     0.84      |    91ms     |    $0.21
```

---

## Contributing

PRs welcome. Follow [CONTRIBUTING.md](CONTRIBUTING.md). Pre-commit hooks run ruff, gitleaks, and typo checks locally; CI gates merges on lint, type-check, unit tests, and security scans.

---

## License

MIT. See [LICENSE](LICENSE).

---

**Author:** Parikshit Reddy — [LinkedIn](https://www.linkedin.com/in/parikshitr/) · [GitHub](https://github.com/parikshit06reddy-cloud)

> Extended fork of [avnlp/vectordb](https://github.com/avnlp/vectordb) with FastAPI gateway, agentic RAG workflows, and production evaluation pipelines.
