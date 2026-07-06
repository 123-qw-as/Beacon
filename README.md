<p align="center">
  <img src="frontend/assets/beacon-logo.png" alt="Beacon Logo" width="180" />
</p>

<h1 align="center">Beacon</h1>
<h3 align="center">Lighting the path for every math modeling student.</h3>

<p align="center">
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="#how-it-works"><strong>How It Works</strong></a> ·
  <a href="#web-ui"><strong>Web UI</strong></a> ·
  <a href="#cli"><strong>CLI</strong></a> ·
  <a href="#project-structure"><strong>Structure</strong></a> ·
  <a href="#configuration"><strong>Configuration</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/node-≥18-green" alt="Node 18+" />
  <img src="https://img.shields.io/badge/framework-LangGraph-orange" alt="LangGraph" />
  <img src="https://img.shields.io/badge/llm-LiteLLM-purple" alt="LiteLLM" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License" />
</p>

---

## What is Beacon?

Beacon is an **end-to-end math modeling automation system** built for students competing in MCM/ICM, GMCM (China Graduate Mathematical Contest in Modeling), and similar contests. Given a competition problem, Beacon orchestrates a multi-agent LangGraph pipeline to analyze the problem, build models, write and execute code, generate figures, produce a complete paper, and compile it to PDF — all with **human-in-the-loop review** at key checkpoints.

> **Why "Beacon"?** In the intense, time-constrained environment of a math modeling competition, teams need clarity and direction. Beacon lights the path from a raw problem statement to a polished paper.

---

## Quick Start

### Prerequisites

- **Python ≥ 3.11** — for the backend agent pipeline
- **Node.js ≥ 18** — for the Web UI server
- **uv** — Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- An **LLM API endpoint** (OpenAI-compatible) — your router, proxy, or cloud provider

### One-command launch

```bash
# Clone
git clone https://github.com/123-qw-as/Beacon.git
cd Beacon

# Configure
cp .env.example .env
# Edit .env → set your LLM API base + model names

# Install & launch
npm install
npm start
```

Open **http://localhost:5173** in your browser — the Web UI will guide you through importing a problem, configuring the pipeline, and monitoring the run in real time.

---

## How It Works

Beacon uses **LangGraph** to orchestrate a 10-node agent pipeline:

```mermaid
graph LR
    A[Analyst] --> B[Modeler]
    B --> C[Model Critic]
    C -->|retry| B
    C -->|advance| D[Coder]
    D --> E[Sensitivity]
    E --> F[Figure Pipeline]
    F --> G[Writer]
    G --> H[Paper Critic]
    H -->|retry| G
    H -->|advance| I[Evaluation]
    I --> J[Human Review]
    J --> K[LaTeX / PDF]
```

| Node | Role |
|------|------|
| **Analyst** | Decomposes the problem, identifies constraints and domains |
| **Modeler** | Builds models through basic → improved → final stages |
| **Model Critic** | Validates assumptions, catches derivation gaps |
| **Coder** | Generates executable Python code for experiments |
| **Sensitivity** | Runs parameter sweeps and robustness analysis |
| **Figure Pipeline** | Renders charts + multi-modal quality review |
| **Writer** | Generates paper sections with structured outlines |
| **Paper Critic** | Reviews paper quality and format |
| **Evaluation** | Produces rubric-aligned scoring (6 dimensions) |
| **Human Review** | Pauses for human approval before finalization |
| **LaTeX** | Compiles paper to PDF via XeLaTeX, falls back to Markdown |

**Key design choices:**
- Every LLM call goes through a **unified `complete()` with retry, timeout, and structured output repair**
- Paper sections use **Jinja2 templates** for consistent formatting
- **RAG (Retrieval-Augmented Generation)** optionally injects classic model patterns and prize-winning paper excerpts into analyst/modeler/writer prompts
- **Checkpoint-based recovery** — if any node crashes, you can `resume` or `recover` from the last saved state

---

## Web UI

Beacon ships with a complete browser-based workspace:

<p align="center">
  <em>Dashboard with pipeline progress, real-time logs, artifact preview, and parameter controls</em>
</p>

**Features:**
- **Problem configuration** — paste a problem or import from JSON/Markdown/PDF
- **Template switching** — Default (standard paper) or GMCM (国赛 gmcmthesis)
- **Live pipeline progress** — 10-stage visualization with real-time node status
- **Run control** — start, monitor logs, stop, and view artifacts
- **RAG toggle** — enable/disable retrieval augmentation per run
- **HITL toggle** — run fully automatic or pause for human approval

---

## CLI

Prefer the terminal? The Python CLI is fully independent:

```bash
# Run with a problem file
uv run math-agent run \
  --problem tests/fixtures/sample_problem.json \
  --out runs/demo \
  --no-interrupt

# Run with human-in-the-loop (default)
uv run math-agent run \
  --problem tests/fixtures/bike_dispatch_full.json \
  --out runs/demo2

# After reviewing intermediate results, approve and continue
uv run math-agent resume --out runs/demo2 --approve --notes "looks good"

# Recover from a crash without manual approval
uv run math-agent recover --out runs/demo2

# View the run report (tokens, timing, per-node breakdown)
uv run math-agent report --out runs/demo

# Index a corpus directory for RAG
uv run math-agent ingest --src corpus/models --db runs/rag.sqlite

# Run the benchmark suite
uv run math-agent bench --out runs/bench
```

---

## Project Structure

```
Beacon/
├── frontend/                   # Web UI
│   ├── assets/
│   │   └── beacon-logo.png     # Project logo
│   ├── server.mjs              # HTTP server + API proxy (connects browser ↔ backend)
│   ├── index.html              # Main page
│   ├── app.js                  # Frontend logic
│   └── styles.css              # Stylesheet
│
├── src/math_agent/             # Python backend
│   ├── cli.py                  # Typer CLI entry point
│   ├── graph.py                # LangGraph graph construction
│   ├── state.py                # Pydantic state schema
│   ├── config.py               # Centralized configuration
│   ├── llm.py                  # Unified LLM client (LiteLLM + retry + repair)
│   ├── tracing.py              # Lightweight run tracer
│   ├── routing.py              # Conditional edge routing (critic loops)
│   ├── errors.py               # Typed exception hierarchy
│   ├── retry.py                # Tenacity retry decorators
│   │
│   ├── nodes/                  # Pipeline nodes (10 total)
│   │   ├── analyst.py          #   Problem decomposition
│   │   ├── modeler.py          #   Model construction
│   │   ├── model_critic.py     #   Model quality review
│   │   ├── coder.py            #   Code generation
│   │   ├── sensitivity.py      #   Parameter sensitivity analysis
│   │   ├── figure_pipeline.py  #   Figure generation + review
│   │   ├── writer.py           #   Paper writing (prep + section loop)
│   │   ├── paper_critic.py     #   Paper quality review
│   │   ├── evaluation.py       #   Rubric-based scoring
│   │   ├── human_review.py     #   HITL gate
│   │   ├── latex.py            #   LaTeX → PDF compilation
│   │   └── table_assembler.py  #   Table formatting
│   │
│   ├── prompts/                # Prompt templates per node
│   │   ├── analyst.py
│   │   ├── modeler.py
│   │   ├── modeler_derivation.py
│   │   ├── model_critic.py
│   │   ├── coder.py
│   │   ├── coder_baseline.py
│   │   ├── coder_figure_one.py
│   │   ├── sensitivity.py
│   │   ├── writer.py
│   │   ├── writer_section.py
│   │   ├── paper_critic.py
│   │   ├── figure_analyst.py
│   │   ├── figure_critic.py
│   │   └── evaluation.py
│   │
│   ├── tools/                  # External tools
│   │   ├── runner.py           #   Python subprocess executor
│   │   ├── latex_compile.py    #   XeLaTeX compiler
│   │   ├── references.py       #   Semantic Scholar lookup
│   │   ├── scholar.py          #   Academic search
│   │   └── image.py            #   Image utilities
│   │
│   └── rag/                    # Retrieval-Augmented Generation
│       ├── ingest.py           #   Corpus ingestion
│       ├── chunking.py         #   Structure-aware text splitting
│       ├── embeddings.py       #   Embedding generation
│       ├── store.py            #   sqlite-vec vector store
│       └── retrieve.py         #   Similarity search
│
├── tests/                      # Test suite
│   └── fixtures/               #   Sample problems
│       ├── sample_problem.json
│       └── bike_dispatch_full.json
│
├── scripts/                    # Utility scripts
│   ├── start.bat               #   Windows one-click launcher
│   └── start.sh                #   Unix/macOS one-click launcher
│
├── package.json                # Node.js configuration
├── pyproject.toml              # Python package configuration
├── .env.example                # Environment variable template
└── .gitignore
```

---

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
# --- LLM API ---
OPENAI_API_BASE=http://localhost:20128/v1   # Your OpenAI-compatible endpoint
OPENAI_API_KEY=your-key-here

# --- Model Selection ---
MATH_AGENT_DEFAULT_MODEL=openai/gpt-4o-mini  # For routine nodes (coder)
MATH_AGENT_STRONG_MODEL=openai/gpt-4o        # For core nodes (analyst, modeler, writer)

# --- RAG (optional) ---
MATH_AGENT_RAG_ENABLED=1
MATH_AGENT_RAG_EMBED=text-embedding-3-small
MATH_AGENT_RAG_DIM=1536

# --- Frontend ---
PORT=5173
MATH_AGENT_COMMAND=uv run math-agent
```

For the full list of tunable parameters, see `.env.example` or `src/math_agent/config.py`.

---

## RAG Setup (Optional)

To enable retrieval augmentation with classic models and prize-winning papers:

```bash
# 1. Prepare your corpus
mkdir -p corpus/models corpus/papers
# Place .md / .txt / .pdf files in these directories

# 2. Index the corpus
uv run math-agent ingest \
  --src corpus/models \
  --db runs/rag.sqlite \
  --embedding-model text-embedding-3-small \
  --dim 1536

# 3. Enable in .env
# MATH_AGENT_RAG_ENABLED=1
```

---

## Development

```bash
# Backend
uv run math-agent run --problem tests/fixtures/sample_problem.json --out runs/dev

# Frontend (with hot reload)
npm run dev

# Tests
uv run pytest -q
```

---

## FAQ

<details>
<summary><strong>What contests does Beacon support?</strong></summary>

Beacon is designed for MCM/ICM and GMCM (中国研究生数学建模竞赛). The default template targets standard English-language papers. The `--template gmcm` flag enables the `gmcmthesis` document class with Chinese-language support, school/team/member fields, and the required cover page format.
</details>

<details>
<summary><strong>Can I use any LLM provider?</strong></summary>

Yes. Beacon uses LiteLLM under the hood, which supports 100+ providers. Any OpenAI-compatible endpoint works. Configure `OPENAI_API_BASE` and `OPENAI_API_KEY` in `.env`. Model names use the `provider/model` format (e.g., `openai/gpt-4o`, `ollama/llama3`).
</details>

<details>
<summary><strong>What happens if the LLM returns poorly formatted JSON?</strong></summary>

Beacon's `complete()` function applies multiple repair strategies: stripping thinking tags (`<think>` blocks), extracting JSON from markdown code fences, escaping illegal backslash sequences in LaTeX math, and retrying with the previous response + error as context. If all retries exhaust, a typed `LLMError` is raised with a saved checkpoint so you can resume.
</details>

<details>
<summary><strong>How do I resume after a crash?</strong></summary>

```bash
# For crashes before human_review
uv run math-agent recover --out runs/your-run

# For crashes at human_review (you need to inject a decision)
uv run math-agent resume --out runs/your-run --approve --notes "approved"
```

The `recover` command restarts from the last saved LangGraph checkpoint. The `resume` command does the same but also injects a human approval decision.
</details>

<details>
<summary><strong>Do I need XeLaTeX installed?</strong></summary>

XeLaTeX produces the highest-quality PDF output. If it's not installed, Beacon automatically falls back to writing `paper.md` (Markdown) instead. The LaTeX node detects missing `xelatex` gracefully and reports it as a non-fatal condition.
</details>

---

## License

MIT © 2026

---

<p align="center">
  <sub>Built with ❤️ for math modeling teams everywhere.</sub>
</p>
