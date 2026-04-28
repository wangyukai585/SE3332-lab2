# Lab 2: Multi-hop QA with LLM and RAG

## Prerequisites

- [Anaconda / Miniconda](https://www.anaconda.com/download) installed
- An **OpenAI API key** (for GPT-4o-mini). Obtain one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
- The dataset file `hotpotqa_longbench.json` placed in the **parent directory** (`lab2/`), one level above `code/`.

## Environment Setup

```bash
# 1. Create and activate a dedicated conda environment
conda create -n se3332_lab2 python=3.11 -y
conda activate se3332_lab2

# 2. Install dependencies
pip install -r requirements.txt
```

## API Key Configuration

Open **both** `main.py` and `improved.py`, and replace the placeholder on line 18 with your actual key:

```python
API_KEY = "YOUR_OPENAI_API_KEY_HERE"   # ← put your key here
```

## Directory Layout

```text
lab2/                          ← project root
├── hotpotqa_longbench.json    ← dataset (place here, NOT inside code/)
├── report.pdf
└── code/
    ├── main.py                ← Basic RAG (BM25 + GPT-4o-mini)
    ├── improved.py            ← Improved RAG (Query Decomposition)
    ├── requirements.txt
    ├── README.md
    └── results/
        ├── outputs.json           ← Basic RAG predictions (200 samples)
        └── outputs_improved.json  ← Query-Decomp predictions (200 samples)
```

## Running

All commands should be run from inside the `code/` directory with the conda environment activated.

### Basic RAG

```bash
cd code
python main.py
```

Runs BM25 retrieval + GPT-4o-mini generation on all 200 samples.  
Prints average EM and F1, then saves predictions to `results/outputs.json`.

### Improved RAG (Query Decomposition)

```bash
python improved.py
```

Decomposes each question into sub-questions, answers them iteratively with RAG, then synthesises a final answer.  
Saves predictions to `results/outputs_improved.json`.

> **Note:** The improved script makes ~4× more API calls than the basic one (~800 calls for 200 samples). At GPT-4o-mini pricing this costs roughly $0.05–$0.10 for the full run.

## System Overview

### Basic RAG (`main.py`)

| Step | Detail |
| --- | --- |
| Chunking | Split context by `Passage N:` markers; sliding window 150 words / 30-word overlap for large passages |
| Retrieval | Okapi BM25, top-5 chunks |
| Generation | GPT-4o-mini, temperature=0.1, concise answer prompt |

### Improved RAG (`improved.py`)

| Step | Detail |
| --- | --- |
| Decompose | GPT-4o-mini breaks multi-hop question into 2-3 atomic sub-questions |
| Iterative RAG | Each sub-question answered with BM25; prior sub-answers enrich next query |
| Synthesise | Final GPT call combines all sub-answers into the definitive answer |

## Results

| System | Avg EM | Avg F1 |
| --- | --- | --- |
| No RAG (baseline) | 0.2000 | 0.2467 |
| Basic RAG (BM25) | 0.4150 | 0.5266 |
| Query-Decomp RAG | 0.3450 | 0.4492 |

## Evaluation Metrics

- **EM (Exact Match)**: 1 if normalised prediction equals normalised gold answer.
- **F1**: Token-level F1 between prediction and gold answer (after lower-casing and removing punctuation/articles).
