# ConvFinQA RAG-Based Question Answering System

This project implements a robust **retrieval-augmented generation (RAG)** pipeline to answer complex financial questions using the **ConvFinQA** dataset. It is designed for structured document understanding (tables + narrative), supporting:

- Dense retrieval  
- Reranking  
- Structured prompts  
- Answer extraction

---

## Prerequisites

- Python 3.12  
- OpenAI API (e.g., GPT-4)  
- Cohere API access

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the vector store

```bash
python -m src.vector_store.builder
```

### 3. Ask a question

```bash
python main.py --question "what was the percentage change in the net cash from operating activities from 2008 to 2009"
```

### 4. Run evaluation

```bash
python evaluate.py
```

---

## Features

- Row-level chunking of financial tables  
- Sub-query generation to enhance retrieval coverage  
- Dual-context prompting (table + narrative)  
- Year-aware document filtering  
- Cohere-based reranker to refine results  
- Numeric-tolerant evaluation logic  
- Table content is embedded row-wise for granular semantic matching  
- Markdown formatting used to assist LLM parsing  
- FAISS-based efficient similarity search  
- Modularized core logic in `src/`  

---

## Evaluation

### Metrics computed:

- **Accuracy** (LLM output vs. gold answer)  
- **Retrieval/Rerank Precision & Recall**  
- **Latency** (per example)

### Example Results (500 samples):

| Metric               | Value   |
|----------------------|---------|
| Average Accuracy     | 41.42%  |
| Retrieval Precision  | 1.76%   |
| Retrieval Recall     | 56.20%  |
| Reranker Precision   | 5.12%   |
| Reranker Recall      | 50.40%  |
| Latency (avg)        | 9.29 sec|

As seen, reranking improves **precision** significantly, even if **recall** slightly drops — this helps LLMs focus on fewer but more relevant contexts, directly improving answer quality.

When `use_ground_truth_retrieval = True` (oracle setup), **accuracy jumps to 79.58%**, indicating that **retrieval quality is the primary bottleneck**, not the generation.

---

## Project Structure

- `src/` – Core modules: agent, llm, evaluation, common utilities  
- `data/` – Parsed ConvFinQA data  
- `outputs/` – Evaluation outputs  
- `main.py` – Entry point for answering questions  
- `evaluate.py` – Evaluation script  

---

## Notes

- The system embeds table rows individually for fine-grained matching.
- Uses Markdown-formatted tables to help the LLM parse structured content more effectively.
