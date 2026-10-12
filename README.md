# Topic: Grounded or Hallucinated? A Small RAG System that Knows When to Say "I Don't Know"

## Overview

Retrieval-Augmented Generation (RAG) allows a language model to answer questions using information retrieved from external documents instead of relying only on its internal memory.

This approach can improve factual accuracy and make responses more grounded in evidence. However, RAG systems still face important challenges:

- relevant information may be distributed across multiple documents,
- retrieval may return incomplete or irrelevant evidence,
- the language model may generate an unsupported answer when the required information is missing.

This project investigates these challenges using the **MultiHop-RAG** dataset.

---

## Dataset

MultiHop-RAG is designed to evaluate retrieval and reasoning across multiple documents.

The dataset contains:

- **2,556 questions**
- **609 news articles**
- evidence that may be distributed across multiple documents
- approximately **12% unanswerable questions**

The unanswerable questions are especially important because they allow us to test whether the system hallucinates an answer or correctly recognizes that there is not enough evidence.

Dataset:
https://huggingface.co/datasets/yixuantt/MultiHopRAG

---

## Goal

The main goal is to build and evaluate a small RAG pipeline that can:

1. retrieve relevant information from the document collection,
2. handle questions that require evidence from multiple documents,
3. generate answers based on retrieved evidence,
4. recognize when the available documents do not contain enough information,
5. avoid unsupported or hallucinated answers.

A simplified RAG pipeline is:

Question  
↓  
Retrieve relevant document chunks  
↓  
Combine and rank evidence  
↓  
Generate an answer from the retrieved context  
↓  
Answer or return "I don't know" when evidence is insufficient

---

# Milestone 1 – Reproduce

The first milestone focuses on reproducing the baseline retrieval experiments from the MultiHop-RAG repository.

The required tasks are:

1. Clone the MultiHop-RAG repository and inspect the README.
2. Install the required `llama-index` version.
3. Download:
   - `corpus.json`
   - `MultiHopRAG.json`
4. Store the files inside the `dataset` directory.
5. Run dense retrieval using:
   - `bge-large-en-v1.5`
   - chunk size = `256`
6. Record the chunk-overlap setting.
7. Evaluate the saved retrieval results.
8. Run BM25 as a keyword-based baseline.

### Target

| Metric | Target |
|---|---:|
| Hits@10 | 0.6718 |
| Hits@4 | 0.5221 |
| MRR@10 | 0.4298 |
| Hits@10 with `bge-reranker-large` | 0.7183 |

### Hints

- If an OpenAI API key is requested during retrieval, a dummy value can be used because retrieval itself does not call OpenAI.
- The repository previously had a chunk-overlap issue, so the overlap value used in the experiment must be recorded explicitly.

---

# Milestone 2 – Improve

The second milestone focuses on improving the baseline system.

Planned improvements include:

- combining BM25 and dense retrieval,
- using hybrid retrieval,
- adding a re-ranker,
- decomposing complex multi-hop questions into simpler sub-questions,
- generating answers using a small open language model in 4-bit precision.

---

# Milestone 3 – Contribute

The third milestone investigates the reliability and practical behavior of the system.

The main questions are:

- Can the system detect unanswerable questions without reducing performance on answerable ones?
- How much speed does each component cost, and how much accuracy does it add?
- Is an automatic LLM-as-judge evaluation reliable compared with manual human evaluation?

---

