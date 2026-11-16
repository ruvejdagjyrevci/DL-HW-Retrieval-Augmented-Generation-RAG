# Milestone 2 – Improve

## Navigation

Use the link below to return to the main project page and access the overall project description and milestone structure. 
[MAIN PAGE](../README.md)

## Team

**Team name:** RAR

**Team members:**

| Name | Neptun Code |
|---|---|
| Ruvejda Gjyrevci | FQL6EH |
| Rahma Nouri | EZDR82 |
| Arlind Bajraktari | D4104N |



## Solution

| Resource | Notebook | Link |
| --- | --- | --- |
| Milestone 2 - Improve | [Notebook](TBA) | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rhFdbKuF2MGeL1bg0Id8Ha4SOH9ZSAYI?usp=sharing) |

## Project Description

Retrieval-Augmented Generation (RAG) lets a language model answer questions using information retrieved from external documents instead of relying only on its internal memory. This can make answers more factual and grounded in evidence, but RAG systems still face important challenges:

- the relevant information may be spread across several documents,
- retrieval may return incomplete or irrelevant evidence,
- the model may produce an unsupported (hallucinated) answer when the required information is missing.

This project studies these challenges on the **MultiHop-RAG** dataset. The goal is to build and evaluate a small RAG pipeline that can:

1. retrieve relevant information from the document collection,
2. handle questions that need evidence from multiple documents,
3. generate answers based on the retrieved evidence,
4. recognize when the documents do not contain enough information,
5. avoid unsupported or hallucinated answers.

```
Question
   ↓
Retrieve relevant document chunks
   ↓
Combine and rank evidence
   ↓
Generate an answer from the retrieved context
   ↓
Answer, or return "I don't know" when evidence is insufficient
```

The work is organized into three milestones.t, prepare the dataset, run the published retrieval pipeline, evaluate the obtained results, and compare them with the expected baseline values.

### Milestone 2 – Improve

Improve the baseline MultiHop-RAG system from Milestone 1 by strengthening retrieval and adding answer generation:

- combine BM25 and dense retrieval using the repository’s hybrid retrieval pipeline,
- add `bge-reranker-large` and measure the retrieval improvement,
- decompose complex multi-hop questions into simpler sub-questions,
- generate answers with a small open language model loaded in **4-bit precision**.

Milestone 2 does not define a fixed numerical target. The goal is to measure whether the added components improve the Milestone 1 baseline in terms of:

- retrieval quality,
- ranking of relevant evidence,
- handling of multi-hop questions,
- and final answer generation.


## Solution


## Output



## Repository Contents

| File / Folder | Purpose |
|---|---|
| `Milestone 2 - Improve/` | Folder containing the Milestone 2 solution. |
| `Milestone 2 - Improve/HW_DL_RAG_Solution.ipynb` | Jupyter Notebook with the full Milestone 2 solution. |
| `Milestone 2 - Improve/hw_dl_rag_solution.py` | Python script version of the notebook. |
| `Milestone 2 - Improve/README.md` | Description of Milestone 2. |

When the notebook runs, it clones the original [MultiHop-RAG repository](https://github.com/yixuantt/MultiHop-RAG) and uses these files from it:

| File / folder | Purpose |
|---|---|
| `simple_retrieval.py` | Chunks the corpus, builds the dense vector index with the chosen embedding model, and retrieves the top-k chunks for every query. |
| `retrieval_evaluate.py` | Official retrieval evaluation script. Our evaluation in T10 uses the same matching rule. |
| `util.py` | Helper code, including the `JSONReader` used to load `corpus.json` into documents. |
| `qa_llama.py`, `qa_evaluate.py` | Question-answering and QA evaluation scripts (used in later milestones). |

The notebook also creates these folders at runtime:

| Folder | Contents |
|---|---|
| `dataset/` | The downloaded `corpus.json` and `MultiHopRAG.json`. |
| `storage/bge-large-en-v1.5_chunk256/` | The persisted dense vector index, so it does not need to be rebuilt every time. |
| `output/` | Retrieval results: `bge-large-en-v1.5_retrieval_test.json` (dense) and `bm25_retrieval_test.json` (BM25). |


## Data Source

The data comes from the **MultiHop-RAG** dataset (Tang & Yang, COLM 2024), hosted on Hugging Face:
https://huggingface.co/datasets/yixuantt/MultiHopRAG

It is licensed under [ODC-BY](https://opendatacommons.org/licenses/by/1-0/).

| File | Contents |
|---|---|
| `corpus.json` | 609 news articles used as the retrieval corpus. |
| `MultiHopRAG.json` | 2,556 questions with their answers, question types and gold evidence. About 12% (301) are unanswerable "null" queries. |

**How it was downloaded:** the notebook (task T5) downloads both files directly from Hugging Face with `wget` into the `dataset/` folder:

```bash
wget -O dataset/corpus.json \
  https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json
wget -O dataset/MultiHopRAG.json \
  https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json
```

Task T6 then loads both files and checks that they contain 609 articles and 2,556 questions.


## Running Instructions

The solution is in the Jupyter Notebook **`HW_DL_RAG_Solution.ipynb`** and is designed for **Google Colab**.

1. Open `HW_DL_RAG_Solution.ipynb` in Google Colab.
2. Select a GPU runtime: **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order from top to bottom (**Runtime → Run all**). The notebook will:
   - clone the MultiHop-RAG repository (T1),
   - install `llama-index==0.9.40` (T3),
   - download the dataset (T4–T6),
   - build the dense index and run retrieval (T7–T9),
   - evaluate dense retrieval (T10),
   - run and evaluate the BM25 baseline (T11).
4. For the BM25 cell (T11), install `rank_bm25` first if it is not available, by uncommenting `!pip install rank_bm25` at the top of the cell.

**Notes:**



## Citation

