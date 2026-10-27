# Milestone 1 – Reproduce

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
| Milestone 1 - Reproduce | [Notebook](Milestone1_Reproduce.ipynb) | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rhFdbKuF2MGeL1bg0Id8Ha4SOH9ZSAYI?usp=sharing) |

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

### Milestone 1 – Reproduce

Reproduce the baseline retrieval results of the original MultiHop-RAG repository:

- dense retrieval with `BAAI/bge-large-en-v1.5`, chunk size **256** tokens, chunk overlap **25** tokens, top-k = 10,
- evaluation with Hits@10, Hits@4 and MRR@10,
- BM25 as a keyword-based baseline on exactly the same chunks.

| Metric | Target (paper) |
|---|---:|
| Hits@10 | 0.6718 |
| Hits@4 | 0.5221 |
| MRR@10 | 0.4298 |
| Hits@10 with `bge-reranker-large` | 0.7183 |


## Solution

The retrieval stage of the MultiHop-RAG paper is reproduced, and a semantic retriever is compared with a keyword-based retriever on identical data.

**Chunking.** The 609 news articles are split into **8,034 chunks** of 256 tokens with a 25-token overlap. Through the overlap, some text is shared between neighbouring chunks, so a fact spanning a chunk boundary is not lost. Each chunk is also annotated with the title, source and publication date of the original article, since this metadata is referenced in many MultiHop-RAG questions (e.g. "according to TechCrunch…").

**Dense retrieval.** Every chunk is embedded with `BAAI/bge-large-en-v1.5`, and the resulting vectors are stored in an index. Each question is embedded in the same way, and the 10 chunks with the most similar vectors are returned. Because *meaning* is captured by the embeddings, relevant passages can be found even when the wording differs from the question.

**BM25 baseline.** For comparison, the *same* 8,034 chunks are ranked with Okapi BM25, a classic keyword-based method. Under BM25, chunks are scored by the number of matching question words and the rarity of those words. Since identical chunks are searched by both retrievers, any difference in scores can be attributed to the retrieval method alone.

**Evaluation.** Retrieval quality is measured by how often the evidence needed to answer each question is found:
- **Hits@10 / Hits@4:** the share of questions with at least one gold evidence passage among the top 10 or top 4 retrieved chunks.
- **MRR@10:** the average reciprocal rank of the first correct chunk (1 for rank 1, ½ for rank 2, …, 0 if no correct chunk is found in the top 10).

A chunk is counted as correct if a gold evidence fact is contained in the chunk, following the same rule as the official evaluation script. The 301 unanswerable questions have no gold evidence and are therefore excluded, so the metrics are computed on the remaining 2,255 questions.

## Output

| Retriever | Hits@10 | Hits@4 | MRR@10 |
|---|---:|---:|---:|
| Paper target (dense) | 0.6718 | 0.5221 | 0.4298 |
| Ours – dense `bge-large-en-v1.5` | 0.7996 | 0.6058 | 0.4372 |
| Ours – BM25 | **0.8683** | **0.7126** | **0.5747** |

All three paper targets are exceeded by the reproduced dense retriever (+0.1278 Hits@10, +0.0837 Hits@4, +0.0074 MRR@10). The BM25 baseline scores higher than dense retrieval on every metric. The largest gain for BM25 is in MRR@10, so correct evidence is ranked higher by keyword matching on this dataset.

## Repository Contents

| File / Folder | Purpose |
|---|---|
| `Milestone 1 - Reproduce/` | Folder containing the Milestone 1 solution. |
| `Milestone 1 - Reproduce/HW_DL_RAG_Solution.ipynb` | Jupyter Notebook with the full Milestone 1 solution. |
| `Milestone 1 - Reproduce/README.md` | Description of Milestone 1. |

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

- Dense retrieval (T8) is the slowest step. On a T4 GPU, embedding the 8,034 chunks and retrieving for all 2,556 queries takes roughly 30 minutes.
- If an OpenAI API key is requested during retrieval, any dummy value works, because retrieval does not call OpenAI.
- The chunk overlap is set explicitly to **25 tokens** because the original repository had a chunk-overlap issue.
- Colab storage is temporary. Download the `output/` folder if you want to keep the retrieval results.


## Citation

```
@misc{tang2024multihoprag,
      title={MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries},
      author={Yixuan Tang and Yi Yang},
      year={2024},
      eprint={2401.15391},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```