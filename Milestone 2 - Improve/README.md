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
| Milestone 2 - Improve | [Notebook](HW_DL_RAG_Solution.ipynb) | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rhFdbKuF2MGeL1bg0Id8Ha4SOH9ZSAYI?usp=sharing) |

The notebook is saved **with all outputs visible**, so every number in this README can be checked without rerunning the code.

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

The work is organized into three milestones.

### Milestone 1 – Reproduce

Clone the MultiHop-RAG repository, prepare the dataset, run the published retrieval pipeline, evaluate the obtained results, and compare them with the expected baseline values.

### Milestone 2 – Improve

Improve the baseline MultiHop-RAG system from Milestone 1 by strengthening retrieval and adding answer generation:

- combine BM25 and dense retrieval using the repository's hybrid retrieval pipeline,
- add `bge-reranker-large` and measure the retrieval improvement,
- decompose complex multi-hop questions into simpler sub-questions,
- generate answers with a small open language model loaded in **4-bit precision**.

Milestone 2 does not define a fixed numerical target. The goal is to measure whether the added components improve the Milestone 1 baseline in terms of retrieval quality, ranking of relevant evidence, handling of multi-hop questions, and final answer generation.



## What was changed and why?

**Baseline (Milestone 1):** dense retrieval with `BAAI/bge-large-en-v1.5`, chunk size 256 tokens, overlap 25, top-10 chunks (`simple_retrieval.py`). The Milestone 1 BM25 run (`bm25_retrieval.py`) is used as a second reference.

| Task | Change | Why was it expected to help? |
|---|---|---|
| **T12** | **Hybrid retrieval:** fuse the dense and BM25 top-10 lists with Reciprocal Rank Fusion (RRF, k = 60, equal weights; `rrf_fusion.py`). | BM25 matches exact words (news source names, people, companies, dates), and these appear in almost every MultiHop-RAG question. Dense retrieval matches meaning. The two retrievers make different mistakes, so fusing them should find more of the gold evidence. |
| **T13** | **Reranking:** build a larger hybrid candidate pool (dense top-20 + BM25 top-20 → RRF top-20), then re-order it with the cross-encoder `BAAI/bge-reranker-large` and keep the top-10. | Retrievers score the question and the chunk separately. A cross-encoder reads the question and the chunk *together*, so it can judge relevance more precisely. This should mainly move relevant chunks higher in the ranking (better Hits@4 and MRR). |
| **T14** | **Question decomposition:** `Qwen2.5-3B-Instruct` (4-bit) splits every question into 1–3 sub-questions. Each sub-question is retrieved separately (dense + BM25), and all lists are fused with RRF into a pool of the same size (20), then reranked as in T13. | A multi-hop question needs evidence from 2–4 different articles. One query often finds one article and misses the others. Each sub-question is about one article, so it should find its own evidence. |
| **T15** | **Answer generation:** `Qwen2.5-3B-Instruct` in 4-bit NF4 (`bitsandbytes`) answers from the top-*k* retrieved chunks, or replies *"Insufficient information."* when the evidence is not enough. | This turns the retrieval system into a full RAG pipeline. Better retrieval should give better answers, and the explicit "I don't know" option should reduce hallucinated answers to unanswerable (null) questions. |

## How each change was tested?
- **Metrics:**
  - Retrieval: **Hits@10**, **Hits@4** and **MRR@10**, computed with the repository's own `retrieval_evaluate.py` / `calculate_metrics` on the answerable questions (null questions have no gold evidence).
  - Answers: **accuracy** (the gold answer appears in the model's first line, after lower-casing and removing punctuation), plus two hallucination metrics: **Null abstained** (null questions where the model correctly says *Insufficient information*, higher is better) and **Wrongly abstained** (answerable questions where the model gives up, lower is better).
- **Data split:** In T14 we split the 2,556 questions **once**, stratified by question type, with a fixed seed (42), and saved the split to `dataset/splits.json`:

  | Split | Questions | Null questions | Answerable | Used for |
  |---|---|---|---|---|
  | train | 510 | 60 | 450 | not used in Milestone 2 (kept for Milestone 3) |
  | validation | 510 | 60 | 450 | comparing settings and **choosing** the best one |
  | test | 1,536 | 181 | 1,355 | **final evaluation only**, run once after the choice |

- **Selection rules, fixed before looking at the results:**
  - T14: among the reranked settings, choose the one with the highest validation **Hits@4** (tie-breaker MRR@10), because T15 gives the LLM the top-4 chunks.
  - T15: among the RAG configurations, choose the one with the highest validation **accuracy on all questions** (null questions included).
- **One factor at a time:** In T12 and T13 only the retrieval method changes (same chunks, same chunk size, same top-10 output). In T13 we also compare against a **fair pre-reranking baseline**: the same hybrid top-20 pool cut to its top-10 without reranking, so the only difference is the reranker. In T14 only *how the queries are written* changes: same chunks, retrievers, RRF, pool size (20) and reranker. In T15 one factor changes per row: no evidence → Milestone 1 evidence → best T14 evidence → more chunks.
- **Comparable budget:** No model is trained or fine-tuned in Milestone 2. All models are used off the shelf, and every comparison uses the same models, the same prompt and greedy decoding (`do_sample=False`), so the outputs are deterministic.
- **Note on T12/T13:** T12 and T13 were run with the repository scripts on **all 2,255 answerable questions**, before we introduced the split in T14. To keep the selection fair, T14 re-implements these retrievers over the same 8,034 chunks and evaluates all settings (R1–R6) again on the **validation** split. Sanity checks (all 2,255 answerable questions) compare the re-implementation with the repository scripts:

  | Retriever | Repository script | Our re-implementation |
  |---|---|---|
  | Dense, Hits@10 / Hits@4 / MRR@10 | 0.7996 / 0.6058 / 0.4372 | 0.7991 / 0.6027 / 0.4371 |
  | BM25, Hits@10 / Hits@4 / MRR@10 | 0.8683 / 0.7126 / 0.5747 | 0.8962 / 0.7424 / 0.6059 |
  | Hybrid + reranker (T13), Hits@10 / Hits@4 / MRR@10 | 0.9020 / 0.7778 / 0.6751 | 0.9091 / 0.7849 / 0.6819 |

  Dense retrieval is reproduced almost exactly. Our BM25 is about 3 points stronger than the repository's BM25 script, probably because it indexes each chunk together with its title/source/date header and uses a different tokenizer. This makes the T14 baselines (R2–R4) *stronger*, not weaker, so the comparison with decomposition stays fair. The final choice is made on validation and reported on test.

## Results

### T12–T13: Retrieval (repository scripts, all 2,255 answerable questions)

| # | Setting | What changed | Hits@10 | Hits@4 | MRR@10 |
|---|---|---|---|---|---|
| B | **Baseline:** dense `bge-large-en-v1.5`, top-10 (Milestone 1) | – | 0.7996 | 0.6058 | 0.4372 |
| B2 | BM25 top-10 (Milestone 1) | keyword retrieval instead of dense | 0.8683 | **0.7126** | **0.5747** |
| T12 | Hybrid RRF (dense + BM25), top-10 | fuse B and B2 with RRF | **0.8718** | 0.7064 | 0.5618 |
| T13a | Hybrid RRF from top-20 lists, cut to top-10 (fair pre-rerank baseline) | larger input lists (20 + 20), no reranker | 0.8647 | 0.7162 | 0.5653 |
| T13 | T13a pool + `bge-reranker-large`, top-10 | add the reranker | **0.9020** | **0.7778** | **0.6751** |

Paper target for dense retrieval (from Milestone 1): Hits@10 0.6718, Hits@4 0.5221, MRR@10 0.4298.

### T14: Question decomposition – **validation** (450 answerable questions)

| ID | Retrieval setting | What changed | Hits@10 | Hits@4 | MRR@10 |
|---|---|---|---|---|---|
| R1 | **Baseline:** dense, original question | – | 0.8156 | 0.6244 | 0.4565 |
| R2 | BM25, original question | retriever | 0.8956 | 0.7378 | 0.6210 |
| R3 | Hybrid RRF top-10, original question | fuse R1 + R2 | 0.8844 | 0.7311 | 0.5861 |
| R4 | Hybrid pool-20 + reranker, original question (T14 baseline) | add reranker to R3 | 0.9156 | 0.7978 | 0.7029 |
| R5 | Hybrid RRF top-10, decomposed | sub-questions added to R3 | 0.9200 | 0.7000 | 0.5083 |
| R6 | Hybrid pool-20 + reranker, decomposed | sub-questions added to R4 | **0.9311** | **0.8111** | **0.7157** |

**Selected on validation: R6** (highest Hits@4).

Per question type (validation), original question (R4) vs. decomposed (R6):

| Question type | n | Hits@10 original | Hits@10 decomposed | MRR@10 original | MRR@10 decomposed |
|---|---|---|---|---|---|
| inference | 163 | **0.9509** | 0.9448 | 0.7889 | **0.8001** |
| comparison | 171 | 0.9064 | **0.9357** | 0.7155 | **0.7278** |
| temporal | 116 | 0.8793 | **0.9052** | 0.5635 | **0.5794** |

Decomposition statistics: 5,939 sub-questions for 2,556 questions (2.32 on average; 1 sub-question: 30, 2: 1,669, 3: 857). Only 4 questions (0.2%) were left unchanged. On average, the original and the decomposed 20-chunk pools share 75.2% of their chunks.

### T14: Final evaluation – **test** (1,355 answerable questions, run once)

| ID | Retrieval setting | Hits@10 | Hits@4 | MRR@10 |
|---|---|---|---|---|
| R1 | **Baseline:** dense, original question (Milestone 1) | 0.8074 | 0.6111 | 0.4377 |
| R4 | Hybrid pool-20 + reranker, original question | **0.9122** | **0.7860** | **0.6788** |
| R6 | Hybrid pool-20 + reranker, decomposed (**selected**) | 0.9041 | 0.7734 | 0.6696 |

### T15: Answer generation – **validation** (510 questions, incl. 60 null)

Model: `Qwen2.5-3B-Instruct`, 4-bit NF4, greedy decoding.

| ID | Evidence | k | What changed | Acc (all) | Acc (answerable) | Inference | Comparison | Temporal | Null abstained ↑ | Wrongly abstained ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| G0 | none (closed-book) | 0 | reference without retrieval | 0.2294 | 0.2467 | 0.3129 | 0.2865 | 0.0948 | 0.1000 | 0.4800 |
| G1 | **Baseline:** Milestone 1 dense | 4 | add retrieved evidence | 0.4510 | 0.4200 | 0.7730 | 0.2573 | 0.1638 | 0.6833 | 0.4422 |
| G2 | best T14 retrieval (R6) | 4 | better retrieval | 0.5039 | 0.4778 | **0.8589** | 0.3158 | 0.1810 | **0.7000** | 0.3533 |
| G3 | best T14 retrieval (R6) | 8 | more chunks | **0.5078** | **0.5044** | 0.8344 | **0.3567** | **0.2586** | 0.5333 | **0.3267** |

**Selected on validation: G3** (highest accuracy on all questions).

### T15: Final evaluation – **test** (1,536 questions, incl. 181 null, run once)

| ID | Evidence | k | Acc (all) | Acc (answerable) | Inference | Comparison | Temporal | Null abstained ↑ | Wrongly abstained ↓ |
|---|---|---|---|---|---|---|---|---|---|
| G0 | none (closed-book) | 0 | 0.2344 | 0.2487 | 0.4143 | 0.2140 | 0.0684 | 0.1271 | 0.5026 |
| G1 | **Baseline:** Milestone 1 dense | 4 | 0.4408 | 0.4059 | 0.7306 | 0.2354 | 0.2023 | **0.7017** | 0.4266 |
| G3 | best T14 retrieval (R6), **selected** | 8 | **0.5026** | **0.4937** | **0.8265** | **0.3327** | **0.2650** | 0.5691 | **0.3358** |

### Compute cost (one run on an L4 GPU)

| Stage | Time |
|---|---|
| Build our dense + BM25 retrievers (T14.2) | 70 s |
| Dense / BM25 retrieval, 2,556 original questions | 4 s / 1 s |
| Question decomposition with the 4-bit LLM, 2,556 questions (T14.4) | 1,157 s (≈0.45 s/question) |
| Dense + BM25 retrieval, 5,939 sub-questions | 7 s |
| Reranking one pool, 51,120 pairs (fp16) | ≈400 s |
| Answer generation, k = 4 / k = 8 | ≈0.30 / ≈0.54 s per question |

### Summary: improvement over the Milestone 1 baseline (test set)

| Stage | Milestone 1 baseline | Best Milestone 2 system | Change |
|---|---|---|---|
| Retrieval Hits@4 | 0.6111 (R1) | 0.7860 (R4) / 0.7734 (R6, selected) | +17.5 / +16.2 points |
| Retrieval MRR@10 | 0.4377 (R1) | 0.6788 (R4) / 0.6696 (R6, selected) | +24.1 / +23.2 points |
| Answer accuracy (all questions) | 0.4408 (G1) | 0.5026 (G3) | +6.2 points |
| Null questions abstained | 0.7017 (G1) | 0.5691 (G3) | −13.3 points (worse) |

## Discussion: learning outcomes

**What helped?**

- **The reranker was the single most useful change.** On the same candidate pool, it improved Hits@10 by +3.7, Hits@4 by +6.2 and MRR@10 by +11.0 points (T13a → T13). The gain is largest at the top of the ranking. This matches what a cross-encoder does: it rarely finds new chunks, but it puts the right ones first.
- **Retrieval is essential for answering.** Without evidence, the 3B model answers only 23% of the test questions correctly (G0). Most articles are from late 2023, so the model does not know them. With the Milestone 1 evidence, accuracy almost doubles (G1, 44%).
- **Better retrieval gave better answers.** With the T14 evidence, test accuracy rose from 0.4408 to 0.5026 (+6.2 points, +8.8 on answerable questions). The gain is largest on inference questions (0.73 → 0.83).

**What did not help, or helped only partly?**

- **BM25 beat dense retrieval clearly** (Hits@4 0.7126 vs 0.6058). MultiHop-RAG questions usually name the news source, the people and the date, and each chunk keeps its title/source/date header. So exact keyword matching is very strong here, probably stronger than in typical semantic search benchmarks.
- **Hybrid RRF (T12) only matched BM25.** With the repository scripts, it found slightly more gold evidence (Hits@10 +0.35 points) but ranked it slightly worse (Hits@4 −0.6, MRR −1.3 points). On the validation split the picture is the same: BM25 (R2) is better than hybrid RRF (R3) on all three metrics (MRR@10 0.6210 vs 0.5861). With equal weights, RRF mixes chunks from the clearly weaker dense list into BM25's top positions. Fusing larger input lists (top-20 instead of top-10, T13a) also did not help by itself. Hybrid retrieval became useful only as the **candidate pool for the reranker**.
- **Question decomposition (T14) did not give a reliable gain.**
  - *Without the reranker* (R3 → R5), decomposition found more evidence (Hits@10 +3.6 points) but ranked it much worse (Hits@4 −3.1, MRR −7.8 points). Mixing the lists of 2–3 sub-questions pushes the best chunk down.
  - *With the reranker* (R4 → R6), it gave a small gain on validation (+1.6 Hits@10, +1.3 Hits@4). Comparison (+2.9 Hits@10) and temporal (+2.6) questions gained, because they name two different articles. Inference questions lost a little in Hits@10 (−0.6), because one article is often enough for them. The two pools still share 75% of their chunks, so decomposition changes only a quarter of the evidence.
  - *On the test set the gain disappeared* (R6 is 0.8 points below R4 in Hits@10). The validation gain of 6–7 questions out of 450 was within noise. This result is the reason we select on validation and use the test set only once: on the test set, the simpler R4 would have been the better choice.
  - *Cost:* decomposition adds one LLM call per question (≈0.45 s per question on an L4 GPU) plus a second reranking run, without a reliable benefit on this dataset.
- **More context (k = 8) trades hallucination against coverage.** G3 answers more answerable questions correctly than G2, but it abstains less often on null questions (validation 0.70 → 0.53). More evidence makes the small model more confident, including when it should say "I don't know". G3 beat G2 by only 2 of 510 validation questions, so the two are effectively tied. The extra chunks mainly help the two-article question types (validation temporal 0.18 → 0.26, comparison 0.32 → 0.36), while inference questions get slightly worse (0.86 → 0.83).

**What explains the remaining errors?**

- Temporal (0.27) and comparison (0.33) questions remain hard. They need two facts *and* a comparison of dates or claims, which is difficult for a 3B model.
- In the validation error analysis (G3), the gold evidence **was** in the model's context in 89% of the false abstentions and 96% of the wrong answers. Most remaining errors therefore come from **generation**, not from retrieval.
- 28 of the 60 null validation questions were answered instead of abstained. Decomposition does not help here either: the LLM splits null questions into confident-looking sub-questions.

**Next step (Milestone 3):** improve how the system decides to answer *"Insufficient information."*, for example with a threshold on the reranker scores tuned on the train split, and compare the string-match metric with manual checks and an LLM judge.

---

## Repository Contents

| File / Folder | Purpose |
|---|---|
| `Milestone 2 - Improve/` | Folder containing the Milestone 2 solution. |
| `Milestone 2 - Improve/HW_DL_RAG_Solution.ipynb` | Jupyter Notebook with the full Milestone 2 solution (T1–T15), saved with outputs. |
| `Milestone 2 - Improve/hw_dl_rag_solution.py` | Python script version of the notebook. |
| `Milestone 2 - Improve/README.md` | This description of Milestone 2. |

When the notebook runs, it clones the original [MultiHop-RAG repository](https://github.com/yixuantt/MultiHop-RAG) and uses these files from it:

| File / folder | Purpose |
|---|---|
| `simple_retrieval.py` | Chunks the corpus, builds the dense vector index with the chosen embedding model, and retrieves the top-k chunks for every query. |
| `bm25_retrieval.py` | BM25 keyword retrieval (T11, T13). |
| `rrf_fusion.py` | Reciprocal Rank Fusion of the dense and BM25 result files (T12, T13). |
| `retrieval_evaluate.py` | Official retrieval evaluation script. All our retrieval metrics use its matching rule (`calculate_metrics`). |
| `util.py` | Helper code, including the `JSONReader` used to load `corpus.json` into documents. |

The notebook creates these folders at runtime (stored on Google Drive, so nothing is lost if Colab disconnects):

| Folder / file | Contents |
|---|---|
| `dataset/` | `corpus.json`, `MultiHopRAG.json` and **`splits.json`** (the fixed train/validation/test split). |
| `storage/bge-large-en-v1.5_chunk256/` | The persisted dense vector index from Milestone 1 (8,034 chunks). |
| `output/` | All retrieval result files (dense, BM25, hybrid, reranked, T14 variants), the cached sub-questions (`t14_subquestions_raw.json`), the cached LLM answers (`t15_*.json`), `t14_results.json`, `t15_results.json` and the final test predictions `t15_final_predictions_test.json` (input for Milestone 3). |

## Data Source

The data comes from the **MultiHop-RAG** dataset (Tang & Yang, COLM 2024), hosted on Hugging Face:
https://huggingface.co/datasets/yixuantt/MultiHopRAG

It is licensed under [ODC-BY](https://opendatacommons.org/licenses/by/1-0/).

| File | Contents |
|---|---|
| `corpus.json` | 609 news articles used as the retrieval corpus. |
| `MultiHopRAG.json` | 2,556 questions with their answers, question types and gold evidence. About 12% (301) are unanswerable "null" queries. |

The notebook (task T5) downloads both files directly from Hugging Face with `wget` into the `dataset/` folder:

```bash
wget -O dataset/corpus.json \
  https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json
wget -O dataset/MultiHopRAG.json \
  https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json
```

Models (downloaded automatically from Hugging Face): [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5), [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large), [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).

## Running Instructions

The solution is in the Jupyter Notebook **`HW_DL_RAG_Solution.ipynb`** and is designed for **Google Colab** with a GPU.

1. Open `HW_DL_RAG_Solution.ipynb` in Google Colab (badge above).
2. Select a GPU runtime: **Runtime → Change runtime type → T4 GPU** or **L4 GPU**. The saved outputs and timings come from a run on an L4.
3. Run the cells in order from top to bottom (**Runtime → Run all**). The notebook:
   - clones the MultiHop-RAG repository (T1) and links `dataset/`, `output/` and `storage/` to Google Drive (T1.1),
   - installs `llama-index==0.9.40` (T3), downloads and checks the dataset (T4–T6),
   - builds the dense index, runs dense and BM25 retrieval and evaluates them (T7–T11, Milestone 1),
   - runs hybrid RRF retrieval (T12) and reranking with `bge-reranker-large` (T13, installs `FlagEmbedding`),
   - creates the train/validation/test split, decomposes the questions with the 4-bit LLM and compares all retrieval settings (T14, installs `bitsandbytes` and `accelerate`),
   - generates and scores answers with the 4-bit LLM (T15).

**Notes:**

- **Saved results are re-used.** Every expensive step checks whether its result file already exists in `output/` (on Drive) and skips the computation if it does. The LLM steps save after every batch, so an interrupted run continues where it stopped.
- **Approximate run times:** dense retrieval (T8) about 25 min, reranking with the repository code (T13) about 2 h, question decomposition (T14.4) about 20 min on an L4 (20–40 min on a T4), and the T15 test generation about 24 min on an L4 (2–3× longer on a T4).
- **After a runtime restart,** run T1, T1.1, T3 and the `FlagEmbedding` install cell, then T14.1 and T14.3 before any later T14/T15 cell.
- **If you change a prompt or an evidence file in T15,** delete the matching `output/t15_*.json` cache files, otherwise the old answers are loaded.
- **The split is fixed.** It is saved in `dataset/splits.json`. If the file is missing, T14.1 creates it again from the same seed (42), which gives the same split.
- A T4 GPU is enough for everything: the 4-bit LLM needs about 2–3 GB of GPU memory. If Colab reports "cannot connect to a GPU due to usage limits", the free GPU quota is used up; wait for it to reset or use Colab compute units.

## Citation

```bibtex
@misc{tang2024multihoprag,
      title={MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries},
      author={Yixuan Tang and Yi Yang},
      year={2024},
      eprint={2401.15391},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```