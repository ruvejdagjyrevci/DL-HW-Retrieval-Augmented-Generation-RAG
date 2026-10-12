# Milestone 1 – Reproduce

## Navigation

[Back to the project overview](../README.md)

## Objective

The goal of Milestone 1 is to reproduce the baseline retrieval results of the MultiHop-RAG project as closely as possible.

During this milestone, the team is expected to set up the required environment, prepare the dataset, run the published retrieval pipeline, evaluate the obtained results, and compare them with the expected baseline values.

## Tasks

The following tasks are expected to be completed during Milestone 1:

1. Clone the official MultiHop-RAG repository.
2. Read the repository README and follow the provided setup instructions.
3. Install the required dependencies, including the specified `llama-index` version.
4. Download the two required dataset files:
   - `corpus.json`
   - `MultiHopRAG.json`
5. Place both files inside a `dataset` folder within the repository.
6. Run dense retrieval using the `bge-large-en-v1.5` embedding model.
7. Use a chunk size of **256 tokens**.
8. Record the chunk-overlap setting used during retrieval.
9. Save the dense retrieval results.
10. Evaluate the results using the repository's retrieval evaluation script.
11. Report the required retrieval metrics:
    - Hits@10
    - Hits@4
    - MRR@10
12. Run BM25 as a keyword-based retrieval baseline.
13. Compare the dense retrieval results with the BM25 baseline.
14. Document the final configuration and obtained results so that the experiment can be reproduced.

## Target Results

The expected baseline results are:

- Hits@10: **0.6718**
- Hits@4: **0.5221**
- MRR@10: **0.4298**
- Hits@10 with `bge-reranker-large`: **0.7183**

## Notes

- If the retrieval script requests an OpenAI API key, a dummy value can be used because the retrieval stage does not call OpenAI.
- The chunk-overlap value must be recorded explicitly because the repository previously had a chunk-overlap issue.
- The final repository should remain clearly organized and reproducible.
