# Retrieval quality metrics

Status: metric definitions and implementation requirements. The examples below are synthetic calculations, not model benchmark results. Binary metric functions and fixed-query aggregation are implemented and tested locally. File-based binary evaluation is also implemented; graded TREC conventions remain pending.

## Shared example

Suppose a query has three known relevant passages: A, B, and C. A retriever returns:

| Rank | Passage | Judged relevant? |
| --- | --- | --- |
| 1 | X | No |
| 2 | A | Yes |
| 3 | Y | No |
| 4 | B | Yes |
| 5 | Z | No |

The example uses binary relevance. A, B, and C have relevance 1; all other results receive zero gain for scoring. In a real sparsely judged collection, zero gain can mean unjudged rather than conclusively irrelevant.

## Recall at K: candidate coverage

Recall measures the fraction of known relevant passages found within the first K results. Let R be the set of known relevant passage IDs and L_K the set of IDs in the first K results:

$$
\mathrm{Recall@K}=\frac{|R\cap L_K|}{|R|}.
$$

The example retrieves A and B but misses C:

$$
\mathrm{Recall@5}=\frac{2}{3}\approx0.667.
$$

Recall ignores the relative order of passages within the cutoff. Moving A from rank 2 to rank 1 would not change Recall@5. Increasing K can increase or preserve recall for the same ranking, while increasing retrieval and reranking work.

This metric is useful for candidate generation: a reranker cannot recover a passage absent from its candidate set. High recall alone does not establish good final ordering.

## Reciprocal rank and MRR: first relevant result

Reciprocal rank uses the position r of the first relevant result. With cutoff K:

$$
\mathrm{RR@K}=\begin{cases}1/r,&r\leq K,\\0,&\text{no relevant result within K}.\end{cases}
$$

In the example, the first relevant passage appears at rank 2, so RR@5 is 0.5. A relevant result at rank 1 earns 1; the first relevant result at rank 4 earns 0.25. Relevant results after the first do not improve reciprocal rank.

Mean reciprocal rank averages over a fixed evaluation query set Q:

$$
\mathrm{MRR@10}=\frac{1}{|Q|}\sum_{q\in Q}\mathrm{RR@10}(q).
$$

The example supplies one query's reciprocal rank, not a measured multi-query MRR. Queries returning no results must contribute zero rather than disappear from the average.

## NDCG: ordering relative to an ideal ranking

Discounted cumulative gain rewards relevant results and reduces their contribution at lower ranks. For binary relevance rel_i:

$$
\mathrm{DCG@K}=\sum_{i=1}^{K}\frac{\mathrm{rel}_i}{\log_2(i+1)}.
$$

Only ranks 2 and 4 contribute in the example:

$$
\mathrm{DCG@5}=\frac{1}{\log_2(3)}+\frac{1}{\log_2(5)}\approx1.061606.
$$

The ideal ranking places the three known relevant passages first. Ideal DCG uses all known judgments for the query, not only the relevant passages retrieved by this system:

$$
\mathrm{IDCG@5}=1+\frac{1}{\log_2(3)}+\frac{1}{\log_2(4)}\approx2.130930.
$$

Normalize actual gain by ideal gain:

$$
\mathrm{NDCG@5}=\frac{\mathrm{DCG@5}}{\mathrm{IDCG@5}}\approx0.498189.
$$

Unlike reciprocal rank, NDCG rewards multiple relevant results. Normalization makes an ideal ranking score 1 for each query with positive ideal gain.

For graded judgments, replace binary relevance with an explicitly documented gain function, such as linear gain rel or exponential gain 2^rel - 1. These conventions differ for grades above 1 and must not be silently interchanged. The TREC graded gain and binary relevance threshold will be fixed and checked against the chosen reference evaluator before reporting TREC results.

## What each metric reveals

| Metric | Question | Limitation |
| --- | --- | --- |
| Recall@K | Did candidate generation find the known relevant passages? | Does not reward ordering within K |
| MRR@10 | How early is the first relevant passage? | Ignores additional relevant results |
| NDCG@10 | How close is the ordering to the ideal judged ranking? | Depends on gain convention and judgment coverage |

For the shared example, Recall@5 is approximately 0.667, RR@5 is 0.5, and NDCG@5 is approximately 0.498. These values describe different properties of the same ranking; they should not be interpreted as interchangeable percentages of overall system quality.

## Implementation requirements

- Use unique passage IDs for relevance sets. Identical source judgment rows must not inflate denominators or ideal gain; conflicting labels must fail.
- Reject duplicate retrieved IDs rather than awarding the same passage gain multiple times.
- Use a fixed query list for comparisons. Missing result lists are empty rankings and score zero for queries with positive judgments. Report missing-query counts.
- Require positive integer cutoffs. Evaluate shorter rankings as returned; do not invent missing results.
- Define deterministic score-tie ordering when converting scored results into rankings, and record that policy.
- For the initial supervised evaluation, require at least one known positive per evaluated query. Fail explicitly if that condition is violated, rather than silently dropping queries or changing the denominator. A future protocol with queries lacking positives must define their treatment separately.
- Average per-query scores equally over the fixed query set. Retain per-query outputs for paired comparisons and confidence intervals.
- Test perfect, empty, missed-positive, cutoff-boundary, multiple-positive, duplicate-ID, and hand-calculated rankings. Check graded conventions against an independent reference before using them for claims.

## Sparse judgments and ANN recall

MS MARCO judgments do not identify every potentially useful passage. Standard scoring gives unjudged passages zero relevance credit, but that scoring rule is not evidence that they are irrelevant. Report Recall@K as recall of known judged positives and acknowledge judgment incompleteness.

Relevance recall is also distinct from ANN approximation recall. Relevance recall compares retrieved IDs with human judgments. ANN top-K overlap compares approximate results with exact vector-search results from the same embeddings and corpus. An ANN index can closely reproduce an exact ranking even when the embedding model's relevance quality is poor.

## Code structure

`src/ltr_system/metrics.py` has two public functions:

- `query_metrics(ranking, relevant, k)` returns recall, reciprocal rank, and binary NDCG for one ordered ranking. Passage IDs are nonempty strings; `relevant` must be a set of known positive IDs. It requires at least one positive and rejects duplicate ranked IDs, including duplicates beyond the scoring cutoff.
- `evaluate(query_ids, rankings, positives, recall_cutoffs, rank_cutoff)` computes macro averages and retains per-query scores. Defaults are Recall@20/50/100/1000 and MRR@10/NDCG@10. It reports missing and empty rankings separately; missing rankings are also empty rankings. Extra ranked query IDs fail explicitly to catch split mixing.

The functions receive ordered IDs rather than scores, so score sorting and tie resolution belong to the future result-file adapter. They have no GPU, model, NumPy, or dataset-download dependency. Binary relevant-ID sets do not carry graded labels; these functions must not be presented as graded TREC evaluation. The file adapter described below supplies a CLI and strict rank parsing.

Eight metric tests verify the worked example, perfect and short rankings, no-hit and empty rankings, cutoff boundaries, invalid arguments, fixed-population macro averages, explicit empty versus missing lists, and population errors. All 42 project tests passed locally. This is evaluator correctness evidence on synthetic examples, not measured retrieval quality or independent reference-evaluator parity.

## File-based evaluation

Use `scripts/evaluate_retrieval.py` or the installed `ltr-evaluate` entry point. Required flags are `--queries`, `--qrels`, `--run`, `--output`, and `--run-id`. Relative paths resolve against `--project-root`. Defaults score Recall@20/50/100/1000 and MRR@10/NDCG@10; override using `--recall-cutoffs` and `--rank-cutoff`.

Inputs are UTF-8, headerless TSV files:

| Input | Fields |
| --- | --- |
| Queries | query ID, query text |
| Judgments | query ID, iteration 0, passage ID, binary label 0 or 1 |
| Rankings | query ID, passage ID, one-based rank |

Ranking file order may vary, but each query must have unique contiguous ranks starting at 1. The producer must resolve tied scores deterministically before writing ranks and record that tie policy with its run configuration. This format is a project rank format, not the six-column TREC run format. Duplicate passage IDs and ranked queries outside the explicit evaluation set fail. Absent queries receive empty rankings. Identical judgment duplicates are counted and deduplicated; conflicting or graded labels fail. Queries without known positives fail under the initial evaluation protocol.

Example command for a future validation run (the ranking file must already exist):

```bash
python -I scripts/evaluate_retrieval.py \
  --project-root "$SCRATCH/learning-to-rank" \
  --queries data/processed/msmarco-passage-v1-splits/validation.queries.tsv \
  --qrels data/processed/msmarco-passage-v1-splits/validation.qrels.tsv \
  --run artifacts/runs/baseline.validation.tsv \
  --output artifacts/evaluation/baseline.validation.json \
  --run-id baseline-validation
```

The JSON report contains macro metrics, per-query scores, missing/empty counts, cutoffs, duplicate-judgment count, evaluator version, run ID, and SHA-256/byte identities for each input. Inputs are hashed before and after evaluation to detect changes; concurrent writers are unsupported. The report refuses an existing output path and is written atomically after successful validation. Input hashes identify the supplied files, but do not verify passage membership in a corpus or authenticate a model/index/split manifest. Keep those identities and the code commit in the producing run's provenance before reporting benchmark claims.

Four adapter tests cover report generation and isolated-mode CLI parity, invalid ranks, conflicting/graded qrels, and failure without report publication. All 46 local project tests and Ruff passed. No official retrieval run has been scored and no reference-evaluator parity is claimed.

## Independent reference evaluation

`scripts/crosscheck_evaluation.py` compares saved validation reports with
`pytrec-eval-terrier==0.5.11`. The reference package is installed with
`--target` in `tools/reference-eval/`, which is ignored by Git. Neither the
GPU retrieval environment nor the BM25 environment is modified. Python
isolated mode requires the runner to add that directory explicitly.

The runner independently parses the query, binary judgment and ranking TSV
files rather than using the project's adapters. Negative ranks become
unique reference scores, preserving the saved order without introducing
score ties. Recall uses the full saved ranking. NDCG uses the requested
cutoff; reciprocal rank uses only the top ten because unrestricted
`recip_rank` is not MRR@10. Missing rankings receive zero on the full frozen
query population. Both per-query values and macro aggregates must agree
within absolute tolerance 1e-12. Input checksums are checked against the
saved evaluation and again after the comparison. The validation files also
must match the split manifest. Inputs must remain immutable during execution.

Default runs are BM25 and exact E5. The output is
`artifacts/evaluation/reference-parity-v1.json`, published only after both
comparisons pass. Existing output is not overwritten. Run this command in
a CPU Slurm allocation. The package import and rank-2/rank-11 cutoff sanity
checks passed on HPC; full saved-run reference agreement remains pending.
Agreement validates metric computation for these saved rankings, not model
generalization, retrieval correctness, or statistical significance.

The full saved-run reference comparison subsequently passed on HPC in job
23625746: both 2,000-query runs matched every per-query metric with zero
observed error, and aggregates passed tolerance 1e-12. This supersedes the
pending comparison status above. The report remains a validation result for
these saved binary rankings rather than a general correctness proof.
