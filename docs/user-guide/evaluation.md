# Evaluation

The golden file has 50 questions: exact identifiers, ordinary technical questions, multi-hop questions, unanswerable questions, and multi-turn questions. Dataset version: `sample-v1`.

## Retrieval metrics

A hit is relevant when its filename is in `expected_files` and its text contains one of `expected_any_substrings`.

Recall@K is fact recall: the fraction of those substrings that appear in the top-k hits from an expected file. A wrong section of a long manual does not count just because the filename matched.

Precision@K uses denominator `k`. Slots that were not returned count as not relevant.

MRR is `1 / rank` of the first relevant hit.

Unanswerable questions are excluded from retrieval averages. They are scored only on the generation side, by abstention.

Multi-turn rows are retrieved with the labeled standalone `question`, not the raw follow-up. Rewrite quality needs an LLM and was not executed.

## Generation proxies

`answer_relevance_proxy` is the fraction of labeled substrings that appear in the answer. For an unanswerable question it is 1 only when the answer contains the abstention sentence.

`faithfulness_proxy` is the fraction of labeled substrings that appear in both the answer and a cited chunk. For an unanswerable question it is 1 only when the model abstains and cites nothing.

`citation_accuracy` is true when every cited filename was expected, at least one citation exists for an answerable question, and no id was rejected. An unanswerable question is correct only with no citations and no rejected ids.

These proxies are not an LLM judge. Judge scores stay null unless a judge is actually run, and a report that used a judge must say so.

## Reports

`evaluation/report.py` writes timestamped JSON. Each file has git revision, timestamp, machine, Python version, model name, configuration, dataset version, methodology, and metrics. Metrics are null when the run did not execute.

| Script | Report prefix | This environment |
| --- | --- | --- |
| `scripts/evaluate.py bm25` | `bm25_retrieval_` | Executed |
| `scripts/evaluate.py chunk-strategies` | `chunk_strategy_` | Executed |
| `scripts/evaluate.py entities` | `entity_extraction_` | Executed |
| `scripts/evaluate.py entity-boost` | `intelligence_ablation_` | Executed for the entity bonus only |
| `scripts/benchmark_models.py` | `embedding_benchmark_`, `llm_benchmark_` | Not executed |
| `scripts/inspect_model.py` | `transformers_random_init_` | Executed on a random tiny GPT-2 |
| `scripts/train_lora.py` | `lora_training_` | Not executed, requires a GPU |
| `scripts/compare_finetune.py` | `finetune_comparison_` | Not executed |
| `scripts/train_classifiers.py` | `classifier_training_` | Not executed |

Do not copy a number from memory. Open the JSON file and read `status` before quoting it.

## How to run the embedding comparison later

```bash
uv run python scripts/benchmark_models.py --download
```

Both `BAAI/bge-base-en-v1.5` and `intfloat/e5-base-v2` must load. Then encode the same chunks, search the golden questions, and record Recall@5, Recall@10, MRR, indexing time, query latency, and memory. Memory for a PyTorch embedder is process RSS plus CUDA peak if a GPU is used. Do not mix that number with an Ollama RSS.

## How to run the LLM comparison later

Start Ollama or vLLM. Retrieve once per question and reuse that context for every model. Record answer relevance, faithfulness, citation accuracy, and client latency. If the server is local and its PID is visible, record process RSS and say so. If the server is remote, record client latency only.

## How to run fine-tuning later

```bash
uv run python scripts/train_lora.py --model <causal-lm-id> --output models/lora
# optional 4-bit base:
uv sync --extra qlora
uv run python scripts/train_lora.py --model <causal-lm-id> --qlora --output models/lora
uv run python scripts/compare_finetune.py
```

The comparison script still needs a generator wired to the four arms. It will not invent the four scores.
