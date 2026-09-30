# Document intelligence

Four modules sit on the ingestion path. Each has its own flag. All four default to false. Base RAG does not require them.

| Flag | Module | Method |
| --- | --- | --- |
| `INTEL_ENTITIES` | `intelligence/entities.py` | Regular expressions and a small gazetteer |
| `INTEL_SECTION_CLASSIFIER` | `intelligence/section_classifier.py` | `nn.Linear` on a frozen chunk embedding |
| `INTEL_DOCUMENT_CLASSIFIER` | `intelligence/document_classifier.py` | `nn.Linear` on a frozen document embedding |
| `INTEL_SUMMARIZE` | `intelligence/summarize.py` | The configured `LLMProvider` |

`ingestion/pipeline.py` calls them only when the matching flag is on. A missing `.pt` file raises `ClassifierNotReadyError` (HTTP 409). The API does not invent a label.

## Entities

The patterns are ordered so a more specific span wins. `POST code 0xD4` is a post code, not also a hex code. `AR-GPU-1.6.0` is a product firmware string, and the version inside that span is not extracted a second time. The gazetteer covers BMC, BIOS, IPMI, Redfish, NVMe, NVIDIA, and firmware, matched on word boundaries and stored once per surface.

`ARX-77` is a tracking code in the firmware guide. It is not a version, and the patterns do not match it. The entity gold file labels it as a false negative on purpose. An endpoint pattern stops before a sentence period, so `/redfish/v1/.` is stored as `/redfish/v1/`.

When the flag is on, entity surfaces are stored on the chunk. Hybrid retrieval can add `1.0` to the BM25 score when a surface overlaps the query, then re-sort before fusion. The ablation report compares BM25 with and without that bonus. The change on this corpus is small. That is the result. It is not assumed to help.

## Classifiers

`LinearClassifier` is one matrix and one bias:

```text
logits = x W^T + b
```

Training is Adam and cross-entropy for a fixed number of epochs. The embedding is a constant. It is not in the optimizer. `model.train()` is used while the loss is stepped. `predict` uses `model.eval()` and `torch.inference_mode()`.

Section labels: overview, procedure, reference, error_definition, warning, log_event, other.

Document labels: manual, api_reference, troubleshooting, release_notes, firmware_notes, log.

`tests/test_intelligence.py` fits the head on two well-separated synthetic clouds and checks that it recovers the labels. That test proves the training loop. It is not a score on the manuals. `scripts/train_classifiers.py` refuses to report accuracy until real embeddings exist. In this environment the embedding weights were not available, so that script is `NOT_EXECUTED`.

## Summaries

The summary prompt tells the model to use only the document text and to stop at six sentences. The summary is stored on the document and also appended as a chunk with `section = Summary` and `metadata.kind = summary`. A later experiment can drop those chunks and measure retrieval again. That experiment was not run, because it needs an LLM.

## What was measured

The hand-labeled entity passages were scored. The numbers are in `evaluation/reports/entity_extraction_*.json`. The BM25 entity-boost ablation is in `evaluation/reports/intelligence_ablation_*.json`. Section labels, document labels, and summaries were not part of that ablation.
