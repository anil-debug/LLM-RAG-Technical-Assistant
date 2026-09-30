# Prompting, RAG, and fine-tuning

These are three different ways to change what a model says. This project uses the first two in the request path and prepares the third without executing it.

## Prompting

The weights stay frozen. The only change is the text in the request.

`generation/prompt.py` puts instructions in the system message and the question plus retrieved chunks in the user message. History is earlier user and assistant turns. Retrieved chunks are not copied into history, and history is not the retrieval query.

Prompting is the right tool when the model already knows the general skill (how to write a careful technical answer) and the missing piece is the instruction (cite `[n]`, abstain with one exact sentence). It cannot teach the model a firmware version that is absent from both its weights and the prompt.

The abstention sentence is fixed:

```text
The provided documents do not contain enough information to answer this question.
```

Generation metrics treat that sentence as the correct behavior on unanswerable questions. Those questions ask for a default BIOS password, an IPMI cipher suite, a warranty period, a chassis serial, an LDAP server, an SNMP community, and a live fan RPM. The manuals say those values are not in the set.

## RAG

RAG puts the missing facts in the prompt at request time. The path is:

```text
ingestion -> chunking -> embedding -> retrieval -> reranking -> context -> generation -> citation
```

The model is still frozen. The new information lives in the index, so updating a manual does not require a training run. The failure modes move to retrieval: the wrong chunk, a chunk that is too wide to cite, or a model that ignores the evidence and answers from memory.

Citations are how a reader checks the second failure. Every packed chunk has an id. Ids the model invents are rejected in `resolve_citations`. The source lines are built from the chunk metadata, not from text the model wrote after "Sources:".

RAG is the right tool for this assistant because the facts are in documents that change (firmware versions, error codes) and the answers must point at a page and a section.

## Fine-tuning

Fine-tuning changes weights. It is the right tool when the model needs a new behavior that prompting does not reliably produce: a house style, a JSON schema, or a domain phrasing that should appear even when no chunk is retrieved.

It is the wrong tool for "what is BMC firmware 01.73.12" if that string is not in the training set. A fine-tuned model can still hallucinate a version. RAG remains the way to bind the answer to a document.

This repository's plan, in `finetune/train.py`, is LoRA:

```text
W' = W + B A
```

`W` is frozen. `A` and `B` are the adapter. Rank 8, alpha 16, target modules `q_proj` and `v_proj`. QLoRA is the same adapter with the base loaded in 4-bit. It requires the `qlora` extra and CUDA.

The comparison that would justify the adapter is four arms on the golden set, with retrieval held fixed:

| Arm | Weights | Retrieved context |
| --- | --- | --- |
| base | original | no |
| base + RAG | original | yes |
| fine-tuned | base + LoRA | no |
| fine-tuned + RAG | base + LoRA | yes |

That comparison was not executed. No adapter was saved. Training loss, if it were measured later, is not a substitute for those four scores.

## How to talk about them together

Prompting sets the rules. RAG supplies the evidence for this question. Fine-tuning would change the model's habits, and only a held-out comparison can say whether that helped. On the sample corpus the first measurement to quote is BM25 retrieval, because that one actually ran.
