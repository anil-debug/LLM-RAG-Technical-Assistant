# Transformers, from tokens to the next token

This note is the map for `generation/transformers_local.py` and `scripts/inspect_model.py`.

## Tokenization

A tokenizer splits text into subword ids. GPT-2 uses byte-pair encoding. A technical token such as `SYS_FAN1` may be one piece or several, depending on the vocabulary. The regex tokenizer in `core/text.py` is a different tool: it is for chunking and BM25, and it keeps `0x1F` and `01.73.12` whole. It is not the model vocabulary.

The inspect script's random GPT-2 does not load a trained tokenizer. Its `input_ids` are hand-written integers so the forward pass can run offline. A pretrained run (`--model`) prints the real tokens.

## The residual stream

Each token id is looked up in an embedding matrix of shape `(vocab, d_model)`. A positional vector of the same width is added. From that point on, the token is a vector that each block updates.

One block does two things, each wrapped in a residual connection:

```text
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

The residual add is why a block can pass information forward even if the attention output is small. Layer normalization keeps the scale of `x` stable across depth. (Some architectures normalize after the residual instead of before it. The equation above is pre-norm, which is what current GPT-style blocks use.)

## Attention

For each token the block builds a query, a key, and a value with learned matrices:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

`X` has shape `(sequence, d_model)`. With `h` heads, each head uses `d_k = d_model / h`.

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

`Q K^T` is a `(sequence, sequence)` matrix of compatibility scores. Dividing by `sqrt(d_k)` keeps the dot products from growing with the head width and saturating the softmax. The softmax weights sum to 1 along the key axis. Multiplying by `V` returns a mixture of value vectors.

Multi-head attention runs `h` of these maps and concatenates them, then applies an output matrix `W_O`. Different heads can track different relations (a nearby identifier versus a section heading).

A causal language model masks scores so position `t` cannot attend to positions after `t`. The mask is applied before the softmax, usually by adding a large negative number to the forbidden entries. A bidirectional encoder such as BERT does not use that mask. BGE and E5 are encoder bi-encoders: they read the whole chunk at once and pool it into one vector. The chat model is a decoder: it predicts the next token.

## Cross-encoder versus bi-encoder

| | Bi-encoder | Cross-encoder |
| --- | --- | --- |
| Input | Query and chunk encoded separately | Query and chunk in one sequence |
| Cost | One vector per chunk, reused at query time | One forward pass per pair |
| Used here | `BAAI/bge-base-en-v1.5` at index and query time | `BAAI/bge-reranker-base` on the top 30 |

The bi-encoder's score is cosine similarity of two pooled vectors. The cross-encoder's score is a classification head on the joint sequence. That is why reranking can reorder a short list more carefully than cosine can.

## Feed-forward network

After attention, each position is passed through a small MLP, independently:

```text
FFN(x) = activation(x W_1) W_2
```

Attention mixes tokens. The feed-forward layer mixes features inside one token. Both are needed.

## From logits to a token

The final hidden state is projected to the vocabulary:

```text
logits = h_last W_vocab
```

`logits` has shape `(batch, sequence, vocab)`. The last position is the distribution for the next token. Greedy decoding takes `argmax`. Sampling uses temperature. `TransformersLocalProvider` sets `do_sample` only when temperature is greater than 0, and it passes `temperature=None` for greedy decoding because a temperature of 0 is not a valid sampler argument.

## Prefill, decode, and the KV cache

The first forward pass consumes the whole prompt. That is prefill. Each new token is then one short forward pass. Keys and values from previous positions are stored in the KV cache so the model does not recompute them. `model.generate` owns that cache. Latency is mostly prefill for a long prompt plus one step per new token. Throughput improves when several sequences share a batch. A long retrieved context increases prefill cost, which is why the context builder has a token budget.

## eval mode and inference mode

`model.eval()` turns off dropout and uses the running statistics for batch norm. Without it, the same prompt can take a different path on every call.

`torch.inference_mode()` disables autograd. The forward pass does not store activations for a backward pass, which cuts memory and adds speed. Training must not use it, because LoRA needs gradients on the adapter weights. Serving must use it.

## Memory

A parameter in float32 is 4 bytes. Float16 or bfloat16 is 2 bytes. Activations and the KV cache sit on top of the weights. The cache grows with batch size, layers, heads, and sequence length. That is why a 7B model can fit its weights in 16-bit on a 16 GB GPU and still run out of memory once the prompt and the cache are large.

Quantization stores weights in 8-bit or 4-bit. QLoRA uses 4-bit storage for the frozen base (NF4) and trains LoRA adapters in higher precision. The optional `qlora` extra installs bitsandbytes. It is not installed in the default environment, and training is not started without CUDA.

`scripts/inspect_model.py` records process RSS from `/proc/self/status` and CUDA peak memory only when CUDA is available. On this machine the random-init run is CPU, so `cuda_peak_bytes` is null. That null is the measurement.

## Encoder versus decoder in this repository

- The embedding model is an encoder. It maps a chunk to one 768-d vector.
- The reranker is an encoder cross-encoder. It maps a pair to one score.
- The chat model is a decoder. It maps the prompt to the next token, repeatedly.

Do not describe BGE as "generating the answer." It only retrieves.
