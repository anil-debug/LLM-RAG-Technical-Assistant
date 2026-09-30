"""Show a real Transformers forward pass.

The default run builds a tiny GPT-2 from a config. The weights are random.
Nothing is downloaded, and the decoded tokens are not a quality result.
Pass --model to load a pretrained checkpoint. If that load fails, the report
says NOT_EXECUTED and the random-init demonstration is still written.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.report import git_revision, machine_info, write_report
from evaluation.runners import process_rss_bytes


EXPLANATION = """
tokenization
  -> token embeddings + positional embeddings
  -> transformer blocks
       attention: softmax(Q K^T / sqrt(d_k)) V
       residual + layer norm
       feed-forward network
  -> logits over the vocabulary
  -> one new token
  -> append that token and repeat (KV cache lives inside generate)

model.eval() disables dropout so the same input does not take random layers.
torch.inference_mode() disables autograd, so the forward pass does not keep
activations for a backward pass. Serving uses both. Training does not.
"""


def inspect_random() -> dict:
    """Run a tiny randomly initialized GPT-2 and record shapes."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=128,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=64,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = GPT2LMHeadModel(config)
    # Scaled-dot-product attention does not return attention weights.
    # Eager attention does, which is what this inspection needs.
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    # These ids stand in for tokenizer output. A pretrained tokenizer is not loaded.
    input_ids = torch.tensor([[1, 10, 11, 12, 13]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        forward = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=config.eos_token_id,
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    attentions = forward.attentions or ()
    attention_shape = list(attentions[0].shape) if attentions else None
    return {
        "status": "EXECUTED",
        "reason": None,
        "model_name": "random-gpt2-config",
        "pretrained": False,
        "configuration": {
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "n_positions": config.n_positions,
        },
        "metrics": {
            "parameter_count": parameter_count,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "input_ids": input_ids[0].tolist(),
            "attention_mask_shape": list(attention_mask.shape),
            "logits_shape": list(forward.logits.shape),
            "attention_layers": len(attentions),
            "attention_shape": attention_shape,
            "generated_ids": generated[0].tolist(),
            "new_token_count": int(generated.shape[1] - input_ids.shape[1]),
            "rss_bytes": process_rss_bytes(),
            "cuda_peak_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
        },
        "methodology": (
            "GPT2LMHeadModel(GPT2Config(...)) with the default random initialization. "
            "No checkpoint download. model.eval() and torch.inference_mode() wrap the "
            "forward pass and generate(). input_ids are hand-written integers, not the "
            "output of a trained tokenizer. RSS is /proc/self/status VmRSS. "
            "cuda_peak_bytes is null because this run is on CPU. "
            "Do not treat generated_ids as language-model quality."
        ),
    }


def inspect_pretrained(model_name: str) -> dict:
    """Load a Hugging Face checkpoint. Failures stay NOT_EXECUTED."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        encoded = tokenizer("AstraRack BMC", return_tensors="pt")
        with torch.inference_mode():
            forward = model(**encoded)
        return {
            "status": "EXECUTED",
            "reason": None,
            "model_name": model_name,
            "pretrained": True,
            "metrics": {
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
                "input_ids": encoded["input_ids"][0].tolist(),
                "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]),
                "logits_shape": list(forward.logits.shape),
                "rss_bytes": process_rss_bytes(),
            },
            "methodology": (
                "Pretrained checkpoint loaded with AutoModelForCausalLM. "
                "One forward pass under torch.inference_mode(). No generation quality score."
            ),
        }
    except Exception as exc:
        return {
            "status": "NOT_EXECUTED",
            "reason": f"Could not load {model_name}: {exc}",
            "model_name": model_name,
            "pretrained": True,
            "metrics": None,
            "methodology": "The pretrained load failed. No shape or latency number was recorded.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Transformers forward pass.")
    parser.add_argument("--model", default="", help="Optional pretrained causal LM id.")
    args = parser.parse_args()
    print(EXPLANATION)
    reports = ROOT / "evaluation" / "reports"
    random_payload = inspect_random()
    random_payload["git_revision"] = git_revision(ROOT)
    random_payload["machine"] = machine_info()
    random_payload["dataset_version"] = "sample-v1"
    random_path = write_report(reports, "transformers_random_init", random_payload)
    print(f"{random_payload['status']} {random_path}")
    metrics = random_payload["metrics"]
    print(
        f"device={metrics['device']} dtype={metrics['dtype']} "
        f"parameters={metrics['parameter_count']} logits_shape={metrics['logits_shape']} "
        f"attention_shape={metrics['attention_shape']}"
    )
    if args.model:
        pretrained = inspect_pretrained(args.model)
        pretrained["git_revision"] = git_revision(ROOT)
        pretrained["machine"] = machine_info()
        pretrained["dataset_version"] = "sample-v1"
        path = write_report(reports, "transformers_pretrained", pretrained)
        print(f"{pretrained['status']} {path}")


if __name__ == "__main__":
    main()
