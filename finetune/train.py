"""LoRA and QLoRA training.

The base weights stay frozen. LoRA learns a low-rank update

    W' = W + B A

where A is (r, in) and B is (out, r), with r much smaller than the hidden
size. QLoRA stores the frozen base in 4-bit and trains the adapters in
higher precision. This function does not start training unless CUDA is
available. A CPU fallback would change the method and the memory story.
"""

from pathlib import Path

import torch

from finetune.dataset import build_pairs


def lora_config(rank: int = 8, alpha: int = 16):
    """PEFT config used when training actually starts."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def training_plan(*, qlora: bool = False) -> dict:
    """Describe the run that would execute on a GPU. This does not train."""
    pairs = build_pairs()
    quantizer = False
    if qlora:
        try:
            import bitsandbytes  # noqa: F401

            quantizer = True
        except ImportError:
            quantizer = False
    return {
        "examples": len(pairs),
        "rank": 8,
        "alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "base_weights": "frozen",
        "adapter": "low-rank matrices A and B",
        "formula": "W' = W + B A",
        "qlora_requested": qlora,
        "bitsandbytes_installed": quantizer,
        "cuda_available": bool(torch.cuda.is_available()),
        "golden_set": "held out; training questions are checked against it",
    }


def train_lora(
    model_name: str,
    output_dir: Path,
    *,
    qlora: bool = False,
    max_steps: int = 40,
) -> dict:
    """Train a LoRA adapter, or return NOT_EXECUTED when CUDA is absent.

    The training loop is real: it loads a causal LM, wraps it with PEFT, and
    runs AdamW on adapter parameters only. It is not invoked on a machine
    without CUDA.
    """
    plan = training_plan(qlora=qlora)
    if not torch.cuda.is_available():
        return {
            "status": "NOT_EXECUTED",
            "reason": (
                "REQUIRES GPU. torch.cuda.is_available() is false, so LoRA/QLoRA "
                "training did not start and no adapter was written. "
                "On a CUDA machine run: uv run python scripts/train_lora.py "
                "--model <causal-lm-id> --output models/lora"
            ),
            "plan": plan,
            "metrics": None,
        }
    if not model_name:
        return {
            "status": "NOT_EXECUTED",
            "reason": "TRANSFORMERS_MODEL or --model is empty.",
            "plan": plan,
            "metrics": None,
        }
    if qlora and not plan["bitsandbytes_installed"]:
        return {
            "status": "NOT_EXECUTED",
            "reason": "QLoRA was requested but bitsandbytes is not installed. Use uv sync --extra qlora.",
            "plan": plan,
            "metrics": None,
        }

    from peft import get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {}
    if qlora:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = get_peft_model(model, lora_config())
    model.train()

    class PairDataset(Dataset):
        def __init__(self) -> None:
            self.rows = build_pairs()

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            row = self.rows[index]
            text = f"Question: {row['question']}\nAnswer: {row['answer']}"
            encoded = tokenizer(text, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    loader = DataLoader(PairDataset(), batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
    steps = 0
    last_loss = None
    while steps < max_steps:
        for batch in loader:
            if steps >= max_steps:
                break
            batch = {key: value.to(model.device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            steps += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {
        "status": "EXECUTED",
        "reason": None,
        "plan": plan,
        "metrics": {"steps": steps, "last_loss": last_loss, "output_dir": str(output_dir)},
        "methodology": (
            "Loss is the causal LM token loss on the training pairs. "
            "It is not a golden-set score. Compare arms with scripts/compare_finetune.py."
        ),
    }
