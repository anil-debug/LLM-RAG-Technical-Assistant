"""Train a LoRA adapter when CUDA is available. Otherwise write NOT_EXECUTED."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.settings import Settings
from evaluation.report import git_revision, machine_info, write_report
from finetune.dataset import write_jsonl
from finetune.train import train_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA training.")
    parser.add_argument("--model", default="")
    parser.add_argument("--output", default="models/lora")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()
    settings = Settings()
    dataset_path = ROOT / "data" / "sample" / "labels" / "train.jsonl"
    count = write_jsonl(dataset_path)
    print(f"wrote {count} training pairs to {dataset_path}")
    result = train_lora(
        args.model or settings.transformers_model,
        ROOT / args.output,
        qlora=args.qlora,
        max_steps=args.max_steps,
    )
    result["git_revision"] = git_revision(ROOT)
    result["machine"] = machine_info()
    result["dataset_version"] = settings.dataset_version
    result["model_name"] = args.model or settings.transformers_model or None
    result["configuration"] = result.get("plan")
    path = write_report(ROOT / "evaluation" / "reports", "lora_training", result)
    print(f"{result['status']} {path}")
    if result["status"] != "EXECUTED":
        print(result["reason"])


if __name__ == "__main__":
    main()
