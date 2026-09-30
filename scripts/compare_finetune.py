"""Write the four-arm fine-tune comparison, or NOT_EXECUTED when it cannot run."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.settings import Settings
from evaluation.report import git_revision, machine_info, write_report
from finetune.compare import comparison_status


def main() -> None:
    settings = Settings()
    adapter = ROOT / "models" / "lora"
    generator_ready = False
    payload = comparison_status(adapter_ready=adapter.exists(), generator_ready=generator_ready)
    payload.update(
        {
            "git_revision": git_revision(ROOT),
            "machine": machine_info(),
            "dataset_version": settings.dataset_version,
            "model_name": settings.transformers_model or settings.llm_model,
            "configuration": {
                "arms": payload.get("arms"),
                "adapter_path": str(adapter),
                "held_out_dataset": "evaluation/datasets/golden.jsonl",
            },
            "methodology": (
                "When executed, each arm answers the golden questions. "
                "base and finetuned see no retrieved chunks. "
                "base_rag and finetuned_rag see the same retrieved context. "
                "Training loss is not copied into these metrics."
            ),
        }
    )
    path = write_report(ROOT / "evaluation" / "reports", "finetune_comparison", payload)
    print(f"{payload['status']} {path}")


if __name__ == "__main__":
    main()
