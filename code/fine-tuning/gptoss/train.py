import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "oss_shared"))
from train_unsloth_lora import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-OSS LoRA fine-tuning wrapper")
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Dataset bin directory (e.g., ./data/in-domain/bin0/)",
    )
    parser.add_argument("--model-id", default=None, help="Optional model ID override")
    parser.add_argument("--output-root", default="artifacts/lora", help="LoRA output root")
    parser.add_argument("--run-id", default=None, help="Optional run id")
    args = parser.parse_args()

    output_dir = run_training(
        dataset_dir=args.directory.rstrip("/"),
        model_family="gptoss",
        model_id_override=args.model_id,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(f"GPT-OSS training completed. Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
