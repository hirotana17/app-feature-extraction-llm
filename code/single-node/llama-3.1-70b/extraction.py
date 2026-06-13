import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import prompt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "oss_shared"))
from infer_local_single import infer_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama 3.1 70B single-node extraction wrapper")
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Dataset bin directory (e.g., ./data/in-domain/bin0/)",
    )
    parser.add_argument("--model-id", default=None, help="Optional model ID override")
    parser.add_argument("--adapter_dir", default=None, help="LoRA adapter directory")
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Optional output filename suffix (defaults to PID)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help=argparse.SUPPRESS,  # Deprecated: kept for backward compatibility.
    )
    args = parser.parse_args()

    output_file = infer_reviews(
        directory=args.directory,
        model_family="llama-3.1-70b",
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        system_prompt=prompt.PROMPTFT1,
        output_suffix=args.output_suffix,
    )
    print(f"Llama 3.1 70B extraction completed. Output saved to: {output_file}")


if __name__ == "__main__":
    main()
