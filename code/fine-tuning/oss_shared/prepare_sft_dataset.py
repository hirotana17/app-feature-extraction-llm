import argparse
import json
import os
import random
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import prompt


SYSTEM_PROMPT = prompt.PROMPTFT1
PROMPT_NAME = "promptft1"
TRAIN_OUTPUT_DIR = "fine_tuning_data"
VALIDATION_SPLIT_NAME = "validation-set"
DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42


def create_chat_messages(review: Dict) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": review["input"]},
        {"role": "assistant", "content": json.dumps(review["output"], ensure_ascii=False)},
    ]


def convert_reviews_to_sft_records(reviews: Iterable[Dict]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for review in reviews:
        messages = create_chat_messages(review)
        # Text field is used by SFTTrainer with tokenizer chat template during training.
        text = json.dumps(messages, ensure_ascii=False)
        records.append({"text": text})
    return records


def write_jsonl(records: Iterable[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_reviews(base_dir: str, split_name: str) -> List[Dict]:
    input_path = os.path.join(base_dir, "formatted_original_data", f"{split_name}.json")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_output_dir(split_name: str) -> str:
    if split_name in {"train-set", VALIDATION_SPLIT_NAME}:
        return TRAIN_OUTPUT_DIR
    raise ValueError(f"Unsupported split name: {split_name}")


def process_split(base_dir: str, split_name: str, reviews: Sequence[Dict] = None) -> str:
    if reviews is None:
        reviews = load_reviews(base_dir, split_name)
    records = convert_reviews_to_sft_records(reviews)
    output_path = os.path.join(
        base_dir, resolve_output_dir(split_name), f"{split_name}-sft-{PROMPT_NAME}.jsonl"
    )
    write_jsonl(records, output_path)
    return output_path


def split_train_reviews(
    reviews: Sequence[Dict], validation_ratio: float, split_seed: int
) -> Tuple[List[Dict], List[Dict]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            f"validation_ratio must be in (0.0, 1.0). Received: {validation_ratio}"
        )
    if len(reviews) < 2:
        raise ValueError("At least 2 train examples are required for validation split.")

    val_count = int(len(reviews) * validation_ratio)
    val_count = max(1, min(len(reviews) - 1, val_count))

    indices = list(range(len(reviews)))
    random.Random(split_seed).shuffle(indices)
    val_indices = set(indices[:val_count])

    train_reviews = [review for idx, review in enumerate(reviews) if idx not in val_indices]
    val_reviews = [review for idx, review in enumerate(reviews) if idx in val_indices]
    return train_reviews, val_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT JSONL data for OSS fine-tuning")
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Dataset bin directory (e.g., ./data/in-domain/bin0/)",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help=(
            "Fraction of train-set to reserve as validation and write to "
            "fine_tuning_data/validation-set-sft-promptft1.jsonl (default: 0.1)"
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Random seed for train/validation split (default: 42)",
    )
    args = parser.parse_args()

    base_dir = args.directory.rstrip("/")
    train_reviews = load_reviews(base_dir, "train-set")
    train_reviews, val_reviews = split_train_reviews(
        train_reviews, validation_ratio=args.validation_ratio, split_seed=args.split_seed
    )
    train_output = process_split(base_dir, "train-set", reviews=train_reviews)
    print(f"Train SFT dataset saved to: {train_output}")
    print(f"Train examples: {len(train_reviews)}")

    val_output = process_split(base_dir, VALIDATION_SPLIT_NAME, reviews=val_reviews)
    print(f"Validation SFT dataset saved to: {val_output}")
    print(f"Validation examples: {len(val_reviews)}")
    print(
        f"Split details: validation_ratio={args.validation_ratio}, split_seed={args.split_seed}"
    )


if __name__ == "__main__":
    main()
