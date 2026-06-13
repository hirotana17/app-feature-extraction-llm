import argparse
import inspect
import json
import os
from datetime import datetime
from typing import Optional

from datasets import load_dataset
import torch

from config import SUPPORTED_MODEL_FAMILIES, get_model_config, normalize_model_family

try:
    import unsloth  # noqa: F401  # Must be imported before transformers/peft.
    from unsloth import FastLanguageModel
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
except ImportError as exc:
    raise ImportError(
        "Missing OSS training dependencies. Install with: pip install -r code/requirements-oss.txt"
    ) from exc


DEFAULT_LOGGING_STEPS = 50
DEFAULT_EVAL_STEPS = 100
DEFAULT_SAVE_STEPS = 100
DEFAULT_SAVE_TOTAL_LIMIT = 3
DEFAULT_RANDOM_SEED = 42
EVAL_STABILITY_MODEL_FAMILY_PREFIXES = ("qwen", "gptoss", "gemma", "llama")


def _apply_eval_stability_workarounds(model_family: str, has_eval_ds: bool) -> None:
    if not any(
        model_family.startswith(prefix) for prefix in EVAL_STABILITY_MODEL_FAMILY_PREFIXES
    ) or not has_eval_ds:
        return
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        # Work around intermittent eval-time cuDNN SDPA plan failures on current stack.
        torch.backends.cuda.enable_cudnn_sdp(False)
        print(f"Info: Disabled cuDNN SDPA for {model_family} eval stability.")


def _format_chat_records(dataset, tokenizer):
    def format_example(example):
        messages = json.loads(example["text"])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"formatted_text": text}

    return dataset.map(format_example, desc="Formatting chat templates")


def _resolve_text_tokenizer(tokenizer):
    if hasattr(tokenizer, "pad"):
        return tokenizer
    nested_tokenizer = getattr(tokenizer, "tokenizer", None)
    if nested_tokenizer is not None and hasattr(nested_tokenizer, "pad"):
        return nested_tokenizer
    raise TypeError(
        f"Tokenizer of type {type(tokenizer).__name__} is not pad-compatible and has no nested tokenizer."
    )


def _tokenize_chat_records(dataset, tokenizer, max_seq_length: int):
    def tokenize_example(example):
        tokenized = tokenizer(
            example["formatted_text"],
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    return dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT text",
    )


def run_training(
    dataset_dir: str,
    model_family: str,
    model_id_override: Optional[str] = None,
    output_root: str = "artifacts/lora",
    run_id: Optional[str] = None,
) -> str:
    model_family = normalize_model_family(model_family)
    cfg = get_model_config(model_family)
    model_id = model_id_override or cfg.model_id

    train_data_dir = os.path.join(dataset_dir, "fine_tuning_data")
    train_path = os.path.join(train_data_dir, "train-set-sft-promptft1.jsonl")
    eval_path = os.path.join(train_data_dir, "validation-set-sft-promptft1.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
    )
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
    )

    train_ds = load_dataset("json", data_files=train_path, split="train")
    train_ds = _format_chat_records(train_ds, tokenizer)
    train_ds = _tokenize_chat_records(train_ds, text_tokenizer, cfg.max_seq_length)

    eval_ds = None
    if os.path.exists(eval_path):
        eval_ds = load_dataset("json", data_files=eval_path, split="train")
        eval_ds = _format_chat_records(eval_ds, tokenizer)
        eval_ds = _tokenize_chat_records(eval_ds, text_tokenizer, cfg.max_seq_length)
    if model_family == "gptoss" and eval_ds is not None:
        # Workaround for current Unsloth gpt-oss eval path bug:
        # during Trainer.evaluate(), attention_mask can be treated as 4D
        # inside unsloth_zoo temporary patch and crash with IndexError.
        print(
            "Warning: Disabling in-training eval for gptoss due to Unsloth eval attention_mask bug. "
            "Training will continue and final task evaluation is still done in pipeline step [4/4]."
        )
        eval_ds = None
    _apply_eval_stability_workarounds(model_family=model_family, has_eval_ds=eval_ds is not None)

    run_label = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, model_family, os.path.basename(dataset_dir), run_label)
    os.makedirs(output_dir, exist_ok=True)

    training_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "warmup_steps": cfg.warmup_steps,
        "weight_decay": cfg.weight_decay,
        "load_best_model_at_end": eval_ds is not None,
        "report_to": "none",
        "bf16": True,
        "logging_steps": DEFAULT_LOGGING_STEPS,
        "save_steps": DEFAULT_SAVE_STEPS,
        "save_total_limit": DEFAULT_SAVE_TOTAL_LIMIT,
        "seed": DEFAULT_RANDOM_SEED,
        "data_seed": DEFAULT_RANDOM_SEED,
    }
    eval_key = (
        "evaluation_strategy"
        if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "eval_strategy"
    )
    training_kwargs[eval_key] = "steps" if eval_ds is not None else "no"
    if eval_ds is not None:
        training_kwargs["eval_steps"] = DEFAULT_EVAL_STEPS
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["greater_is_better"] = False
        if any(
            model_family.startswith(prefix) for prefix in EVAL_STABILITY_MODEL_FAMILY_PREFIXES
        ):
            training_kwargs["per_device_eval_batch_size"] = 1
            print(f"Info: Using per_device_eval_batch_size=1 for {model_family} eval stability.")
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": training_args,
        "data_collator": DataCollatorForLanguageModeling(tokenizer=text_tokenizer, mlm=False),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = text_tokenizer
    else:
        # Backward compatibility with older Transformers versions.
        trainer_kwargs["tokenizer"] = text_tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    final_output_dir = os.path.join(output_dir, "final")
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    metadata = {
        "model_family": model_family,
        "model_id": model_id,
        "dataset_dir": dataset_dir,
        "train_path": train_path,
        "eval_path": eval_path if os.path.exists(eval_path) else None,
        "output_dir": final_output_dir,
        "training_args": {
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_train_epochs,
            "warmup_steps": cfg.warmup_steps,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "max_seq_length": cfg.max_seq_length,
            "logging_steps": DEFAULT_LOGGING_STEPS,
            "eval_steps": DEFAULT_EVAL_STEPS if eval_ds is not None else None,
            "per_device_eval_batch_size": (
                training_kwargs.get("per_device_eval_batch_size") if eval_ds is not None else None
            ),
            "save_steps": DEFAULT_SAVE_STEPS,
            "save_total_limit": DEFAULT_SAVE_TOTAL_LIMIT,
            "seed": DEFAULT_RANDOM_SEED,
        },
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return final_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA adapters with Unsloth + PEFT")
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Dataset bin directory (e.g., ./data/in-domain/bin0/)",
    )
    parser.add_argument(
        "--model-family",
        required=True,
        choices=SUPPORTED_MODEL_FAMILIES,
        help="Model family preset to use",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional override model ID",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/lora",
        help="Root output directory for LoRA artifacts",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id for reproducible artifact paths",
    )
    args = parser.parse_args()

    output_dir = run_training(
        dataset_dir=args.directory.rstrip("/"),
        model_family=args.model_family,
        model_id_override=args.model_id,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(f"Training completed. Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
