import argparse
import json
import os
import re
import sys
from typing import Optional

import torch

sys.path.append(os.path.dirname(__file__))
from response_parser import parse_feature_response

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "fine-tuning", "oss_shared")
)
from config import SUPPORTED_MODEL_FAMILIES, get_model_config, normalize_model_family


def _resolve_output_suffix(output_suffix: Optional[str]) -> str:
    if output_suffix:
        sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", output_suffix).strip("-")
        if sanitized:
            return sanitized
    return str(os.getpid())


def _configure_generation(model) -> None:
    # Avoid repetitive warnings when greedy decoding ignores sampling flags.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, field):
                setattr(generation_config, field, None)


def _load_with_unsloth(resolved_model_id: str, cfg, adapter_dir: Optional[str]):
    try:
        import unsloth  # noqa: F401  # Must be imported before transformers/peft for patching.
        from peft import PeftModel
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Missing OSS inference dependencies. Install with: pip install -r code/requirements-oss.txt"
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolved_model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    FastLanguageModel.for_inference(model)
    _configure_generation(model)
    return model, tokenizer


def load_model_and_tokenizer(model_family: str, model_id: Optional[str], adapter_dir: Optional[str]):
    model_family = normalize_model_family(model_family)
    cfg = get_model_config(model_family)
    resolved_model_id = model_id or cfg.model_id

    if model_family == "gptoss":
        # Must be set before importing Unsloth in this process.
        # This avoids current gpt-oss inference crash in Unsloth temporary patches.
        if os.environ.get("UNSLOTH_COMPILE_DISABLE") != "1":
            os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
            print(
                "Info: Set UNSLOTH_COMPILE_DISABLE=1 for gptoss inference "
                "to avoid current Unsloth attention_mask generation bug."
            )
    model, tokenizer = _load_with_unsloth(
        resolved_model_id=resolved_model_id,
        cfg=cfg,
        adapter_dir=adapter_dir,
    )
    return model, tokenizer, cfg, resolved_model_id


def generate_features(
    review_text: str,
    system_prompt: str,
    model,
    tokenizer,
    cfg,
) -> list:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": review_text},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": False,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    response_text = tokenizer.decode(generated, skip_special_tokens=True)
    return parse_feature_response(response_text)


def infer_reviews(
    directory: str,
    model_family: str,
    model_id: Optional[str],
    adapter_dir: Optional[str],
    system_prompt: str,
    output_suffix: Optional[str] = None,
) -> str:
    model_family = normalize_model_family(model_family)
    base_dir = directory.rstrip("/")
    input_file = os.path.join(base_dir, "formatted_original_data", "test-set.json")
    output_dir = os.path.join(base_dir, "feature_extracted_data")
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    model, tokenizer, cfg, resolved_model_id = load_model_and_tokenizer(
        model_family=model_family, model_id=model_id, adapter_dir=adapter_dir
    )
    model_label = re.sub(r"[^a-z0-9]+", "-", model_family.lower()).strip("-")
    suffix = _resolve_output_suffix(output_suffix)
    output_file = os.path.join(output_dir, f"test-set-{model_label}-singleagent-{suffix}.json")

    for index, review in enumerate(reviews, start=1):
        review["review_number"] = index
        extracted = generate_features(
            review_text=review["input"],
            system_prompt=system_prompt,
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
        )
        review["extracted_features"] = extracted
        print(
            f"[{index}] model={resolved_model_id} extracted={extracted} "
            f"gt={review.get('output', [])}"
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-node local inference for OSS models")
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
    parser.add_argument("--model-id", default=None, help="Optional base model ID override")
    parser.add_argument("--adapter-dir", default=None, help="Optional LoRA adapter directory")
    parser.add_argument("--system-prompt", required=True, help="System prompt for extraction")
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Optional output filename suffix (defaults to current process PID)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help=argparse.SUPPRESS,  # Deprecated: output uses model-based filenames.
    )
    args = parser.parse_args()

    output_file = infer_reviews(
        directory=args.directory,
        model_family=args.model_family,
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        system_prompt=args.system_prompt,
        output_suffix=args.output_suffix,
    )
    print(f"Inference completed. Output saved to: {output_file}")


if __name__ == "__main__":
    main()
