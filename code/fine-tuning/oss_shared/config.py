from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelRuntimeConfig:
    # Model identity
    family: str
    model_id: str

    # Primary tuning knobs (tuned per model scale/task fit)
    max_seq_length: int
    lora_r: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float

    # Other important runtime settings (kept explicit for reproducibility)
    load_in_4bit: bool
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: Tuple[str, ...]
    warmup_steps: int
    weight_decay: float
    max_new_tokens: int


DEFAULT_LORA_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


# Per-model tuned presets for the main OSS fine-tuning experiments.
MODEL_PRESETS: Dict[str, ModelRuntimeConfig] = {
    "qwen2.5-32b": ModelRuntimeConfig(
        # Model identity
        family="qwen2.5-32b",
        model_id="Qwen/Qwen2.5-32B-Instruct",

        # Primary tuning knobs
        max_seq_length=768,
        lora_r=16,
        learning_rate=7e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1.0,

        # Other important runtime settings
        load_in_4bit=True,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        warmup_steps=30,
        weight_decay=0.01,
        max_new_tokens=64,
    ),
    "qwen2.5-7b": ModelRuntimeConfig(
        # Model identity
        family="qwen2.5-7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",

        # Primary tuning knobs
        max_seq_length=768,
        lora_r=32,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1.5,

        # Other important runtime settings
        load_in_4bit=True,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        warmup_steps=40,
        weight_decay=0.01,
        max_new_tokens=64,
    ),
    "gptoss": ModelRuntimeConfig(
        # Model identity
        family="gptoss",
        model_id="openai/gpt-oss-20b",

        # Primary tuning knobs
        max_seq_length=768,
        lora_r=16,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1.0,

        # Other important runtime settings
        load_in_4bit=True,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        warmup_steps=30,
        weight_decay=0.01,
        max_new_tokens=64,
    ),
    "gemma-3-27b": ModelRuntimeConfig(
        # Model identity
        family="gemma-3-27b",
        model_id="google/gemma-3-27b-it",

        # Primary tuning knobs
        max_seq_length=768,
        lora_r=16,
        learning_rate=7e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1.0,

        # Other important runtime settings
        load_in_4bit=True,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        warmup_steps=30,
        weight_decay=0.01,
        max_new_tokens=64,
    ),
    "gemma-3-4b": ModelRuntimeConfig(
        # Model identity
        family="gemma-3-4b",
        model_id="google/gemma-3-4b-it",

        # Primary tuning knobs
        max_seq_length=768,
        lora_r=32,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1.5,

        # Other important runtime settings
        load_in_4bit=True,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        warmup_steps=40,
        weight_decay=0.01,
        max_new_tokens=64,
    ),
    "llama-3.1-8b": ModelRuntimeConfig(
        # Model identity
        family="llama-3.1-8b",
        model_id="meta-llama/Llama-3.1-8B-Instruct",

        # Primary tuning knobs
        max_seq_length=768,
        lora_r=32,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1.5,

        # Other important runtime settings
        load_in_4bit=True,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        warmup_steps=40,
        weight_decay=0.01,
        max_new_tokens=64,
    ),
    "llama-3.1-70b": ModelRuntimeConfig(
        # Model identity
        family="llama-3.1-70b",
        model_id="meta-llama/Llama-3.1-70B-Instruct",

        # Primary tuning knobs
        max_seq_length=768,
        lora_r=16,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1.0,

        # Other important runtime settings
        load_in_4bit=True,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        warmup_steps=40,
        weight_decay=0.01,
        max_new_tokens=64,
    ),
}

LEGACY_MODEL_FAMILY_ALIASES: Dict[str, str] = {
    "qwen": "qwen2.5-32b",
    "gemma": "gemma-3-27b",
    "llama": "llama-3.1-8b",
}

SUPPORTED_MODEL_FAMILIES: Tuple[str, ...] = tuple(sorted(MODEL_PRESETS.keys()))


def normalize_model_family(model_family: str) -> str:
    family = model_family.lower()
    return LEGACY_MODEL_FAMILY_ALIASES.get(family, family)


def get_model_config(model_family: str) -> ModelRuntimeConfig:
    family = normalize_model_family(model_family)
    if family not in MODEL_PRESETS:
        supported = ", ".join(SUPPORTED_MODEL_FAMILIES)
        raise ValueError(f"Unsupported model family '{model_family}'. Supported: {supported}")
    return MODEL_PRESETS[family]
